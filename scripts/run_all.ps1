<#
.SYNOPSIS
    Windows/PowerShell equivalent of scripts/run_eval.sh + scripts/build_report.sh.

.DESCRIPTION
    Runs the full pipeline locally: Hugging Face preflight, data sanity check,
    checkpoint sweep, figures/tables, and the LaTeX build.  Sets HF_HOME to the
    in-project cache so nothing lands in the user profile directory.

.EXAMPLE
    .\scripts\run_all.ps1
    .\scripts\run_all.ps1 -NAnswerable 100 -Revisions @('step1000','step143000')
#>
[CmdletBinding()]
param(
    [string]   $Model         = 'dhgottesman/LMEnt-170M-6E',
    [string[]] $Revisions     = @('step10000','step40000','step100000','step200000','step400000','step658032'),
    [int]      $NAnswerable   = 300,
    [int]      $NAdversarial  = 120,
    [int]      $BatchSize     = 16,
    [int]      $MaxNewTokens  = 16,
    [string]   $PromptStyle   = 'abstain_fewshot',
    [string]   $Device        = 'auto',
    [string]   $DeviceMap     = '',
    [string]   $Dtype         = 'float32',
    [string]   $OutputDir     = 'results',
    [string]   $PlotsDir      = 'plots',
    [string]   $TablesDir     = 'report\tables',
    [int]      $Limit         = 0,
    [switch]   $Overwrite,
    [switch]   $Pilot,
    [switch]   $SkipPreflight,
    [switch]   $SkipReport
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

# ---- Persistent Hugging Face cache --------------------------------------------------
$env:HF_HOME                     = Join-Path $ProjectRoot '.cache\huggingface'
$env:HUGGINGFACE_HUB_CACHE       = Join-Path $env:HF_HOME 'hub'
$env:HF_DATASETS_CACHE           = Join-Path $env:HF_HOME 'datasets'
$env:TRANSFORMERS_CACHE          = $env:HUGGINGFACE_HUB_CACHE
$env:TOKENIZERS_PARALLELISM      = 'false'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:PYTHONUNBUFFERED            = '1'
$env:PYTHONPATH                  = Join-Path $ProjectRoot 'src'

foreach ($dir in @($env:HF_HOME, $env:HUGGINGFACE_HUB_CACHE, $env:HF_DATASETS_CACHE,
                   'logs', $OutputDir, 'plots', 'data')) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

$Py = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $Py)) { $Py = 'python' }

function Invoke-Native {
    <#
      Run a native executable and judge success by its exit code only.
      Native programs (Python, pdflatex, the HF Hub client) write warnings and
      progress bars to stderr; under $ErrorActionPreference = 'Stop' PowerShell
      would otherwise convert those lines into terminating errors whenever the
      output is redirected or transcribed.
    #>
    param([string]$Exe, [string[]]$Arguments, [switch]$Quiet)
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        # Write-Host keeps the function's output stream clean so that the exit
        # code is the only value returned to the caller.
        if ($Quiet) { & $Exe @Arguments 2>&1 | Out-Null }
        else {
            & $Exe @Arguments 2>&1 | ForEach-Object {
                if ($_ -is [System.Management.Automation.ErrorRecord]) { Write-Host $_.Exception.Message }
                else { Write-Host "$_" }
            }
        }
    } finally {
        $ErrorActionPreference = $prev
    }
    return $LASTEXITCODE
}

function Invoke-Step {
    param([string]$Label, [string[]]$Arguments)
    Write-Host ''
    Write-Host ">>> $Label" -ForegroundColor Cyan
    $code = Invoke-Native -Exe $Py -Arguments $Arguments
    if ($code -ne 0) { throw "$Label failed with exit code $code" }
}

Write-Host '=====================================================================' -ForegroundColor Green
Write-Host ' Knowledge and Hallucinations across Training Dynamics'  -ForegroundColor Green
Write-Host '=====================================================================' -ForegroundColor Green
Write-Host " project  : $ProjectRoot"
Write-Host " python   : $Py"
Write-Host " HF_HOME  : $($env:HF_HOME)"
Write-Host " model    : $Model"
Write-Host " revisions: $($Revisions -join ' ')"
Write-Host " device   : $Device$(if ($DeviceMap) { " (device_map=$DeviceMap)" })   dtype: $Dtype"
Write-Host " output   : $OutputDir$(if ($Limit -gt 0) { "   (limit $Limit examples per checkpoint)" })"

$started = Get-Date

if (-not $SkipPreflight) { Invoke-Step '[1/4] Hugging Face preflight' @('src\setup_hf.py') }

Invoke-Step '[2/4] Data sanity check' @('src\data.py', '--sanity-check', '--n', '10')
Invoke-Step '[2/4] Build evaluation set' @(
    'src\data.py', '--build',
    '--n-answerable', "$NAnswerable", '--n-adversarial', "$NAdversarial")

$evalArgs = @(
    'src\evaluator.py',
    '--model', $Model,
    '--revisions') + $Revisions + @(
    '--device', $Device,
    '--dtype', $Dtype,
    '--batch-size', "$BatchSize",
    '--max-new-tokens', "$MaxNewTokens",
    '--n-answerable', "$NAnswerable",
    '--n-adversarial', "$NAdversarial",
    '--prompt-style', $PromptStyle,
    '--output-dir', $OutputDir)
if ($DeviceMap)  { $evalArgs += @('--device-map', $DeviceMap) }
if ($Limit -gt 0) { $evalArgs += @('--limit', "$Limit") }
if ($Overwrite)  { $evalArgs += '--overwrite' }
Invoke-Step '[3/4] Checkpoint sweep' $evalArgs

if ($Pilot) {
    foreach ($rev in @($Revisions[0], $Revisions[-1]) | Select-Object -Unique) {
        Invoke-Step "[3b] Prompt-format pilot @ $rev" @(
            'scripts\probe_prompt.py', '--model', $Model, '--revision', $rev, '--n', '40',
            '--device', $Device, '--dtype', $Dtype, '--batch-size', "$BatchSize",
            '--output', "logs\prompt_pilot_$rev.json")
    }
}

Invoke-Step '[4/4] Analysis, figures and tables' @(
    'scripts\plot_results.py', '--results-dir', $OutputDir,
    '--plots-dir', $PlotsDir, '--tables-dir', $TablesDir)

if (-not $SkipReport) {
    Write-Host ''
    Write-Host '>>> Compiling report/main.tex' -ForegroundColor Cyan
    Push-Location (Join-Path $ProjectRoot 'report')
    try {
        $code = Invoke-Native -Exe 'pdflatex' -Arguments @('-interaction=nonstopmode', '-halt-on-error', 'main.tex') -Quiet
        if ($code -ne 0) { throw "pdflatex (pass 1) failed with exit code $code; see report\main.log" }
        Invoke-Native -Exe 'bibtex'   -Arguments @('main') -Quiet | Out-Null   # warnings are non-fatal
        Invoke-Native -Exe 'pdflatex' -Arguments @('-interaction=nonstopmode', 'main.tex') -Quiet | Out-Null
        Invoke-Native -Exe 'pdflatex' -Arguments @('-interaction=nonstopmode', 'main.tex') -Quiet | Out-Null
        if (Test-Path 'main.pdf') {
            Write-Host "    -> $(Join-Path $PWD 'main.pdf')" -ForegroundColor Green
        } else {
            Write-Warning 'main.pdf was not produced; inspect report/main.log'
        }
    } finally { Pop-Location }
}

$elapsed = (Get-Date) - $started
Write-Host ''
Write-Host '=====================================================================' -ForegroundColor Green
Write-Host (" Completed in {0:mm}m {0:ss}s" -f $elapsed) -ForegroundColor Green
Write-Host " results -> $([System.IO.Path]::Combine($ProjectRoot, $OutputDir))"
Write-Host " figures -> $([System.IO.Path]::Combine($ProjectRoot, $PlotsDir))"
Write-Host " paper   -> $ProjectRoot\report\main.pdf"
Write-Host '=====================================================================' -ForegroundColor Green
