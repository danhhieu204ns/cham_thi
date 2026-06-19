param(
    [string]$Python = "",
    [string]$RunId = "",
    [switch]$Sample,
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$BaselineRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Push-Location $BaselineRoot

try {
    if ([string]::IsNullOrWhiteSpace($Python)) {
        $VenvPython = Join-Path $BaselineRoot ".venv\Scripts\python.exe"
        if (Test-Path $VenvPython) {
            $Python = $VenvPython
        } else {
            $Python = "python"
        }
    }

    if ([string]::IsNullOrWhiteSpace($RunId)) {
        $RunId = Get-Date -Format "yyyyMMdd_HHmmss"
    }

    $RunDir = "reports\test_runs\$RunId"
    New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

    $SelectionArgs = @("--all")
    if ($Sample) {
        $SelectionArgs = @()
    }

    function Invoke-PipelineStep {
        param(
            [Parameter(Mandatory = $true)]
            [string]$Name,
            [Parameter(Mandatory = $true)]
            [string]$Script,
            [Parameter(Mandatory = $true)]
            [string[]]$Arguments
        )

        Write-Host ""
        Write-Host "==> $Name"
        & $Python $Script @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "$Name failed with exit code $LASTEXITCODE"
        }
    }

    Write-Host "Baseline root: $BaselineRoot"
    Write-Host "Python: $Python"
    Write-Host "Run dir: $RunDir"
    if ($Sample) {
        Write-Host "Selection: data\labels\template_samples.txt"
    } else {
        Write-Host "Selection: all ok sheets"
    }

    Invoke-PipelineStep "Extract sheets" "scripts\extract_sheets.py" @(
        @($SelectionArgs)
        "--output-jsonl", "data\processed\results\sheet_extraction_baseline.jsonl"
        "--warped-output-dir", "data\processed\warped\sheet_extraction"
        "--crop-output-dir", "data\processed\crops\sheet_extraction"
        "--run-dir", $RunDir
    )

    Invoke-PipelineStep "Build web demo data" "scripts\build_web_demo_data.py" @(
        "--extraction-jsonl", "data\processed\results\sheet_extraction_baseline.jsonl"
        "--run-dir", $RunDir
    )

    if (-not $SkipTests) {
        Invoke-PipelineStep "Run tests" "-m" @("unittest", "discover", "-s", "tests")
    }

    Write-Host ""
    Write-Host "Baseline completed."
    Write-Host "Run dir: $RunDir"
    Write-Host "Extraction output: data\processed\results\sheet_extraction_baseline.jsonl"
    Write-Host "Web demo data: ..\web_demo\data\demo_data.json"
} finally {
    Pop-Location
}
