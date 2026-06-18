param(
    [string]$Python = "",
    [string]$RunId = "",
    [switch]$Sample
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$BaselineRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$RepoRoot = Resolve-Path (Join-Path $BaselineRoot "..")
Push-Location $BaselineRoot

try {
    if ([string]::IsNullOrWhiteSpace($Python)) {
        $ClassifierPython = Join-Path $RepoRoot "bubble_classifier\.venv\Scripts\python.exe"
        $BaselinePython = Join-Path $BaselineRoot ".venv\Scripts\python.exe"
        if (Test-Path $ClassifierPython) {
            $Python = $ClassifierPython
        } elseif (Test-Path $BaselinePython) {
            $Python = $BaselinePython
        } else {
            $Python = "python"
        }
    }

    if ([string]::IsNullOrWhiteSpace($RunId)) {
        $RunId = Get-Date -Format "yyyyMMdd_HHmmss"
    }

    $RunDir = "reports\compare_methods\$RunId"
    $RuleRunDir = Join-Path $RunDir "rule"
    $ClassifierRunDir = Join-Path $RunDir "classifier"
    $CompareDir = Join-Path $RunDir "comparison"
    New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

    $SelectionArgs = @("--all")
    if ($Sample) {
        $SelectionArgs = @()
    }

    function Invoke-Step {
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

    Invoke-Step "Extract rule baseline" "scripts\extract_sheets.py" @(
        @($SelectionArgs)
        "--output-jsonl", (Join-Path $RuleRunDir "extraction.jsonl")
        "--warped-output-dir", (Join-Path $RuleRunDir "warped")
        "--run-dir", $RuleRunDir
        "--visual-limit", "10"
    )

    Invoke-Step "Extract classifier" "scripts\extract_sheets.py" @(
        @($SelectionArgs)
        "--bubble-classifier"
        "--output-jsonl", (Join-Path $ClassifierRunDir "extraction.jsonl")
        "--warped-output-dir", (Join-Path $ClassifierRunDir "warped")
        "--run-dir", $ClassifierRunDir
        "--visual-limit", "10"
    )

    Invoke-Step "Compare methods" "scripts\compare_extractions.py" @(
        "--method-a", (Join-Path $RuleRunDir "extraction.jsonl")
        "--method-b", (Join-Path $ClassifierRunDir "extraction.jsonl")
        "--label-a", "rule"
        "--label-b", "classifier"
        "--output-dir", $CompareDir
    )

    Invoke-Step "Visualize comparison" "scripts\visualize_comparison.py" @(
        $CompareDir
    )

    Write-Host ""
    Write-Host "Comparison completed."
    Write-Host "Summary: $(Join-Path $CompareDir "summary.md")"
    Write-Host "Differences: $(Join-Path $CompareDir "differences.csv")"
    Write-Host "Side-by-side: $(Join-Path $CompareDir "side_by_side.html")"
} finally {
    Pop-Location
}
