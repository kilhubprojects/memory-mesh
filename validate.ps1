# MemoryMesh validation script
# Runs: uv sync, ruff check, pytest, benchmark imports

$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "MemoryMesh — Test & Lint Validation" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: uv sync
Write-Host "[1/5] Running 'uv sync'…" -ForegroundColor Yellow
Set-Location $projectDir
& uv sync 2>&1 | Tee-Object -Variable syncOutput
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: uv sync failed with code $LASTEXITCODE" -ForegroundColor Red
    exit 1
}
Write-Host "✓ uv sync completed" -ForegroundColor Green
Write-Host ""

# Step 2: ruff check
Write-Host "[2/5] Running 'uv run ruff check .'…" -ForegroundColor Yellow
& uv run ruff check . 2>&1 | Tee-Object -Variable ruffOutput
$ruffCode = $LASTEXITCODE
if ($ruffCode -ne 0) {
    Write-Host "ERROR: ruff found issues (exit code: $ruffCode)" -ForegroundColor Red
    Write-Host ""
    Write-Host "Ruff output:" -ForegroundColor Red
    Write-Host $ruffOutput
    exit 1
}
Write-Host "✓ ruff check passed (0 errors)" -ForegroundColor Green
Write-Host ""

# Step 3: pytest
Write-Host "[3/5] Running 'uv run pytest tests/ -q --tb=short'…" -ForegroundColor Yellow
& uv run pytest tests/ -q --tb=short 2>&1 | Tee-Object -Variable pytestOutput
$pytestCode = $LASTEXITCODE
Write-Host ""
if ($pytestCode -ne 0 -and $pytestCode -ne 5) {
    Write-Host "ERROR: pytest failed with exit code $pytestCode" -ForegroundColor Red
    Write-Host ""
    Write-Host "Pytest output:" -ForegroundColor Red
    Write-Host $pytestOutput
    exit 1
}
Write-Host "✓ pytest completed" -ForegroundColor Green
Write-Host ""

# Step 4: Check benchmark imports
Write-Host "[4/5] Checking benchmark file imports…" -ForegroundColor Yellow
$benchmarkFiles = @(
    "benchmarks.bench_indexing",
    "benchmarks.bench_search_latency",
    "benchmarks.bench_embedding_models"
)

foreach ($module in $benchmarkFiles) {
    Write-Host "  • Testing import: $module" -ForegroundColor Gray
    & uv run python -c "import $module" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to import $module" -ForegroundColor Red
        exit 1
    }
}
Write-Host "✓ All benchmark files importable" -ForegroundColor Green
Write-Host ""

# Summary
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Summary" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "✓ uv sync: success" -ForegroundColor Green
Write-Host "✓ ruff check: 0 errors" -ForegroundColor Green

# Parse pytest output for counts
if ($pytestOutput -match "(\d+) passed") {
    $passed = [int]$matches[1]
} else {
    $passed = 0
}
if ($pytestOutput -match "(\d+) skipped") {
    $skipped = [int]$matches[1]
} else {
    $skipped = 0
}
if ($pytestOutput -match "(\d+) failed") {
    $failed = [int]$matches[1]
} else {
    $failed = 0
}

Write-Host "✓ pytest: $passed passed, $skipped skipped, $failed failed" -ForegroundColor Green
Write-Host "✓ benchmarks: 3/3 files importable" -ForegroundColor Green
Write-Host ""
Write-Host "All checks passed!" -ForegroundColor Green
