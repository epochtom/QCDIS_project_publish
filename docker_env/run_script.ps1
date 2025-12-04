# PowerShell script to execute any training script in Docker
# Usage: .\run_script.ps1 <script_name> [additional_args]

param(
    [Parameter(Mandatory=$true)]
    [string]$ScriptName,
    
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$AdditionalArgs
)

# Check if script exists
$scriptPath = Join-Path (Split-Path $PSScriptRoot -Parent) "universal_training_script\$ScriptName"
if (-not (Test-Path $scriptPath)) {
    Write-Host "Error: Script '$ScriptName' not found in universal_training_script folder" -ForegroundColor Red
    exit 1
}

# Build Docker image if not exists
$imageExists = docker images universal-training:latest --format "{{.Repository}}:{{.Tag}}" 2>$null
if (-not $imageExists) {
    Write-Host "Building Docker image..." -ForegroundColor Yellow
    $contextPath = Split-Path $PSScriptRoot -Parent
    docker build -t universal-training:latest -f "$PSScriptRoot\Dockerfile" $contextPath
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Failed to build Docker image" -ForegroundColor Red
        exit 1
    }
}

# Prepare arguments
$allArgs = @()
if ($AdditionalArgs) {
    $allArgs = $AdditionalArgs
}

# Get absolute paths
$uploadPath = Join-Path (Split-Path $PSScriptRoot -Parent) "upload"
$outputPath = Join-Path (Split-Path $PSScriptRoot -Parent) "output"
$scriptDir = Join-Path (Split-Path $PSScriptRoot -Parent) "universal_training_script"

# Create directories if they don't exist
if (-not (Test-Path $uploadPath)) { New-Item -ItemType Directory -Path $uploadPath | Out-Null }
if (-not (Test-Path $outputPath)) { New-Item -ItemType Directory -Path $outputPath | Out-Null }

# Run the script in Docker with 24-hour timeout
Write-Host "Executing $ScriptName in Docker container..." -ForegroundColor Green
Write-Host "Timeout: 24 hours" -ForegroundColor Green
Write-Host ""

$dockerArgs = @(
    "run", "--rm",
    "--name", "universal-training-run",
    "--stop-timeout", "86400",
    "-v", "${uploadPath}:/app/data:ro",
    "-v", "${outputPath}:/app/output:rw",
    "-v", "${scriptDir}:/app/universal_training_script:ro",
    "-w", "/app",
    "universal-training:latest",
    "python", "universal_training_script/$ScriptName"
)

if ($allArgs.Count -gt 0) {
    $dockerArgs += $allArgs
}

docker $dockerArgs

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Execution completed successfully!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "Execution failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

