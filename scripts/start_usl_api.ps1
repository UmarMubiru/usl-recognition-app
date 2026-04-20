param(
    [string]$PythonPath = ".\.venv\Scripts\python.exe",
    [string]$ApiKey = "",
    [string]$ModelArtifactPath = "",
    [string]$FallbackModelArtifactPath = "",
    [double]$FallbackConfidenceThreshold = 0.75,
    [bool]$EnableFallback = $true,
    [string]$ApiHost = "0.0.0.0",
    [int]$Port = 8000
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$workspaceRoot = Split-Path -Parent $PSScriptRoot
Set-Location $workspaceRoot

$pythonResolved = if ([System.IO.Path]::IsPathRooted($PythonPath)) {
    $PythonPath
}
else {
    Join-Path $workspaceRoot $PythonPath
}

if (-not (Test-Path $pythonResolved)) {
    throw "Python executable not found: $pythonResolved"
}

if ($ApiKey -ne "") {
    $env:API_KEY = $ApiKey
}

if ($ModelArtifactPath -ne "") {
    $env:MODEL_ARTIFACT_PATH = $ModelArtifactPath
}

if ($FallbackModelArtifactPath -ne "") {
    $env:FALLBACK_MODEL_ARTIFACT_PATH = $FallbackModelArtifactPath
}

$env:FALLBACK_CONFIDENCE_THRESHOLD = [string]$FallbackConfidenceThreshold
$env:ENABLE_FALLBACK = if ($EnableFallback) { "true" } else { "false" }

& $pythonResolved -m uvicorn models_dataset1.deployment.api:app --host $ApiHost --port $Port
