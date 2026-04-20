param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('export', 'serve', 'health', 'predict-smoke', 'compose-up', 'compose-down', 'ci-local', 'svc-install', 'svc-start', 'svc-stop', 'svc-status', 'svc-remove', 'svc-logs')]
    [string]$Action,

    [string]$PythonPath = ".\.venv\Scripts\python.exe",
    [string]$ApiHost = "127.0.0.1",
    [int]$Port = 8000,
    [string]$ApiKey = "",
    [string]$ModelArtifactPath = "",
    [string]$FallbackModelArtifactPath = "",
    [double]$FallbackConfidenceThreshold = 0.75,
    [bool]$EnableFallback = $true,
    [string]$Metric = "test_accuracy",
    [string]$ServiceName = "USLDiseaseAPI"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Assert-PathExists {
    param([string]$PathToCheck, [string]$Label)
    if (-not (Test-Path $PathToCheck)) {
        throw "$Label not found: $PathToCheck"
    }
}

function Assert-DockerAvailable {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw "Docker CLI is not installed or not in PATH."
    }
}

function Invoke-Export {
    Assert-PathExists $PythonPath "Python executable"
    & $PythonPath "models_dataset1\csv_models\export_best_model.py" --metric $Metric --augment
}

function Invoke-Serve {
    Assert-PathExists $PythonPath "Python executable"
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
    & $PythonPath -m uvicorn models_dataset1.deployment.api:app --host $ApiHost --port $Port
}

function Invoke-Health {
    $url = "http://$ApiHost`:$Port/health"
    (Invoke-WebRequest -UseBasicParsing -Uri $url).Content
}

function Invoke-PredictSmoke {
    $url = "http://$ApiHost`:$Port/predict"
    $body = @{ features = @{ dummy = 0.0 }; top_k = 3 } | ConvertTo-Json -Depth 5

    if ($ApiKey -ne "") {
        $headers = @{ "x-api-key" = $ApiKey }
        Invoke-RestMethod -Method Post -Uri $url -Headers $headers -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 6
    }
    else {
        Invoke-RestMethod -Method Post -Uri $url -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 6
    }
}

function Invoke-ComposeUp {
    Assert-DockerAvailable
    docker compose up -d --build
}

function Invoke-ComposeDown {
    Assert-DockerAvailable
    docker compose down
}

function Invoke-CiLocal {
    Assert-PathExists $PythonPath "Python executable"

    & $PythonPath -m pip install -r "models_dataset1\deployment\requirements.txt"
    & $PythonPath -m pip install pandas
    & $PythonPath "models_dataset1\csv_models\export_best_model.py" --metric $Metric --augment
    & $PythonPath -c "import models_dataset1.deployment.api as api; print(api.MODEL_NAME, len(api.FEATURE_COLS))"

    $testSnippet = @'
from fastapi.testclient import TestClient
from models_dataset1.deployment.api import app

client = TestClient(app)

ready = client.get('/ready')
assert ready.status_code == 200, ready.text

payload = {'features': {'dummy': 0.0}, 'top_k': 2}
pred = client.post('/predict', json=payload)
assert pred.status_code == 200, pred.text

print('local-ci-smoke-ok')
'@
    & $PythonPath -c $testSnippet
}

function Invoke-ServiceAction {
    param([string]$SvcAction)

    $scriptPath = ".\scripts\manage_usl_api_service.ps1"
    Assert-PathExists $scriptPath "Service manager script"

    $svcParams = @{
        Action = $SvcAction
        ServiceName = $ServiceName
        PythonPath = $PythonPath
        ModelArtifactPath = $ModelArtifactPath
        FallbackModelArtifactPath = $FallbackModelArtifactPath
        FallbackConfidenceThreshold = $FallbackConfidenceThreshold
        EnableFallback = $EnableFallback
        ApiHost = $ApiHost
        Port = $Port
    }

    if ($ApiKey -ne "") {
        $svcParams.ApiKey = $ApiKey
    }

    & $scriptPath @svcParams
}

function Invoke-ServiceLogs {
    Get-WinEvent -LogName System -MaxEvents 300 |
        Where-Object {
            $_.ProviderName -eq "Service Control Manager" -and
            $_.Message -match $ServiceName
        } |
        Select-Object -First 20 TimeCreated, Id, LevelDisplayName, Message |
        Format-List
}

switch ($Action) {
    'export' { Invoke-Export }
    'serve' { Invoke-Serve }
    'health' { Invoke-Health }
    'predict-smoke' { Invoke-PredictSmoke }
    'compose-up' { Invoke-ComposeUp }
    'compose-down' { Invoke-ComposeDown }
    'ci-local' { Invoke-CiLocal }
    'svc-install' { Invoke-ServiceAction -SvcAction 'install' }
    'svc-start' { Invoke-ServiceAction -SvcAction 'start' }
    'svc-stop' { Invoke-ServiceAction -SvcAction 'stop' }
    'svc-status' { Invoke-ServiceAction -SvcAction 'status' }
    'svc-remove' { Invoke-ServiceAction -SvcAction 'remove' }
    'svc-logs' { Invoke-ServiceLogs }
    default { throw "Unsupported action: $Action" }
}
