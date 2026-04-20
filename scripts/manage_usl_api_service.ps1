param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('install', 'start', 'stop', 'status', 'remove')]
    [string]$Action,

    [string]$ServiceName = "USLDiseaseAPI",
    [string]$DisplayName = "USL Disease Model API",
    [string]$Description = "FastAPI service for Uganda Sign Language disease model inference",
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

function Test-IsAdmin {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Get-ServiceObject {
    param([string]$Name)
    return Get-Service -Name $Name -ErrorAction SilentlyContinue
}

$workspaceRoot = Split-Path -Parent $PSScriptRoot
$launcher = Join-Path $PSScriptRoot "start_usl_api.ps1"

$pythonResolved = if ([System.IO.Path]::IsPathRooted($PythonPath)) {
    $PythonPath
}
else {
    Join-Path $workspaceRoot $PythonPath
}

switch ($Action) {
    'install' {
        if (-not (Test-IsAdmin)) {
            throw "Admin rights required. Re-run PowerShell as Administrator."
        }
        if (-not (Test-Path $launcher)) {
            throw "Launcher script not found: $launcher"
        }
        if (-not (Test-Path $pythonResolved)) {
            throw "Python executable not found: $pythonResolved"
        }
        if (Get-ServiceObject -Name $ServiceName) {
            throw "Service already exists: $ServiceName"
        }

        $psExe = Join-Path $env:WINDIR "System32\WindowsPowerShell\v1.0\powershell.exe"
        $enableFallbackLiteral = if ($EnableFallback) { '$true' } else { '$false' }
        $binaryPath = "`"$psExe`" -NoProfile -ExecutionPolicy Bypass -File `"$launcher`" -PythonPath `"$pythonResolved`" -ApiHost `"$ApiHost`" -Port $Port -FallbackConfidenceThreshold $FallbackConfidenceThreshold -EnableFallback $enableFallbackLiteral"
        if ($ApiKey -ne "") {
            $binaryPath += " -ApiKey `"$ApiKey`""
        }
        if ($ModelArtifactPath -ne "") {
            $binaryPath += " -ModelArtifactPath `"$ModelArtifactPath`""
        }
        if ($FallbackModelArtifactPath -ne "") {
            $binaryPath += " -FallbackModelArtifactPath `"$FallbackModelArtifactPath`""
        }

        & sc.exe create $ServiceName binPath= $binaryPath start= auto DisplayName= $DisplayName | Out-Host
        & sc.exe description $ServiceName $Description | Out-Host
        Write-Host "Installed service: $ServiceName"
    }

    'start' {
        if (-not (Test-IsAdmin)) {
            throw "Admin rights required. Re-run PowerShell as Administrator."
        }
        Start-Service -Name $ServiceName
        Start-Sleep -Seconds 2
        (Get-Service -Name $ServiceName) | Format-Table -AutoSize Name, Status, StartType
    }

    'stop' {
        if (-not (Test-IsAdmin)) {
            throw "Admin rights required. Re-run PowerShell as Administrator."
        }
        Stop-Service -Name $ServiceName -Force
        Start-Sleep -Seconds 2
        (Get-Service -Name $ServiceName) | Format-Table -AutoSize Name, Status, StartType
    }

    'status' {
        $svc = Get-ServiceObject -Name $ServiceName
        if (-not $svc) {
            Write-Host "Service not found: $ServiceName"
            exit 0
        }
        $svc | Format-Table -AutoSize Name, Status, StartType
        & sc.exe queryex $ServiceName | Out-Host
    }

    'remove' {
        if (-not (Test-IsAdmin)) {
            throw "Admin rights required. Re-run PowerShell as Administrator."
        }
        $svc = Get-ServiceObject -Name $ServiceName
        if (-not $svc) {
            Write-Host "Service not found: $ServiceName"
            exit 0
        }
        if ($svc.Status -ne 'Stopped') {
            Stop-Service -Name $ServiceName -Force
            Start-Sleep -Seconds 2
        }
        & sc.exe delete $ServiceName | Out-Host
        Write-Host "Removed service: $ServiceName"
    }
}
