<#
.SYNOPSIS
    Start the job worker and the HTTP API. No administrator rights needed.

.DESCRIPTION
    Two long-lived processes:
      * worker.py  claims queued jobs and runs them (default 4 concurrent)
      * api.py     serves the queue over HTTP

    By default the API binds to this machine's Tailscale address, so it is
    reachable from your tailnet and invisible to the local network. Pass
    -BindLocal to bind 127.0.0.1 instead (useful before Tailscale is set up),
    or -BindAddress to choose explicitly.

    Both processes log to server/logs/. Use -Detached to run them under Task
    Scheduler so they survive you closing the SSH session or logging out --
    see the note below, because this is the part Windows does differently
    from Linux.

.NOTES
    Durability: on Linux you would use nohup or tmux. On Windows, children of
    an SSH session are generally killed when the session ends, so a plain
    Start-Process here dies when you disconnect. -Detached registers scheduled
    tasks instead, which run in their own session and survive disconnect and
    logout. An RDP *disconnect* (as opposed to signing out) also preserves a
    session, if you prefer that route.

.NOTES
    -ExecutionPolicy Bypass is required unless you have already run
    `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`; by default Windows
    refuses to run script files at all. Bypass applies to this invocation only
    and changes no system setting.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File server\serve.ps1
    powershell -ExecutionPolicy Bypass -File server\serve.ps1 -BindLocal -Slots 2
    powershell -ExecutionPolicy Bypass -File server\serve.ps1 -Detached
    powershell -ExecutionPolicy Bypass -File server\serve.ps1 -Stop
    powershell -ExecutionPolicy Bypass -File server\serve.ps1 -Status
#>

[CmdletBinding()]
param(
    [string]$BindAddress,
    [switch]$BindLocal,
    [int]$Port = 8000,
    [int]$Slots = 4,
    [switch]$Detached,
    [switch]$Stop,
    [switch]$Status
)

$ErrorActionPreference = 'Stop'

$ServerDir  = $PSScriptRoot
$ProjectDir = Split-Path $ServerDir -Parent
$LogDir     = Join-Path $ServerDir 'logs'

# The MSYS2 python on PATH has no pandas; the Anaconda one is the interpreter
# the solver actually runs under. Prefer an explicit override, then Anaconda,
# then whatever python resolves to.
function Resolve-Python {
    if ($env:ITCM_PYTHON -and (Test-Path $env:ITCM_PYTHON)) { return $env:ITCM_PYTHON }
    $anaconda = 'N:\ML\Anaconda\Install\python.exe'
    if (Test-Path $anaconda) { return $anaconda }
    $p = Get-Command python -ErrorAction SilentlyContinue
    if ($p) {
        Write-Warning "falling back to $($p.Source); set ITCM_PYTHON if this lacks pandas/numba"
        return $p.Source
    }
    throw 'no python found; set ITCM_PYTHON to a full interpreter path'
}

function Get-TailscaleIp {
    $ts = Get-Command tailscale -ErrorAction SilentlyContinue
    if (-not $ts) { return $null }
    try {
        $ip = (& tailscale ip -4 2>$null | Select-Object -First 1)
        if ($ip -match '^\d+\.\d+\.\d+\.\d+$') { return $ip.Trim() }
    } catch { }
    return $null
}

$TaskWorker = 'ITCM-worker'
$TaskApi    = 'ITCM-api'

# --------------------------------------------------------------------- -Status
if ($Status) {
    Write-Host 'Scheduled tasks:'
    foreach ($t in @($TaskWorker, $TaskApi)) {
        $task = Get-ScheduledTask -TaskName $t -ErrorAction SilentlyContinue
        if ($task) {
            $info = Get-ScheduledTaskInfo -TaskName $t
            Write-Host ("  {0,-14} {1,-10} last run {2}" -f $t, $task.State, $info.LastRunTime)
        }
        else { Write-Host ("  {0,-14} not registered" -f $t) }
    }
    Write-Host "`nForeground processes:"
    $procs = Get-CimInstance Win32_Process -Filter "Name like '%python%'" |
             Where-Object { $_.CommandLine -match 'worker\.py|api\.py' }
    if ($procs) {
        foreach ($p in $procs) {
            $what = if ($p.CommandLine -match 'worker\.py') { 'worker' } else { 'api' }
            Write-Host ("  pid {0,-8} {1}" -f $p.ProcessId, $what)
        }
    }
    else { Write-Host '  none' }
    return
}

# ----------------------------------------------------------------------- -Stop
if ($Stop) {
    foreach ($t in @($TaskWorker, $TaskApi)) {
        if (Get-ScheduledTask -TaskName $t -ErrorAction SilentlyContinue) {
            Stop-ScheduledTask -TaskName $t -ErrorAction SilentlyContinue
            Write-Host "stopped task $t"
        }
    }
    $procs = Get-CimInstance Win32_Process -Filter "Name like '%python%'" |
             Where-Object { $_.CommandLine -match 'worker\.py|api\.py' }
    foreach ($p in $procs) {
        Write-Host "stopping pid $($p.ProcessId)"
        Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
    }
    Write-Host 'Note: solver child processes already running are left alone;'
    Write-Host 'their jobs are requeued next time the worker starts.'
    return
}

# ------------------------------------------------------------------- bind addr
if ($BindLocal) {
    $bind = '127.0.0.1'
}
elseif ($BindAddress) {
    $bind = $BindAddress
}
else {
    $bind = Get-TailscaleIp
    if (-not $bind) {
        Write-Warning 'no Tailscale address found; binding 127.0.0.1 (local only).'
        Write-Warning 'Run setup_remote.ps1 and `tailscale up`, or pass -BindAddress.'
        $bind = '127.0.0.1'
    }
}

$python = Resolve-Python
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$stamp      = Get-Date -Format 'yyyyMMdd-HHmmss'
$workerLog  = Join-Path $LogDir "worker-$stamp.log"
$apiLog     = Join-Path $LogDir "api-$stamp.log"

Write-Host "python : $python"
Write-Host "api    : http://${bind}:$Port"
Write-Host "slots  : $Slots"
Write-Host "logs   : $LogDir"

# Agg so a headless solver child never tries to open a GUI backend.
$env:MPLBACKEND = 'Agg'
$env:PYTHONUNBUFFERED = '1'

# ------------------------------------------------------------------- -Detached
if ($Detached) {
    # A scheduled task with -LogonType S4U runs in its own session, so it
    # outlives the SSH/RDP session that registered it. Plain Start-Process
    # would not: Windows tears down an SSH session's children on disconnect.
    $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" `
        -LogonType S4U -RunLevel Limited
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) `
        -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 5)

    $specs = @(
        @{ Name = $TaskWorker
           Args = "-u `"$ServerDir\worker.py`" --slots $Slots"
           Log  = $workerLog },
        @{ Name = $TaskApi
           Args = "-u `"$ServerDir\api.py`" --host $bind --port $Port"
           Log  = $apiLog }
    )

    foreach ($s in $specs) {
        # cmd wrapper only to get output redirection into the log file.
        $cmd = "/c `"`"$python`" $($s.Args) >> `"$($s.Log)`" 2>&1`""
        $action = New-ScheduledTaskAction -Execute 'cmd.exe' -Argument $cmd `
            -WorkingDirectory $ProjectDir
        if (Get-ScheduledTask -TaskName $s.Name -ErrorAction SilentlyContinue) {
            Unregister-ScheduledTask -TaskName $s.Name -Confirm:$false
        }
        Register-ScheduledTask -TaskName $s.Name -Action $action `
            -Principal $principal -Settings $settings | Out-Null
        Start-ScheduledTask -TaskName $s.Name
        Write-Host "started detached task $($s.Name)"
    }

    Write-Host "`nThese survive SSH disconnect and logout."
    Write-Host "Check:  powershell -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Status"
    Write-Host "Stop:   powershell -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Stop"
    return
}

# ------------------------------------------------------------------ foreground
Write-Host "`nStarting in the foreground (Ctrl+C to stop)."
Write-Host 'These die when this session ends -- use -Detached to survive disconnect.'

$wp = Start-Process -FilePath $python `
    -ArgumentList "-u", "$ServerDir\worker.py", "--slots", "$Slots" `
    -WorkingDirectory $ProjectDir -NoNewWindow -PassThru `
    -RedirectStandardOutput $workerLog -RedirectStandardError "$workerLog.err"
Write-Host "worker pid $($wp.Id)"

$ap = Start-Process -FilePath $python `
    -ArgumentList "-u", "$ServerDir\api.py", "--host", $bind, "--port", "$Port" `
    -WorkingDirectory $ProjectDir -NoNewWindow -PassThru `
    -RedirectStandardOutput $apiLog -RedirectStandardError "$apiLog.err"
Write-Host "api    pid $($ap.Id)"

Write-Host "`nTailing status. Ctrl+C to stop both."
try {
    while (-not $wp.HasExited -and -not $ap.HasExited) { Start-Sleep -Seconds 2 }
    if ($wp.HasExited) { Write-Warning "worker exited with $($wp.ExitCode); see $workerLog" }
    if ($ap.HasExited) { Write-Warning "api exited with $($ap.ExitCode); see $apiLog" }
}
finally {
    foreach ($p in @($wp, $ap)) {
        if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
    }
    Write-Host 'stopped.'
}
