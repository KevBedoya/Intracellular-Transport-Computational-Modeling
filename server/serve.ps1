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
$PidDir     = $LogDir

function Get-PidFile($role) { Join-Path $PidDir "$role.pid" }

function Write-PidFile($role, $processId) {
    New-Item -ItemType Directory -Force -Path $PidDir | Out-Null
    Set-Content -Path (Get-PidFile $role) -Value $processId -Encoding ascii
}

function Read-PidFile($role) {
    $f = Get-PidFile $role
    if (-not (Test-Path $f)) { return $null }
    $v = (Get-Content $f -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($v -match '^\d+$') { return [int]$v }
    return $null
}

function Find-ServerProcesses {
    <#
      Locate the worker and API processes.

      Deliberately does NOT match on CommandLine. A scheduled task registered
      with -LogonType S4U runs in another session, where CommandLine reads back
      empty for an unelevated caller -- so the command-line match this script
      used previously found nothing, and -Stop quietly did nothing at all.

      Three sources, in order of reliability:
        port  -- whoever is listening on the API port really is the API
        pid   -- recorded at launch, before the process becomes unreadable
        name  -- python processes started by a task, as a last resort
    #>
    $out = @{}

    $conns = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
    foreach ($c in $conns) {
        if ($c.OwningProcess) { $out[[int]$c.OwningProcess] = @{ Role = 'api'; Source = 'port' } }
    }

    foreach ($role in @('api', 'worker')) {
        $rp = Read-PidFile $role
        if ($rp -and (Get-Process -Id $rp -ErrorAction SilentlyContinue)) {
            if (-not $out.ContainsKey($rp)) { $out[$rp] = @{ Role = $role; Source = 'pidfile' } }
        }
    }

    $results = @()
    foreach ($processId in $out.Keys) {
        $proc = Get-Process -Id $processId -ErrorAction SilentlyContinue
        if (-not $proc) { continue }
        $ci = Get-CimInstance Win32_Process -Filter "ProcessId=$processId" -ErrorAction SilentlyContinue
        $cpu = if ($ci) { ($ci.KernelModeTime + $ci.UserModeTime) / 1e7 } else { 0 }
        $results += [PSCustomObject]@{
            Pid    = $processId
            Role   = $out[$processId].Role
            Source = $out[$processId].Source
            RamMB  = [math]::Round($proc.WorkingSet64 / 1MB, 0)
            CpuSec = [math]::Round($cpu, 1)
        }
    }
    return @($results | Sort-Object Role, Pid)
}

function Get-SolverChildren {
    <#
      Python processes that look like a running solve rather than the worker or
      API, identified by meaningful CPU time. Virtual size is no help: every
      python process reserves ~4 GB of it. Reported so -Status can show a long
      job is alive, since its working set is misleading too -- the timeseries
      arrays are touched rarely enough to be trimmed out of it.
    #>
    $serverPids = @(@(Find-ServerProcesses) | ForEach-Object { $_.Pid })
    $kids = @()
    foreach ($ci in (Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue)) {
        if ($serverPids -contains [int]$ci.ProcessId) { continue }
        $cpuHr = (($ci.KernelModeTime + $ci.UserModeTime) / 1e7) / 3600
        $virtMB = [math]::Round($ci.VirtualSize / 1MB, 0)
        if ($cpuHr -lt 0.05) { continue }
        $kids += [PSCustomObject]@{
            Pid = [int]$ci.ProcessId; VirtMB = $virtMB
            CpuHr = [math]::Round($cpuHr, 2); Created = $ci.CreationDate
        }
    }
    return @($kids | Sort-Object CpuHr -Descending)
}

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
    Write-Host "`nProcesses:"
    $found = @(Find-ServerProcesses)
    if ($found.Count) {
        foreach ($p in $found) {
            Write-Host ("  pid {0,-8} {1,-7} {2,8:N0} MB  cpu {3,8:N1} s  via {4}" -f
                $p.Pid, $p.Role, $p.RamMB, $p.CpuSec, $p.Source)
        }
    }
    else { Write-Host '  none' }

    # A task reading Running while its process is unaccounted for is worth
    # saying out loud: it is exactly the state that makes a restart appear to
    # work while the old process keeps serving.
    foreach ($pair in @(@{T = $TaskApi; R = 'api'}, @{T = $TaskWorker; R = 'worker'})) {
        $task = Get-ScheduledTask -TaskName $pair.T -ErrorAction SilentlyContinue
        if ($task -and $task.State -eq 'Running' -and
            -not ($found | Where-Object { $_.Role -eq $pair.R })) {
            Write-Host ("  note: {0} reads Running but no {1} process was located." -f
                $pair.T, $pair.R)
            Write-Host ("        It holds no port and has no recorded pid (pid files are" )
            Write-Host ("        written by -Detached), so -Stop cannot target it.")
        }
    }

    Write-Host "`nSolver children:"
    $kids = @(Get-SolverChildren)
    if ($kids.Count) {
        # Working set is misleading here: a long solve's timeseries arrays get
        # trimmed out of it, so a healthy job can look like a few MB. Virtual
        # size and CPU time are the honest signals.
        foreach ($k in $kids) {
            Write-Host ("  pid {0,-8} {1,8:N0} MB virt  cpu {2,7:N2} hr  started {3}" -f
                $k.Pid, $k.VirtMB, $k.CpuHr, $k.Created)
        }
    }
    else { Write-Host '  none' }
    return
}

# ----------------------------------------------------------------------- -Stop
if ($Stop) {
    # Stopping the scheduled task is not enough on its own: the task action runs
    # python under cmd.exe for output redirection, so stopping the task kills
    # cmd and ORPHANS the python child, which keeps holding its port. The port
    # then blocks the next start, and the task exits immediately looking like it
    # simply "did not run".
    foreach ($t in @($TaskWorker, $TaskApi)) {
        if (Get-ScheduledTask -TaskName $t -ErrorAction SilentlyContinue) {
            Stop-ScheduledTask -TaskName $t -ErrorAction SilentlyContinue
            Write-Host "stopped task $t"
        }
    }

    $found = @(Find-ServerProcesses)
    if (-not $found.Count) { Write-Host 'no server processes found' }

    $stuck = @()
    foreach ($p in $found) {
        try {
            Stop-Process -Id $p.Pid -Force -ErrorAction Stop
            Write-Host ("stopped pid {0} ({1})" -f $p.Pid, $p.Role)
        }
        catch {
            $stuck += $p
            Write-Host ("COULD NOT STOP pid {0} ({1}): {2}" -f
                $p.Pid, $p.Role, $_.Exception.Message)
        }
    }

    if ($stuck.Count) {
        # Silently skipping this is how a "restart" ends up still serving the old
        # code, so say it plainly rather than returning as if it worked.
        Write-Host ''
        Write-Warning @"
$($stuck.Count) process(es) survived. A scheduled task registered with -LogonType
S4U runs under a token an unelevated shell cannot terminate, so the old process
keeps its port and the next start will exit immediately.

Finish from an ELEVATED PowerShell:
$($stuck | ForEach-Object { "    Stop-Process -Id $($_.Pid) -Force" } | Out-String)    Start-ScheduledTask -TaskName $TaskApi
    Start-ScheduledTask -TaskName $TaskWorker
"@
    }

    Write-Host ''
    Write-Host 'Solver child processes already running are left alone; their jobs'
    Write-Host 'are requeued the next time the worker starts.'
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
        @{ Name = $TaskWorker; Role = 'worker'
           Args = "-u `"$ServerDir\worker.py`" --slots $Slots"
           Log  = $workerLog },
        @{ Name = $TaskApi; Role = 'api'
           Args = "-u `"$ServerDir\api.py`" --host $bind --port $Port"
           Log  = $apiLog }
    )

    foreach ($s in $specs) {
        # cmd wrapper only to get output redirection into the log file. Note the
        # consequence handled in -Stop: stopping the task kills cmd.exe and
        # orphans this python child, so -Stop has to terminate the process too.
        $cmd = "/c `"`"$python`" $($s.Args) >> `"$($s.Log)`" 2>&1`""
        $action = New-ScheduledTaskAction -Execute 'cmd.exe' -Argument $cmd `
            -WorkingDirectory $ProjectDir
        if (Get-ScheduledTask -TaskName $s.Name -ErrorAction SilentlyContinue) {
            Unregister-ScheduledTask -TaskName $s.Name -Confirm:$false
        }
        Register-ScheduledTask -TaskName $s.Name -Action $action `
            -Principal $principal -Settings $settings | Out-Null
        Remove-Item (Get-PidFile $s.Role) -ErrorAction SilentlyContinue
        Start-ScheduledTask -TaskName $s.Name
        Write-Host "started detached task $($s.Name)"
    }

    # Record the pids now. Once a task is running, its CommandLine is unreadable
    # from an unelevated shell, so this is the only cheap way for a later -Stop
    # to know which process belongs to which role.
    Start-Sleep -Seconds 4
    $conns = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
    if ($conns) { Write-PidFile 'api' ([int]($conns | Select-Object -First 1).OwningProcess) }
    $cands = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
             Where-Object { $_.CreationDate -gt (Get-Date).AddSeconds(-30) }
    $apiPid = Read-PidFile 'api'
    $workerCand = $cands | Where-Object { [int]$_.ProcessId -ne $apiPid } |
                  Select-Object -First 1
    if ($workerCand) { Write-PidFile 'worker' ([int]$workerCand.ProcessId) }

    Write-Host "`nThese survive SSH disconnect and logout."
    Write-Host "Check:  powershell -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Status"
    Write-Host "Stop:   powershell -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Stop"
    Write-Host ''
    Write-Host 'A later -Stop needs the same rights these were started with. If it'
    Write-Host 'reports processes it could not stop, rerun it elevated.'
    return
}

# ------------------------------------------------------------------ foreground
Write-Host "`nStarting in the foreground (Ctrl+C to stop)."
Write-Host 'These die when this session ends -- use -Detached to survive disconnect.'

$wp = Start-Process -FilePath $python `
    -ArgumentList "-u", "$ServerDir\worker.py", "--slots", "$Slots" `
    -WorkingDirectory $ProjectDir -NoNewWindow -PassThru `
    -RedirectStandardOutput $workerLog -RedirectStandardError "$workerLog.err"
Write-PidFile 'worker' $wp.Id
Write-Host "worker pid $($wp.Id)"

$ap = Start-Process -FilePath $python `
    -ArgumentList "-u", "$ServerDir\api.py", "--host", $bind, "--port", "$Port" `
    -WorkingDirectory $ProjectDir -NoNewWindow -PassThru `
    -RedirectStandardOutput $apiLog -RedirectStandardError "$apiLog.err"
Write-PidFile 'api' $ap.Id
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
