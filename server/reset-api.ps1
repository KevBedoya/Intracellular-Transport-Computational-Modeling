<#
.SYNOPSIS
    Restart the API so it picks up backend changes, leaving the worker alone.

.DESCRIPTION
    Run this after pushing a change that the API itself has to see: a new
    computation in COMPUTATION_FUNCTIONS, an edit to PARAMETER_SCHEMAS or
    PARAMETER_HINTS, a validation rule, or the cost model in estimate.py.

    WHY A RESTART IS NEEDED AT ALL

    api.py imports computation_router, params_config and estimate lazily, inside
    the request handlers -- but Python caches modules in sys.modules, so the lazy
    import only defers the *first* one. After that the process holds those module
    objects for its lifetime. A long-running API therefore serves the code as it
    was when the process started, however many times the files change on disk.
    (ui.js is exempt: it is a static file, re-read per request.)

    The job worker is NOT restarted, and does not need to be. worker.py launches
    every job as a fresh child process, and that child imports the router from
    disk at run time -- so a job submitted after this restart runs the new code
    even under a worker process from weeks ago. Restarting the worker would also
    requeue anything in flight, which is exactly what you do not want after a
    backend change.

    HOW IT RESTARTS

    The task action is `cmd.exe /c "python api.py ... >> log 2>&1"`, so the tree
    is Task Scheduler -> cmd.exe -> python. Killing the *leaf* collapses the
    whole tree: python exits, the port frees, cmd's /c command completes, cmd
    exits, and the task drops Running -> Ready. serve.ps1 -Stop has to do two
    steps only because it kills cmd first, which orphans python and leaves it
    holding the port.

    The task is registered MultipleInstances=IgnoreNew, so Start-ScheduledTask
    against a task still reading Running is silently ignored -- no error, no
    instance. That is the one race here, and waiting for Ready is what avoids it.

.PARAMETER Pull
    Fast-forward from the remote before restarting. Off by default: this script
    is also useful after a local edit, and a pull is not something to do to
    someone's working copy without being asked. Refuses a dirty working tree
    unless -Force.

.PARAMETER Force
    Allow -Pull to proceed with local modifications present. `git pull --ff-only`
    will still refuse to overwrite them; this only skips this script's own check.

.PARAMETER Port
    API port. Used to find the running process and to health-check the new one.

.NOTES
    -ExecutionPolicy Bypass is required unless you have already run
    `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. It applies to this
    invocation only and changes no system setting.

    Needs the API to have been started once with `serve.ps1 -Detached`, which is
    what registers the scheduled task. If the task does not exist this script
    says so and stops rather than starting an API that dies with your session.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File server\reset-api.ps1
    powershell -ExecutionPolicy Bypass -File server\reset-api.ps1 -Pull
#>

[CmdletBinding()]
param(
    [int]$Port = 8000,
    [string]$TaskName = 'ITCM-api',
    [switch]$Pull,
    [switch]$Force,
    [int]$TimeoutSeconds = 90
)

$ErrorActionPreference = 'Stop'

$ServerDir  = $PSScriptRoot
$ProjectDir = Split-Path $ServerDir -Parent
$PidDir     = Join-Path $ServerDir 'logs'

# Duplicated from serve.ps1 rather than shared. serve.ps1 cannot be dot-sourced:
# with no switches it falls through to its default action and starts both
# servers in the foreground, which is the opposite of what this script wants.
function Get-PidFile($role) { Join-Path $PidDir "$role.pid" }

function Write-PidFile($role, $processId) {
    New-Item -ItemType Directory -Force -Path $PidDir | Out-Null
    Set-Content -Path (Get-PidFile $role) -Value $processId -Encoding ascii
}

function Get-ApiListener {
    <# The process listening on the API port. Identifying it by port rather than
       by command line is deliberate: a task registered -LogonType S4U runs in
       another session, where CommandLine reads back empty for an unelevated
       caller. Whoever holds the port really is the API. #>
    Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue |
        Select-Object -First 1
}

function Get-HeadSha {
    (& git -C $ProjectDir rev-parse --short HEAD 2>$null)
}

# ------------------------------------------------------------------ preflight
$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if (-not $task) {
    Write-Host "No scheduled task '$TaskName'." -ForegroundColor Red
    Write-Host 'The API has never been started detached. Start it once with:'
    Write-Host "    powershell -ExecutionPolicy Bypass -File `"$ServerDir\serve.ps1`" -Detached"
    exit 1
}

# ----------------------------------------------------------------------- pull
if ($Pull) {
    $dirty = @(& git -C $ProjectDir status --porcelain)
    if ($dirty.Count -and -not $Force) {
        Write-Host 'Working tree has local modifications:' -ForegroundColor Red
        $dirty | Select-Object -First 10 | ForEach-Object { Write-Host "    $_" }
        if ($dirty.Count -gt 10) { Write-Host "    ... and $($dirty.Count - 10) more" }
        Write-Host 'Commit, stash, or re-run with -Force to attempt the pull anyway.'
        exit 1
    }
    $branch = (& git -C $ProjectDir rev-parse --abbrev-ref HEAD).Trim()
    Write-Host "pulling origin/$branch ..."
    & git -C $ProjectDir pull --ff-only origin $branch
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'Pull failed; not restarting -- the API would come back on the old code.' -ForegroundColor Red
        exit 1
    }
}

# Report what the restarted API will actually be running, and say so loudly if
# that is not what is on the remote. A restart that silently brings back the
# same code is the failure this script exists to prevent, and it looks
# identical to a successful one.
& git -C $ProjectDir fetch origin --quiet 2>$null
$branch  = (& git -C $ProjectDir rev-parse --abbrev-ref HEAD).Trim()
$behind  = (& git -C $ProjectDir rev-list --count "HEAD..origin/$branch" 2>$null)
$headSha = Get-HeadSha
Write-Host "repo   : $ProjectDir"
Write-Host "commit : $headSha ($branch)"
if ($behind -and [int]$behind -gt 0) {
    Write-Host "WARNING: $behind commit(s) behind origin/$branch." -ForegroundColor Yellow
    Write-Host '         The restart will NOT include what was pushed. Re-run with -Pull.' -ForegroundColor Yellow
}

# -------------------------------------------------------------------- restart
$listener = Get-ApiListener
if ($listener) {
    $oldPid = [int]$listener.OwningProcess
    Write-Host "stopping api pid $oldPid ..."
    Stop-Process -Id $oldPid -Force
} else {
    $oldPid = $null
    Write-Host "nothing listening on port $Port; starting the task anyway"
}

# Wait for Running -> Ready. Without this the Start below is dropped silently
# (MultipleInstances=IgnoreNew) and you are left with the API down and a task
# that reads Ready, which looks exactly like a start that "did not run".
$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
while ((Get-ScheduledTask -TaskName $TaskName).State -eq 'Running') {
    if ((Get-Date) -gt $deadline) {
        Write-Host "Task '$TaskName' still Running after ${TimeoutSeconds}s." -ForegroundColor Red
        Write-Host 'Something else is holding the action open. Fall back to:'
        Write-Host "    powershell -ExecutionPolicy Bypass -File `"$ServerDir\serve.ps1`" -Stop"
        exit 1
    }
    Start-Sleep -Milliseconds 300
}

# The task carries -RestartCount 3 -RestartInterval 5min, and a force-kill exits
# non-zero, so Task Scheduler may already be bringing it back on its own.
# IgnoreNew makes this Start harmless if it has -- but it is not immediate, so
# do not rely on it.
Start-ScheduledTask -TaskName $TaskName
Write-Host "started task $TaskName"

# --------------------------------------------------------------------- verify
# Poll for the port rather than sleeping a fixed interval: numba is not imported
# at API startup, so this is normally a second or two, but a cold filesystem can
# make it longer.
$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
do {
    Start-Sleep -Milliseconds 500
    $listener = Get-ApiListener
} while (-not $listener -and (Get-Date) -lt $deadline)

if (-not $listener) {
    Write-Host "API did not come back on port $Port within ${TimeoutSeconds}s." -ForegroundColor Red
    Write-Host "Check the newest log in $PidDir."
    exit 1
}

$newPid = [int]$listener.OwningProcess
if ($oldPid -and $newPid -eq $oldPid) {
    # Same pid means nothing actually restarted, so the module cache is intact
    # and the whole exercise achieved nothing.
    Write-Host "WARNING: same pid $newPid as before; the API did not restart." -ForegroundColor Yellow
}

# Refresh the pid file serve.ps1 -Stop and -Status fall back on. They resolve by
# port first, so a stale entry is not fatal -- but it would point at a recycled
# pid, and this is the cheap moment to correct it.
Write-PidFile 'api' $newPid

# 0.0.0.0 / :: mean "all interfaces", which is not a usable target for a request.
$hostAddress = $listener.LocalAddress
if ($hostAddress -in @('0.0.0.0', '::')) { $hostAddress = '127.0.0.1' }
$base = "http://${hostAddress}:$Port"

try {
    $health = Invoke-RestMethod -Uri "$base/health" -TimeoutSec 10
    $names  = (Invoke-RestMethod -Uri "$base/computations" -TimeoutSec 10).computations
    Write-Host ''
    Write-Host "API up : $base  (pid $newPid)" -ForegroundColor Green
    Write-Host "commit : $headSha"
    Write-Host "queue  : $($health.queue | ConvertTo-Json -Compress)"
    Write-Host "serving: $($names.Count) computations"
} catch {
    Write-Host "Port $Port is open but the API did not answer: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Check the newest log in $PidDir."
    exit 1
}
