<#
.SYNOPSIS
    One-time setup for reaching this machine's job server remotely.

.DESCRIPTION
    Run ONCE from an elevated PowerShell. Everything here needs administrator
    rights, which is why it is a separate script rather than part of serve.ps1.

    It is idempotent: re-running skips anything already in place.

    What it does:
      1. Installs Tailscale (via winget) if absent.
      2. Installs and starts the OpenSSH server, sets PowerShell as the
         default SSH shell, and opens port 22 to the private profile only.
      3. Leaves the API firewall alone on purpose -- see the note below.

    What it deliberately does NOT do:
      * open any port to the internet
      * add a port-forward on your router
      * expose RDP

.NOTES
    Why Tailscale rather than port-forwarding:
      Forwarding 22 or 3389 to the public internet gets the machine scanned and
      brute-forced within hours; exposed RDP in particular is a leading
      ransomware entry point. Tailscale is a WireGuard mesh -- the machine gets
      a private 100.x address reachable only by devices signed into your
      tailnet, with no inbound ports open at the router at all.

    Why the API gets no firewall rule:
      Bind the API to the Tailscale address (serve.ps1 does this by default),
      so it is reachable over the tailnet and invisible to the local network.
      Adding a rule for port 8000 would expose it to every device on your LAN
      for no benefit.

.EXAMPLE
    # From an elevated PowerShell, in the repo root:
    powershell -ExecutionPolicy Bypass -File June-2026\server\setup_remote.ps1
#>

[CmdletBinding()]
param(
    [switch]$SkipTailscale,
    [switch]$SkipSsh
)

$ErrorActionPreference = 'Stop'

function Assert-Admin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]$id
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        Write-Error @'
This script must run elevated.

Open PowerShell as Administrator (Win+X, "Windows PowerShell (Admin)"), then:
    powershell -ExecutionPolicy Bypass -File June-2026\server\setup_remote.ps1
'@
        exit 1
    }
}

function Step($n, $msg) { Write-Host "`n[$n] $msg" -ForegroundColor Cyan }
function Ok($msg)       { Write-Host "    OK   $msg" -ForegroundColor Green }
function Skip($msg)     { Write-Host "    skip $msg" -ForegroundColor DarkGray }
function Note($msg)     { Write-Host "    note $msg" -ForegroundColor Yellow }

Assert-Admin
Write-Host "Remote-access setup for the intracellular-transport job server"

# ---------------------------------------------------------------- 1. Tailscale
if (-not $SkipTailscale) {
    Step 1 'Tailscale (private network, no open ports)'
    if (Get-Command tailscale -ErrorAction SilentlyContinue) {
        Skip 'tailscale already installed'
    }
    else {
        if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
            Note 'winget unavailable; install manually from https://tailscale.com/download'
        }
        else {
            Write-Host '    installing via winget...'
            # --silent so it does not block on a UI; accepts both agreements
            winget install --id Tailscale.Tailscale --silent `
                --accept-package-agreements --accept-source-agreements
            if ($LASTEXITCODE -eq 0) { Ok 'tailscale installed' }
            else { Note "winget exited $LASTEXITCODE -- install manually if this failed" }
        }
    }
    Note 'Next, sign in (interactive, opens a browser):  tailscale up'
    Note 'Then note this machine''s tailnet address:      tailscale ip -4'
}
else { Step 1 'Tailscale -- skipped by request' }

# ------------------------------------------------------------------ 2. OpenSSH
if (-not $SkipSsh) {
    Step 2 'OpenSSH server'

    $cap = Get-WindowsCapability -Online -Name 'OpenSSH.Server*'
    if ($cap.State -eq 'Installed') {
        Skip 'OpenSSH.Server capability already installed'
    }
    else {
        Write-Host '    installing capability...'
        Add-WindowsCapability -Online -Name $cap.Name | Out-Null
        Ok 'OpenSSH.Server installed'
    }

    Set-Service -Name sshd -StartupType Automatic
    if ((Get-Service sshd).Status -ne 'Running') {
        Start-Service sshd
        Ok 'sshd started'
    }
    else { Skip 'sshd already running' }
    Ok 'sshd set to start automatically'

    # PowerShell as the login shell; the default is cmd, which makes the
    # server-side commands in these scripts awkward to run.
    $reg = 'HKLM:\SOFTWARE\OpenSSH'
    if (-not (Test-Path $reg)) { New-Item -Path $reg -Force | Out-Null }
    $psPath = 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe'
    New-ItemProperty -Path $reg -Name DefaultShell -Value $psPath `
        -PropertyType String -Force | Out-Null
    Ok 'default SSH shell set to PowerShell'

    # Private profile only. Domain/Public are left closed so a coffee-shop
    # network never sees port 22.
    $ruleName = 'ITCM-sshd-private'
    if (Get-NetFirewallRule -Name $ruleName -ErrorAction SilentlyContinue) {
        Skip "firewall rule $ruleName already present"
    }
    else {
        New-NetFirewallRule -Name $ruleName `
            -DisplayName 'OpenSSH Server (ITCM, private only)' `
            -Enabled True -Direction Inbound -Protocol TCP -Action Allow `
            -LocalPort 22 -Profile Private | Out-Null
        Ok 'opened TCP 22 on the private profile only'
    }

    Note 'Set up key auth, then disable passwords in C:\ProgramData\ssh\sshd_config:'
    Note '    PasswordAuthentication no'
    Note '    PubkeyAuthentication yes'
    Note 'and restart:  Restart-Service sshd'
}
else { Step 2 'OpenSSH -- skipped by request' }

# --------------------------------------------------------------------- 3. Power
Step 3 'Power settings (long jobs must survive an idle machine)'
$ac = (powercfg /query SCHEME_CURRENT SUB_SLEEP STANDBYIDLE |
       Select-String 'Current AC Power Setting Index').ToString()
if ($ac -match '0x00000000') {
    Skip 'AC sleep already disabled'
}
else {
    powercfg /change standby-timeout-ac 0
    Ok 'disabled AC sleep'
}
powercfg /change hibernate-timeout-ac 0 2>$null
Ok 'disabled AC hibernate'

# ------------------------------------------------------------------- 4. Summary
Write-Host "`nDone." -ForegroundColor Cyan
Write-Host @'
Remaining manual steps:

  1. tailscale up                      sign in (opens a browser)
  2. tailscale ip -4                   note the 100.x address
  3. Install the same on your laptop and sign into the same tailnet.
  4. Start the server (NOT elevated):
         powershell -File June-2026\server\serve.ps1
  5. From your laptop:
         curl http://<100.x address>:8000/health
         ssh <user>@<100.x address>

Sanity check that nothing is publicly exposed: from outside your network,
nothing should answer on 22, 3389 or 8000. Tailscale opens no router ports.
'@
