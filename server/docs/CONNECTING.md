# Connecting to the compute server

The solver runs on a Windows workstation (12 cores, 32 GB, RTX 3060) that
exposes a small HTTP job API. You submit a job, disconnect, and collect results
later — nothing runs on your own machine.

The server is **not on the public internet**. It is reachable only from a
private Tailscale network (a "tailnet"), so joining that network is the whole
of the access control. There is currently **no API password**: anyone on the
tailnet can queue jobs, cancel jobs, and download any job's output. Access is
therefore limited to people the admin invites directly.

## Where to go once you are on the tailnet

| | |
|---|---|
| **Web UI** | **http://100.83.174.69:8000/ui** |
| API root | `http://100.83.174.69:8000` |
| Machine name on the tailnet | `pop` |

The UI is the normal way to use the server: pick a computation, fill a form,
queue it, and download the results from the same page. Everything it does is
also available over the API if you prefer `curl` — see
[`COMPUTATION_MENU.md`](COMPUTATION_MENU.md).

### Finding the server's address

The `100.x.y.z` address belongs to the tailnet and means nothing outside it. It
**changes if the server is moved to a different tailnet**, so if the UI stops
loading, re-check it rather than trusting a bookmark.

**From your own machine** (once you are on the tailnet) — this is the reliable
way, because it reports what your device can actually reach:

```bash
tailscale status | grep pop
```

```
100.83.174.69   pop   codingendeavors88@   windows   -
```

The first column is the address. So the UI is that address followed by
`:8000/ui`.

**On the server itself**, if you are sitting at the workstation:

```powershell
tailscale ip -4
```

Either way, confirm before opening a browser:

```bash
curl -s http://<address>:8000/health
# {"status": "ok", "queue": {...}}
```

---

## 1. Ask the admin for an invite

The tailnet admin invites you from the Tailscale admin console
(**Users → Invite**). You will get an email with a join link.

You cannot add yourself. If you sign in to Tailscale with an account that
belongs to some *other* tailnet (a university or company Google Workspace
account often does), you will end up on that tailnet instead of this one and
the server will not be visible. If in doubt, sign in with a personal account.

## 2. Install Tailscale

| Platform | How |
|---|---|
| macOS | [tailscale.com/download/mac](https://tailscale.com/download/mac) — or `brew install --cask tailscale` |
| Windows | [tailscale.com/download/windows](https://tailscale.com/download/windows) — or `winget install Tailscale.Tailscale` |
| Linux | `curl -fsSL https://tailscale.com/install.sh \| sh` |

Then sign in with the account the invite was sent to:

```bash
tailscale up
```

This opens a browser. Accept the invite first if you have not already.

### If the CLI is not found (macOS)

The Mac app does not always put `tailscale` on your `PATH`:

```bash
sudo ln -s /Applications/Tailscale.app/Contents/MacOS/Tailscale /usr/local/bin/tailscale
```

## 3. Wait for approval — and do not trust the UI

If the browser says:

> **User approval required:** admins of this tailnet must approve you before
> you can join.

then you are **not** on the tailnet yet, even though the Tailscale app will
happily say "Connected" and your system settings will look fine. Those only
mean the client authenticated and is running. Being *authorised to carry
traffic* is a separate gate.

**The only reliable check is whether you can see the server:**

```bash
tailscale status
```

If the output does not list `pop`, you are not in yet — wait for the admin.
Once `pop` appears:

```bash
tailscale ping pop
```

Replies here are the real confirmation. A device can appear in `status` and
still fail to route, so do not skip this.

## 4. Test that the server is up

```bash
curl -s http://100.83.174.69:8000/health
```

Expected:

```json
{"status": "ok", "queue": {"queued": 0, "succeeded": 12}}
```

`queue` is a live count of jobs by status, so a response also tells you the
worker has a database to talk to. Then list what you can run:

```bash
curl -s http://100.83.174.69:8000/computations
```

If that returns 18 names, you are fully connected.

## 5. Open the UI

**http://100.83.174.69:8000/ui**

That is all the setup there is. The page loads the list of computations from the
server, so if the dropdown fills in, everything behind it is working.

A good first run is a 32×32 characteristic-time job — about a minute, and it
exercises the whole path including the plot preview:

| field | value |
|---|---|
| Computation | Characteristic Time (mass vs v) |
| `rg_param`, `ry_param` | 32, 32 |
| `v_LIST` | 10000 |
| `w_param` | 100 |
| `T_param` | 1 |
| `N_LIST` | use the helper: enter 4 tubes, click *compute positions* |

Fill in **submitted by** at the top right first — it is saved in your browser, so
you only do it once, and it is how anyone else sharing the queue can tell whose
job is whose.

If you would rather drive it from a terminal, set this once per shell (or add it
to your shell profile) and see [`COMPUTATION_MENU.md`](COMPUTATION_MENU.md):

```bash
export ITCM=http://100.83.174.69:8000
curl -s $ITCM/health
```

---

## Troubleshooting

**`tailscale ping pop` says "no matching peer"**
You are on the wrong tailnet. Check which account you are signed in as:

```bash
tailscale switch --list
```

The `Tailnet` column must match the one the invite came from. Fix with
`tailscale logout` then `tailscale up`, making sure the browser signs in as the
right account — sign out of tailscale.com first, otherwise it silently reuses
your existing session and puts you straight back on the wrong tailnet.

**`curl` hangs or "connection refused"**
`tailscale ping pop` succeeding but `curl` failing means the tailnet is fine and
the server process is not running. Ask the admin to check
`serve.ps1 -Status`. A hang on *both* means Tailscale is down on one end; this
is not a firewall problem, because Tailscale opens no router ports.

**Everything worked yesterday and now nothing answers**
The server address changes if the tailnet changes. Re-run
`tailscale status | grep pop` and use the address listed there rather than a
bookmark.

**The UI loads but the computation dropdown is empty**
The page reached the server but `GET /computations` failed. Check
`curl -s <address>:8000/computations` — if that works and the dropdown still does
not fill, it is a browser-side fault worth reporting.

**`/ui` returns 404 while `/health` works**
The server is running code from before the UI existed. It needs restarting on the
workstation — note that restarting picks up whatever is in the working tree, and
that stopping the scheduled task alone leaves the old process holding the port
(see the admin notes below).

**A job stays `queued` forever**
The API accepted it but no worker is consuming the queue. That is a
server-side condition — the admin restarts it with `serve.ps1 -Detached`.

---

## Notes for the admin

Bringing the server up (on the workstation, no administrator rights needed):

```powershell
powershell -ExecutionPolicy Bypass -File server\serve.ps1 -Detached
powershell -ExecutionPolicy Bypass -File server\serve.ps1 -Status
powershell -ExecutionPolicy Bypass -File server\serve.ps1 -Stop
```

`-ExecutionPolicy Bypass` is required unless `Set-ExecutionPolicy -Scope
CurrentUser RemoteSigned` has been run; Windows otherwise refuses to run
script files at all.

`-Detached` registers scheduled tasks rather than starting plain background
processes. This matters: Windows kills the children of an SSH session when the
session ends, so there is no `nohup` equivalent — a plain background process
would die the moment you disconnect. Scheduled tasks run in their own session
and survive both disconnect and logout.

**Restarting is fiddlier than it looks.** The task action runs python under
`cmd.exe` to redirect output, so stopping the task kills `cmd` and *orphans* the
python child, which keeps holding port 8000. The next start then exits
immediately and the task reads `Ready`, looking as though it simply did not run
while the old code carries on serving. `-Stop` handles this by locating processes
via the listening port and recorded pid files, and it reports anything it could
not terminate — a task registered with `-LogonType S4U` runs under a token an
unelevated shell cannot kill, so finish those from an elevated prompt:

```powershell
Stop-Process -Id <pid> -Force
Start-ScheduledTask -TaskName ITCM-api
```

`-Status` shows how each process was found, and lists running solver children
with their virtual size and CPU time. Do not judge a long solve by its working
set: a healthy 112×112 job showed 21 MB resident while holding 5.4 GB committed,
because its timeseries arrays are touched rarely enough to be trimmed out of the
working set. CPU time is the honest signal.

One-time machine setup (Tailscale, OpenSSH, firewall, sleep settings) is
`server/setup_remote.ps1`, which must run from an elevated prompt.

**Before inviting anyone beyond a small trusted group**, note again that the API
is unauthenticated. Two things worth adding first:

1. Tailscale ACLs restricting each user to `pop:8000`, so an invited account
   cannot reach anything else on the workstation.
2. A shared token on the API. Until then `submitted_by` is caller-supplied and
   unverified, so with several users you cannot reliably tell who queued what.
