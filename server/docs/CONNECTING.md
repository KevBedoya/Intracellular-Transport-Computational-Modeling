# Connecting to the compute server

The solver runs on a Windows workstation (12 cores, 32 GB, RTX 3060) that
exposes a small HTTP job API. You submit a job, disconnect, and collect results
later — nothing runs on your own machine.

The server is **not on the public internet**. It is reachable only from a
private Tailscale network (a "tailnet"), so joining that network is the whole
of the access control. There is currently **no API password**: anyone on the
tailnet can queue jobs, cancel jobs, and download any job's output. Access is
therefore limited to people the admin invites directly.

- **Server address:** `http://100.83.174.69:8000`
- **Machine name on the tailnet:** `pop`

> The `100.x.y.z` address belongs to the tailnet and is meaningless outside it.
> It changes if the server is moved to a different tailnet — ask the admin if
> `/health` stops answering.

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

If that returns 18 names, you are fully connected. Continue to
[`COMPUTATION_MENU.md`](COMPUTATION_MENU.md) to launch something.

For convenience, set this once per shell (or add it to your shell profile):

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
`tailscale status` and use the address listed for `pop` rather than a
hard-coded one.

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

One-time machine setup (Tailscale, OpenSSH, firewall, sleep settings) is
`server/setup_remote.ps1`, which must run from an elevated prompt.

**Before inviting anyone beyond a small trusted group**, note again that the API
is unauthenticated. Two things worth adding first:

1. Tailscale ACLs restricting each user to `pop:8000`, so an invited account
   cannot reach anything else on the workstation.
2. A shared token on the API. Until then `submitted_by` is caller-supplied and
   unverified, so with several users you cannot reliably tell who queued what.
