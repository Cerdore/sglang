"""Multi-node RAS dead-rank timeline: poll node0's RAS, kill rank1 on node1 via SSH.

Runs on node0 (or any host that can reach node0's RAS socket AND SSH into
node1). Steps:

1. Poll ``localhost:28028`` until the RAS socket is up; record the healthy
   baseline and pick rank1's pid + host from the STATUS JSON.
2. Sleep ~5s for a stable baseline, then ``ssh <node1> kill -9 <rank1_pid>``.
3. Poll 28028 every 5s for ~120s, recording ``missing_ranks_count``,
   ``unresponsive``, ``considered_dead`` per rank.
4. Assert: rank1 moves to ``missing_ranks[]`` with ``unresponsive=True`` within
   ~15s, and ``considered_dead`` flips True within ~75s. These are the cross-
   node dead-rank signals a single-node hard-crash cannot reproduce (the job
   tears down before RAS's ~60s considered_dead).

Usage (see ras_multinode_README.md for the full 2-node launch)::

    NODE1_SSH=user@node1-host python ras_multinode_timeline.py

Env:
    RAS_HOST / RAS_PORT   - where node0's RAS socket is reachable from this
                            process (default 127.0.0.1 / 28028).
    NODE1_SSH             - ssh target for the node running rank1 (e.g.
                            ``user@10.0.0.2``). Required to kill rank1.
    POLL_SECONDS          - how long to poll after the kill (default 120).
"""
import json
import os
import shlex
import socket
import subprocess
import sys
import time

HOST = os.environ.get("RAS_HOST", "127.0.0.1")
PORT = int(os.environ.get("RAS_PORT", "28028"))
NODE1_SSH = os.environ.get("NODE1_SSH")
POLL_SECONDS = int(os.environ.get("POLL_SECONDS", "120"))

# Soft assertion windows (NCCL ~60s considered_dead + margin).
UNRESPONSIVE_WITHIN_SEC = 30
CONSIDERED_DEAD_WITHIN_SEC = 90


def poll():
    """One RAS STATUS poll -> parsed dict, or None on refuse/timeout/malformed."""
    try:
        s = socket.create_connection((HOST, PORT), timeout=2)
    except OSError:
        return None
    s.settimeout(2)
    s.sendall(b"SET FORMAT json\n")
    s.sendall(b"STATUS\n")
    buf = b""
    while True:
        try:
            d = s.recv(4096)
        except socket.timeout:
            break
        if not d:
            break
        buf += d
        if len(buf) > 16 * 1024 * 1024:  # bound memory
            break
    s.close()
    i = buf.find(b"{")
    if i < 0:
        return None
    try:
        return json.loads(buf[i:])
    except json.JSONDecodeError:
        return None


def rank1_info(r):
    """Return (pid, host) for rank1 from a STATUS dict, or (None, None)."""
    for c in r.get("communicators", []):
        for rk in c.get("ranks", []):
            if rk.get("rank") == 1:
                return rk.get("pid"), rk.get("host")
        for mr in c.get("missing_ranks", []):
            if mr.get("rank") == 1:
                return mr.get("pid"), mr.get("host")
    return None, None


def rank1_state(r):
    """Return (missing_count, unresponsive, considered_dead, in_missing)."""
    for c in r.get("communicators", []):
        for rk in c.get("ranks", []):
            if rk.get("rank") == 1:
                return c.get("missing_ranks_count"), False, False, False
        for mr in c.get("missing_ranks", []):
            if mr.get("rank") == 1:
                st = mr.get("status", {})
                return (
                    c.get("missing_ranks_count"),
                    bool(st.get("unresponsive")),
                    bool(st.get("considered_dead")),
                    True,
                )
    return None, None, None, None


def kill_rank1_remote(pid, host):
    """ssh into the node hosting rank1 and SIGKILL it."""
    target = NODE1_SSH or (None if host in (None, os.uname().nodename) else f"{host}")
    if target is None:
        # rank1 is on this host — kill locally
        os.kill(pid, 9)
        return "local SIGKILL"
    cmd = ["ssh", "-o", "ConnectTimeout=8", target, f"kill -9 {pid}"]
    subprocess.run(cmd, check=False, timeout=20)
    return f"ssh {target} kill -9 {pid}"


def main():
    if not NODE1_SSH:
        print("Set NODE1_SSH=user@node1-host (or run on node0 where rank1 is "
              "local).", flush=True)
        sys.exit(2)

    t0 = time.time()
    # 1. wait for RAS socket + capture baseline + rank1 pid
    r = None
    for _ in range(30):
        r = poll()
        if r:
            break
        time.sleep(1)
    if not r:
        print(f"RAS socket at {HOST}:{PORT} never came up", flush=True)
        sys.exit(1)
    pid, host = rank1_info(r)
    if pid is None:
        print("rank1 not found in baseline STATUS", flush=True)
        sys.exit(1)
    print(f"[t={time.time()-t0:5.1f}s] baseline: rank1 pid={pid} host={host}",
          flush=True)
    print(f"           {rank1_state(r)}", flush=True)
    time.sleep(5)

    # 2. kill rank1 on its node
    print(f"[t={time.time()-t0:5.1f}s] KILL rank1 via {kill_rank1_remote(pid, host)}",
          flush=True)

    # 3. timeline + 4. assertions
    seen_unresp = seen_dead = False
    t_unresp = t_dead = None
    for i in range(POLL_SECONDS // 5):
        time.sleep(5)
        t = time.time() - t0
        r = poll()
        if r is None:
            print(f"[t={t:6.1f}s] poll FAILED/refused (survivor socket down?)",
                  flush=True)
            continue
        mc, unresp, dead, in_missing = rank1_state(r)
        print(f"[t={t:6.1f}s] missing={mc} rank1_in_missing={in_missing} "
              f"unresponsive={unresp} considered_dead={dead}", flush=True)
        if unresp and not seen_unresp:
            seen_unresp = True
            t_unresp = t
            print(f"  >>> unresponsive turned True at ~t={t:.1f}s", flush=True)
        if dead and not seen_dead:
            seen_dead = True
            t_dead = t
            print(f"  >>> considered_dead turned True at ~t={t:.1f}s", flush=True)

    # assertions
    print("\n=== assertions ===", flush=True)
    ok = True
    if not seen_unresp:
        print(f"FAIL: unresponsive never turned True within {POLL_SECONDS}s", flush=True)
        ok = False
    elif t_unresp > UNRESPONSIVE_WITHIN_SEC:
        print(f"FAIL: unresponsive took {t_unresp:.1f}s > {UNRESPONSIVE_WITHIN_SEC}s",
              flush=True)
        ok = False
    else:
        print(f"PASS: unresponsive within {UNRESPENSIVE_WITHIN_SEC}s "
              f"(actual ~{t_unresp:.1f}s)", flush=True)
    if not seen_dead:
        print(f"FAIL: considered_dead never turned True within {POLL_SECONDS}s",
              flush=True)
        ok = False
    elif t_dead > CONSIDERED_DEAD_WITHIN_SEC:
        print(f"FAIL: considered_dead took {t_dead:.1f}s > "
              f"{CONSIDERED_DEAD_WITHIN_SEC}s", flush=True)
        ok = False
    else:
        print(f"PASS: considered_dead within {CONSIDERED_DEAD_WITHIN_SEC}s "
              f"(actual ~{t_dead:.1f}s)", flush=True)
    print("RESULT: " + ("PASS" if ok else "FAIL"), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
