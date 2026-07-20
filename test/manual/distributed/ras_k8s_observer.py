"""K8s live-server NCCL RAS observer for a 2-pod SGLang deployment.

Unlike ras_multinode_timeline.py (which launches its own idle NCCL workers and
asserts the clean ~60s considered_dead path), THIS script runs against your
already-running SGLang pods. A live serving server has pending collectives, so
killing node1's scheduler usually triggers NCCL abort + the forward watchdog
BEFORE RAS's ~60s considered_dead — so this script does NOT assert
considered_dead. It records whatever RAS/metrics signals surface (poll_success
drop, missing_ranks brief rise, /health non-200, job teardown) within a short
window, and softly asserts that *something* changed (otherwise RAS gave no
signal at all).

Prereqs on the SGLang pods (set in the pod env):
- SGLANG_NCCL_RAS_ENABLE=1        (so the collector runs on world rank 0)
- --enable-metrics                (so /metrics exposes nccl_ras_* series)
- NCCL_RAS_ENABLE is setdefault'd by sglang; no need to set it yourself.

Usage (run from anywhere with kubectl access to the cluster)::

    KUBE_NODE0_POD=sglang-0 KUBE_NODE1_POD=sglang-1 NAMESPACE=default \
        python ras_k8s_observer.py

Optional env:
    RAS_PORT_FWD / METRICS_PORT_FWD - local ports for the port-forwards
                                     (default 28028 / 30000).
    OBSERVE_SECONDS  - how long to observe after the kill (default 90).
    KILL_MODE        - "exec" (kubectl exec kill -9, default) or "delete"
                      (kubectl delete pod node1 — more disruptive, simulates
                      a real node loss).

The script starts `kubectl port-forward` for node0's pod (RAS 28028 + metrics
30000), polls both, reads rank1's pid from RAS STATUS, kills it on node1 via
kubectl, and records the timeline. Ctrl-C or OBSERVE_SECONDS ends it; the port
forwards are cleaned up on exit.
"""
import json
import os
import shlex
import socket
import subprocess
import sys
import time

NODE0 = os.environ.get("KUBE_NODE0_POD")
NODE1 = os.environ.get("KUBE_NODE1_POD")
NS = os.environ.get("NAMESPACE", "default")
RAS_PORT = int(os.environ.get("RAS_PORT_FWD", "28028"))
METRICS_PORT = int(os.environ.get("METRICS_PORT_FWD", "30000"))
OBSERVE_SECONDS = int(os.environ.get("OBSERVE_SECONDS", "90"))
KILL_MODE = os.environ.get("KILL_MODE", "exec")

# Live server: assert SOMETHING signals the loss within this window (poll drop,
# missing_ranks, ranks_in_error, /health). Do not assert considered_dead.
SIGNAL_WITHIN_SEC = 40


def sh(cmd, timeout=20):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def poll_ras():
    try:
        s = socket.create_connection(("127.0.0.1", RAS_PORT), timeout=2)
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
        if len(buf) > 16 * 1024 * 1024:
            break
    s.close()
    i = buf.find(b"{")
    if i < 0:
        return None
    try:
        return json.loads(buf[i:])
    except json.JSONDecodeError:
        return None


def metrics_nccl_ras():
    try:
        out = subprocess.run(
            ["curl", "-s", "-m", "3", f"http://127.0.0.1:{METRICS_PORT}/metrics"],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except Exception:
        return ""
    return "\n".join(l for l in out.splitlines() if "nccl_ras" in l)


def health_status():
    try:
        r = subprocess.run(
            ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "-m", "3",
             f"http://127.0.0.1:{METRICS_PORT}/health"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except Exception:
        return "ERR"
    return r or "ERR"


def rank1_pid(r):
    for c in r.get("communicators", []):
        for rk in c.get("ranks", []):
            if rk.get("rank") == 1:
                return rk.get("pid")
        for mr in c.get("missing_ranks", []):
            if mr.get("rank") == 1:
                return mr.get("pid")
    return None


def rank1_state(r):
    for c in r.get("communicators", []):
        for rk in c.get("ranks", []):
            if rk.get("rank") == 1:
                return ("live", c.get("missing_ranks_count"))
        for mr in c.get("missing_ranks", []):
            if mr.get("rank") == 1:
                st = mr.get("status", {})
                return (
                    f"missing(unresp={st.get('unresponsive')},"
                    f"dead={st.get('considered_dead')})",
                    c.get("missing_ranks_count"),
                )
    return ("gone", None)


def start_port_forward():
    # Forward node0 pod's RAS + metrics ports to localhost.
    pf = subprocess.Popen(
        ["kubectl", "port-forward", "-n", NS, NODE0,
         f"{RAS_PORT}:28028", f"{METRICS_PORT}:30000"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    # wait for "Forwarding" line or fail
    for _ in range(30):
        line = pf.stderr.readline() if pf.stderr else ""
        if "Forwarding" in line or "error" in line.lower():
            break
        if poll_ras() is not None:
            break
        time.sleep(0.5)
    return pf


def kill_node1(pid):
    if KILL_MODE == "delete":
        cmd = ["kubectl", "delete", "pod", "-n", NS, NODE1, "--wait=false"]
        subprocess.run(cmd, timeout=30)
        return "kubectl delete pod " + NODE1
    cmd = ["kubectl", "exec", "-n", NS, NODE1, "--", "kill", "-9", str(pid)]
    subprocess.run(cmd, timeout=20)
    return f"kubectl exec {NODE1} kill -9 {pid}"


def main():
    if not NODE0 or not NODE1:
        print("Set KUBE_NODE0_POD and KUBE_NODE1_POD (and optionally NAMESPACE).",
              flush=True)
        sys.exit(2)
    t0 = time.time()
    pf = start_port_forward()
    try:
        # baseline
        r = None
        for _ in range(30):
            r = poll_ras()
            if r:
                break
            time.sleep(1)
        if not r:
            print(f"RAS socket (port-forward {RAS_PORT}) never came up. "
                  "Is SGLANG_NCCL_RAS_ENABLE=1 set on the pods?", flush=True)
            sys.exit(1)
        pid = rank1_pid(r)
        if pid is None:
            print("rank1 not found in baseline STATUS", flush=True)
            sys.exit(1)
        print(f"[t={time.time()-t0:5.1f}s] baseline: rank1 pid={pid} "
              f"state={rank1_state(r)} health={health_status()}", flush=True)
        print(f"           metrics:\n{metrics_nccl_ras()}", flush=True)
        time.sleep(3)

        print(f"[t={time.time()-t0:5.1f}s] KILL node1 via {kill_node1(pid)}",
              flush=True)

        saw_signal = False
        for i in range(OBSERVE_SECONDS // 5):
            time.sleep(5)
            t = time.time() - t0
            r = poll_ras()
            rstatus = rank1_state(r) if r else ("poll-failed", None)
            hs = health_status()
            m = metrics_nccl_ras()
            poll_succ = next(
                (l for l in m.splitlines() if "nccl_ras_poll_success" in l), ""
            )
            print(f"[t={t:6.1f}s] rank1={rstatus} health={hs} {poll_succ.strip()}",
                  flush=True)
            # any signal of the loss?
            if r is None or "missing" in rstatus[0] or hs != "200":
                saw_signal = True

        print("\n=== result ===", flush=True)
        if saw_signal:
            print(f"PASS: RAS/health reflected the node1 loss within "
                  f"{OBSERVE_SECONDS}s (live-server path — job may have torn "
                  f"down before considered_dead, which is expected).", flush=True)
            sys.exit(0)
        print(f"FAIL: no RAS/health signal within {OBSERVE_SECONDS}s "
              "(did node0's collector survive? is the comm actually across "
              "these 2 pods?)", flush=True)
        sys.exit(1)
    finally:
        pf.terminate()
        try:
            pf.wait(timeout=5)
        except Exception:
            pf.kill()


if __name__ == "__main__":
    main()
