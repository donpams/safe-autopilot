"""Fault-injection test: does the safety monitor catch a FAILED learned controller?

Faults are applied to the LEARNED controller's output only; the PID fallback
is never faulted (it stands for an independent, known-good channel). The PID
alone is run as an unfaulted reference. Conditions follow the v1 harness
(v1_vs_v2.py): same turn sizes, 90 s, 20 identical episodes per cell, monitor
reading the sensed state.

  elev_bias_0.4     constant +0.4 added to the elevator command (calm air)
  aileron_stuck     aileron command frozen at +0.3 (a held roll input), calm air
  elev_bias+gusts   +0.4 elevator bias in severity-2 turbulence
"""
import json
import os

import numpy as np

import v1_vs_v2 as V
from safety_monitor import SafetyMonitor

FAULTS = {
    "elev_bias_0.4":    dict(turb=0, kind="elev_bias", val=0.4),
    "aileron_stuck":    dict(turb=0, kind="aileron_stuck", val=0.3),
    "elev_bias+gusts":  dict(turb=2, kind="elev_bias", val=0.4),
}


def fault(a, f):
    a = np.array(a, dtype=np.float64)
    if f["kind"] == "elev_bias":
        a[0] += f["val"]
    else:
        a[1] = f["val"]
    return np.clip(a, -1, 1)


def run(ctrl, opts, f):
    name = ctrl.split("+")[0]
    monitored = ctrl.endswith("+mon")
    V.env.reset(seed=opts["seed"], options=opts)
    V.pid.reset(trim_pitch_deg=V.env.trim_pitch_deg)
    mon = SafetyMonitor()
    max_alt = max_bank = 0.0
    for _ in range(V.SECONDS * V.env.CONTROL_HZ):
        s = V.env.raw_state()
        use_pid = name == "pid" or (monitored and mon.check(s, V.env.t))
        a = V.pid.act(s) if use_pid else fault(V.learned_act(name, V.obs_of(s)), f)
        _, _, term, _, _ = V.env.step(a)
        max_alt = max(max_alt, abs(s["alt_err_ft"]))
        max_bank = max(max_bank, abs(s["bank_deg"]))
        if term:
            return dict(crashed=True, trip_t=mon.trip_time, max_alt=max_alt, max_bank=max_bank,
                        final_alt=abs(s["alt_err_ft"]))
    fin = V.env.raw_state()
    return dict(crashed=False, trip_t=mon.trip_time, max_alt=max_alt, max_bank=max_bank,
                final_alt=abs(fin["alt_err_ft"]))


if __name__ == "__main__":
    ctrls = ["pid", "bc", "bc+mon", "rl0", "rl0+mon", "rl1", "rl1+mon"]
    res = {}
    for fi, (fname, f) in enumerate(FAULTS.items()):
        for c in ctrls:
            runs = [run(c, V.episode_opts(70 + fi, i, 30, f["turb"]), f) for i in range(V.N_EPISODES)]
            trips = [r["trip_t"] for r in runs if r["trip_t"] is not None]
            res[f"{fname}|{c}"] = dict(
                crash=100 * float(np.mean([r["crashed"] for r in runs])),
                trip=100 * len(trips) / len(runs) if c.endswith("+mon") else None,
                trip_t=float(np.mean(trips)) if trips else None,
                max_alt=float(np.mean([r["max_alt"] for r in runs])),
                max_bank=float(np.mean([r["max_bank"] for r in runs])),
                final_alt=float(np.mean([r["final_alt"] for r in runs])))
        print(fname, "done", flush=True)
    json.dump(res, open("faults_124.json", "w"), indent=2)
    print(f"\n{'fault':18s}{'ctrl':9s}{'crash%':>7s}{'trip%':>7s}{'trip t(s)':>10s}{'max|alt|':>10s}{'max bank':>9s}{'final|alt|':>11s}")
    for k, r in res.items():
        fn, c = k.split("|")
        t = "-" if r["trip"] is None else f"{r['trip']:.0f}"
        tt = "-" if r["trip_t"] is None else f"{r['trip_t']:.1f}"
        print(f"{fn:18s}{c:9s}{r['crash']:7.0f}{t:>7s}{tt:>10s}{r['max_alt']:10.1f}{r['max_bank']:9.1f}{r['final_alt']:11.1f}")
