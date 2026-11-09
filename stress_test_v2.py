"""v2 weekend 5: stress-test the RL policy, with and without the safety monitor.

Three controllers fly identical conditions (same seeds):
  pid        - the fallback controller alone
  rl         - the PPO policy alone (no monitor)
  rl+monitor - PPO policy, latching fallback to the PID when the monitor trips

Conditions (same five categories as the v1 stress test; parameters here are
defined fresh for v2, so absolute numbers are NOT directly comparable to v1):
  baseline        calm air, start within +/-25 ft
  gusts           turbulence severity 2
  large_start     start 60-90 ft off altitude, heading target 25-35 deg off
  sensor_noise    Gaussian noise on what the controllers SENSE
  combined        gusts + large start + sensor noise

Noise is applied to the inputs of both policy and PID. The monitor and the
crash check read the TRUE state (a monitor on a separate/validated sensor
path is the usual design assumption; documented here, not hidden).
"""
import argparse
import json

import numpy as np
from stable_baselines3 import PPO

from flight_env import C172Env, wrap180
from pid_controller import C172PID
from safety_monitor import SafetyMonitor

HORIZON_S = 120.0
# 1-sigma noise in physical units, applied to sensed values
NOISE = dict(alt_err_ft=3.0, hdg_err_deg=1.0, bank_deg=0.5, pitch_deg=0.3,
             p_rps=0.01, q_rps=0.01, r_rps=0.01, cas_kts=1.0, vs_fps=0.5)

CONDITIONS = {
    "baseline":     dict(sev=0, big=False, noise=False),
    "gusts":        dict(sev=2, big=False, noise=False),
    "large_start":  dict(sev=0, big=True,  noise=False),
    "sensor_noise": dict(sev=0, big=False, noise=True),
    "combined":     dict(sev=2, big=True,  noise=True),
}

# Out-of-distribution suite: things the policy was NOT trained on. `bias` adds a
# constant elevator offset to the LEARNED controller's output only (simulating
# a degraded/faulty policy); the PID fallback is never biased.
OOD_CONDITIONS = {
    "turb_sev3":       dict(sev=3, big=False, noise=False, bias=0.0),
    "turb_sev4":       dict(sev=4, big=False, noise=False, bias=0.0),
    "policy_fault":    dict(sev=0, big=False, noise=False, bias=0.15),
    "fault+turb_sev2": dict(sev=2, big=False, noise=False, bias=0.15),
}


def noisy(s, rng):
    out = dict(s)
    for k, sig in NOISE.items():
        out[k] = s[k] + rng.normal(0, sig)
    return out


def obs_from_raw(s):
    return np.array([s["alt_err_ft"] / 100.0, s["hdg_err_deg"] / 30.0, s["bank_deg"] / 30.0,
                     s["pitch_deg"] / 15.0, s["p_rps"], s["q_rps"], s["r_rps"],
                     (s["cas_kts"] - 100.0) / 10.0], dtype=np.float32)


def fly(kind, model, seed, cond):
    rng = np.random.default_rng(seed + 11)
    env = C172Env(seed=seed)
    env.MAX_TIME_S = HORIZON_S
    sign = lambda: rng.choice([-1, 1])
    if cond["big"]:
        init_alt = 5000.0 + sign() * rng.uniform(60, 90)
        init_hdg = rng.uniform(0, 360)
        target_hdg = (init_hdg + sign() * rng.uniform(25, 35)) % 360
    else:
        init_alt = 5000.0 + rng.uniform(-25, 25)
        init_hdg = rng.uniform(0, 360)
        target_hdg = (init_hdg + rng.uniform(-20, 20)) % 360
    env.reset(seed=seed, options=dict(turb_severity=cond["sev"], init_alt_ft=init_alt,
                                      init_hdg_deg=init_hdg, target_hdg_deg=target_hdg))
    pid = C172PID(dt=1.0 / env.CONTROL_HZ)
    pid.reset(trim_pitch_deg=env.trim_pitch_deg)
    mon = SafetyMonitor()
    fallback = False
    alt = []
    crashed = False
    terminated = truncated = False
    while not (terminated or truncated):
        true_s = env.raw_state()
        sensed = noisy(true_s, rng) if cond["noise"] else true_s
        if kind == "rl+monitor" and not fallback:
            fallback = mon.check(true_s, env.t)
        if kind == "pid" or fallback:
            action = pid.act(sensed)
        else:
            action, _ = model.predict(obs_from_raw(sensed), deterministic=True)
            action = np.clip(action + np.array([cond.get("bias", 0.0), 0.0]), -1, 1)
        _, _, terminated, truncated, info = env.step(action)
        alt.append(info["raw"]["alt_err_ft"])
        crashed = terminated
    alt = np.abs(np.array(alt))
    return dict(crashed=bool(crashed), tripped=bool(fallback) if kind == "rl+monitor" else None,
                trip_t=mon.trip_time if kind == "rl+monitor" else None,
                final_alt=float(np.mean(alt[-100:])), rms_alt=float(np.sqrt(np.mean(alt ** 2))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--episodes", type=int, default=25)
    ap.add_argument("--out", required=True)
    ap.add_argument("--suite", choices=["main", "ood"], default="main")
    args = ap.parse_args()
    conditions = CONDITIONS if args.suite == "main" else OOD_CONDITIONS
    model = PPO.load(args.model, device="cpu")
    res = {}
    for cname, cond in conditions.items():
        for kind in ("pid", "rl", "rl+monitor"):
            runs = [fly(kind, model, 2000 + i, cond) for i in range(args.episodes)]
            res[f"{cname}|{kind}"] = dict(
                n=len(runs),
                crash_pct=100 * float(np.mean([r["crashed"] for r in runs])),
                trip_pct=(100 * float(np.mean([r["tripped"] for r in runs])) if kind == "rl+monitor" else None),
                final_alt_ft=float(np.mean([r["final_alt"] for r in runs])),
                rms_alt_ft=float(np.mean([r["rms_alt"] for r in runs])))
        print(f"{cname} done", flush=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"\n{'condition':<14}{'controller':<12}{'crash%':>7}{'trip%':>7}{'final|alt| ft':>15}{'rms alt ft':>12}")
    for k, r in res.items():
        c, kind = k.split("|")
        t = "-" if r["trip_pct"] is None else f"{r['trip_pct']:.0f}"
        print(f"{c:<14}{kind:<12}{r['crash_pct']:>7.0f}{t:>7}{r['final_alt_ft']:>15.1f}{r['rms_alt_ft']:>12.1f}")


if __name__ == "__main__":
    main()
