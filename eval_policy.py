"""v2 weekend 4: evaluate the trained RL policy against the PID baseline.

Two-horizon check, mirroring the process that caught the v1 BC model's
slow drift: one 120s flight per episode, with metrics computed over the
first 60s AND over the full 120s. (The first 60s of a 120s flight are the
same trajectory as a 60s flight, so one run gives both.)

PID and RL fly identical conditions: same seeds, same turbulence severity.
The safety monitor is NOT in the loop here -- this measures the raw
controller. We only record when the monitor *would* have tripped, which
tells us how often the RL policy would need the fallback.
"""
import argparse
import json

import numpy as np
from stable_baselines3 import PPO

from flight_env import C172Env
from pid_controller import C172PID
from safety_monitor import SafetyMonitor

HORIZON_S = 120.0
CHECK_S = (60.0, 120.0)
SEVERITIES = (0, 1, 2)


def fly(controller, seed, severity):
    env = C172Env(seed=seed)
    env.MAX_TIME_S = HORIZON_S
    # Start within +/-25 ft of target (training used +/-50). The monitor's
    # altitude limit is 35 ft, so a start >35 ft off would "trip" at t=0
    # for ANY controller and swamp the trip-rate metric with start-condition noise.
    init_alt = env.target_alt_ft + np.random.default_rng(seed + 7).uniform(-25, 25)
    obs, info = env.reset(seed=seed, options={"turb_severity": severity, "init_alt_ft": init_alt})
    pid = None
    if controller == "pid":
        pid = C172PID(dt=1.0 / env.CONTROL_HZ)
        pid.reset(trim_pitch_deg=env.trim_pitch_deg)
    mon = SafetyMonitor()
    alt, hdg, cas = [], [], []
    crash_t = None
    terminated = truncated = False
    while not (terminated or truncated):
        s = env.raw_state()
        if pid is not None:
            action = pid.act(s)
        else:
            action, _ = controller.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        s = info["raw"]
        alt.append(s["alt_err_ft"]); hdg.append(s["hdg_err_deg"]); cas.append(s["cas_kts"])
        mon.check(s, env.t)
        if terminated:
            crash_t = env.t
    return dict(alt=np.array(alt), hdg=np.array(hdg), cas=np.array(cas),
                crash_t=crash_t, trip_t=mon.trip_time, trip_reason=mon.trip_reason)


def summarize(runs, horizon_s):
    n = int(horizon_s * C172Env.CONTROL_HZ)
    crashed = [r["crash_t"] is not None and r["crash_t"] <= horizon_s for r in runs]
    tripped = [r["trip_t"] is not None and r["trip_t"] <= horizon_s for r in runs]
    alive = [r for r, c in zip(runs, crashed) if not c]
    rms_alt = [float(np.sqrt(np.mean(r["alt"][:n] ** 2))) for r in alive]
    rms_hdg = [float(np.sqrt(np.mean(r["hdg"][:n] ** 2))) for r in alive]
    max_alt = [float(np.max(np.abs(r["alt"][:n]))) for r in alive]
    return dict(n=len(runs), crash_rate=float(np.mean(crashed)),
                monitor_trip_rate=float(np.mean(tripped)),
                rms_alt_ft=float(np.mean(rms_alt)) if rms_alt else None,
                rms_hdg_deg=float(np.mean(rms_hdg)) if rms_hdg else None,
                worst_alt_ft=float(np.max(max_alt)) if max_alt else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/ppo_v2/final_model.zip")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--out", default="eval_results.json")
    args = ap.parse_args()

    model = PPO.load(args.model, device="cpu")
    results, traces = {}, {}
    for sev in SEVERITIES:
        seeds = [1000 + i for i in range(args.episodes)]
        for name, ctrl in (("pid", "pid"), ("rl", model)):
            runs = [fly(ctrl, sd, sev) for sd in seeds]
            for h in CHECK_S:
                results[f"{name}_sev{sev}_{int(h)}s"] = summarize(runs, h)
            if sev == 0:
                traces[name] = runs[0]
        print(f"severity {sev} done", flush=True)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    np.savez("eval_traces.npz", **{f"{k}_{f2}": v[f2] for k, v in traces.items() for f2 in ("alt", "hdg", "cas")})

    print(f"\n{'condition':<18}{'horizon':>8}{'crash%':>8}{'trip%':>8}{'rmsAlt ft':>11}{'rmsHdg deg':>12}{'worstAlt':>10}")
    for sev in SEVERITIES:
        for h in CHECK_S:
            for name in ("pid", "rl"):
                r = results[f"{name}_sev{sev}_{int(h)}s"]
                f = lambda x: "  n/a" if x is None else f"{x:.1f}"
                print(f"{name.upper()+' sev'+str(sev):<18}{int(h):>7}s{100*r['crash_rate']:>8.0f}"
                      f"{100*r['monitor_trip_rate']:>8.0f}{f(r['rms_alt_ft']):>11}{f(r['rms_hdg_deg']):>12}{f(r['worst_alt_ft']):>10}")


if __name__ == "__main__":
    main()
