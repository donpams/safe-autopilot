"""v1-vs-v2 comparison, ported from the user's step7_stress_test.py.

Kept identical to the v1 harness: condition definitions (turbulence severity,
altitude-offset range, heading-target range +-40 deg, sensor-noise sigmas),
90 s horizon, 20 episodes per cell, noise applied to the SENSED state, and the
monitor reading the SENSED (noisy) state.

One deliberate change: v1 drew each episode's start from a shared RNG that
advanced between modes, so "bc" and "monitored" flew different episodes.
Here every controller flies the SAME episode (start, turbulence seed and
sensor-noise sequence all derived from the episode index), so differences
between controllers are not luck of the draw.

Controllers: pid, bc, bc+mon, rl0, rl0+mon, rl1, rl1+mon
(rl0 / rl1 = PPO policies trained with seed 0 / seed 1).
"""
import argparse
import json
import os

import numpy as np
import torch
from stable_baselines3 import PPO

from flight_env import C172Env
from pid_controller import C172PID
from bc_model import BCPolicy
from safety_monitor import SafetyMonitor

NOISE = dict(alt_ft=2.0, hdg_deg=0.3, bank_deg=0.2, pitch_deg=0.2, cas_kts=0.5,
             p_rps=0.01, q_rps=0.01, r_rps=0.01)
N_EPISODES = int(os.environ.get('N_EP', 20))
SECONDS = 90

CONDITIONS = {
    "baseline (calm, small offset)":     dict(turb=0, offset_max=30,  noise=False),
    "moderate gusts (severity 2)":       dict(turb=2, offset_max=30,  noise=False),
    "large off-nominal start (+-150ft)": dict(turb=2, offset_max=150, noise=False),
    "sensor noise":                      dict(turb=2, offset_max=30,  noise=True),
    "combined stress":                   dict(turb=2, offset_max=150, noise=True),
}

env = C172Env()
pid = C172PID()
bc = BCPolicy()
bc.load_state_dict(torch.load("bc_policy.pt"))
bc.eval()
rl = {"rl0": PPO.load("runs/ppo_v2/final_model.zip", device="cpu"),
      "rl1": PPO.load("runs/ppo_v2_seed1/final_model.zip", device="cpu")}


def noisy_state(s, rng, on):
    if not on:
        return s
    n = dict(s)
    for k, sigma in NOISE.items():
        n[k] = s[k] + rng.normal(0, sigma)
    n["alt_err_ft"] = n["alt_ft"] - env.target_alt_ft
    n["hdg_err_deg"] = ((n["hdg_deg"] - env.target_hdg_deg + 180) % 360) - 180
    return n


def obs_of(s):
    return np.array([
        s["alt_err_ft"] / 100.0, s["hdg_err_deg"] / 30.0, s["bank_deg"] / 30.0, s["pitch_deg"] / 15.0,
        s["p_rps"], s["q_rps"], s["r_rps"], (s["cas_kts"] - env.target_cas_kts) / 10.0,
    ], dtype=np.float32)


def learned_act(name, obs):
    if name == "bc":
        with torch.no_grad():
            return bc(torch.tensor(obs).unsqueeze(0)).squeeze(0).numpy()
    return rl[name].predict(obs, deterministic=True)[0]


def run_episode(ctrl, opts, noise_on, noise_seed):
    rng = np.random.default_rng(noise_seed)
    name = ctrl.split("+")[0]
    monitored = ctrl.endswith("+mon")
    env.reset(seed=opts["seed"], options=opts)
    pid.reset(trim_pitch_deg=env.trim_pitch_deg)
    mon = SafetyMonitor()
    max_alt, min_cas = 0.0, 1e9
    for _ in range(SECONDS * env.CONTROL_HZ):
        true_s = env.raw_state()
        sensed = noisy_state(true_s, rng, noise_on)
        use_pid = name == "pid" or (monitored and mon.check(sensed, env.t))
        a = pid.act(sensed) if use_pid else learned_act(name, obs_of(sensed))
        _, _, terminated, _, _ = env.step(a)
        max_alt = max(max_alt, abs(true_s["alt_err_ft"]))
        min_cas = min(min_cas, true_s["cas_kts"])
        if terminated:
            return dict(crashed=True, max_alt=max_alt, final_alt=abs(true_s["alt_err_ft"]),
                        tripped=mon.tripped, speed_crash=bool(true_s["cas_kts"] < 51))
    fin = env.raw_state()
    return dict(crashed=False, max_alt=max_alt, final_alt=abs(fin["alt_err_ft"]),
                tripped=mon.tripped, speed_crash=False)


def episode_opts(tag, i, offset_max, turb):
    r = np.random.default_rng(1000 * tag + i)
    init_hdg = r.uniform(0, 360)
    offset = r.uniform(-offset_max, offset_max)
    target = init_hdg + r.uniform(-40, 40)
    return dict(seed=5000 + i, init_hdg_deg=init_hdg, init_alt_ft=5000.0 + offset,
                target_hdg_deg=target, turb_severity=turb, turb_windspeed_fps=15)


def main_part(controllers):
    res = {}
    for ci, (cname, cfg) in enumerate(CONDITIONS.items()):
        for ctrl in controllers:
            runs = [run_episode(ctrl, episode_opts(ci + 1, i, cfg["offset_max"], cfg["turb"]),
                                cfg["noise"], 90000 + 100 * ci + i) for i in range(N_EPISODES)]
            res[f"{cname}|{ctrl}"] = dict(
                crash=100 * float(np.mean([r["crashed"] for r in runs])),
                trip=100 * float(np.mean([r["tripped"] for r in runs])) if ctrl.endswith("+mon") else None,
                mean_max=float(np.mean([r["max_alt"] for r in runs])),
                mean_final=float(np.mean([r["final_alt"] for r in runs])))
        print(cname, "done", flush=True)
    return res


def reliability_part(controllers):
    res = {}
    for sev in range(5):
        for ctrl in controllers:
            runs = [run_episode(ctrl, episode_opts(50 + sev, i, 30, sev), False, 0) for i in range(N_EPISODES)]
            res[f"sev{sev}|{ctrl}"] = dict(crash=100 * float(np.mean([r["crashed"] for r in runs])),
                                           speed=int(sum(r["speed_crash"] for r in runs)))
        print("severity", sev, "done", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["main", "reliability"], required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.part == "main":
        ctrls = ["pid", "bc", "bc+mon", "rl0", "rl0+mon", "rl1", "rl1+mon"]
        res = main_part(ctrls)
        print(f"\n{'condition':36s}{'ctrl':9s}{'crash%':>7s}{'trip%':>7s}{'max|alt|':>10s}{'final|alt|':>12s}")
        for k, r in res.items():
            c, m = k.split("|")
            t = "-" if r["trip"] is None else f"{r['trip']:.0f}"
            print(f"{c:36s}{m:9s}{r['crash']:7.0f}{t:>7s}{r['mean_max']:10.1f}{r['mean_final']:12.1f}")
    else:
        ctrls = ["pid", "bc", "rl0", "rl1"]
        res = reliability_part(ctrls)
        print(f"\n{'severity':10s}" + "".join(f"{c:>14s}" for c in ctrls) + "   (crash%, [airspeed-floor crashes])")
        for sev in range(5):
            print(f"{sev:<10d}" + "".join(
                f"{res[f'sev{sev}|{c}']['crash']:>8.0f}%[{res[f'sev{sev}|{c}']['speed']:>2d}]  " for c in ctrls))
    json.dump(res, open(args.out, "w"), indent=2)
