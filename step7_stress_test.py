import numpy as np
import torch
import matplotlib.pyplot as plt
from flight_env import C172Env
from pid_controller import C172PID
from bc_model import BCPolicy
from safety_monitor import SafetyMonitor

env = C172Env()
pid = C172PID()
model = BCPolicy()
model.load_state_dict(torch.load("bc_policy.pt"))
model.eval()

NOISE = dict(alt_ft=2.0, hdg_deg=0.3, bank_deg=0.2, pitch_deg=0.2, cas_kts=0.5,
             p_rps=0.01, q_rps=0.01, r_rps=0.01)

def noisy_state(s, rng, on):
    if not on:
        return s
    n = dict(s)
    for k, sigma in NOISE.items():
        n[k] = s[k] + rng.normal(0, sigma)
    n["alt_err_ft"] = n["alt_ft"] - env.target_alt_ft
    n["hdg_err_deg"] = ((n["hdg_deg"] - env.target_hdg_deg + 180) % 360) - 180
    return n

def noisy_obs(s):
    return np.array([
        s["alt_err_ft"] / 100.0, s["hdg_err_deg"] / 30.0, s["bank_deg"] / 30.0, s["pitch_deg"] / 15.0,
        s["p_rps"], s["q_rps"], s["r_rps"], (s["cas_kts"] - env.target_cas_kts) / 10.0,
    ], dtype=np.float32)

def bc_act(obs):
    with torch.no_grad():
        return model(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()

def run_episode(mode, opts, seconds, noise_on, rng):
    obs, _ = env.reset(seed=opts["seed"], options=opts)
    pid.reset(trim_pitch_deg=env.trim_pitch_deg)
    monitor = SafetyMonitor()
    max_alt_err = 0.0
    for _ in range(int(seconds * env.CONTROL_HZ)):
        true_s = env.raw_state()
        sensed = noisy_state(true_s, rng, noise_on)
        if mode == "bc":
            a = bc_act(noisy_obs(sensed))
        else:
            use_pid = monitor.check(sensed, env.t)
            a = pid.act(sensed) if use_pid else bc_act(noisy_obs(sensed))
        obs, _, terminated, _, _ = env.step(a)
        max_alt_err = max(max_alt_err, abs(true_s["alt_err_ft"]))
        if terminated:
            return dict(crashed=True, max_alt_err=max_alt_err, final_alt_err=abs(true_s["alt_err_ft"]),
                        tripped=monitor.tripped, trip_time=monitor.trip_time)
    final_s = env.raw_state()
    return dict(crashed=False, max_alt_err=max_alt_err, final_alt_err=abs(final_s["alt_err_ft"]),
                tripped=monitor.tripped, trip_time=monitor.trip_time)

N_EPISODES = 20
SECONDS = 90
rng = np.random.default_rng(7)

conditions = {
    "baseline (calm, small offset)":        dict(turb=0, offset_max=30,  noise=False),
    "moderate gusts (severity 2)":          dict(turb=2, offset_max=30,  noise=False),
    "large off-nominal start (+-150ft)":    dict(turb=2, offset_max=150, noise=False),
    "sensor noise":                          dict(turb=2, offset_max=30,  noise=True),
    "combined stress":                       dict(turb=2, offset_max=150, noise=True),
}

results = {}
print(f"{'condition':32s} {'mode':4s} {'crash%':>7s} {'trip%':>6s} {'mean max|alt|':>14s} {'mean final|alt|':>16s}")
for cond_name, cfg in conditions.items():
    for mode in ["bc", "monitored"]:
        stats = []
        for i in range(N_EPISODES):
            seed = 5000 + i
            init_hdg = rng.uniform(0, 360)
            offset = rng.uniform(-cfg["offset_max"], cfg["offset_max"])
            target_hdg = init_hdg + rng.uniform(-40, 40)
            opts = dict(seed=seed, init_hdg_deg=init_hdg, init_alt_ft=5000.0 + offset,
                        target_hdg_deg=target_hdg, turb_severity=cfg["turb"], turb_windspeed_fps=15)
            stats.append(run_episode(mode, opts, SECONDS, cfg["noise"], rng))
        crash_rate = 100 * np.mean([s["crashed"] for s in stats])
        trip_rate = 100 * np.mean([s["tripped"] for s in stats]) if mode == "monitored" else float("nan")
        mean_max = np.mean([s["max_alt_err"] for s in stats])
        mean_final = np.mean([s["final_alt_err"] for s in stats])
        results[(cond_name, mode)] = dict(crash_rate=crash_rate, trip_rate=trip_rate,
                                           mean_max=mean_max, mean_final=mean_final)
        trip_str = f"{trip_rate:5.0f}%" if mode == "monitored" else "   -  "
        print(f"{cond_name:32s} {mode:4s} {crash_rate:6.0f}% {trip_str} {mean_max:13.1f}ft {mean_final:15.1f}ft")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
names = list(conditions.keys())
x = np.arange(len(names))
w = 0.35
bc_crash = [results[(n, "bc")]["crash_rate"] for n in names]
mon_crash = [results[(n, "monitored")]["crash_rate"] for n in names]
axes[0].bar(x - w/2, bc_crash, w, label="BC alone", color="tab:orange")
axes[0].bar(x + w/2, mon_crash, w, label="BC + monitor", color="tab:green")
axes[0].set_ylabel("crash rate (%)"); axes[0].set_xticks(x); axes[0].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
axes[0].legend(); axes[0].grid(alpha=0.3, axis="y")

bc_final = [results[(n, "bc")]["mean_final"] for n in names]
mon_final = [results[(n, "monitored")]["mean_final"] for n in names]
axes[1].bar(x - w/2, bc_final, w, label="BC alone", color="tab:orange")
axes[1].bar(x + w/2, mon_final, w, label="BC + monitor", color="tab:green")
axes[1].set_ylabel("mean final |alt error| (ft)"); axes[1].set_xticks(x); axes[1].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
axes[1].legend(); axes[1].grid(alpha=0.3, axis="y")

plt.tight_layout(); plt.savefig("step7_stress_summary.png", dpi=150)
print("\nsaved step7_stress_summary.png")

print("\n=== PID-alone reliability vs turbulence severity (no model, no monitor) ===")
print("(turns + turbulence combined, same episode conditions as above, N=20 each)")
pid_reliability = {}
for sev in [0, 1, 2, 3, 4]:
    crashes, speed_floor = 0, 0
    for i in range(N_EPISODES):
        seed = 6000 + i
        init_hdg = rng.uniform(0, 360)
        offset = rng.uniform(-30, 30)
        target_hdg = init_hdg + rng.uniform(-40, 40)
        obs, _ = env.reset(seed=seed, options=dict(init_hdg_deg=init_hdg, init_alt_ft=5000.0 + offset,
                            target_hdg_deg=target_hdg, turb_severity=sev, turb_windspeed_fps=15))
        pid.reset(trim_pitch_deg=env.trim_pitch_deg)
        crashed = False
        for _ in range(SECONDS * env.CONTROL_HZ):
            s = env.raw_state()
            _, _, te, _, _ = env.step(pid.act(s))
            if te:
                crashed = True
                if s["cas_kts"] < 51:
                    speed_floor += 1
                break
        crashes += crashed
    pid_reliability[sev] = 100 * crashes / N_EPISODES
    print(f"  severity {sev}: crash rate {crashes}/{N_EPISODES} = {100*crashes/N_EPISODES:.0f}%  (airspeed-floor crashes: {speed_floor})")

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(list(pid_reliability.keys()), list(pid_reliability.values()), "o-", color="tab:blue")
ax.set_xlabel("turbulence severity"); ax.set_ylabel("PID-alone crash rate (%)")
ax.set_title("PID fallback's own reliability limit\n(no throttle loop -- airspeed bleeds off in a turn + gust)")
ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("step7_pid_fallback_limit.png", dpi=150)
print("saved step7_pid_fallback_limit.png")
