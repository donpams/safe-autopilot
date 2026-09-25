import numpy as np
import matplotlib.pyplot as plt
from flight_env import C172Env
from pid_controller import C172PID

N_EPISODES = 300
SEED0 = 1000          # offset so these seeds never overlap the step3 eval seeds (0-7)
VAL_FRACTION = 0.15

env = C172Env()
pid = C172PID()
rng = np.random.default_rng(42)   # controls episode conditions, not the sim/turbulence itself

obs_log, act_log, ep_id_log = [], [], []
summaries = []

for ep in range(N_EPISODES):
    seed = SEED0 + ep
    init_hdg = rng.uniform(0, 360)
    target_hdg = init_hdg + rng.uniform(-40, 40)          # up to a 40 deg turn
    init_alt = 5000.0 + rng.uniform(-100, 100)             # up to 100 ft off target
    turb_sev = int(rng.choice([0, 1, 2], p=[0.3, 0.35, 0.35]))   # 30% calm, 70% light gusts
    turb_ws = float(rng.uniform(10, 20))

    opts = dict(init_hdg_deg=init_hdg, init_alt_ft=init_alt, target_hdg_deg=target_hdg,
                turb_severity=turb_sev, turb_windspeed_fps=turb_ws)
    obs, _ = env.reset(seed=seed, options=opts)
    pid.reset()

    n_steps = 0
    lost = False
    while True:
        s = env.raw_state()
        a = pid.act(s)
        obs_log.append(obs.copy())
        act_log.append(a.copy())
        ep_id_log.append(ep)
        obs, _, terminated, truncated, _ = env.step(a)
        n_steps += 1
        if terminated:
            lost = True
            break
        if truncated:
            break

    summaries.append(dict(ep=ep, seed=seed, steps=n_steps, lost=lost, turb_sev=turb_sev))
    if (ep + 1) % 50 == 0:
        print(f"  {ep+1}/{N_EPISODES} episodes generated")

obs_arr = np.asarray(obs_log, dtype=np.float32)
act_arr = np.asarray(act_log, dtype=np.float32)
ep_arr = np.asarray(ep_id_log, dtype=np.int32)

lost_eps = {s["ep"] for s in summaries if s["lost"]}
print(f"\n{N_EPISODES} episodes, {len(obs_arr)} (obs, action) pairs total")
print(f"PID lost control in {len(lost_eps)} episode(s): {sorted(lost_eps) if lost_eps else 'none'}")
print("Excluding any lost episodes from the training set (a controller mid-failure is not a demonstration to imitate).")

keep_mask = ~np.isin(ep_arr, list(lost_eps))
obs_arr, act_arr, ep_arr = obs_arr[keep_mask], act_arr[keep_mask], ep_arr[keep_mask]
kept_eps = sorted(set(ep_arr.tolist()))

# Split by EPISODE, not by row, so validation rows are never adjacent-in-time to training rows.
rng.shuffle(kept_eps)
n_val = max(1, int(len(kept_eps) * VAL_FRACTION))
val_eps = set(kept_eps[:n_val])
is_val = np.isin(ep_arr, list(val_eps))

np.savez_compressed(
    "pid_dataset.npz",
    obs=obs_arr, act=act_arr, episode=ep_arr, is_val=is_val,
)
print(f"Saved pid_dataset.npz: {len(obs_arr)} rows, {len(kept_eps)} episodes "
      f"({len(kept_eps)-n_val} train / {n_val} val)")

# --- sanity plot: a few example episodes across the turbulence range ---
example_eps = [s["ep"] for s in summaries if not s["lost"]][:6]
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 7))
for ep in example_eps:
    rows = np.where(ep_arr == ep)[0]
    if len(rows) == 0:
        continue
    t = np.arange(len(rows)) / env.CONTROL_HZ
    o = obs_arr[rows]
    axes[0].plot(t, o[:, 0] * 100, alpha=0.7, label=f"ep {ep} (sev {summaries[ep]['turb_sev']})")
    axes[1].plot(t, o[:, 1] * 30, alpha=0.7)
    axes[2].plot(t, o[:, 2] * 30, alpha=0.7)
for ax, label in zip(axes, ["alt error (ft)", "heading error (deg)", "bank (deg)"]):
    ax.set_ylabel(label); ax.grid(alpha=0.3)
axes[0].legend(fontsize=7, ncol=2)
axes[-1].set_xlabel("time (s)")
axes[0].set_title("Sample training episodes (PID + turbulence)")
plt.tight_layout(); plt.savefig("step4_sample_episodes.png", dpi=150)
print("saved step4_sample_episodes.png")

# --- action distribution, since step5's model needs to know the target range ---
print(f"elevator cmd: min {act_arr[:,0].min():+.3f} max {act_arr[:,0].max():+.3f} std {act_arr[:,0].std():.3f}")
print(f"aileron  cmd: min {act_arr[:,1].min():+.3f} max {act_arr[:,1].max():+.3f} std {act_arr[:,1].std():.3f}")
