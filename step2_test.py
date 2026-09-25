import numpy as np
import matplotlib.pyplot as plt
from flight_env import C172Env

env = C172Env()
obs, info = env.reset(seed=0, options={"init_hdg_deg": 0.0, "init_alt_ft": 5000.0, "target_hdg_deg": 0.0})

# Test 1: zero action (hands off). Should look like the step 1 flight.
log = {k: [] for k in ["alt_ft", "hdg_err_deg", "bank_deg", "pitch_deg", "cas_kts"]}
t = []
done = False
while not done:
    obs, r, term, trunc, info = env.step(np.zeros(2))
    done = term or trunc
    t.append(env.t)
    for k in log:
        log[k].append(info["raw"][k])
print(f"Test 1 (hands off): ran {t[-1]:.1f} s, terminated={term}, truncated={trunc}")
print(f"  altitude drift {log['alt_ft'][-1]-5000:.2f} ft, speed drift {log['cas_kts'][-1]-100:.3f} kts")

fig, axes = plt.subplots(5, 1, sharex=True, figsize=(8, 10))
for ax, k in zip(axes, log):
    ax.plot(t, log[k]); ax.set_ylabel(k)
axes[-1].set_xlabel("time (s)")
plt.tight_layout(); plt.savefig("step2_handsoff.png", dpi=150)

# Test 2: random actions. Should eventually terminate, and never crash Python.
obs, _ = env.reset(seed=1)
steps = 0
done = False
while not done:
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    steps += 1
    done = term or trunc
print(f"Test 2 (random actions): ended after {steps} steps ({steps/20:.1f} s), terminated={term}, truncated={trunc}")
print(f"  obs shape {obs.shape}, finite={np.isfinite(obs).all()}")