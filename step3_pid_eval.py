import numpy as np
import matplotlib.pyplot as plt
from flight_env import C172Env
from pid_controller import C172PID

env = C172Env()
pid = C172PID()

def fly(opts, seconds=60):
    env.reset(seed=0, options=opts); pid.reset()
    rows = []
    for _ in range(int(seconds * env.CONTROL_HZ)):
        s = env.raw_state()
        a = pid.act(s)
        _, _, terminated, _, _ = env.step(a)
        rows.append((env.t, s["alt_err_ft"], s["hdg_err_deg"], s["bank_deg"], s["pitch_deg"],
                     s["cas_kts"], a[0], a[1]))
        if terminated:
            break
    return np.array(rows), terminated

def last_outside(t, err, band):
    bad = np.where(np.abs(err) > band)[0]
    return 0.0 if len(bad) == 0 else t[bad[-1]]

cases = {
    "hold, no offset":         dict(init_hdg_deg=90.0,  init_alt_ft=5000.0, target_hdg_deg=90.0),
    "altitude +50 ft":         dict(init_hdg_deg=90.0,  init_alt_ft=5050.0, target_hdg_deg=90.0),
    "altitude -50 ft":         dict(init_hdg_deg=90.0,  init_alt_ft=4950.0, target_hdg_deg=90.0),
    "heading +20 deg":         dict(init_hdg_deg=90.0,  init_alt_ft=5000.0, target_hdg_deg=110.0),
    "heading -20 deg":         dict(init_hdg_deg=90.0,  init_alt_ft=5000.0, target_hdg_deg=70.0),
    "cross north 350->10":     dict(init_hdg_deg=350.0, init_alt_ft=5000.0, target_hdg_deg=10.0),
    "combined hdg+30, alt-80": dict(init_hdg_deg=200.0, init_alt_ft=4920.0, target_hdg_deg=230.0),
}

print(f"{'case':26s} {'lost?':6s} {'alt settle':>10s} {'alt over':>9s} {'hdg settle':>10s} {'max bank':>9s} {'min cas':>8s}")
results = {}
for name, opts in cases.items():
    d, lost = fly(opts)
    t = d[:, 0]
    init_alt = d[0, 1]
    over = max(0.0, (-d[:, 1].min() if init_alt > 0 else d[:, 1].max())) if abs(init_alt) > 1 else float(np.abs(d[:, 1]).max())
    print(f"{name:26s} {str(lost):6s} {last_outside(t, d[:,1], 5):9.1f}s {over:8.1f}ft "
          f"{last_outside(t, d[:,2], 2):9.1f}s {np.abs(d[:,3]).max():8.1f}d {d[:,5].min():7.1f}k")
    results[name] = d

# Plot the combined case
d = results["combined hdg+30, alt-80"]
fig, axes = plt.subplots(5, 1, sharex=True, figsize=(8, 10))
for ax, col, label in zip(axes, [1, 2, 3, 5, 6], ["alt error (ft)", "heading error (deg)", "bank (deg)", "airspeed (kts)", "elevator cmd"]):
    ax.plot(d[:, 0], d[:, col]); ax.set_ylabel(label); ax.grid(alpha=0.3)
axes[0].set_title("PID baseline: heading +30 deg and altitude -80 ft at the same time")
axes[-1].set_xlabel("time (s)")
plt.tight_layout(); plt.savefig("step3_pid_combined.png", dpi=150)
print("saved step3_pid_combined.png")
