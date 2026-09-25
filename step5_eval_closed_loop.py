import numpy as np
import torch
import matplotlib.pyplot as plt
from flight_env import C172Env
from pid_controller import C172PID
from bc_model import BCPolicy

env = C172Env()
pid = C172PID()
model = BCPolicy()
model.load_state_dict(torch.load("bc_policy.pt"))
model.eval()

def bc_act(obs):
    with torch.no_grad():
        return model(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()

def fly(controller, opts, seconds=60):
    obs, _ = env.reset(seed=0, options=opts)
    if controller == "pid":
        pid.reset()
    rows = []
    for _ in range(int(seconds * env.CONTROL_HZ)):
        s = env.raw_state()
        a = pid.act(s) if controller == "pid" else bc_act(obs)
        obs, _, terminated, _, _ = env.step(a)
        rows.append((env.t, s["alt_err_ft"], s["hdg_err_deg"], s["bank_deg"], s["pitch_deg"], s["cas_kts"], a[0]))
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

print(f"--- 60 s closed-loop comparison ---")
print(f"{'case':26s} {'ctrl':4s} {'lost?':6s} {'alt settle':>10s} {'alt |err| @60s':>14s} {'hdg settle':>10s} {'max bank':>9s} {'min cas':>8s}")
results = {}
for name, opts in cases.items():
    for ctrl in ["pid", "bc"]:
        d, lost = fly(ctrl, opts, seconds=60)
        t = d[:, 0]
        final_alt = np.abs(d[t >= 50, 1]).mean() if (t >= 50).any() else np.abs(d[-1, 1])
        print(f"{name:26s} {ctrl:4s} {str(lost):6s} {last_outside(t, d[:,1], 5):9.1f}s {final_alt:13.1f}ft "
              f"{last_outside(t, d[:,2], 2):9.1f}s {np.abs(d[:,3]).max():8.1f}d {d[:,5].min():7.1f}k")
        results[(name, ctrl)] = d

fig, axes = plt.subplots(4, 1, sharex=True, figsize=(8, 9))
for ctrl, style in [("pid", "-"), ("bc", "--")]:
    d = results[("combined hdg+30, alt-80", ctrl)]
    axes[0].plot(d[:, 0], d[:, 1], style, label=ctrl)
    axes[1].plot(d[:, 0], d[:, 2], style, label=ctrl)
    axes[2].plot(d[:, 0], d[:, 3], style, label=ctrl)
    axes[3].plot(d[:, 0], d[:, 5], style, label=ctrl)
for ax, label in zip(axes, ["alt error (ft)", "heading error (deg)", "bank (deg)", "airspeed (kts)"]):
    ax.set_ylabel(label); ax.grid(alpha=0.3); ax.legend()
axes[0].set_title("PID vs behavior-cloned model, 60 s: heading +30, altitude -80")
axes[-1].set_xlabel("time (s)")
plt.tight_layout(); plt.savefig("step5_pid_vs_bc.png", dpi=150)
print("saved step5_pid_vs_bc.png")

print("\n--- 120 s longer-horizon diagnostic (altitude +50 ft, no heading change) ---")
opts = dict(init_hdg_deg=90.0, init_alt_ft=5050.0, target_hdg_deg=90.0)
long_rows = {}
for ctrl in ["pid", "bc"]:
    d, lost = fly(ctrl, opts, seconds=120)
    long_rows[ctrl] = d
    print(f"{ctrl}: lost={lost}, alt error at 60s={d[d[:,0]<=60][-1,1]:+.1f} ft, "
          f"at 120s={d[-1,1]:+.1f} ft (or ended early at t={d[-1,0]:.0f}s)")

fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 7))
for ctrl, style in [("pid", "-"), ("bc", "--")]:
    d = long_rows[ctrl]
    axes[0].plot(d[:, 0], d[:, 1], style, label=ctrl)
    axes[1].plot(d[:, 0], d[:, 4], style, label=ctrl)
    axes[2].plot(d[:, 0], d[:, 6], style, label=ctrl)
for ax, label in zip(axes, ["alt error (ft)", "pitch (deg)", "elevator cmd"]):
    ax.set_ylabel(label); ax.grid(alpha=0.3); ax.legend()
axes[0].axhline(0, color="k", lw=0.5)
axes[0].set_title("120 s altitude capture from +50 ft: PID vs behavior-cloned model")
axes[-1].set_xlabel("time (s)")
plt.tight_layout(); plt.savefig("step5_bc_longrun_divergence.png", dpi=150)
print("saved step5_bc_longrun_divergence.png")
