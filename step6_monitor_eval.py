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

def bc_act(obs):
    with torch.no_grad():
        return model(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()

def fly(mode, opts, seconds):
    """mode: 'bc' (model only), 'pid' (baseline only), 'monitored' (model + monitor + PID fallback)."""
    obs, _ = env.reset(seed=0, options=opts)
    pid.reset(trim_pitch_deg=env.trim_pitch_deg)   # correct even if PID is engaged later, mid-disturbance
    monitor = SafetyMonitor()
    rows = []
    for _ in range(int(seconds * env.CONTROL_HZ)):
        s = env.raw_state()
        if mode == "pid":
            a, active = pid.act(s), "pid"
        elif mode == "bc":
            a, active = bc_act(obs), "bc"
        else:
            use_pid = monitor.check(s, env.t)
            a = pid.act(s) if use_pid else bc_act(obs)
            active = "pid" if use_pid else "bc"
        obs, _, terminated, _, _ = env.step(a)
        rows.append((env.t, s["alt_err_ft"], s["hdg_err_deg"], s["bank_deg"], s["cas_kts"], 1 if active == "pid" else 0))
        if terminated:
            break
    trip_info = (monitor.trip_time, monitor.trip_reason) if mode == "monitored" else (None, None)
    return np.array(rows), terminated, trip_info


def plot_case(name, opts, seconds, fname):
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    d_bc, _, _ = fly("bc", opts, seconds)
    d_mon, _, (trip_t, trip_reason) = fly("monitored", opts, seconds)
    axes[0].plot(d_bc[:, 0], d_bc[:, 1], "--", color="tab:orange", label="BC alone")
    axes[0].plot(d_mon[:, 0], d_mon[:, 1], "-", color="tab:green", label="BC + monitor")
    axes[0].axhline(35, color="gray", lw=0.8, ls=":"); axes[0].axhline(-35, color="gray", lw=0.8, ls=":")
    axes[0].set_ylabel("alt error (ft)"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].fill_between(d_mon[:, 0], d_mon[:, 5], step="post", alpha=0.4, color="tab:blue")
    axes[1].set_ylabel("PID active (1) /\nBC active (0)"); axes[1].set_ylim(-0.1, 1.1)
    axes[1].set_xlabel("time (s)"); axes[1].grid(alpha=0.3)
    title = f"{name}\nmonitor "
    title += f"tripped at t={trip_t:.1f}s: {trip_reason}" if trip_t is not None else "never tripped"
    axes[0].set_title(title, fontsize=10)
    plt.tight_layout(); plt.savefig(fname, dpi=150)
    print(f"{name}: {title}")
    return d_bc, d_mon


print("=== Case A: starts inside the envelope, drifts out mid-flight ===")
plot_case("Case A (start 25 ft high, level flight, 120s)",
          dict(init_hdg_deg=90.0, init_alt_ft=5025.0, target_hdg_deg=90.0),
          120, "step6_case_a_midflight_catch.png")

print("\n=== Case B: starts already outside the envelope ===")
plot_case("Case B (start 80 ft high, heading +30, 90s)",
          dict(init_hdg_deg=90.0, init_alt_ft=5080.0, target_hdg_deg=120.0),
          90, "step6_case_b_immediate_fallback.png")

print("\n=== Table: monitor performance across the step3/5 case set (120s each) ===")
cases = {
    "hold, no offset":         dict(init_hdg_deg=90.0,  init_alt_ft=5000.0, target_hdg_deg=90.0),
    "altitude +50 ft":         dict(init_hdg_deg=90.0,  init_alt_ft=5050.0, target_hdg_deg=90.0),
    "altitude -50 ft":         dict(init_hdg_deg=90.0,  init_alt_ft=4950.0, target_hdg_deg=90.0),
    "heading +20 deg":         dict(init_hdg_deg=90.0,  init_alt_ft=5000.0, target_hdg_deg=110.0),
    "heading -20 deg":         dict(init_hdg_deg=90.0,  init_alt_ft=5000.0, target_hdg_deg=70.0),
    "cross north 350->10":     dict(init_hdg_deg=350.0, init_alt_ft=5000.0, target_hdg_deg=10.0),
    "combined hdg+30, alt-80": dict(init_hdg_deg=200.0, init_alt_ft=4920.0, target_hdg_deg=230.0),
}
print(f"{'case':26s} {'lost(bc)':9s} {'lost(mon)':10s} {'trip t':>8s} {'reason':40s} {'max|alt|(bc)':>13s} {'max|alt|(mon)':>13s}")
for name, opts in cases.items():
    d_bc, lost_bc, _ = fly("bc", opts, 120)
    d_mon, lost_mon, (trip_t, reason) = fly("monitored", opts, 120)
    reason = reason or "-"
    trip_str = f"{trip_t:.1f}s" if trip_t is not None else "none"
    print(f"{name:26s} {str(lost_bc):9s} {str(lost_mon):10s} {trip_str:>8s} {reason:40s} "
          f"{np.abs(d_bc[:,1]).max():12.1f}ft {np.abs(d_mon[:,1]).max():12.1f}ft")

print("\n=== Sanity: does the monitor ever trip on the PID itself (nominal, no BC involved)? ===")
for name, opts in cases.items():
    obs, _ = env.reset(seed=0, options=opts); pid.reset(trim_pitch_deg=env.trim_pitch_deg)
    m2 = SafetyMonitor()
    for _ in range(120 * env.CONTROL_HZ):
        s = env.raw_state()
        m2.check(s, env.t)
        _, _, te, _, _ = env.step(pid.act(s))
        if te: break
    print(f"{name:26s} PID-alone against monitor thresholds: tripped={m2.tripped}"
          + (f" at t={m2.trip_time:.1f}s ({m2.trip_reason})" if m2.tripped else ""))
