import jsbsim
import matplotlib.pyplot as plt

fdm = jsbsim.FGFDMExec(None)   # None = use the package's bundled data
fdm.set_debug_level(0)
fdm.load_model("c172x")

# Initial conditions: level flight, 5000 ft, ~100 kts, heading north
fdm["ic/h-sl-ft"] = 5000
fdm["ic/vc-kts"] = 100
fdm["ic/psi-true-deg"] = 0
fdm.run_ic()

fdm["propulsion/set-running"] = -1      # start engine
fdm["fcs/throttle-cmd-norm"] = 0.65
fdm["simulation/do_simple_trim"] = 1    # let JSBSim find a steady trim

print("Trim controls:")
for p in ["fcs/throttle-cmd-norm", "fcs/elevator-cmd-norm",
          "fcs/aileron-cmd-norm", "fcs/rudder-cmd-norm"]:
    print(f"  {p} = {fdm[p]:.4f}")

log = {k: [] for k in ["t", "alt_ft", "hdg_deg", "bank_deg", "pitch_deg", "cas_kts"]}
while fdm.get_sim_time() < 60:
    fdm.run()
    log["t"].append(fdm.get_sim_time())
    log["alt_ft"].append(fdm["position/h-sl-ft"])
    log["hdg_deg"].append(fdm["attitude/psi-deg"])
    log["bank_deg"].append(fdm["attitude/phi-deg"])
    log["pitch_deg"].append(fdm["attitude/theta-deg"])
    log["cas_kts"].append(fdm["velocities/vc-kts"])

# Wrap heading to [-180, 180) so 359.99 deg shows as -0.01 deg, not a 360 jump
log["hdg_deg"] = [((h + 180) % 360) - 180 for h in log["hdg_deg"]]

fig, axes = plt.subplots(5, 1, sharex=True, figsize=(8, 10))
for ax, k in zip(axes, list(log)[1:]):
    ax.plot(log["t"], log[k]); ax.set_ylabel(k)
axes[-1].set_xlabel("time (s)")
plt.tight_layout(); plt.savefig("step1_trim_flight.png", dpi=150)