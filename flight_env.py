import tempfile

import gymnasium as gym
import numpy as np
import jsbsim
from gymnasium import spaces

def wrap180(deg):
    """Wrap an angle in degrees to [-180, 180)."""
    return (deg + 180.0) % 360.0 - 180.0

class C172Env(gym.Env):
    """JSBSim Cessna 172 wrapped as a Gymnasium environment.

    Actions  (2, in [-1, 1]): elevator command, aileron delta from trim.
    Obs      (8, roughly in [-1, 1]): normalized errors and attitude/rates.
    Throttle and rudder are held at their trim values (kept simple on purpose).
    """

    SIM_HZ = 120
    CONTROL_HZ = 20
    MAX_TIME_S = 60.0

    def __init__(self, seed=None):
        super().__init__()
        self.fdm = None
        self.substeps = self.SIM_HZ // self.CONTROL_HZ

        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)

        self.target_alt_ft = 5000.0
        self.target_hdg_deg = 0.0
        self.target_cas_kts = 100.0
        self.rng = np.random.default_rng(seed)

    def _new_fdm(self):
        """Fresh sim per episode: ~12 ms, and guarantees no state leaks between episodes."""
        f = jsbsim.FGFDMExec(None)
        f.set_debug_level(0)
        f.set_output_path(tempfile.gettempdir())   # keep the model's CSV log out of your repo
        f.load_model("c172x")
        return f

    # ---------- state access ----------
    def raw_state(self):
        """Physical units. The PID and the safety monitor will read this."""
        f = self.fdm
        return {
            "alt_ft": f["position/h-sl-ft"],
            "hdg_deg": f["attitude/psi-deg"],
            "bank_deg": f["attitude/phi-deg"],
            "pitch_deg": f["attitude/theta-deg"],
            "cas_kts": f["velocities/vc-kts"],
            "vs_fps": f["velocities/h-dot-fps"],   # vertical speed, ft/s (+ = climbing)
            "p_rps": f["velocities/p-rad_sec"],
            "q_rps": f["velocities/q-rad_sec"],
            "r_rps": f["velocities/r-rad_sec"],
            "alt_err_ft": f["position/h-sl-ft"] - self.target_alt_ft,
            "hdg_err_deg": wrap180(f["attitude/psi-deg"] - self.target_hdg_deg),
        }

    def _obs(self):
        s = self.raw_state()
        return np.array([
            s["alt_err_ft"] / 100.0,
            s["hdg_err_deg"] / 30.0,
            s["bank_deg"] / 30.0,
            s["pitch_deg"] / 15.0,
            s["p_rps"], s["q_rps"], s["r_rps"],
            (s["cas_kts"] - self.target_cas_kts) / 10.0,
        ], dtype=np.float32)

    # ---------- Gymnasium API ----------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        opts = options or {}
        f = self.fdm = self._new_fdm()

        init_hdg = opts.get("init_hdg_deg", self.rng.uniform(0, 360))
        init_alt = opts.get("init_alt_ft", self.target_alt_ft + self.rng.uniform(-50, 50))
        self.target_hdg_deg = wrap180(opts.get("target_hdg_deg", init_hdg + self.rng.uniform(-20, 20))) % 360

        f["ic/h-sl-ft"] = init_alt
        f["ic/vc-kts"] = self.target_cas_kts
        f["ic/psi-true-deg"] = init_hdg
        f.run_ic()
        f["propulsion/set-running"] = -1
        f["fcs/throttle-cmd-norm"] = 0.65
        f["simulation/do_simple_trim"] = 1   # must trim BEFORE enabling turbulence, or trim fails

        # Turbulence (Tustin/Dryden-family gust model). severity: 0 = off,
        # else JSBSim's MIL-F-8785C-style scale, roughly 1=light .. 6=severe.
        turb_severity = opts.get("turb_severity", 0)
        if turb_severity > 0:
            # atmosphere/randomseed must be set or every episode gets the
            # identical gust sequence (JSBSim defaults to a fixed seed).
            f["atmosphere/randomseed"] = int(self.rng.integers(0, 2**31 - 1))
            f["atmosphere/turb-type"] = 4
            f["atmosphere/turbulence/milspec/severity"] = turb_severity
            f["atmosphere/turbulence/milspec/windspeed_at_20ft_AGL-fps"] = opts.get("turb_windspeed_fps", 15)
        self.turb_severity = turb_severity

        # Remember trim so actions are deviations from it.
        self.trim_aileron = f["fcs/aileron-cmd-norm"]
        self.trim_pitch_deg = f["attitude/theta-deg"]   # true level-flight trim, for any controller engaged later
        f["fcs/elevator-cmd-norm"] = 0.0
        self.t = 0.0
        self.n_steps = 0
        return self._obs(), {"raw": self.raw_state()}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        f = self.fdm
        f["fcs/elevator-cmd-norm"] = float(a[0])
        f["fcs/aileron-cmd-norm"] = float(np.clip(self.trim_aileron + a[1], -1.0, 1.0))
        for _ in range(self.substeps):
            f.run()
        self.n_steps += 1
        self.t = self.n_steps / self.CONTROL_HZ   # count steps; don't trust the sim clock across resets

        s = self.raw_state()

        # "Crashed / lost control" limits, deliberately looser than the
        # safety monitor's envelope, which we define in step 6.
        terminated = bool(abs(s["bank_deg"]) > 70 or abs(s["pitch_deg"]) > 40
                          or abs(s["alt_err_ft"]) > 500 or s["cas_kts"] < 50)
        truncated = bool(self.t >= self.MAX_TIME_S)

        reward = self._reward(s, a, terminated)

        return self._obs(), float(reward), terminated, truncated, {"raw": s}

    def _reward(self, s, action, terminated):
        """v2 shaped reward for RL (BC never used the placeholder; it only
        copied the PID's actions).

        REVISION NOTE (after the first 500k-step training run): the first
        version of this reward had an unbounded per-step tracking/airspeed
        penalty plus a flat -50 one-time crash penalty. That let PPO find a
        real exploit: once a rollout starts diverging, the *per-step*
        penalty (up to ~-16/step right at the termination boundary) adds up
        to far more than -50 over even a few dozen steps, so "crash on
        purpose, eat the -50 once" was the rational move whenever the
        policy hadn't yet learned to recover. ep_len_mean collapsed from
        ~400 steps early in training down to ~25 steps and stayed there --
        the policy learned to end episodes fast, not to fly. This is the
        textbook "reward hacking via early termination" failure mode,
        caught here only because the full 500k-step run was actually
        inspected rather than trusted on the strength of the pre-training
        sanity check (which only tested PID-good and instant-max-crash,
        never a slowly-diverging mediocre policy -- that gap is exactly
        where this broke).

        Fix, now in two parts instead of one:
          1. Tracking + airspeed penalties are CAPPED per step, so no
             matter how far off course the aircraft gets, one step can
             never cost more than a bounded amount -- there's no more
             "run away unboundedly, so better to just crash" arithmetic.
          2. A small positive ALIVE bonus is paid every step the episode
             continues. This makes survival itself valuable (the standard
             fix used in e.g. Gymnasium's Hopper/Ant "healthy_reward"),
             so ending the episode early always costs the policy real,
             foregone future reward -- without needing a hand-tuned crash
             constant large enough to dominate every possible bad patch.
        The crash penalty stays, but now only needs to be a clear, modest
        deterrent (not the sole thing standing between "fly" and "dive"),
        since the capped per-step cost + alive bonus already remove the
        incentive to end the episode on purpose.
        """
        track = min(abs(s["alt_err_ft"]) / 100.0, 2.0) + min(abs(s["hdg_err_deg"]) / 30.0, 1.0)
        airspeed = min(abs(s["cas_kts"] - self.target_cas_kts) / 10.0, 1.0)
        effort = 0.05 * float(np.sum(np.square(action)))
        alive_bonus = 1.0

        reward = alive_bonus - (track + airspeed + effort)
        if terminated:
            reward -= 20.0
        return reward
