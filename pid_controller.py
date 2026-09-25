import numpy as np


def clip(x, lo, hi):
    return max(lo, min(hi, x))


class C172PID:
    """Cascaded altitude-hold + heading-hold controller for the C172 env.

    Outer loops turn errors into attitude targets; inner loops turn attitude
    errors into stick commands, using gyro rates (q, p) as damping.

        altitude error --PI--> pitch target --P + q damping--> elevator
        heading error  --P---> bank target  --P + p damping--> aileron

    Sign conventions (measured in JSBSim): +elevator = nose down,
    +aileron = roll right.
    """

    def __init__(self, dt=0.05):
        self.dt = dt
        # altitude / pitch loops
        self.k_alt_p = 0.060      # deg of pitch target per ft of altitude error
        self.k_alt_i = 0.006      # deg per (ft*s)
        self.k_vs = 0.05          # deg of pitch target per ft/s of vertical speed (damping)
        self.max_pitch_cmd = 6.0  # deg, limit on pitch target
        self.int_zone_ft = 15.0   # integrate only when this close to target (prevents windup overshoot)
        self.k_pitch_p = 0.16     # elevator per deg of pitch error
        self.k_q = 0.5            # elevator per rad/s of pitch rate (damping)
        # heading / bank loops
        self.k_hdg_p = 1.0        # deg of bank target per deg of heading error
        self.max_bank_cmd = 20.0  # deg, limit on bank target
        self.pitch_rate_limit = 2.0   # deg/s, max change rate of the pitch target (no step commands)
        self.bank_rate_limit = 8.0    # deg/s, max change rate of the bank target
        self.k_bank_p = 0.03      # aileron per deg of bank error
        self.k_p_roll = 0.10      # aileron per rad/s of roll rate (damping)
        self.reset()

    def reset(self, trim_pitch_deg=None):
        """trim_pitch_deg: the aircraft's true level-flight pitch (from
        env.trim_pitch_deg). Pass it whenever the PID might be engaged
        away from a fresh reset -- e.g. as a monitor's fallback -- so it
        doesn't mistake a momentarily disturbed attitude for "level."
        If None, it falls back to capturing whatever pitch it sees on
        its first act() call, which is only correct if that first call
        happens right at a stable reset."""
        self.alt_int = 0.0
        self.pitch_trim = trim_pitch_deg
        self.pitch_cmd_prev = trim_pitch_deg
        self.bank_cmd_prev = 0.0

    def act(self, s):
        """s = env.raw_state(). Returns action array [elevator, aileron_delta]."""
        if self.pitch_trim is None:
            self.pitch_trim = s["pitch_deg"]           # trimmed level-flight pitch at engagement
            self.pitch_cmd_prev = self.pitch_trim
        # ---- altitude -> pitch target ----
        alt_err = s["alt_err_ft"]                      # + means too high
        # integrate only while the pitch target is not saturated (anti-windup)
        pitch_cmd_unsat = self.pitch_trim - (self.k_alt_p * alt_err + self.k_alt_i * self.alt_int + self.k_vs * s["vs_fps"])
        pitch_cmd = clip(pitch_cmd_unsat, self.pitch_trim - self.max_pitch_cmd, self.pitch_trim + self.max_pitch_cmd)
        if pitch_cmd == pitch_cmd_unsat and abs(alt_err) < self.int_zone_ft:
            self.alt_int += alt_err * self.dt
        elif abs(alt_err) >= self.int_zone_ft:
            self.alt_int = 0.0                          # far from target: start fresh
        step = self.pitch_rate_limit * self.dt                  # slew-limit the target
        pitch_cmd = self.pitch_cmd_prev + clip(pitch_cmd - self.pitch_cmd_prev, -step, step)
        self.pitch_cmd_prev = pitch_cmd
        # ---- pitch -> elevator (+elevator = nose down) ----
        elevator = -(self.k_pitch_p * (pitch_cmd - s["pitch_deg"]) - self.k_q * s["q_rps"])

        # ---- heading -> bank target ----
        hdg_err = -s["hdg_err_deg"]                    # desired minus current, already wrapped
        bank_cmd = clip(self.k_hdg_p * hdg_err, -self.max_bank_cmd, self.max_bank_cmd)
        step = self.bank_rate_limit * self.dt
        bank_cmd = self.bank_cmd_prev + clip(bank_cmd - self.bank_cmd_prev, -step, step)
        self.bank_cmd_prev = bank_cmd
        # ---- bank -> aileron (+aileron = roll right) ----
        aileron = self.k_bank_p * (bank_cmd - s["bank_deg"]) - self.k_p_roll * s["p_rps"]

        return np.array([clip(elevator, -1, 1), clip(aileron, -1, 1)])
