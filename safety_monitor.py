class SafetyMonitor:
    """Watches the aircraft's raw state and decides whether the primary
    (learned) controller may keep flying, or whether the PID fallback
    must take over.

    Design choice: LATCHING fallback. Once the envelope is violated, the
    monitor stays on the PID for the rest of the episode rather than
    handing control back once the state re-enters the envelope. A
    controller that has already produced one unsafe excursion hasn't
    demonstrated it won't do it again a few seconds later, and switching
    back and forth ("chattering") right at a boundary is its own hazard.
    This is a simplifying choice, not a fundamental one -- see the
    writeup for what a hysteresis / re-entry-band design would add.

    Thresholds are set from evidence, not guesses: they sit with real
    margin above the PID's own worst-case numbers in nominal flight
    (see step 3 / step 5 eval tables), and well inside the environment's
    hard "lost control" limits (bank 70 deg, pitch 40 deg, alt err 500 ft,
    airspeed 50 kts) so the monitor always has room to recover.
    """

    def __init__(self,
                 max_bank_deg=30.0,
                 max_pitch_deg=12.0,
                 max_alt_err_ft=35.0,
                 min_cas_kts=85.0,
                 max_cas_kts=120.0):
        self.max_bank_deg = max_bank_deg
        self.max_pitch_deg = max_pitch_deg
        self.max_alt_err_ft = max_alt_err_ft
        self.min_cas_kts = min_cas_kts
        self.max_cas_kts = max_cas_kts
        self.reset()

    def reset(self):
        self.tripped = False
        self.trip_reason = None
        self.trip_time = None

    def _violation(self, s):
        if abs(s["bank_deg"]) > self.max_bank_deg:
            return f"bank {s['bank_deg']:+.1f} deg (limit {self.max_bank_deg})"
        if abs(s["pitch_deg"]) > self.max_pitch_deg:
            return f"pitch {s['pitch_deg']:+.1f} deg (limit {self.max_pitch_deg})"
        if abs(s["alt_err_ft"]) > self.max_alt_err_ft:
            return f"altitude error {s['alt_err_ft']:+.1f} ft (limit {self.max_alt_err_ft})"
        if s["cas_kts"] < self.min_cas_kts:
            return f"airspeed {s['cas_kts']:.1f} kts (floor {self.min_cas_kts})"
        if s["cas_kts"] > self.max_cas_kts:
            return f"airspeed {s['cas_kts']:.1f} kts (ceiling {self.max_cas_kts})"
        return None

    def check(self, s, t):
        """Call once per control step with the CURRENT raw state, before
        deciding which controller's action to use. Returns True if the
        PID fallback should fly this step (either just tripped, or
        already tripped earlier in this episode)."""
        if not self.tripped:
            reason = self._violation(s)
            if reason is not None:
                self.tripped = True
                self.trip_reason = reason
                self.trip_time = t
        return self.tripped
