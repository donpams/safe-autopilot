"""Domain randomization wrapper for training.

Matches the same randomization the v1 BC dataset used, so a v1-vs-v2
comparison later is apples-to-apples: randomized initial heading,
randomized initial altitude offset, randomized heading-target offset
(all already inside C172Env.reset's defaults), PLUS a turbulence-severity
mix layered on top here.

Turbulence mix rationale: step7's stress testing found the PID fallback
itself (no model, no monitor) crashes 55-75% of the time at turbulence
severity >=3, purely from airspeed bleeding through the 50kt floor --
a limit of the fallback controller, not something a monitor or a better
learned policy can be expected to fix. Training the RL policy on
severities where even the safety net can't guarantee a safe outcome
would teach it to solve an un-survivable problem instead of a flyable
one. So the training mix stays inside severity 0-2, where the fallback
itself is proven reliable, matching the envelope the final system will
actually be evaluated and deployed within.
"""
import gymnasium as gym


class DomainRandomizedC172Env(gym.Wrapper):
    def __init__(self, env, turb_weights=(0.4, 0.3, 0.3), seed=None):
        """turb_weights: probability of turb_severity = 0, 1, 2 respectively."""
        super().__init__(env)
        assert len(turb_weights) == 3
        self.turb_weights = turb_weights
        self._rng = gym.utils.seeding.np_random(seed)[0]

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = gym.utils.seeding.np_random(seed)[0]
        opts = dict(options or {})
        if "turb_severity" not in opts:
            opts["turb_severity"] = int(self._rng.choice([0, 1, 2], p=self.turb_weights))
        return self.env.reset(seed=seed, options=opts)
