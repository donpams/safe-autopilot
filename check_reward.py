"""Sanity-check the v2 reward shaping before trusting it to train anything.

Three checks:
  1. PID flying nominally -> reward should stay small-magnitude and stable
     (the policy we'd eventually want to match/beat starts here).
  2. Random actions -> reward should be clearly worse (more negative) than
     the PID, since random elevator/aileron is a worse controller.
  3. A forced crash (hold full nose-down elevator) -> confirm the -50
     terminal penalty actually fires and dominates.
"""
import numpy as np
from flight_env import C172Env
from pid_controller import C172PID


def run_episode(policy_fn, label, seed=0, max_steps=None):
    env = C172Env(seed=seed)
    obs, info = env.reset(seed=seed, options={"init_hdg_deg": 0.0, "init_alt_ft": 5000.0, "target_hdg_deg": 0.0})
    total_reward = 0.0
    n = 0
    terminated = truncated = False
    while not (terminated or truncated):
        s = env.raw_state()
        action = policy_fn(env, s)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        n += 1
        if max_steps and n >= max_steps:
            break
    print(f"{label:20s} steps={n:4d} terminated={str(terminated):5s} "
          f"total_reward={total_reward:8.1f} mean_reward/step={total_reward/n:6.3f}")
    return total_reward, n, terminated


pid = C172PID(dt=1.0 / 20)
pid_env_ref = {}

def pid_policy(env, s):
    if "trim" not in pid_env_ref:
        pid.reset(trim_pitch_deg=env.trim_pitch_deg)
        pid_env_ref["trim"] = True
    return pid.act(s)

rng = np.random.default_rng(1)
def random_policy(env, s):
    return rng.uniform(-1, 1, size=2)

def crash_policy(env, s):
    return np.array([1.0, 0.0])   # full nose-down elevator, held -> should trip terminated quickly

def mediocre_policy(env, s):
    """PID action plus a small constant bias -- never recovers, drifts
    slowly off course for the WHOLE episode without ever quite tripping
    termination early. This is exactly the case the first reward design
    missed: it's not clean (like PID) and it's not instant death (like
    crash_policy) -- it's a long, slow, never-ending bad patch. The whole
    point of the fix is that THIS case should score clearly worse than
    crashing fast would, per step, but the policy should have no
    incentive to deliberately trip termination to escape it."""
    a = pid_policy(env, s)
    return np.clip(a + np.array([0.08, 0.0]), -1, 1)

def quick_crash_then_stop_policy(env, s):
    """Simulates what the exploit policy learned: dive hard enough to
    trip termination almost immediately, then (implicitly) stop -- the
    episode ends on its own once terminated=True."""
    return np.array([1.0, 0.0])

pid_env_ref.clear()
r_pid, n_pid, _ = run_episode(pid_policy, "PID (nominal)")
pid_env_ref.clear()
r_med, n_med, term_med = run_episode(mediocre_policy, "Mediocre (biased)")
r_rand, n_rand, _ = run_episode(random_policy, "Random actions")
r_crash, n_crash, _ = run_episode(crash_policy, "Forced crash (nose-down)")

print()
print("Key check: does a long mediocre flight score BETTER than a fast")
print("deliberate crash, per step and in total? (it must, or the exploit")
print("from the first reward design is still there)")
print(f"  mediocre: total={r_med:8.1f} over {n_med:4d} steps (terminated={term_med})  mean/step={r_med/n_med:.3f}")
print(f"  crash:    total={r_crash:8.1f} over {n_crash:4d} steps                 mean/step={r_crash/n_crash:.3f}")
if n_med > n_crash * 3 and r_med > r_crash:
    print("  -> OK: surviving the mediocre patch both lasts longer AND scores higher in total.")
else:
    print("  -> WARNING: crashing may still look attractive to the optimizer.")
