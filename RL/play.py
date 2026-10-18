"""Watch or evaluate a trained policy.
   python play.py --run runs/stand                 (opens the MuJoCo viewer)
   python play.py --run runs/stand --no-render --episodes 20 --push 1.0   (headless stats)"""
import argparse, os, time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stand_env import StandEnv

p = argparse.ArgumentParser()
p.add_argument("--run", default="runs/stand")
p.add_argument("--model", default="final_model.zip")
p.add_argument("--norm", default="vecnormalize.pkl")
p.add_argument("--episodes", type=int, default=10)
p.add_argument("--push", type=float, default=0.0)
p.add_argument("--no-render", action="store_true")
p.add_argument("--device", default="cpu", help="cpu is fastest for single-step playback; cuda also works")
a = p.parse_args()

venv = DummyVecEnv([lambda: StandEnv(push_force=a.push)])
venv = VecNormalize.load(os.path.join(a.run, a.norm), venv)
venv.training = False; venv.norm_reward = False
model = PPO.load(os.path.join(a.run, a.model), device=a.device)
env = venv.venv.envs[0].unwrapped

viewer = None
if not a.no_render:
    import mujoco.viewer
    viewer = mujoco.viewer.launch_passive(env.model, env.data)

obs = venv.reset(); ep_ret = ep_len = 0; lens, rets = [], []
next_t = time.perf_counter()
while len(lens) < a.episodes and (viewer is None or viewer.is_running()):
    action, _ = model.predict(obs, deterministic=True)
    obs, r, done, info = venv.step(action)
    ep_ret += float(r[0]); ep_len += 1
    if viewer is not None:
        viewer.sync()
        next_t += env.control_dt                      # sleep only the time left in this step
        time.sleep(max(0.0, next_t - time.perf_counter()))
    if done[0]:
        lens.append(ep_len); rets.append(ep_ret); ep_ret = ep_len = 0
        print(f"episode {len(lens)}: {lens[-1]} steps ({lens[-1]*env.control_dt:.1f} s), return {rets[-1]:.0f}")
if lens:
    print(f"mean length {np.mean(lens):.0f}/{env.max_steps} steps | stood full episode: "
          f"{sum(l >= env.max_steps for l in lens)}/{len(lens)}")
