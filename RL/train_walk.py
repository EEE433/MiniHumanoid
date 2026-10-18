"""Train a joystick-commandable walking policy with PPO.

  python train_walk.py --steps 30000000 --envs 8 --cmd-scale 0.4 --name walk     # stage 1: gentle commands
  python train_walk.py --steps 30000000 --envs 8 --cmd-scale 1.0 --push 1.0 \
         --resume runs/walk/final_model.zip --name walk2                          # stage 2: full range + pushes
Results: runs/<name>/ (checkpoints, final_model.zip, vecnormalize.pkl, tensorboard)."""
import argparse, os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from walk_env import WalkEnv


def make_env(rank, push, cmd_scale, mass_scale):
    def _init():
        env = WalkEnv(push_force=push, cmd_scale=cmd_scale, mass_scale=mass_scale)
        env.reset(seed=rank)
        return Monitor(env)
    return _init


if __name__ == "__main__":          # required on Windows (SubprocVecEnv uses spawn)
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=30_000_000)
    p.add_argument("--envs", type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument("--push", type=float, default=0.0, help="max random push force in N (0 = off)")
    p.add_argument("--cmd-scale", type=float, default=1.0, help="fraction of the full command range to sample (curriculum)")
    p.add_argument("--mass-scale", type=float, default=1.0, help="multiply all link masses (what-if for a lighter build)")
    p.add_argument("--name", default="walk")
    p.add_argument("--resume", default=None, help="model .zip to continue from (also loads its vecnormalize.pkl)")
    p.add_argument("--device", default="auto", help='"cuda" for your NVIDIA GPU, "cuda", or "auto" (cuda if available)')
    a = p.parse_args()
    import torch; torch.set_num_threads(1)

    out = os.path.join("runs", a.name); os.makedirs(out, exist_ok=True)
    Vec = SubprocVecEnv if a.envs > 1 else DummyVecEnv
    venv = Vec([make_env(i, a.push, a.cmd_scale, a.mass_scale) for i in range(a.envs)])
    if a.resume:
        base = os.path.basename(a.resume)                      # ckpt_<N>_steps.zip -> ckpt_vecnormalize_<N>_steps.pkl
        stats = os.path.join(os.path.dirname(a.resume),
                             base.replace("ckpt_", "ckpt_vecnormalize_").replace(".zip", ".pkl")
                             if base.startswith("ckpt_") else "vecnormalize.pkl")
        venv = VecNormalize.load(stats, venv); venv.training = True
        model = PPO.load(a.resume, env=venv, device=a.device)
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
        model = PPO("MlpPolicy", venv, n_steps=512, batch_size=4096, n_epochs=5,
                    learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                    ent_coef=0.0, max_grad_norm=0.5,
                    policy_kwargs=dict(net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]), log_std_init=-1.0),
                    tensorboard_log=os.path.join(out, "tb"), device=a.device, verbose=1)
    print("training on:", model.device)
    ckpt = CheckpointCallback(save_freq=max(1, 500_000 // a.envs), save_path=out,
                              name_prefix="ckpt", save_vecnormalize=True)
    try:
        model.learn(total_timesteps=a.steps, callback=ckpt, reset_num_timesteps=not a.resume)
    except KeyboardInterrupt:                      # Ctrl+C: stop cleanly and still save below
        print("\nInterrupted - saving the current model...")
    model.save(os.path.join(out, "final_model")); venv.save(os.path.join(out, "vecnormalize.pkl"))
    print("saved to", out)
