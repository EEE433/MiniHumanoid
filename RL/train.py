"""Train PPO to stand.   python train.py --steps 3000000 --envs 8
Results land in runs/<name>/ : checkpoints, final_model.zip, vecnormalize.pkl, tensorboard logs."""
import argparse, os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stand_env import StandEnv
from controller import resolve_device


def make_env(rank, push, sim_backend, device, nconmax, njmax):
    def _init():
        env = StandEnv(push_force=push, sim_backend=sim_backend, device=device,
                       nconmax=nconmax, njmax=njmax)
        env.reset(seed=rank)
        return Monitor(env)
    return _init


if __name__ == "__main__":          # required on Windows (SubprocVecEnv uses spawn)
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=3_000_000)
    p.add_argument("--envs", type=int, default=min(8, os.cpu_count() or 1), help="parallel envs (<= physical cores)")
    p.add_argument("--push", type=float, default=0.0, help="max random push force in N (0 = off)")
    p.add_argument("--name", default="stand")
    p.add_argument("--resume", default=None, help="path to a model .zip to continue training")
    p.add_argument("--device", default="cuda", help="PPO device (cuda when available; falls back to cpu)")
    p.add_argument("--sim-backend", choices=("warp", "mujoco"), default="warp",
                   help="warp runs MuJoCo physics on the NVIDIA GPU; mujoco uses CPU")
    p.add_argument("--nconmax", type=int, default=256, help="MJWarp maximum contacts per world")
    p.add_argument("--njmax", type=int, default=512, help="MJWarp maximum constraints per world")
    a = p.parse_args()
    a.device = resolve_device(a.device)
    import torch; torch.set_num_threads(1)   # small MLP: 1 thread is faster than fighting the env workers

    out = os.path.join("runs", a.name); os.makedirs(out, exist_ok=True)
    Vec = SubprocVecEnv if a.envs > 1 else DummyVecEnv
    venv = Vec([make_env(i, a.push, a.sim_backend, a.device, a.nconmax, a.njmax)
                for i in range(a.envs)])
    print(f"simulation backend: {a.sim_backend}" + (f" ({a.device})" if a.sim_backend == "warp" else " (CPU)"))
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    if a.resume:
        model = PPO.load(a.resume, env=venv, device=a.device)
    else:
        model = PPO("MlpPolicy", venv, n_steps=512, batch_size=2048, n_epochs=10,
                    learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                    ent_coef=0.0, max_grad_norm=0.5,
                    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), log_std_init=-1.0),
                    tensorboard_log=os.path.join(out, "tb"), device=a.device, verbose=1)
    print("training on:", model.device)
    ckpt = CheckpointCallback(save_freq=max(1, 250_000 // a.envs), save_path=out,
                              name_prefix="ckpt", save_vecnormalize=True)
    try:
        model.learn(total_timesteps=a.steps, callback=ckpt, reset_num_timesteps=not a.resume)
    except KeyboardInterrupt:                      # Ctrl+C: stop cleanly and still save below
        print("\nInterrupted - saving the current model...")
    model.save(os.path.join(out, "final_model")); venv.save(os.path.join(out, "vecnormalize.pkl"))
    print("saved to", out)
