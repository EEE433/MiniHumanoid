"""Drive the trained walking policy with a game controller (or keyboard) in the MuJoCo viewer.

  python play_gamepad.py --run runs/walk                 # gamepad: left stick = move/strafe, right stick X = turn
  python play_gamepad.py --run runs/walk --keyboard      # WASD / arrows move, Q/E turn, T stand, G wave
  python play_gamepad.py --list-axes                     # find your stick axis numbers
  python play_gamepad.py --run runs/walk --selftest      # headless tracking test, no viewer
Keyboard: R reset, Space stop. Xbox: A reset, B stop, X wave, Y standing toggle."""
import argparse, os, time
import numpy as np
from controller import GamepadReader, KeyboardPad, print_axes, resolve_device, run_selftest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/walk")
    p.add_argument("--model", default="final_model.zip")
    p.add_argument("--keyboard", action="store_true")
    p.add_argument("--list-axes", action="store_true")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--device", default="cuda", help="PPO inference device (cuda when available; falls back to cpu)")
    p.add_argument("--push", type=float, default=0.0)
    p.add_argument("--mass-scale", type=float, default=1.0, help="must match what you trained with")
    p.add_argument("--axis-lx", type=int, default=0); p.add_argument("--axis-ly", type=int, default=1)
    p.add_argument("--axis-rx", type=int, default=2); p.add_argument("--deadzone", type=float, default=0.12)
    a = p.parse_args()
    if a.list_axes: return print_axes()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from walk_env import WalkEnv
    env = WalkEnv(push_force=a.push, mass_scale=a.mass_scale)
    vn = VecNormalize.load(os.path.join(a.run, "vecnormalize.pkl"), DummyVecEnv([lambda: WalkEnv(mass_scale=a.mass_scale)]))
    vn.training = False; vn.norm_reward = False
    model = PPO.load(os.path.join(a.run, a.model), device=resolve_device(a.device))
    print("policy inference on:", model.device)
    policy = lambda obs: model.predict(vn.normalize_obs(obs), deterministic=True)[0]

    if a.selftest:
        falls = run_selftest(env, policy); print("total falls:", falls); return

    pad = None
    if not a.keyboard:
        try:
            pad = GamepadReader(a.axis_lx, a.axis_ly, a.axis_rx, a.deadzone)
            print("Gamepad:", pad.name)
        except RuntimeError as e:
            print(e, "-> using the keyboard")
    if pad is None:
        kb = KeyboardPad()
        print("Keyboard: click the control window, then use WASD / arrows, Q/E, T (stand), G (wave), Space (stop), R (reset)")
    import mujoco.viewer
    viewer = mujoco.viewer.launch_passive(env.model, env.data)
    obs, _ = env.reset(); next_t = time.perf_counter(); n = 0
    while viewer.is_running():
        if pad:
            cmd = pad.command(env.vx_range, env.vy_range, env.wz_range)
            reset = pad.reset_pressed()
            standing, wave = pad.standing(), pad.wave_active()
        else:
            cmd = kb.command(env.vx_range, env.vy_range, env.wz_range, env.control_dt)
            reset = kb.reset_pressed()
            standing, wave = kb.standing(), kb.wave_active()
            if kb.quit: break
        env.set_command(cmd)
        env.set_gesture(standing=standing, wave=wave)
        if reset: obs, _ = env.reset()
        obs, r, te, tr, info = env.step(policy(obs))
        if te or tr: obs, _ = env.reset()
        viewer.sync(); n += 1
        if n % 10 == 0:
            c = env.cmd; print(f"\rcmd vx {c[0]:+.2f} vy {c[1]:+.2f} wz {c[2]:+.2f} | measured {info['v_fwd']:+.2f} {info['v_left']:+.2f} {info['wz']:+.2f}   ", end="", flush=True)
        next_t += env.control_dt
        time.sleep(max(0.0, next_t - time.perf_counter()))


if __name__ == "__main__":
    main()
