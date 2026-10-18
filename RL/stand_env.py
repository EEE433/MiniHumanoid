"""Gymnasium environment: teach the 13-inch humanoid (test.xml / STLv2_13in.xml) to stand.

Observation (63, no privileged info - could come from an IMU + encoders on the real robot):
    gravity vector in base frame (3), base gyro (3), joint pos rel. to neutral (19),
    joint vel * 0.1 (19), previous action (19)
Action (19, in [-1, 1]): servo target = neutral + action_scale_deg * action (degrees), clipped to joint range.
Servos are modelled on the TowerPro MG90S (see apply_fixes.py): 0.1765 N*m stall, ~10.5 rad/s no-load.
"""
import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

HERE = os.path.dirname(os.path.abspath(__file__))


class StandEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, xml_path=os.path.join(HERE, "test.xml"), render_mode=None,
                 control_dt=0.02, episode_seconds=10.0, action_scale_deg=30.0,
                 init_noise=0.05, push_force=0.0, timestep=0.004):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        if timestep: self.model.opt.timestep = timestep   # 0.004 is ~4x faster than the XML default of 0.001
        self.frame_skip = max(1, int(round(control_dt / self.model.opt.timestep)))
        self.control_dt = self.frame_skip * self.model.opt.timestep
        self.max_steps = int(round(episode_seconds / self.control_dt))
        self.action_scale = np.deg2rad(action_scale_deg)   # servo target = neutral +/- this many degrees
        self.init_noise, self.push_force = init_noise, push_force

        m = self.model
        self.base_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link_1")
        jid = m.actuator_trnid[:, 0]
        self.qadr, self.vadr = m.jnt_qposadr[jid], m.jnt_dofadr[jid]
        self.ctrl_lo, self.ctrl_hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
        self.nu = m.nu

        mujoco.mj_resetDataKeyframe(m, self.data, 0)          # "neutral" keyframe
        mujoco.mj_forward(m, self.data)
        self.home_q = self.data.qpos[self.qadr].copy()
        self.h0 = float(self.data.xpos[self.base_id][2])      # standing height of base origin

        obs_dim = 6 + 3 * self.nu
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float64)
        self.action_space = spaces.Box(-1.0, 1.0, (self.nu,), np.float32)
        self._viewer = self._renderer = None
        self._prev_action = np.zeros(self.nu)
        self._steps = 0
        self._push_left = 0

    # ------------------------------------------------------------------
    def _obs(self):
        d = self.data
        R = d.xmat[self.base_id].reshape(3, 3)
        grav_b = R.T @ np.array([0.0, 0.0, -1.0])
        return np.concatenate([grav_b, d.qvel[3:6], d.qpos[self.qadr] - self.home_q,
                               0.1 * d.qvel[self.vadr], self._prev_action])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        m, d = self.model, self.data
        mujoco.mj_resetDataKeyframe(m, d, 0)
        n = self.init_noise                                     # radians (0.05 ~ 3 deg)
        d.qpos[self.qadr] += self.np_random.uniform(-n, n, self.nu)
        ang = self.np_random.uniform(0, 0.03)                  # small random tilt (rad)
        axis = self.np_random.normal(size=3); axis[2] = 0; axis /= (np.linalg.norm(axis) + 1e-9)
        d.qpos[3:7] = np.r_[np.cos(ang / 2), axis * np.sin(ang / 2)]
        d.qvel[:] = self.np_random.uniform(-0.01, 0.01, m.nv)
        d.ctrl[:] = np.clip(self.home_q, self.ctrl_lo, self.ctrl_hi)
        d.xfrc_applied[:] = 0
        mujoco.mj_forward(m, d)
        self._prev_action[:] = 0; self._steps = 0; self._push_left = 0
        return self._obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        d, m = self.data, self.model
        d.ctrl[:] = np.clip(self.home_q + self.action_scale * action, self.ctrl_lo, self.ctrl_hi)

        # optional random pushes on the base (robustness)
        if self.push_force > 0:
            if self._push_left == 0 and self.np_random.random() < 0.01:
                self._push_left = 5
                ang = self.np_random.uniform(0, 2 * np.pi)
                self._push = self.push_force * self.np_random.uniform(0.3, 1.0) * np.array([np.cos(ang), np.sin(ang), 0])
            d.xfrc_applied[self.base_id, :3] = self._push if self._push_left > 0 else 0
            self._push_left = max(0, self._push_left - 1)

        for _ in range(self.frame_skip):
            mujoco.mj_step(m, d)
        self._steps += 1

        R = d.xmat[self.base_id].reshape(3, 3)
        up = float(R[2, 2]); h = float(d.xpos[self.base_id][2])
        q_dev = d.qpos[self.qadr] - self.home_q
        v_xy = d.qvel[0:2]
        r_up = 1.5 * max(up, 0.0) ** 2
        r_h = 2.0 * np.exp(-((h - self.h0) / 0.03) ** 2)
        pen = (0.01 * float(d.qvel[3:6] @ d.qvel[3:6])           # base spin
               + 0.001 * float(d.qvel[self.vadr] @ d.qvel[self.vadr])
               + 0.1 * float(np.mean((action - self._prev_action) ** 2))
               + 0.2 * float(np.mean(q_dev ** 2))
               + 2.0 * float(v_xy @ v_xy))
        reward = 0.5 + r_up + r_h - pen
        self._prev_action = action.copy()

        terminated = bool(up < 0.6 or h < 0.6 * self.h0 or not np.isfinite(d.qpos).all())
        truncated = self._steps >= self.max_steps
        return self._obs(), float(reward), terminated, truncated, {"up": up, "height": h}

    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, 480, 640)
            self._renderer.update_scene(self.data); return self._renderer.render()
        if self.render_mode == "human":
            import mujoco.viewer
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()

    def close(self):
        if self._viewer is not None: self._viewer.close()
        if self._renderer is not None: self._renderer.close()
