"""Gymnasium environment: velocity-commanded walking and turning for the 13-inch humanoid.

Command (what the joystick sets):   vx  forward/back  [m/s]
                                    vy  left/right    [m/s]  (+ = robot's left)
                                    wz  turn rate     [rad/s] (+ = counter-clockwise seen from above)

Observation (68):  gravity in base frame (3), gyro (3), command / max (3), gait clock sin,cos (2, zero when
                   standing), joint pos rel. to home pose (19), joint vel*0.1 (19), previous action (19)
Action (19, [-1,1]): servo target = home pose + action_scale_deg * action, clipped to joint range.

The robot faces -y of the CAD frame (toes point to -y, left side is +x). If yours faces +y, change FORWARD_B.
"""
import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from stand_env import StandEnv, HERE


class WalkEnv(StandEnv):
    FORWARD_B = np.array([0.0, -1.0, 0.0])      # robot-forward in the base_link frame
    DUTY = 0.55                                  # fraction of the gait cycle a foot is on the ground

    def __init__(self, xml_path=os.path.join(HERE, "test.xml"), render_mode=None,
                 control_dt=0.02, episode_seconds=20.0, action_scale_deg=30.0,
                 init_noise=0.05, push_force=0.0, timestep=0.004,
                 vx_range=(-0.08, 0.16), vy_range=(-0.06, 0.06), wz_range=(-0.6, 0.6),
                 cmd_scale=1.0, crouch_deg=15.0, arms_down_deg=70.0,
                 gait_hz=2.0, step_height=0.015, mass_scale=1.0,
                 sim_backend="mujoco", device="cuda", nworld=1, nconmax=256, njmax=512):
        super().__init__(xml_path=xml_path, render_mode=render_mode, control_dt=control_dt,
                         episode_seconds=episode_seconds, action_scale_deg=action_scale_deg,
                         init_noise=init_noise, push_force=push_force, timestep=timestep,
                         sim_backend=sim_backend, device=device, nworld=nworld,
                         nconmax=nconmax, njmax=njmax,
                         initialize_warp=(mass_scale == 1.0))
        if mass_scale != 1.0:                       # quick what-if for lighter builds (printed parts instead of steel)
            self.model.body_mass[:] *= mass_scale; self.model.body_inertia[:] *= mass_scale
            mujoco.mj_setConst(self.model, self.data)
        self.vx_range, self.vy_range, self.wz_range = tuple(vx_range), tuple(vy_range), tuple(wz_range)
        self.cmd_lo = np.array([vx_range[0], vy_range[0], wz_range[0]], dtype=np.float64)
        self.cmd_hi = np.array([vx_range[1], vy_range[1], wz_range[1]], dtype=np.float64)
        self.cmd_max = np.maximum(np.abs(self.cmd_lo), np.abs(self.cmd_hi))
        self.cmd_scale, self.gait_hz, self.step_height = cmd_scale, gait_hz, step_height
        self.cmd = np.zeros(3); self.manual_command = False; self._cmd_left = 0
        self._phase = 0.0; self._push = np.zeros(3)
        self._standing = False; self._wave = False

        m = self.model
        self.floor_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.foot_gids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
                          for n in ("LeftFoot_1_col", "RightFoot_1_col")]   # [left, right]
        self.tau_max = float(m.actuator_forcerange[0, 1])
        self._build_roles()
        self._build_home(crouch_deg, arms_down_deg)
        if self.sim_backend == "warp" and mass_scale != 1.0:
            self._init_warp()
        self.observation_space = spaces.Box(-np.inf, np.inf, (11 + 3 * self.nu,), np.float64)

    # ---------------------------------------------------------------- setup
    def _body_name(self, a):
        j = self.model.actuator_trnid[a, 0]
        return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, self.model.jnt_bodyid[j])

    def _build_roles(self):
        roles = []
        for a in range(self.nu):
            bn = self._body_name(a)
            if bn.endswith("HipJoint_1"): r = "hip_pitch"
            elif bn.endswith("UpperThigh_1"): r = "hip_roll"
            elif bn.endswith("LowerThigh_1"): r = "hip_yaw"
            elif bn.endswith("Shin_1"): r = "knee"
            elif bn.endswith("Foot_1"): r = "ankle"
            elif bn.endswith("Torso_1"): r = "waist"
            elif bn.endswith("UpperUpperArm_1"): r = "shoulder_roll"
            else: r = "arm"
            roles.append(r)
        self.roles = roles
        self.arm_actuators = {
            role: next(i for i in range(self.nu) if self._body_name(i) == body)
            for role, body in (("wave_shoulder", "RightUpperUpperArm_1"),
                               ("wave_yaw", "RightShoulderJoint_1"))
        }
        w = {"hip_roll": 1.0, "hip_yaw": 1.0, "waist": 1.0, "arm": 0.3, "shoulder_roll": 0.3}
        self.pose_w = np.array([w.get(r, 0.0) for r in roles])

    def _geom_min_z(self, g):
        d = self.data
        R = d.geom_xmat[g].reshape(3, 3); half = self.model.geom_size[g]
        corners = [d.geom_xpos[g] + R @ (half * np.array([sx, sy, sz]))
                   for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        return min(c[2] for c in corners)

    def _build_home(self, crouch_deg, arms_down_deg):
        """Slightly crouched stance (hip a, knee 2a, ankle keeps feet flat), arms hanging down."""
        m, d = self.model, self.data
        mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d)
        a = np.deg2rad(crouch_deg); b = 2 * a
        q = np.zeros(self.nu)
        for i in range(self.nu):
            j = m.actuator_trnid[i, 0]; sx, sy = d.xaxis[j][0], d.xaxis[j][1]   # world axis at neutral
            r = self.roles[i]
            if r == "hip_pitch":   q[i] = -a / sx            # thigh forward by a
            elif r == "knee":      q[i] = b / sx             # shank back by b relative to thigh
            elif r == "ankle":     q[i] = -(b - a) / sx      # keep the sole flat
            elif r == "shoulder_roll":
                q[i] = np.deg2rad(arms_down_deg) * np.sign(d.xanchor[j][0]) / sy   # arm swings down
        q = np.clip(q, self.ctrl_lo, self.ctrl_hi)
        d.qpos[self.qadr] = q; mujoco.mj_forward(m, d)
        lowest = min(self._geom_min_z(g) for g in self.foot_gids)
        d.qpos[2] -= (lowest - 0.002)                           # soles 2 mm above the floor
        mujoco.mj_forward(m, d)
        self.home_q = q
        self.base_z0 = float(d.qpos[2]); self.h0 = float(d.xpos[self.base_id][2])
        self.foot_z0 = np.array([d.geom_xpos[g][2] for g in self.foot_gids])

    # ------------------------------------------------------------- commands
    def _is_moving(self, cmd=None):
        c = self.cmd if cmd is None else cmd
        return bool(np.linalg.norm(c[:2]) > 0.02 or abs(c[2]) > 0.1)

    def _sample_command(self):
        r = self.np_random
        comp = lambda lo, hi: r.uniform(lo, hi) if r.random() < 0.6 else 0.0
        cmd = np.array([comp(*self.vx_range), comp(*self.vy_range), comp(*self.wz_range)]) * self.cmd_scale
        if r.random() < 0.15 or not self._is_moving(cmd): cmd[:] = 0.0      # standing still examples
        self.cmd = cmd
        self._cmd_left = int(r.uniform(2.0, 5.0) / self.control_dt)

    def set_command(self, cmd):
        """Used by the joystick: [vx, vy, wz]. Stops random command resampling."""
        self.cmd = np.clip(np.asarray(cmd, dtype=np.float64), self.cmd_lo, self.cmd_hi)
        self.manual_command = True

    def set_gesture(self, standing=False, wave=False):
        """Select a zero-velocity stand command or overlay a right-arm wave on the policy."""
        self._standing, self._wave = bool(standing), bool(wave)
        if self._standing or self._wave:
            self.cmd[:] = 0.0

    # --------------------------------------------------------------- gym API
    def _obs(self):
        d = self.data
        R = d.xmat[self.base_id].reshape(3, 3)
        grav_b = R.T @ np.array([0.0, 0.0, -1.0])
        moving = float(self._is_moving())
        clock = moving * np.array([np.sin(2 * np.pi * self._phase), np.cos(2 * np.pi * self._phase)])
        return np.concatenate([grav_b, d.qvel[3:6], self.cmd / self.cmd_max, clock,
                               d.qpos[self.qadr] - self.home_q, 0.1 * d.qvel[self.vadr], self._prev_action])

    def reset(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        m, d = self.model, self.data
        if self.sim_backend == "warp":
            import mujoco_warp as mjw
            mjw.reset_data_keyframe(self._warp_model, self._warp_data, 0)
        else:
            mujoco.mj_resetDataKeyframe(m, d, 0)
        n = self.init_noise
        d.qpos[2] = self.base_z0
        d.qpos[self.qadr] = np.clip(self.home_q + self.np_random.uniform(-n, n, self.nu), self.ctrl_lo, self.ctrl_hi)
        ang = self.np_random.uniform(0, 0.03)
        axis = self.np_random.normal(size=3); axis[2] = 0; axis /= (np.linalg.norm(axis) + 1e-9)
        d.qpos[3:7] = np.r_[np.cos(ang / 2), axis * np.sin(ang / 2)]
        d.qvel[:] = self.np_random.uniform(-0.01, 0.01, m.nv)
        d.ctrl[:] = np.clip(self.home_q, self.ctrl_lo, self.ctrl_hi)
        d.xfrc_applied[:] = 0
        if self.sim_backend == "warp":
            self._upload_warp_state()
            self._sync_warp_state()
        else:
            mujoco.mj_forward(m, d)
        self._prev_action[:] = 0; self._steps = 0; self._push_left = 0
        self._phase = float(self.np_random.uniform(0, 1))
        if not self.manual_command: self._sample_command()
        if self._standing or self._wave: self.cmd[:] = 0.0
        return self._obs(), {}

    def _contacts(self):
        d, m = self.data, self.model
        foot = np.zeros(2, dtype=bool); bad = False
        for k in range(d.ncon):
            c = d.contact[k]; g1, g2 = c.geom1, c.geom2
            if g1 != self.floor_gid and g2 != self.floor_gid: continue
            other = g2 if g1 == self.floor_gid else g1
            if other == self.foot_gids[0]: foot[0] = True
            elif other == self.foot_gids[1]: foot[1] = True
            else: bad = True                                    # knee, torso, hand ... touching the floor
        return foot, bad

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        d, m = self.data, self.model
        d.ctrl[:] = np.clip(self.home_q + self.action_scale * action, self.ctrl_lo, self.ctrl_hi)
        if self._wave:
            shoulder = self.arm_actuators["wave_shoulder"]
            yaw = self.arm_actuators["wave_yaw"]
            phase = 2 * np.pi * 1.5 * self._steps * self.control_dt
            d.ctrl[shoulder] = np.clip(self.home_q[shoulder] + 0.8, self.ctrl_lo[shoulder], self.ctrl_hi[shoulder])
            d.ctrl[yaw] = np.clip(0.55 * np.sin(phase), self.ctrl_lo[yaw], self.ctrl_hi[yaw])

        if self.push_force > 0:                                  # random shoves for robustness
            if self._push_left == 0 and self.np_random.random() < 0.01:
                self._push_left = 5
                ang = self.np_random.uniform(0, 2 * np.pi)
                self._push = self.push_force * self.np_random.uniform(0.3, 1.0) * np.array([np.cos(ang), np.sin(ang), 0])
            d.xfrc_applied[self.base_id, :3] = self._push if self._push_left > 0 else 0
            self._push_left = max(0, self._push_left - 1)

        self._step_simulation()
        self._steps += 1

        if not self.manual_command:
            self._cmd_left -= 1
            if self._cmd_left <= 0: self._sample_command()
        moving = self._is_moving() and not self._standing
        if moving: self._phase = (self._phase + self.gait_hz * self.control_dt) % 1.0

        # ---- state in the heading frame
        R = d.xmat[self.base_id].reshape(3, 3)
        up = float(R[2, 2]); h = float(d.xpos[self.base_id][2])
        f_w = R @ self.FORWARD_B; psi = np.arctan2(f_w[1], f_w[0]); c, s = np.cos(psi), np.sin(psi)
        v = d.qvel[0:2]; v_fwd = c * v[0] + s * v[1]; v_left = -s * v[0] + c * v[1]
        wz = float((R @ d.qvel[3:6])[2])

        foot_contact, bad = self._contacts()
        if moving:
            ph = np.array([self._phase, (self._phase + 0.5) % 1.0])          # [left, right]
            expect_stance = ph < self.DUTY
            sw = np.clip((ph - self.DUTY) / (1 - self.DUTY), 0, 1)
            z_ref = np.where(expect_stance, 0.0, self.step_height * np.sin(np.pi * sw))
        else:
            expect_stance = np.array([True, True]); z_ref = np.zeros(2)
        zf = np.array([d.geom_xpos[g][2] for g in self.foot_gids]) - self.foot_z0
        slip = 0.0; res = np.zeros(6)
        for i, g in enumerate(self.foot_gids):
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_GEOM, g, res, 0)
            slip += float(foot_contact[i]) * float(res[3:5] @ res[3:5])

        err = np.array([self.cmd[0] - v_fwd, self.cmd[1] - v_left])
        r_lin = np.exp(-(err @ err) / 0.05 ** 2)
        r_yaw = np.exp(-((self.cmd[2] - wz) / 0.3) ** 2)
        r_up = max(up, 0.0) ** 2
        r_h = np.exp(-((h - self.h0) / 0.025) ** 2)
        r_contact = float(np.mean(foot_contact == expect_stance))
        r_foot = float(np.mean(np.exp(-((zf - z_ref) / 0.008) ** 2)))
        q_dev = d.qpos[self.qadr] - self.home_q
        pen = (0.05 * float(np.mean((action - self._prev_action) ** 2))
               + 0.05 * float(np.mean((d.actuator_force / self.tau_max) ** 2))      # servo load
               + 0.0005 * float(d.qvel[self.vadr] @ d.qvel[self.vadr])
               + 0.2 * float(np.mean(self.pose_w * q_dev ** 2))
               + 0.01 * float(d.qvel[3] ** 2 + d.qvel[4] ** 2)
               + 5.0 * float(d.qvel[2] ** 2)
               + 5.0 * slip)
        reward = (3.0 * r_lin + 1.5 * r_yaw + 1.0 * r_up + 1.0 * r_h
                  + 1.0 * r_contact + 1.0 * r_foot + 0.3 - pen)
        self._prev_action = action.copy()

        terminated = bool(up < 0.6 or h < 0.65 * self.h0 or bad or not np.isfinite(d.qpos).all())
        truncated = self._steps >= self.max_steps
        info = {"up": up, "height": h, "v_fwd": float(v_fwd), "v_left": float(v_left), "wz": wz,
                "cmd": self.cmd.copy()}
        return self._obs(), float(reward), terminated, truncated, info
