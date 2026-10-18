"""Joystick / keyboard -> velocity command, plus a headless self-test. No RL libraries needed here."""
import numpy as np


def resolve_device(requested="cuda"):
    """Prefer CUDA for PPO, but keep scripts usable on machines without an NVIDIA GPU."""
    if requested == "auto":
        requested = "cuda"
    if requested.startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            print("CUDA requested but unavailable; using CPU.")
            return "cpu"
        if ":" in requested and int(requested.split(":", 1)[1]) >= torch.cuda.device_count():
            print(f"{requested} is unavailable; using CPU.")
            return "cpu"
    return requested


def _dead(x, dz):
    return 0.0 if abs(x) < dz else float(np.sign(x) * (abs(x) - dz) / (1.0 - dz))


def stick_to_command(lx, ly, rx, vx_range, vy_range, wz_range, deadzone=0.12):
    """Left stick: up/down = forward/back, left/right = strafe.  Right stick X: turn.
    pygame axes: stick pushed UP gives a NEGATIVE y value, pushed RIGHT gives a POSITIVE x value."""
    fwd = -_dead(ly, deadzone)
    vx = fwd * vx_range[1] if fwd >= 0 else fwd * abs(vx_range[0])
    left = -_dead(lx, deadzone)                              # stick right -> robot's right
    vy = left * (vy_range[1] if left >= 0 else abs(vy_range[0]))
    turn = -_dead(rx, deadzone)                              # stick right -> clockwise -> negative wz
    wz = turn * (wz_range[1] if turn >= 0 else abs(wz_range[0]))
    return np.array([vx, vy, wz])


class GamepadReader:
    """pygame 2 default axes (Xbox / PlayStation, wired or Bluetooth): LX=0, LY=1, RX=2. Override if yours differ."""
    def __init__(self, axis_lx=0, axis_ly=1, axis_rx=2, deadzone=0.12, index=0,
                 btn_reset=0, btn_stop=1, btn_wave=2, btn_stand=3):
        import pygame
        self.pg = pygame
        pygame.init(); pygame.joystick.init()
        if pygame.joystick.get_count() <= index:
            raise RuntimeError("No gamepad found. Plug it in / pair it, or run with --keyboard.")
        self.joy = pygame.joystick.Joystick(index)
        self.name = self.joy.get_name()
        self.ax = (axis_lx, axis_ly, axis_rx); self.dz = deadzone
        self.btn_reset, self.btn_stop = btn_reset, btn_stop
        self.btn_wave, self.btn_stand = btn_wave, btn_stand
        self._buttons = {"reset": False, "wave": False, "stand": False}
        self._previous = {name: False for name in self._buttons}
        self._wave = False; self._stand = False

    def command(self, vx_range, vy_range, wz_range):
        self.pg.event.pump()
        for name, button in (("reset", self.btn_reset), ("wave", self.btn_wave), ("stand", self.btn_stand)):
            now = bool(self.joy.get_button(button))
            self._buttons[name] = now and not self._previous[name]
            self._previous[name] = now
        if self._buttons["wave"]: self._wave = not self._wave
        if self._buttons["stand"]: self._stand = not self._stand

        lx, ly, rx = (self.joy.get_axis(i) for i in self.ax)
        cmd = stick_to_command(lx, ly, rx, vx_range, vy_range, wz_range, self.dz)
        if self._wave or self._stand or self.joy.get_button(self.btn_stop): cmd[:] = 0.0
        return cmd

    def reset_pressed(self):                                 # edge-triggered: A button
        return self._buttons["reset"]

    def wave_active(self):
        return self._wave

    def standing(self):
        return self._stand


def print_axes(index=0):
    """Run this to find your stick axis numbers: move each stick and watch which value changes."""

    import time, pygame
    pygame.init(); pygame.joystick.init()
    if pygame.joystick.get_count() <= index: raise SystemExit("No gamepad found.")
    j = pygame.joystick.Joystick(index); print("Gamepad:", j.get_name(), "| Ctrl+C to quit")
    while True:
        pygame.event.pump()
        print("\r" + "  ".join(f"ax{i}:{j.get_axis(i):+.2f}" for i in range(j.get_numaxes())), end="", flush=True)
        time.sleep(0.05)


def keys_to_target(fwd, back, left, right, turn_l, turn_r, slow, vx_range, vy_range, wz_range):
    """Held keys -> target command. Opposite keys cancel. `slow` halves the speed."""
    k = 0.5 if slow else 1.0
    vx = k * (vx_range[1] * (fwd and not back) + vx_range[0] * (back and not fwd))
    vy = k * (vy_range[1] * (left and not right) + vy_range[0] * (right and not left))   # +vy = robot's left
    wz = k * (wz_range[1] * (turn_l and not turn_r) + wz_range[0] * (turn_r and not turn_l))  # +wz = counter-clockwise
    return np.array([vx, vy, wz], dtype=np.float64)


def slew(cmd, target, max_step):
    """Move cmd toward target by at most max_step (per axis) so key taps don't jerk the robot."""
    return cmd + np.clip(target - cmd, -max_step, max_step)


class KeyboardPad:
    """Hold-to-move keyboard control in a small pygame window (click it so it has focus; no OS permissions needed).
         W / S  (or Up / Down)      forward / back        A / D   strafe left / right
         Q / E  (or Left / Right)   turn left / right     Shift   half speed
         T   toggle standing        G   toggle wave        Space   stop / R reset"""
    LEGEND = ["W / S  or  Up / Down     forward / back", "A / D                    strafe left / right",
              "Q / E  or  Left / Right   turn left / right", "Shift                   half speed",
              "T                       toggle standing", "G                       toggle wave",
              "Space                   stop", "R                       reset robot"]

    def __init__(self, ramp_per_s=2.0):
        import pygame
        self.pg = pygame; pygame.init(); pygame.font.init()
        self.screen = pygame.display.set_mode((460, 340)); pygame.display.set_caption("Robot keyboard control - keep this window focused")
        self.font = pygame.font.Font(None, 24); self.small = pygame.font.Font(None, 21)
        self.cmd = np.zeros(3); self.ramp = ramp_per_s; self.quit = False
        self._prev_r = False; self._reset = False; self._standing = False; self._wave = False; self._n = 0

    def command(self, vx_range, vy_range, wz_range, dt):
        pg = self.pg
        for e in pg.event.get():
            if e.type == pg.QUIT: self.quit = True
            elif e.type == pg.KEYDOWN and e.key == pg.K_t: self._standing = not self._standing
            elif e.type == pg.KEYDOWN and e.key == pg.K_g: self._wave = not self._wave
        k = pg.key.get_pressed()
        shift = k[pg.K_LSHIFT] or k[pg.K_RSHIFT]
        target = keys_to_target(k[pg.K_w] or k[pg.K_UP], k[pg.K_s] or k[pg.K_DOWN], k[pg.K_a], k[pg.K_d],
                                k[pg.K_q] or k[pg.K_LEFT], k[pg.K_e] or k[pg.K_RIGHT], shift,
                                vx_range, vy_range, wz_range)
        if self._standing or self._wave: target[:] = 0.0
        full = np.maximum(np.abs([vx_range[0], vy_range[0], wz_range[0]]), np.abs([vx_range[1], vy_range[1], wz_range[1]]))
        if k[pg.K_SPACE]: self.cmd[:] = 0.0; target[:] = 0.0
        self.cmd = slew(self.cmd, target, self.ramp * full * dt)       # reaches full speed in 1/ramp seconds
        r = bool(k[pg.K_r]); self._reset = r and not self._prev_r; self._prev_r = r
        self._n += 1
        if self._n % 3 == 0: self._draw(full)
        return self.cmd.copy()

    def reset_pressed(self):
        return self._reset

    def wave_active(self):
        return self._wave

    def standing(self):
        return self._standing

    def _draw(self, full):
        pg, scr = self.pg, self.screen
        scr.fill((24, 26, 32))
        for i, t in enumerate(self.LEGEND): scr.blit(self.small.render(t, True, (190, 195, 205)), (16, 14 + 22 * i))
        for i, (name, v, f) in enumerate(zip(("forward", "left", "turn"), self.cmd, full)):
            y = 205 + 36 * i
            scr.blit(self.font.render(f"{name:8s}{v:+.2f}", True, (235, 235, 240)), (16, y))
            pg.draw.rect(scr, (60, 64, 74), (170, y + 2, 270, 16))
            pg.draw.rect(scr, (90, 170, 255), (305, y + 2, int(135 * np.clip(v / f, -1, 1)), 16))   # bar grows right (+) or left (-)
            pg.draw.line(scr, (220, 220, 225), (305, y), (305, y + 20), 2)
        pg.display.flip()


def run_selftest(env, policy_fn, seconds=4.0):
    """Headless: command a fixed sequence and report how well the robot tracks it."""
    hi, lo = env.cmd_hi, env.cmd_lo
    seq = [("stand", (0, 0, 0)), ("forward", (0.6 * hi[0], 0, 0)), ("backward", (0.6 * lo[0], 0, 0)),
           ("strafe left", (0, 0.6 * hi[1], 0)), ("strafe right", (0, 0.6 * lo[1], 0)),
           ("turn left", (0, 0, 0.6 * hi[2])), ("turn right", (0, 0, 0.6 * lo[2])),
           ("forward + turn", (0.5 * hi[0], 0, 0.5 * hi[2]))]
    obs, _ = env.reset(); falls_total = 0
    print(f"{'test':15s} {'command vx vy wz':>22s} {'measured vx vy wz':>24s} {'falls':>6s}")
    for name, c in seq:
        env.set_command(c); meas = []; falls = 0
        for k in range(int(seconds / env.control_dt)):
            obs, r, te, tr, info = env.step(policy_fn(obs))
            if k * env.control_dt > 1.0: meas.append([info["v_fwd"], info["v_left"], info["wz"]])   # skip 1 s transient
            if te or tr: falls += 1; obs, _ = env.reset()
        m = np.mean(meas, axis=0) if meas else np.zeros(3); falls_total += falls
        print(f"{name:15s} {c[0]:+7.3f}{c[1]:+7.3f}{c[2]:+7.2f}   {m[0]:+8.3f}{m[1]:+8.3f}{m[2]:+8.2f}   {falls:6d}")
    return falls_total
