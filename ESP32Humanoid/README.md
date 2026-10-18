# ESP32 Humanoid Motion Playback

This PlatformIO project targets a classic ESP32 Dev Module (`esp32dev`, ESP32-WROOM-32 class) and two PCA9685 boards. It plays a compact, open-loop motion table exported from the MuJoCo PPO walk policy in `../RL/`.

> **Safety first:** a learned simulation trajectory is not a validated real-robot gait. Secure the robot on a stand, disconnect servo power during initial uploads, test one servo at a time, use conservative limits, and keep a physical power switch within reach. Hobby servos provide no position feedback; this firmware does not stabilize balance or detect falls.

## Hardware and wiring

- ESP32 GPIO 21 (SDA) / GPIO 22 (SCL) to SDA/SCL on both PCA9685 boards; these are common defaults for ESP32 Dev Module boards and can be changed in firmware. Connect grounds together.
- Set the PCA9685 addresses to `0x40` and `0x46`. Keep the first board at default `0x40`. On a standard PCA9685 address scheme, configure the second board with A1 and A2 high/bridged and A0 low/open for `0x46`; verify the specific board's jumper labels/documentation before changing them. The two I²C addresses must be unique.
- Power servo V+ from a separate regulated supply sized for simultaneous servo stall current. Do **not** power the servos from the ESP32 3V3 pin or USB. Join servo-supply ground, PCA9685 ground, and ESP32 ground.
- PCA9685 logic VCC must be compatible with the ESP32's 3.3 V I2C logic. Avoid 5 V pull-ups on SDA/SCL.
- Firmware maps logical channels 0–15 to PCA9685 `0x40` outputs 0–15, and logical channels 16–31 to PCA9685 `0x46` outputs 0–15 (for example, logical 16 is address `0x46`, output 0).
- The **left arm** mapping is shoulder pitch=logical 15, shoulder roll=14, shoulder yaw=13, elbow=12. All four use PCA9685 `0x40`, outputs 15/14/13/12 respectively.
- The **left leg** mapping is pitch=4, roll=3, yaw=2, knee=1, ankle=0, all on PCA9685 `0x40`. Its mirror, the **right leg**, uses PCA9685 `0x46` outputs 4/3/2/1/0 for pitch/roll/yaw/knee/ankle.
- Torso yaw uses PCA9685 `0x46`, output 8. The right arm mirrors the left-arm channel pattern on PCA9685 `0x46`: shoulder pitch/roll/yaw/elbow use outputs 15/14/13/12 respectively.
- Verify every cable matches the `kServos` table and calibrate direction, zero, and pulse limits before attaching or powering the robot. These assignments were not tested on physical hardware.

## Calibrate before movement

Edit `src/main.cpp`:

1. Confirm the SDA/SCL pins for your exact ESP32 Dev Module board; firmware defaults to GPIO 21 / GPIO 22.
2. Verify both PCA9685 I2C addresses with a scanner.
3. Correct the `kServos` map to match your harness and the actuator order listed above the table. Set each direction to `+1` or `-1` and calibrate its pulse center/limits and `zeroDeg` for that physical servo. `minUs`/`maxUs` initially default to 600–2400 us; many servos cannot safely reach these endpoints, so start narrower and calibrate gradually.
4. Check the sim-to-hardware joint zero and sign. `kHomeDeg`, `kJointMinDeg`, and `kJointMaxDeg` are derived from `RL/walk_env.py` and are *simulation* references, not a guarantee that printed joints or servo horns are aligned the same way.
5. Keep `kOffsetLimitDeg` low during first tests. The firmware clamps learned offsets, applies model joint limits, and slew-limits target changes.

## Exporting an RL movement

Install Python dependencies in your environment with `pip install -r RL/requirements.txt`. From the repository root, export the supplied trained walk policy:

```sh
python RL/export_walk.py --run RL/runs/walk --output ESP32Humanoid/include/generated_walk.h \
  --seconds 2 --stride 3 --max-angle-deg 15 --vx 0.08
```

The exporter loads `final_model.zip` and `vecnormalize.pkl`, runs the policy in `WalkEnv`, and writes 19 joint offsets relative to `WalkEnv.home_q`. Frames use signed centidegrees and a duration field; defaults sample every third 20 ms control step (60 ms per frame). The generated header is included by the firmware and stored in flash (`PROGMEM`). The exporter refuses a command outside the training envelope and refuses to write a trajectory if the simulation falls. Check its clamp summary: frequent clamps indicate the selected trajectory is not suitable to transfer as-is. A safe first hardware try should be a brief, low-amplitude clip, not an assumption that the simulation gait is safe.

To train a new walk policy, follow `RL/train_walk.py`'s staged curriculum; validate in MuJoCo with `RL/drive.py --selftest` before export. The checkpoint currently in this repository did **not** pass a recent self-test (one fall and poor command tracking), so treat its generated header as a firmware/export smoke-test only—not a walking motion. Retrain and verify tracking/fall resistance before attempting hardware. A policy trained with different model normalization, joint order, or home pose needs matching exporter/environment settings.

## Build, upload and play in VS Code

Open the `ESP32Humanoid` folder in VS Code with the PlatformIO IDE extension. In the PlatformIO toolbar use **Build**, then **Upload**. CLI equivalents from this folder:

```sh
pio run -e esp32dev
pio run -e esp32dev -t upload
pio device monitor -b 115200
```

With the robot supported and servo supply off, flash and inspect serial startup. After calibration, turn servo power on with the robot held/secured. Send `p` in the serial monitor to play once; the firmware holds the final pose when the table ends. Send `s` or space to ramp back to calibrated home; `?` prints help. This is open-loop playback: there is no IMU, encoder, current, or fall feedback yet. Do not use it for unsupported walking until a feedback/stability controller is added and tested.
