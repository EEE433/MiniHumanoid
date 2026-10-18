#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <math.h>
#include "generated_walk.h"

// PCA9685 board addresses: first board default 0x40, second board configured as 0x46 (A1+A2 high, A0 low).
constexpr uint8_t kPcaAddress[2] = {0x40, 0x46};
constexpr uint8_t kI2cSdaPin = 21; // Common ESP32 Dev Module I2C SDA; change for your wiring.
constexpr uint8_t kI2cSclPin = 22; // Common ESP32 Dev Module I2C SCL; change for your wiring.
constexpr uint8_t kServoCount = 19;
constexpr uint16_t kPwmFrequencyHz = 50;
constexpr float kServoMinUs = 600.0f;  // Calibrate per servo; never assume full travel is safe.
constexpr float kServoMaxUs = 2400.0f;
constexpr float kOffsetLimitDeg = 25.0f; // Independent firmware safety envelope for learned offsets.
constexpr float kSlewLimitDegPerSecond = 90.0f;

// Simulation actuator order: Revolute_5,4,3,2,1,10,9,8,7,6,19,14,13,12,11,18,17,16,15.
// Left arm (actuators 15..18) is assumed pitch, roll, yaw, elbow, mapped to logical channels 15,14,13,12.
// These four channels fit on PCA9685 0x40 outputs 12..15. Logical channels 16..31 map to PCA 0x46 outputs 0..15.
// Left leg (actuators 5..9) maps pitch, roll, yaw, knee, ankle to PCA 0x40 outputs 4..0.
// Right leg mirrors this layout on PCA 0x46 outputs 4..0; right arm mirrors the left arm on 0x46 outputs 15..12.
// zeroDeg=0 and pulse limits are placeholders; calibrate before connecting servos.
struct ServoConfig { uint8_t board; uint8_t channel; int8_t direction; float zeroDeg; float minUs; float maxUs; };
// Joint home and model limits in the RL actuator order above. Learned motion is an offset from home.
static const float kHomeDeg[kServoCount] = {-15, 0, 0, -30, 15, 15, 0, 0, 30, -15, 0, 0, -70, 0, 0, 0, 70, 0, 0};
static const float kJointMinDeg[kServoCount] = {-90, -45, -35, -140, -45, -90, -45, -35, -5, -45, -90, -85, -90, -140, -60, -85, -90, -5, -60};
static const float kJointMaxDeg[kServoCount] = {90, 45, 35, 5, 45, 90, 45, 35, 140, 45, 90, 85, 90, 5, 60, 85, 90, 140, 60};
static const ServoConfig kServos[kServoCount] = {
  // Right leg: pitch, roll, yaw, knee, ankle -> PCA 0x46 channels 4..0.
  {1, 4, 1, 0, kServoMinUs, kServoMaxUs}, {1, 3, 1, 0, kServoMinUs, kServoMaxUs},
  {1, 2, 1, 0, kServoMinUs, kServoMaxUs}, {1, 1, 1, 0, kServoMinUs, kServoMaxUs},
  {1, 0, 1, 0, kServoMinUs, kServoMaxUs},
  // Left leg: pitch, roll, yaw, knee, ankle -> PCA 0x40 channels 4..0.
  {0, 4, 1, 0, kServoMinUs, kServoMaxUs}, {0, 3, 1, 0, kServoMinUs, kServoMaxUs},
  {0, 2, 1, 0, kServoMinUs, kServoMaxUs}, {0, 1, 1, 0, kServoMinUs, kServoMaxUs},
  {0, 0, 1, 0, kServoMinUs, kServoMaxUs},
  // Torso yaw -> PCA 0x46 channel 8.
  {1, 8, 1, 0, kServoMinUs, kServoMaxUs},
  // Right arm: pitch, roll, yaw, elbow -> PCA 0x46 channels 15..12.
  {1, 15, 1, 0, kServoMinUs, kServoMaxUs}, {1, 14, 1, 0, kServoMinUs, kServoMaxUs},
  {1, 13, 1, 0, kServoMinUs, kServoMaxUs}, {1, 12, 1, 0, kServoMinUs, kServoMaxUs},
  // Left arm: pitch, roll, yaw, elbow -> PCA 0x40 channels 15..12.
  {0, 15, 1, 0, kServoMinUs, kServoMaxUs}, {0, 14, 1, 0, kServoMinUs, kServoMaxUs},
  {0, 13, 1, 0, kServoMinUs, kServoMaxUs}, {0, 12, 1, 0, kServoMinUs, kServoMaxUs},
};

Adafruit_PWMServoDriver pca0(kPcaAddress[0]);
Adafruit_PWMServoDriver pca1(kPcaAddress[1]);
Adafruit_PWMServoDriver *const kPca[2] = {&pca0, &pca1};
float currentDeg[kServoCount] = {0};
float targetDeg[kServoCount] = {0};
uint16_t motionIndex = 0;
uint16_t activeFrameDurationMs = 0;
bool playing = false;
uint32_t lastMotionMs = 0;
uint32_t lastServoUpdateMs = 0;

uint16_t pulseUsToTicks(float pulseUs) {
  const float ticks = pulseUs * 4096.0f * kPwmFrequencyHz / 1000000.0f;
  return static_cast<uint16_t>(constrain(static_cast<int>(lroundf(ticks)), 0, 4095));
}

void writeServo(uint8_t index, float requestedDeg) {
  const ServoConfig &servo = kServos[index];
  const float offset = constrain(requestedDeg, -kOffsetLimitDeg, kOffsetLimitDeg);
  const float jointDeg = constrain(kHomeDeg[index] + offset, kJointMinDeg[index], kJointMaxDeg[index]);
  const float signedAngle = servo.direction * jointDeg + servo.zeroDeg;
  const float centerUs = (servo.minUs + servo.maxUs) * 0.5f;
  const float pulseUs = constrain(centerUs + signedAngle * ((servo.maxUs - servo.minUs) / 180.0f),
                                  servo.minUs, servo.maxUs);
  kPca[servo.board]->setPWM(servo.channel, 0, pulseUsToTicks(pulseUs));
}

void stopSafely() {
  playing = false;
  motionIndex = 0;
  for (uint8_t i = 0; i < kServoCount; ++i) targetDeg[i] = 0.0f;
  Serial.println(F("Stopped; ramping to calibrated home."));
}

void startMotion() {
  if (kMotionFrameCount == 0) {
    Serial.println(F("No exported motion frames."));
    return;
  }
  const MotionFrame &first = kMotionFrames[0];
  for (uint8_t joint = 0; joint < kServoCount; ++joint)
    targetDeg[joint] = static_cast<float>(first.angleCdeg[joint]) / 100.0f;
  activeFrameDurationMs = first.durationMs;
  motionIndex = 1;
  playing = motionIndex < kMotionFrameCount;
  lastMotionMs = millis();
  Serial.println(F("Playing exported motion."));
}

void handleSerial() {
  while (Serial.available()) {
    const char command = static_cast<char>(Serial.read());
    if (command == 'p' || command == 'P') startMotion();
    else if (command == 's' || command == 'S' || command == ' ') stopSafely();
    else if (command == '?') Serial.println(F("Commands: p=play, s/space=stop and ramp to home, ?=help"));
  }
}

void advanceMotion(uint32_t now) {
  if (!playing || motionIndex >= kMotionFrameCount) return;
  if (static_cast<uint32_t>(now - lastMotionMs) < activeFrameDurationMs) return;
  const MotionFrame &frame = kMotionFrames[motionIndex];
  lastMotionMs += activeFrameDurationMs;
  activeFrameDurationMs = frame.durationMs;
  for (uint8_t joint = 0; joint < kServoCount; ++joint)
    targetDeg[joint] = static_cast<float>(frame.angleCdeg[joint]) / 100.0f;
  ++motionIndex;
  if (motionIndex >= kMotionFrameCount) playing = false; // Hold final pose until stop is sent.
}

void updateServos(uint32_t now) {
  const uint32_t elapsedMs = now - lastServoUpdateMs;
  if (elapsedMs < 20) return;
  lastServoUpdateMs = now;
  const float maxStep = kSlewLimitDegPerSecond * elapsedMs / 1000.0f;
  for (uint8_t i = 0; i < kServoCount; ++i) {
    currentDeg[i] += constrain(targetDeg[i] - currentDeg[i], -maxStep, maxStep);
    writeServo(i, currentDeg[i]);
  }
}

void setup() {
  Serial.begin(115200);
  Wire.begin(kI2cSdaPin, kI2cSclPin);
  Wire.setClock(400000);
  for (uint8_t board = 0; board < 2; ++board) {
    if (!kPca[board]->begin()) {
      Serial.printf("PCA9685 not found at 0x%02X; check power and address.\n", kPcaAddress[board]);
      while (true) delay(1000);
    }
    kPca[board]->setPWMFreq(kPwmFrequencyHz);
  }
  // Board should be mechanically supported and servo power switched off during first upload.
  for (uint8_t i = 0; i < kServoCount; ++i) writeServo(i, 0.0f);
  Serial.println(F("ESP32 humanoid ready. Send p to play, s to ramp to home."));
  Serial.println(F("Use external servo power with shared ground; calibrate PCA channels, direction, zero and pulse range."));
}

void loop() {
  const uint32_t now = millis();
  handleSerial();
  advanceMotion(now);
  updateServos(now);
}
