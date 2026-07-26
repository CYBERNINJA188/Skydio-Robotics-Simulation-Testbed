# Deterministic 6-DoF Drone Flight Controller & Sensor Emulation Testbed

A high-performance MuJoCo drone simulation environment featuring deterministic physics time-stepping, hardware sensor emulation (IMU noise & bus latency), closed-loop PID attitude/altitude stabilization, and asynchronous zero-shot perception.

---

## 🛠️ System Architecture & Key Features

* **Deterministic Loop Synchronization:** Flight controller integration is strictly locked to `dt = model.opt.timestep` (100 Hz), completely decoupling control loop math from rendering frame drops or VLM inference latency.
* **Hardware Sensor Emulation:** 
  * **IMU Noise Injection:** Injects zero-mean Gaussian noise ($\mathcal{N}(0, \sigma^2)$) into raw gyroscope and velocity readouts to simulate physical sensor imperfections.
  * **Bus Latency Buffering:** Uses a `collections.deque` ring buffer to emulate I2C/SPI bus transmission delays and filtering lag before data reaches the PID loop.
* **Closed-Loop Attitude & Position Hold:** Nested PID controllers for pitch, roll, yaw, and altitude, combined with rotation matrix ($R$) velocity mapping for active VIO-style drift braking.
* **Asynchronous Perception Pipeline:** Multiprocessed CPU worker running zero-shot YOLO-World tracking in parallel over thread-safe queues (`multiprocessing.Queue`).

---

## 🎥 Live Demonstration & Codebase

* **Live FPV Perception Demo:** [Watch on YouTube](https://youtu.be/Jd8fH74Pz9s)

---

## 💻 Tech Stack & Requirements

* **Language:** Python 3.10+
* **Physics Engine:** MuJoCo (`mujoco`)
* **Computer Vision & VLM:** OpenCV, Ultralytics YOLO-World
* **Concurrency:** `multiprocessing`, `threading`
