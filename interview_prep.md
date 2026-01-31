# Systems Software Engineer - Mock Interview Modules

Based on the Job Description and the `finger_detection` codebase, here are 5 "Interview Modules" to practice different aspects of the role.

## 1. System Reliability & Recovery (Robustness)
**The Scenario:** The camera cable is loose or the driver crashes. Currently, if `cv2.VideoCapture` fails or disconnects, the app crashes or exits immediately.
**The Task:** 
- Refactor `VideoSource` (or `JetsonCameraSource`) to implement a **fail-over and self-healing mechanism**.
- If the stream dies, the application should not crash. It should enter a "reconnecting" state and attempt to re-initialize the camera indefinitely (e.g., exponential backoff).
- Challenge: Ensure the main loop in `main.py` handles `None` frames gracefully without freezing.
**Skills:** Error handling, state machines, resource cleanup, robustness.

## 2. Edge-to-Cloud Telemetry (Networking/API)
**The Scenario:** The Operations team needs to monitor the health of 1000 deployed devices. They need to know if the app is actually processing frames or if it's hung.
**The Task:**
- Add a background thread or process that exposes a local heartbeat HTTP endpoint (e.g., `GET /health` on port 5000).
- The endpoint should return a JSON response with:
  - `status`: "ok" or "error"
  - `current_fps`: float
  - `hands_detected`: int (most recent count)
  - `uptime_seconds`: float
- Use `Flask`, `FastAPI`, or standard library `http.server`.
**Skills:** Concurrency (Thread safety with shared state), API design, non-blocking I/O.

## 3. The Deployment Artifact (DevOps/Linux)
**The Scenario:** We need to ship this application to a fleet of NVIDIA Jetson devices. We cannot rely on manually running `python main.py`.
**The Task:**
- **Dockerization:** Write a `Dockerfile` that packages the app.
  - *Constraint:* Must handle complex dependencies like OpenCV/GStreamer on ARM64 (Jetson). (You can mock the base image selection for now).
- **Service Management:** Write a `pianist.service` systemd unit file.
  - It should start the container (or script) on boot.
  - It must automatically restart the service if it crashes (Restart=always).
  - Configure logging to `journald`.
**Skills:** Linux systems, Docker, systemd, environment management.

## 4. Mock OTA Implementation (System Design)
**The Scenario:** A critical bug was found in the hand detection logic. We need to push an update to all devices without manual SSH access.
**The Task:**
- Design and implement a lightweight **OTA Agent** (`ota_agent.py`) that runs alongside the main app.
- Logic:
  1. Every 5 minutes, check a remote URL (mocked) for a version manifest (e.g., `{"version": "1.0.1", "git_hash": "..."}`).
  2. If the remote version > local version:
     - Perform a `git pull`.
     - Restart the `pianist.service` (using `subprocess` to call `systemctl`).
     - Report success/failure back to the server.
**Skills:** System design, shell interaction, OTA workflows, secure updates.

## 5. Hybrid Performance Tuning (Optimization)
**The Scenario:** The `AsyncWriter` is saving JPEGs for debugging, but it's too slow. `cv2.imwrite` is CPU-blocking even in a thread, causing the Global Interpreter Lock (GIL) to contend with the main loop, dropping the inference FPS.
**The Task:**
- **Profile:** Demonstrate how you would measure the time spent in `imwrite`.
- **Optimize:** Implement a solution to increase throughput.
  - *Option A:* Use a `multiprocessing.Process` queue instead of `threading` to bypass the GIL.
  - *Option B:* Use GStreamer hardware-accelerated encoding ( `appsrc` -> `omxh264enc` -> `filesink`) instead of OpenCV's CPU JPEG encoder.
**Skills:** Performance profiling, Python GIL understanding, multiprocessing vs threading, hardware acceleration.
