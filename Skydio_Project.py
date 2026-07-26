import os
import sys
import time
import queue
import threading
import multiprocessing
import numpy as np
import keyboard
from collections import deque  # Added for hardware latency buffering


# --- ADDED: Sensor Hardware Emulation Helper ---
def apply_imu_noise(raw_data, noise_std=0.01):
    """
    Simulates hardware sensor imperfections by adding zero-mean Gaussian noise
    to raw readings before feeding them to the PID controller.
    """
    return raw_data + np.random.normal(0, noise_std, size=raw_data.shape)
# -----------------------------------------------

def yolo_worker(image_queue, coords_queue, prompt_queue):
    """
        RUNS ON CPU CORE #2 (Isolated Process)
        Safely initializes YOLOv10 directly from your local Hugging Face folder files.
    """
    from ultralytics import YOLO  # Ensure YOLO is imported in the worker
    print("[+] (Perception Process) Initializing YOLO-World Small V2 Brain...")

    # Direct path to your safely downloaded .pt file
    LOCAL_FILE_PATH = r"C:\Users\-TECHNO-\AGI_env\yolov8s-worldv2.pt"

    # Explicit check to verify the weights file is physically present
    if not os.path.exists(LOCAL_FILE_PATH):
        print(f"[!] CRITICAL ERROR: Could not find model weights at {LOCAL_FILE_PATH}")
        return

    try:
        # Pass the file path directly to the YOLO object; it automatically detects YOLO-World format!
        model = YOLO(LOCAL_FILE_PATH)
        print(f"[+] (Perception Process) YOLO-World v2 loaded perfectly from local file!")
    except Exception as err:
        print(f"[!] CRITICAL ERROR: Failed to load YOLO-World. {err}")
        return

    # Setting target class tracking default to an empty list (Blank Slate)
    active_targets = []

    while True:
        try:
            # Check for a target query update from the Tkinter text tab
            try:
                gui_query = prompt_queue.get_nowait()

                # Split by comma, strip whitespace, lower case, and keep up to 5 targets
                raw_targets = [t.strip().lower() for t in gui_query.split(',') if t.strip()]
                active_targets = raw_targets[:5]

                if active_targets:
                    # Dynamically update YOLO-World's active vocabulary with the full list
                    model.set_classes(active_targets)
                    model.predictor = None  # Force YOLO to wipe the old embeddings
                    print(f"\n[+] (YOLO Process) Focus shifted to tracking: {active_targets}")
                else:
                    print("\n[+] (YOLO Process) Tracking cleared. Blank slate mode.")

            except Exception:
                pass

            # Pull frame from camera queue (Note: MuJoCo outputs RGB arrays)
            img_np = image_queue.get()

            # Convert MuJoCo's RGB frame array to BGR so YOLO sees the true color signatures
            import cv2
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            if not active_targets:
                annotated_frame = img_bgr
            else:
                # 1. Direct prediction on the corrected BGR frame array tracking all active classes
                results_generator = model.predict(source=img_bgr, verbose=False, device='cuda', stream=True,
                                                  conf=0.01)
                results = next(results_generator)

                # 2. Let YOLO-World natively draw bounding boxes, labels, and scores for all targets (returns BGR)
                annotated_frame = results.plot()

            # 3. Ship the fully processed frame backward using the coords_queue slot
            if coords_queue.full():
                try:
                    coords_queue.get_nowait()
                except:
                    pass
            coords_queue.put(annotated_frame)

        except Exception as e:
            pass


def main():
    import mujoco
    import mujoco.viewer

    xml_path = r"C:\Users\-TECHNO-\Mujoco\Skydio Quad Copter\Industrial_world.xml"

    if not os.path.exists(xml_path):
        print(f"[-] Error: Cannot locate file at {xml_path}")
        return

    print("==================================================")
    print("     MANUAL OVERRIDE TESTBED (GAME CONTROLS)      ")
    print("==================================================")

    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)

        # --- PID TUNING PARAMETERS ---
        BASE_GRAVITY_COMP = 3.12

        Kp = 2.5
        Ki = 0.1
        Kd = 1.2

        target_z = 0.3
        integral_error = 0.0
        last_error = 0.0

        VERT_RATE = 1.0
        MOVE_POWER = 0.05
        TILT_LIMIT = 0.15
        K_brake = 0.12
        # ------------------------

        # === SENSOR HARDWARE EMULATION (LATENCY BUFFER) ===
        # Simulates the I2C/SPI bus transmission delay and filtering latency
        LATENCY_TICKS = 3
        gyro_buffer = deque(maxlen=LATENCY_TICKS)
        vel_buffer = deque(maxlen=LATENCY_TICKS)
        # ==================================================

        # === MULTIPROCESSING INTERFACE HOOKS ===
        image_queue = multiprocessing.Queue(maxsize=1)
        coords_queue = multiprocessing.Queue(maxsize=1)
        prompt_queue = multiprocessing.Queue(maxsize=1)

        vlm_process = multiprocessing.Process(target=yolo_worker, args=(image_queue, coords_queue, prompt_queue),
                                              daemon=True)
        vlm_process.start()

        shared_state = {
            "query": None,
            "is_typing": False
        }

        def console_input_listener():
            import tkinter as tk
            from tkinter import simpledialog

            root = tk.Tk()
            root.withdraw()

            while True:
                if shared_state["is_typing"]:
                    new_target = simpledialog.askstring(
                        "Flyability Drone Console",
                        "Enter up to 5 targets separated by commas (e.g. red wall, blue wall, drone gate):",
                        parent=root
                    )

                    if new_target:
                        clean_target = new_target.strip().lower()
                        shared_state["query"] = clean_target

                        if prompt_queue.full():
                            try:
                                prompt_queue.get_nowait()
                            except:
                                pass
                        prompt_queue.put(clean_target)
                        print(f"\n[+] Target cleanly updated via GUI to: '{clean_target}'")

                    shared_state["is_typing"] = False
                time.sleep(0.1)

        input_thread = threading.Thread(target=console_input_listener, daemon=True)
        input_thread.start()

        current_vlm_coords = None
        vlm_query_str = None
        # =======================================

        print("[+] PID Flight Controller Initialized!")

        fpv_mode = False
        camera_toggle_pressed = False

        try:
            fpv_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "fpv_cam")
        except:
            fpv_cam_id = -1
            print("[-] Warning: 'fpv_cam' not found in XML. FPV toggle disabled.")

        print(" -> Hold [Space] to raise target altitude.")
        print(" -> Hold [Shift] to lower target altitude.")

        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                step_start = time.time()

                # 0. STRICT DETERMINISTIC TIME-STEPPING (Skydio Requirement)
                # Lock PID integration exactly to the physics engine tick rate,
                # completely decoupling it from fluctuating wall-clock rendering times.
                dt = model.opt.timestep

                # 1. UPDATE TARGET ALTITUDE
                if keyboard.is_pressed('space'):
                    target_z += VERT_RATE * dt
                elif keyboard.is_pressed('shift'):
                    target_z -= VERT_RATE * dt

                target_z = max(0.1, target_z)

                # 1.5 CAMERA TOGGLE LOGIC
                if keyboard.is_pressed('c'):
                    if not camera_toggle_pressed and fpv_cam_id != -1:
                        fpv_mode = not fpv_mode
                        camera_toggle_pressed = True

                        if fpv_mode:
                            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                            viewer.cam.fixedcamid = fpv_cam_id
                        else:
                            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                else:
                    camera_toggle_pressed = False

                # 2. READ ACTUAL SENSOR DATA & INJECT NOISE/LATENCY
                current_z = data.qpos[2]  # Barometer/GPS approximation

                # Extract raw gyro and inject Gaussian noise
                raw_gyro = data.qvel[3:6]
                noisy_gyro = apply_imu_noise(raw_gyro, noise_std=0.015)
                gyro_buffer.append(noisy_gyro)

                # Extract raw global velocity and inject noise (simulating VIO drift)
                raw_vel = data.qvel[0:2]
                noisy_vel = apply_imu_noise(raw_vel, noise_std=0.02)
                vel_buffer.append(noisy_vel)

                # Fetch delayed sensor readings from the ring buffer
                delayed_gyro = gyro_buffer[0] if len(gyro_buffer) == LATENCY_TICKS else noisy_gyro
                delayed_vel = vel_buffer[0] if len(vel_buffer) == LATENCY_TICKS else noisy_vel

                # 3. PID CALCULATIONS (Using deterministic dt)
                error = target_z - current_z

                integral_error += error * dt
                integral_error = max(-2.0, min(integral_error, 2.0))

                derivative = (error - last_error) / dt if dt > 0 else 0.0
                last_error = error

                pid_adjustment = (Kp * error) + (Ki * integral_error) + (Kd * derivative)
                target_thrust = BASE_GRAVITY_COMP + pid_adjustment
                target_thrust = max(1.0, min(target_thrust, 6.0))

                t0 = t1 = t2 = t3 = target_thrust

                # 4. DIRECTIONAL INPUT WITH NESTED ATTITUDE PID CONTROLLER
                target_pitch = 0.0
                target_roll = 0.0

                res_mat = np.zeros(9)
                mujoco.mju_quat2Mat(res_mat, data.qpos[3:7])
                R = res_mat.reshape(3, 3)

                user_moving_pitch = False
                user_moving_roll = False

                if keyboard.is_pressed('w'):
                    target_pitch = -MOVE_POWER
                    user_moving_pitch = True
                elif keyboard.is_pressed('s'):
                    target_pitch = MOVE_POWER
                    user_moving_pitch = True

                if keyboard.is_pressed('d'):
                    target_roll = -MOVE_POWER
                    user_moving_roll = True
                elif keyboard.is_pressed('a'):
                    target_roll = MOVE_POWER
                    user_moving_roll = True

                    # ACTIVE POSITION HOLD (Using delayed hardware-emulated velocity)
                    global_vel_x = delayed_vel[0]
                    global_vel_y = delayed_vel[1]

                    local_vel_x = R[0, 0] * global_vel_x + R[0, 1] * global_vel_y
                    local_vel_y = R[1, 0] * global_vel_x + R[1, 1] * global_vel_y

                    if not user_moving_pitch:
                        target_pitch = -local_vel_x * K_brake
                    if not user_moving_roll:
                        target_roll = local_vel_y * K_brake

                actual_roll = np.arctan2(R[2, 1], R[2, 2])
                actual_pitch = -np.arcsin(R[2, 0])

                # Use delayed, noisy gyro rates for D-term
                gyro_roll = delayed_gyro[0]
                gyro_pitch = delayed_gyro[1]
                gyro_yaw = delayed_gyro[2]

                Kp_angle = 8.0
                Kd_angle = 1.5

                roll_error = target_roll - actual_roll
                pitch_error = target_pitch - actual_pitch

                roll_offset = -((Kp_angle * roll_error) - (Kd_angle * gyro_roll))
                pitch_offset = (Kp_angle * pitch_error) - (Kd_angle * gyro_pitch)

                pitch_offset = max(-TILT_LIMIT, min(pitch_offset, TILT_LIMIT))
                roll_offset = max(-TILT_LIMIT, min(roll_offset, TILT_LIMIT))

                yaw_input = 0.0
                YAW_POWER = 0.88

                if keyboard.is_pressed('right'): yaw_input = YAW_POWER
                if keyboard.is_pressed('left'): yaw_input = -YAW_POWER

                Kd_yaw = 0.6
                yaw_offset = yaw_input + (Kd_yaw * gyro_yaw)

                # 5. MIXING MATRIX WITH YAW INTEGRATION
                t0 += pitch_offset + roll_offset + yaw_offset
                t1 += pitch_offset - roll_offset - yaw_offset
                t2 += -pitch_offset - roll_offset + yaw_offset
                t3 += -pitch_offset + roll_offset - yaw_offset

                data.ctrl[0] = t0
                data.ctrl[1] = t1
                data.ctrl[2] = t2
                data.ctrl[3] = t3

                mujoco.mj_step(model, data)
                viewer.sync()

                # === NON-BLOCKING ASYNC VLM PROCESSING & HUD OVERLAY ===
                if fpv_mode and fpv_cam_id != -1:
                    import cv2
                    if not hasattr(main, "vlm_renderer"):
                        main.vlm_renderer = mujoco.Renderer(model, height=480, width=640)

                    main.vlm_renderer.update_scene(data, camera="fpv_cam")
                    rgb_buffer = main.vlm_renderer.render()

                    if not image_queue.full():
                        image_queue.put(rgb_buffer.copy())

                    try:
                        current_vlm_coords = coords_queue.get_nowait()
                    except queue.Empty:
                        pass

                    display_frame = current_vlm_coords if current_vlm_coords is not None else cv2.cvtColor(
                        rgb_buffer, cv2.COLOR_RGB2BGR)

                    center_x, center_y = 320, 240
                    cv2.line(display_frame, (center_x - 15, center_y), (center_x + 15, center_y), (0, 255, 0), 2)
                    cv2.line(display_frame, (center_x, center_y - 15), (center_x, center_y + 15), (0, 255, 0), 2)
                    cv2.circle(display_frame, (center_x, center_y), 2, (0, 0, 255), -1)

                    cv2.imshow("FPV Live Perception Feed", display_frame)
                    cv2.waitKey(1)
                else:
                    current_vlm_coords = None
                    try:
                        import cv2
                        cv2.destroyAllWindows()
                    except:
                        pass
                # =======================================================

                if keyboard.is_pressed('t') and not shared_state["is_typing"]:
                    shared_state["is_typing"] = True

                    x, y, z = data.qpos[0:3]
                    metrics = f"Target Z: {target_z:4.2f}m | Actual Z: {z:4.2f}m | Err: {error:5.2f} | [T]: Open Prompt Window"
                    sys.stdout.write("\r" + metrics)
                    sys.stdout.flush()

                # Pace the wall-clock loop to strictly match the physics engine tick
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    except KeyboardInterrupt:
        print("\n\n[-] Manual testbed stopped cleanly.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()