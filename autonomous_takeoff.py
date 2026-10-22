import time
import math
from pymavlink import mavutil
from ultralytics import YOLO
import cv2
import numpy as np

# ----------------- CONFIG -----------------
MODEL_PATH = "qr_ncnn_model"
VIDEO_DEVICE = 0
FRAME_W = 1280
FRAME_H = 720
TARGET_ALTITUDE = 1.2  # m (takeoff altitude)
ALT_MIN = 1.0
ALT_MAX = 1.5
CONF_THRESHOLD = 0.5

KP_YAW = 0.001
KP_ROLL = 0.0008
KP_PITCH = 0.0008

ALIGN_TOL = 20               # px tolerance for aligned
ALIGN_HOLD = 2.5             # seconds hold while aligned -> LAND
MAX_ALIGN_WAIT = 5.0         # seconds to try align before QLAND
NO_DET_TIMEOUT = 3.0         # seconds no detection -> QLAND

OVERRIDE_MIN_INTERVAL = 0.05  # not used but kept for completeness

# ----------------- UTILITIES -----------------
def to_quaternion(roll, pitch, yaw):
    cr = math.cos(roll/2)
    sr = math.sin(roll/2)
    cp = math.cos(pitch/2)
    sp = math.sin(pitch/2)
    cy = math.cos(yaw/2)
    sy = math.sin(yaw/2)
    return [cr*cp*cy + sr*sp*sy,
            sr*cp*cy - cr*sp*sy,
            cr*sp*cy + sr*cp*sy,
            cr*cp*sy - sr*sp*cy]

def send_attitude_target(conn, roll, pitch, yaw_rate, thrust):
    q = to_quaternion(roll, pitch, 0)
    try:
        conn.mav.set_attitude_target_send(
            0, conn.target_system, conn.target_component,
            0b00000000, q, 0, 0, yaw_rate, thrust
        )
    except Exception as e:
        print("send_attitude_target error:", e)

def set_mode(conn, mode_name):
    """Set flight mode by name"""
    mode_mapping = conn.mode_mapping()
    if mode_name not in mode_mapping:
        print(f"Unknown mode: {mode_name}")
        return False
    mode_id = mode_mapping[mode_name]
    conn.mav.set_mode_send(
        conn.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id
    )
    print(f"Mode change requested: {mode_name}")
    return True

def box_center(box):
    x1,y1,x2,y2 = box
    return ((x1+x2)//2, (y1+y2)//2)

def euclid(a,b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def draw_text(img, text, org, size=0.6, thickness=1):
    # white text with slight black shadow for readability
    x,y = org
    cv2.putText(img, text, (x+1, y+1), cv2.FONT_HERSHEY_SIMPLEX, size, (0,0,0), thickness+1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, (255,255,255), thickness, cv2.LINE_AA)

# ----------------- CAMERA -----------------
cap = cv2.VideoCapture(VIDEO_DEVICE)
if not cap.isOpened():
    print("ERROR: cannot open camera")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
time.sleep(0.5)  # short settle, but camera will remain open until program ends

# get actual frame size
ret, tmp = cap.read()
if not ret:
    print("ERROR: cannot read first frame")
    cap.release()
    exit(1)
FRAME_H_ACT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FRAME_W_ACT = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
FRAME_CX = FRAME_W_ACT // 2
FRAME_CY = FRAME_H_ACT // 2

# ----------------- MAVLINK -----------------
print("Connecting MAVLink...")
the_connection = mavutil.mavlink_connection("/dev/ttyACM0", baud=115200, mavlink_version=1)
print("Waiting for heartbeat...")
the_connection.wait_heartbeat()
print(f"Heartbeat from sys {the_connection.target_system} comp {the_connection.target_component}")

# ----------------- WAIT FOR ARM THEN TAKEOFF -----------------
mode = "WAITING_FOR_ARM"
print("Waiting for ARM... (do not send ARM from this script)")

# non-blocking heartbeat loop that also updates UI (we will transition to full loop later)
while True:
    hb = the_connection.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
    if hb and (hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED):
        mode = "ARMED -> TAKING_OFF"
        print("System ARMED. Initiating TAKEOFF.")
        # TAKEOFF (no ARM command sent by this script)
        the_connection.mav.command_long_send(
            the_connection.target_system,
            the_connection.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0, 0, 0, 0, 0, 0, 0, TARGET_ALTITUDE
        )
        break
    # keep waiting until armed
    # small sleep to avoid busy-loop
    time.sleep(0.2)

# wait until reach altitude
def get_alt_m(conn):
    msg = conn.recv_match(type='GLOBAL_POSITION_INT', blocking=False)
    if msg and hasattr(msg, 'relative_alt'):
        return msg.relative_alt / 1000.0
    return None

print("Waiting to reach target altitude...")
while True:
    alt = get_alt_m(the_connection)
    if alt is not None:
        print(f"Alt: {alt:.2f} m")
        if abs(alt - TARGET_ALTITUDE) < 0.1:
            mode = "SEARCHING"
            print("Target altitude reached. Entering detection.")
            break
    time.sleep(0.5)

# ----------------- MODEL LOAD -----------------
print("Loading detection model...")
try:
    model = YOLO(MODEL_PATH)
    print("Model loaded.")
except Exception as e:
    print("Model load error:", e)
    model = None

# ----------------- STATE VARIABLES -----------------
tracking_active = False
last_detection_time = 0.0
no_detection_start = None
alignment_attempt_start = None
alignment_start_time = None
previous_target_center = None
land_executed = False
land_time_sent = 0.0
last_override_time = 0.0

# FPS calc
fps = 0.0
frame_count = 0
t0 = time.time()

# ----------------- MAIN LOOP -----------------
print("Starting main loop. Press 'q' to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            # keep camera open; try again
            time.sleep(0.01)
            continue

        frame_count += 1
        now = time.time()
        elapsed = now - t0
        if elapsed >= 0.5:
            fps = frame_count / elapsed
            frame_count = 0
            t0 = now

        # read altitude (non-blocking)
        alt = get_alt_m(the_connection)
        alt_display = f"{alt:.2f} m" if alt is not None else "N/A"

        # if altitude out of tracking band, hold position and set mode
        if alt is None:
            current_mode = mode
        elif alt < ALT_MIN or alt > ALT_MAX:
            current_mode = "HOLDING (alt out of 1.0-1.5m)"
            # hold attitude
            send_attitude_target(the_connection, 0, 0, 0, 0.5)
        else:
            current_mode = mode  # SEARCHING / TRACKING / etc.

        # DETECTION
        detections = []
        if model is not None:
            try:
                results = model.predict(frame, imgsz=640, verbose=False, conf=CONF_THRESHOLD)
                # results may contain multiple results; use first
                for result in results:
                    for box in getattr(result, "boxes", []):
                        try:
                            conf = float(box.conf)
                        except:
                            conf = 0.0
                        if conf >= CONF_THRESHOLD:
                            coords = None
                            try:
                                coords = box.xyxy[0].cpu().numpy().astype(int)
                            except:
                                try:
                                    coords = np.array(box.xyxy).astype(int).flatten()
                                except:
                                    continue
                            if coords is None or len(coords) < 4:
                                continue
                            x1,y1,x2,y2 = map(int, coords[:4])
                            detections.append({'box':(x1,y1,x2,y2), 'conf':conf})
            except Exception as e:
                # detection error -> treat as no detection
                print("Model predict error:", e)
                detections = []

        # select best target (closest to previous for stability)
        target = None
        if detections:
            if previous_target_center:
                best = min(detections, key=lambda d: euclid(box_center(d['box']), previous_target_center))
                if euclid(box_center(best['box']), previous_target_center) < 200:
                    target = best
                else:
                    target = max(detections, key=lambda d: d['conf'])
            else:
                target = max(detections, key=lambda d: d['conf'])

        # update tracking / timers - SADECE land_executed False ise
        if not land_executed:
            if target:
                target_center = box_center(target['box'])
                previous_target_center = target_center
                last_detection_time = now
                if no_detection_start is not None:
                    no_detection_start = None
                if alignment_attempt_start is None:
                    alignment_attempt_start = now
                tracking_active = True
                mode = "TRACKING"
            else:
                # no target
                if tracking_active:
                    # just lost target
                    if no_detection_start is None:
                        no_detection_start = now
                    tracking_active = False
                    mode = "SEARCHING"
                else:
                    # still searching
                    if no_detection_start is None:
                        # keep waiting
                        mode = "SEARCHING"

        # If no detection for NO_DET_TIMEOUT -> QLAND
        if no_detection_start and (now - no_detection_start) >= NO_DET_TIMEOUT and not land_executed:
            print("No detection for 3s -> switching to QLAND")
            set_mode(the_connection, "QLAND")
            land_executed = True
            mode = "QLAND (no detection)"
            land_time_sent = now

        # If alignment attempt started but not aligned within MAX_ALIGN_WAIT -> QLAND
        if alignment_attempt_start and (now - alignment_attempt_start) >= MAX_ALIGN_WAIT and not land_executed:
            # check if currently aligned; if not, send LAND
            # compute current alignment if target exists
            aligned_now = False
            if target:
                tx, ty = box_center(target['box'])
                x_err = tx - FRAME_CX
                y_err = ty - FRAME_CY
                aligned_now = (abs(x_err) < ALIGN_TOL and abs(y_err) < ALIGN_TOL)
            if not aligned_now:
                print("Alignment not achieved within 5s -> switching to QLAND")
                set_mode(the_connection, "QLAND")
                land_executed = True
                mode = "QLAND (align timeout)"
                land_time_sent = now

        # If tracking and target exists -> control & alignment checks
        offset_text = "X: N/A  Y: N/A"
        align_text = ""
        if tracking_active and target and not land_executed:
            tx, ty = box_center(target['box'])
            x_err = tx - FRAME_CX
            y_err = ty - FRAME_CY
            offset_text = f"X: {x_err:+d}  Y: {y_err:+d}"

            roll = KP_ROLL * x_err
            pitch = -KP_PITCH * y_err
            yaw_rate = KP_YAW * x_err * 0.3
            thrust = 0.5

            # send attitude command to keep tracking
            send_attitude_target(the_connection, roll, pitch, yaw_rate, thrust)

            aligned = (abs(x_err) < ALIGN_TOL and abs(y_err) < ALIGN_TOL)

            if aligned:
                if alignment_start_time is None:
                    alignment_start_time = now
                else:
                    hold = now - alignment_start_time
                    align_text = f"ALIGNED HOLD: {hold:.1f}s"
                    # if held long enough -> QLAND
                    if hold >= ALIGN_HOLD and not land_executed:
                        print("Aligned hold reached -> switching to QLAND")
                        set_mode(the_connection, "QLAND")
                        land_executed = True
                        mode = "QLAND (aligned)"
                        land_time_sent = now
            else:
                alignment_start_time = None
                # provide time left until alignment timeout
                if alignment_attempt_start:
                    time_left = max(0.0, MAX_ALIGN_WAIT - (now - alignment_attempt_start))
                    align_text = f"ALIGN WAIT: {time_left:.1f}s left"
        else:
            alignment_start_time = None

        # DRAW DETECTION BOXES
        disp = frame.copy()
        if target:
            x1,y1,x2,y2 = target['box']
            cv2.rectangle(disp, (x1,y1), (x2,y2), (0,0,255), 2)
            draw_text(disp, f"TARGET {target['conf']:.2f}", (x1, y1-10), size=0.6, thickness=1)

        # UI overlay (English, white text)
        draw_text(disp, f"MODE: {mode}", (10, 25), size=0.7, thickness=2)
        draw_text(disp, f"FPS: {fps:.1f}", (10, 55))
        draw_text(disp, f"ALT: {alt_display}", (10, 85))
        draw_text(disp, f"{offset_text}", (10, 115))
        draw_text(disp, f"{align_text}", (10, 145))

        # show countdowns if any
        if no_detection_start and not land_executed:
            dt = now - no_detection_start
            left = max(0.0, NO_DET_TIMEOUT - dt)
            draw_text(disp, f"NO TARGET -> QLAND in {left:.1f}s", (10, 175))
        if alignment_attempt_start and not land_executed:
            dt = now - alignment_attempt_start
            left = max(0.0, MAX_ALIGN_WAIT - dt)
            draw_text(disp, f"ALIGN TIMELEFT {left:.1f}s (max {MAX_ALIGN_WAIT}s)", (10, 205))

        if land_executed:
            draw_text(disp, "QLAND MODE ACTIVATED", (10, 235), size=0.7, thickness=2)

        # small footer
        draw_text(disp, "Press 'q' to quit", (10, FRAME_H_ACT-10), size=0.5)

        cv2.imshow("Detection & Status", disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("User requested exit.")
            break

        # if we sent land, show overlay a bit and then exit
        if land_executed:
            # keep showing for 3 seconds then break
            if time.time() - land_time_sent >= 3.0:
                break

except KeyboardInterrupt:
    print("Keyboard interrupt, exiting...")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Program terminated.")
