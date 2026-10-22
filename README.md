# 📡 UAV Telemetry & Mission Computer Suite

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Protocol](https://img.shields.io/badge/Protocol-MAVLink%20%7C%20TCP%2FUDP%20Sockets-orange.svg)](https://mavlink.io/en/)
[![UI](https://img.shields.io/badge/UI-PyQt5%20GUI-green.svg)](https://riverbankcomputing.com/software/pyqt/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An avionics communications, telemetry streaming, and mission coordination software stack designed for companion computers (Raspberry Pi / Radxa Rock) onboard autonomous Unmanned Aerial Vehicles (UAVs).

This repository contains real-time MAVLink telemetry decoders, high-frequency TCP/UDP socket servers, graphical IMU monitoring tools, and autonomous flight state dispatchers developed and validated in competition environments.

---

## 📌 Architecture Overview

```
 ┌──────────────────────┐         UART / USB         ┌──────────────────────┐
 │  Flight Controller   │ ─────────────────────────▶ │  Companion Computer  │
 │ (Pixhawk / ArduPilot)│    MAVLink Stream (57.6k)  │   (Debian Linux)     │
 └──────────────────────┘                            └──────────┬───────────┘
                                                                │
                     ┌──────────────────────────────────────────┴───────────────┐
                     │                                                          │
                     ▼                                                          ▼
        ┌─────────────────────────┐                                ┌─────────────────────────┐
        │ telemetry_broadcaster.py│                                │ lockon_socket_server.py │
        │  (MAVLink Packet Parser)│                                │  (Inter-UAV Engagement) │
        └────────────┬────────────┘                                └────────────┬────────────┘
                     │ UDP / JSON                                               │ TCP Sockets
                     ▼                                                          ▼
        ┌─────────────────────────┐                                ┌─────────────────────────┐
        │  imu_telemetry_gui.py   │                                │   telemetry_client.py   │
        │ (Ground Station Display)│                                │   (Referee / Server)    │
        └─────────────────────────┘                                └─────────────────────────┘
```

---

## ✨ Modules & Functionality

*   **`telemetry_broadcaster.py`:**
    *   Connects to ArduPilot via MAVLink (`/dev/ttyAMA0` or serial USB).
    *   Extracts vehicle attitude (`roll`, `pitch`, `yaw`), GPS coordinates (`lat`, `lon`, `alt`), airspeed, and battery voltage.
    *   Serializes telemetry packets and broadcasts them over UDP/IP sockets to ground control networks.
*   **`lockon_socket_server.py`:**
    *   Real-time multi-threaded TCP server coordinating air-to-air target lock notifications.
    *   Provides lock duration verification, timestamp synchronization, and engagement validation.
*   **`imu_telemetry_gui.py`:**
    *   A responsive desktop GUI displaying live artificial horizon, gyroscope rates, accelerometer loads, and Euler angles.
*   **`autonomous_takeoff.py`:**
    *   Guided mode armed takeoff sequencer with safety checks (pre-arm status, satellite lock verification, altitude hold).

---

## 🚀 Getting Started

### 1. Installation
```bash
git clone https://github.com/<your-username>/uav-telemetry-mission-suite.git
cd uav-telemetry-mission-suite
pip install -r requirements.txt
```

### 2. Broadcasting Telemetry
Launch the telemetry broadcaster pointing to your autopilot serial port:
```bash
python telemetry_broadcaster.py --device /dev/ttyAMA0 --baud 57600 --port 14550
```

### 3. Launching the IMU Ground Station GUI
```bash
python imu_telemetry_gui.py
```

---

## 👩‍💻 Author

**Begüm Aydoğan Çiçin**  
*Mechatronics Engineer | Embedded Software & Avionics Specialist*  
* Contact: `aydgnbegm@gmail.com`  
* [LinkedIn](https://www.linkedin.com/in/begüm-aydoğan-çiçin-a94098201)
