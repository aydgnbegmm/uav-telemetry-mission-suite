import sys
import cv2
import json
import socket
import threading
import time
import os
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGroupBox, QTextEdit,
    QWidget, QProgressBar, QGridLayout, QFrame, QRadioButton,
    QSizePolicy, QScrollArea
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QImage, QPixmap, QFont
import numpy as np
from collections import deque
from typing import Optional, Tuple, List, Dict, Any
from ultralytics import YOLO
import logging
from dataclasses import dataclass
from enum import Enum, auto
import torch

# SistemDurumu Enum
class SistemDurumu(Enum):
    BASLATILIYOR = auto()
    OTONOM_MOD = auto()
    DOST_BALON_ARAMA = auto()
    HEDEF_KILITLEME = auto()
    BEKLEME = auto()
    KAPATILIYOR = auto()

# Config sınıfı
@dataclass
class Config:
    CAMERA_ID: int = 0
    CAMERA_WIDTH: int = 640
    CAMERA_HEIGHT: int = 480
    MODEL_PATH: str = "b.onnx"
    MAX_MISSING_FRAMES: int = 15
    KALMAN_PROCESS_NOISE: float = 0.001
    KALMAN_MEASUREMENT_NOISE: float = 5.0
    MAX_VELOCITY_PPS: float = 200.0
    MAX_ACCELERATION_PPS2: float = 100.0
    SMOOTHING_ALPHA: float = 0.7
    MEASUREMENT_QUALITY_THRESHOLD: float = 0.5
    DEBUG_MODE: bool = True
    CONFIDENCE: float = 0.6
    TARGET_FPS: int = 30
    USE_CUDA: bool = True
    BEKLEME_SURESI: float = 3.0
    LOG_PATH: str = os.path.expanduser("~/Desktop/balloon_tracker")
    LOG_LEVEL: str = "DEBUG"
    ATIS_ESIK_PIKSEL: int = 10
    DYNAMIC_CONFIG: bool = False
    NAMLU_OFFSET_MM: float = 57.0
    PIXEL_PER_MM: float = 2.0
    VELOCITY_THRESHOLD: float = 5.0
    ANGLE_STEP: float = 5.0  # New: Degree increment for manual controls

# EnhancedLogger sınıfı
class EnhancedLogger:
    def __init__(self, config: Config):
        self.config = config
        self._setup_logging()

    def _setup_logging(self) -> None:
        os.makedirs(self.config.LOG_PATH, exist_ok=True)
        log_file = os.path.join(
            self.config.LOG_PATH,
            f"tracker_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        logging.root.handlers = []
        formatter = logging.Formatter(
            fmt='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handlers = []
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)
        logging.basicConfig(
            level=getattr(logging, self.config.LOG_LEVEL),
            handlers=handlers
        )
        self.log("Logger initialized", "INFO", {"config": vars(self.config)})

    def log(self, message: str, level: str, extra_data: Optional[Dict[str, Any]] = None) -> None:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "level": level,
            **({"data": extra_data} if extra_data else {})
        }
        try:
            log_msg = json.dumps(log_entry, ensure_ascii=False, default=str)
            getattr(logging, level.lower())(log_msg)
        except Exception as e:
            logging.error(f"Logging error: {str(e)} - Original: {message}")

# Stream sınıfı
class Stream(QObject):
    newText = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._buffer = []

    def write(self, text):
        if text and text.strip():
            self.newText.emit(str(text))
            self._buffer.append(text)

    def flush(self):
        self._buffer.clear()

# UdpReceiver sınıfı
class UdpReceiver(QObject):
    imu_data_received = pyqtSignal(dict)
    control_signal = pyqtSignal(str)

    def __init__(self, listen_port=8888, target_ip='10.0.0.131', target_port=8888):
        super().__init__()
        self.listen_ip = '0.0.0.0'
        self.listen_port = listen_port
        self.target_ip = target_ip
        self.target_port = target_port
        self.running = False
        self.control_socket = None
        self.fixed_mode = False

    def start(self):
        self.running = True
        thread = threading.Thread(target=self.listen_udp, daemon=True)
        thread.start()
        self.control_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_control_command(self, command, control_mode="AUTONOMOUS", x_offset=None, y_offset=None, yaw=None, pitch=None):
        if self.control_socket:
            try:
                if control_mode == "AUTONOMOUS" and x_offset is not None and y_offset is not None:
                    data = {
                        "mode": "AUTONOMOUS",
                        "x_offset": float(x_offset),
                        "y_offset": float(y_offset)
                    }
                else:
                    # Manual mode: Send command with IMU data (yaw, pitch) if available
                    data = {
                        "mode": "MANUAL",
                        "command": command,
                        "yaw": float(yaw) if yaw is not None else None,
                        "pitch": float(pitch) if pitch is not None else None
                    }
                message = json.dumps(data)
                self.control_socket.sendto(message.encode(), (self.target_ip, self.target_port))
                logger.log(f"Komut gönderildi: {message}", "INFO")
            except Exception as e:
                logger.log(f"Komut gönderme hatası: {e}", "ERROR")

    def listen_udp(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind((self.listen_ip, self.listen_port))
            sock.settimeout(1.0)
            logger.log(f"UDP dinleme başlatıldı - {self.listen_ip}:{self.listen_port}", "INFO")
            while self.running:
                try:
                    data, _ = sock.recvfrom(4096)
                    data_str = data.decode('utf-8')
                    imu_dict = json.loads(data_str)
                    self.imu_data_received.emit(imu_dict)
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        logger.log(f"UDP alma hatası: {e}", "ERROR")
        except Exception as e:
            logger.log(f"UDP soket hatası: {e}", "ERROR")
        finally:
            if 'sock' in locals() and sock:
                sock.close()

    def stop(self):
        self.running = False
        if self.control_socket:
            self.control_socket.close()

# GelismisKalmanFiltresi sınıfı
class GelismisKalmanFiltresi:
    def __init__(self, config: Config):
        self.config = config
        self.kalman = cv2.KalmanFilter(6, 2)
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ], np.float32)
        self.kalman.statePost = np.array([[0], [0], [0], [0], [0], [0]], dtype=np.float32)
        self.tahmin_gecmisi = deque(maxlen=20)
        self.measurement_history = deque(maxlen=10)
        self.guven_skoru = 0.0
        self.consecutive_good_measurements = 0
        self.measurement_variance = 0.0
        self.last_update_time = time.time()
        self.last_position = None
        self.velocity_magnitude = 0.0
        self.acceleration_magnitude = 0.0
        self.base_process_noise = config.KALMAN_PROCESS_NOISE
        self.base_measurement_noise = config.KALMAN_MEASUREMENT_NOISE
        self.is_initialized = False
        self.max_velocity = 200.0
        self.max_acceleration = 100.0
        self._initialize_matrices()

    def _initialize_matrices(self):
        dt = 1.0 / 30.0
        self.kalman.transitionMatrix = np.array([
            [1, 0, dt, 0, 0.5 * dt * dt, 0],
            [0, 1, 0, dt, 0, 0.5 * dt * dt],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], np.float32)
        self.kalman.processNoiseCov = np.eye(6, dtype=np.float32)
        self.kalman.processNoiseCov[0:2, 0:2] *= self.base_process_noise
        self.kalman.processNoiseCov[2:4, 2:4] *= self.base_process_noise * 2
        self.kalman.processNoiseCov[4:6, 4:6] *= self.base_process_noise * 5
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * self.base_measurement_noise
        self.kalman.errorCovPost = np.eye(6, dtype=np.float32) * 1000

    def _update_transition_matrix(self, dt):
        dt = max(0.001, min(dt, 0.1))
        self.kalman.transitionMatrix[0, 2] = dt
        self.kalman.transitionMatrix[0, 4] = 0.5 * dt * dt
        self.kalman.transitionMatrix[1, 3] = dt
        self.kalman.transitionMatrix[1, 5] = 0.5 * dt * dt
        self.kalman.transitionMatrix[2, 4] = dt
        self.kalman.transitionMatrix[3, 5] = dt

    def _calculate_measurement_quality(self, measurement: Tuple[int, int]) -> float:
        if len(self.measurement_history) < 2:
            return 0.5
        recent_positions = list(self.measurement_history)[-3:]
        if len(recent_positions) >= 2:
            positions = np.array(recent_positions)
            variance = np.var(positions, axis=0)
            self.measurement_variance = np.mean(variance)
            quality = 1.0 / (1.0 + self.measurement_variance / 100.0)
            return min(1.0, max(0.1, quality))
        return 0.5

    def _adaptive_noise_adjustment(self, measurement_quality: float):
        if measurement_quality > 0.8:
            measurement_noise_factor = 0.5
            self.consecutive_good_measurements += 1
        elif measurement_quality > 0.5:
            measurement_noise_factor = 1.0
            self.consecutive_good_measurements = max(0, self.consecutive_good_measurements - 1)
        else:
            measurement_noise_factor = 2.0
            self.consecutive_good_measurements = 0
        if self.velocity_magnitude > 50:
            process_noise_factor = 2.0
        elif self.velocity_magnitude < 10:
            process_noise_factor = 0.5
        else:
            process_noise_factor = 1.0
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * (
                self.base_measurement_noise * measurement_noise_factor
        )
        self.kalman.processNoiseCov[0:2, 0:2] = np.eye(2, dtype=np.float32) * (
                self.base_process_noise * process_noise_factor
        )
        self.kalman.processNoiseCov[2:4, 2:4] = np.eye(2, dtype=np.float32) * (
                self.base_process_noise * process_noise_factor * 2
        )
        self.kalman.processNoiseCov[4:6, 4:6] = np.eye(2, dtype=np.float32) * (
                self.base_process_noise * process_noise_factor * 5
        )

    def _validate_and_constrain_state(self):
        state = self.kalman.statePost
        state[0, 0] = max(-50, min(state[0, 0], self.config.CAMERA_WIDTH + 50))
        state[1, 0] = max(-50, min(state[1, 0], self.config.CAMERA_HEIGHT + 50))
        state[2, 0] = max(-self.max_velocity, min(state[2, 0], self.max_velocity))
        state[3, 0] = max(-self.max_velocity, min(state[3, 0], self.max_velocity))
        state[4, 0] = max(-self.max_acceleration, min(state[4, 0], self.max_acceleration))
        state[5, 0] = max(-self.max_acceleration, min(state[5, 0], self.max_acceleration))
        self.kalman.statePost = state

    def guncelle(self, olcum: Tuple[int, int]) -> Tuple[int, int]:
        try:
            current_time = time.time()
            dt = current_time - self.last_update_time
            dt = max(0.01, min(dt, 0.1))
            self._update_transition_matrix(dt)
            if self.last_position:
                dx = olcum[0] - self.last_position[0]
                dy = olcum[1] - self.last_position[1]
                self.velocity_magnitude = np.sqrt(dx ** 2 + dy ** 2) / dt
                if hasattr(self, 'last_velocity'):
                    dvx = (dx / dt) - self.last_velocity[0]
                    dvy = (dy / dt) - self.last_velocity[1]
                    self.acceleration_magnitude = np.sqrt(dvx ** 2 + dvy ** 2) / dt
                self.last_velocity = (dx / dt, dy / dt)
            self.measurement_history.append(olcum)
            measurement_quality = self._calculate_measurement_quality(olcum)
            self._adaptive_noise_adjustment(measurement_quality)
            if not self.is_initialized:
                self.kalman.statePost[0, 0] = float(olcum[0])
                self.kalman.statePost[1, 0] = float(olcum[1])
                self.is_initialized = True
                self.last_position = olcum
                self.last_update_time = current_time
                return olcum
            prediction = self.kalman.predict()
            measurement = np.array([[np.float32(olcum[0])], [np.float32(olcum[1])]])
            self.kalman.correct(measurement)
            self._validate_and_constrain_state()
            self.guven_skoru = min(1.0, self.guven_skoru + (0.1 * measurement_quality))
            self.last_position = olcum
            self.last_update_time = current_time
            return self._tahmin_olustur(use_smoothing=True)
        except Exception as e:
            logger.log("Kalman filtresi güncelleme hatası", "ERROR", {"error": str(e)})
            return (int(olcum[0]), int(olcum[1]))

    def tahmin_et(self) -> Tuple[int, int]:
        try:
            if not self.is_initialized:
                return (self.config.CAMERA_WIDTH // 2, self.config.CAMERA_HEIGHT // 2)
            current_time = time.time()
            dt = current_time - self.last_update_time
            dt = max(0.01, min(dt, 0.2))
            self._update_transition_matrix(dt)
            prediction = self.kalman.predict()
            self._validate_and_constrain_state()
            confidence_decay = 0.05 * (dt / 0.033)
            self.guven_skoru = max(0.0, self.guven_skoru - confidence_decay)
            return self._tahmin_olustur(use_smoothing=False)
        except Exception as e:
            logger.log("Kalman filtresi tahmin hatası", "ERROR", {"error": str(e)})
            if self.tahmin_gecmisi:
                return self.tahmin_gecmisi[-1]
            return (self.config.CAMERA_WIDTH // 2, self.config.CAMERA_HEIGHT // 2)

    def _tahmin_olustur(self, use_smoothing: bool = True) -> Tuple[int, int]:
        try:
            state = self.kalman.statePost
            predicted_x = int(state[0, 0])
            predicted_y = int(state[1, 0])
            predicted_x = max(0, min(predicted_x, self.config.CAMERA_WIDTH))
            predicted_y = max(0, min(predicted_y, self.config.CAMERA_HEIGHT))
            tahmin_pozisyon = (predicted_x, predicted_y)
            if use_smoothing and len(self.tahmin_gecmisi) > 0:
                alpha = 0.7
                last_prediction = self.tahmin_gecmisi[-1]
                smoothed_x = int(alpha * predicted_x + (1 - alpha) * last_prediction[0])
                smoothed_y = int(alpha * predicted_y + (1 - alpha) * last_prediction[1])
                diff = np.sqrt((smoothed_x - predicted_x) ** 2 + (smoothed_y - predicted_y) ** 2)
                if diff < 30:
                    tahmin_pozisyon = (smoothed_x, smoothed_y)
            self.tahmin_gecmisi.append(tahmin_pozisyon)
            return tahmin_pozisyon
        except Exception as e:
            logger.log("Tahmin oluşturma hatası", "ERROR", {"error": str(e)})
            if self.tahmin_gecmisi:
                return self.tahmin_gecmisi[-1]
            return (self.config.CAMERA_WIDTH // 2, self.config.CAMERA_HEIGHT // 2)

    def get_tracking_stats(self) -> Dict[str, Any]:
        return {
            "confidence": self.guven_skoru,
            "velocity_magnitude": self.velocity_magnitude,
            "acceleration_magnitude": self.acceleration_magnitude,
            "measurement_variance": self.measurement_variance,
            "consecutive_good_measurements": self.consecutive_good_measurements,
            "prediction_history_length": len(self.tahmin_gecmisi),
            "is_initialized": self.is_initialized
        }

    def reset(self):
        self.kalman.statePost = np.array([[0], [0], [0], [0], [0], [0]], dtype=np.float32)
        self.kalman.errorCovPost = np.eye(6, dtype=np.float32) * 1000
        self.tahmin_gecmisi.clear()
        self.measurement_history.clear()
        self.guven_skoru = 0.0
        self.consecutive_good_measurements = 0
        self.is_initialized = False
        self.last_position = None
        self._initialize_matrices()

# GelismisTakipci sınıfı
class GelismisTakipci:
    def __init__(self, config: Config):
        self.config = config
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.prev_gray = None
        self.prev_pts = None

    def update_tracking(self, frame: np.ndarray, target_bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y, w, h = target_bbox
        center = (x + w // 2, y + h // 2)
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_pts = np.array([[center[0], center[1]]], dtype=np.float32)
            return center
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None, **self.lk_params)
        if status[0] == 1:
            predicted_center = tuple(next_pts[0].astype(int))
        else:
            predicted_center = center
        self.prev_gray = gray.copy()
        self.prev_pts = next_pts
        return predicted_center

# BalonTakipSistemi sınıfı
class BalonTakipSistemi:
    def __init__(self, config: Config):
        self.config = config
        self.model = self._model_yukle()
        self.kamera = self._kamera_ayarla()
        self.merkez = (config.CAMERA_WIDTH // 2, config.CAMERA_HEIGHT // 2)
        self.namlu_offset_px = int(config.NAMLU_OFFSET_MM * config.PIXEL_PER_MM)
        self.kf = GelismisKalmanFiltresi(config)
        self.takipci = GelismisTakipci(config)
        self.data_lock = threading.Lock()
        self._latest_data = None
        self.son_pozisyon = None
        self.son_boyut = None
        self.kacirilan_kareler = 0
        self.fps = 0
        self.frame_count = 0
        self.calistiriyor = False
        self.sistem_durumu = SistemDurumu.BASLATILIYOR
        self.hedef_balon = None
        self.bekleme_zamani = 0
        self.start_time = time.time()
        self.last_frame_time = time.time()

    @property
    def latest_data(self):
        with self.data_lock:
            return self._latest_data

    @latest_data.setter
    def latest_data(self, value):
        with self.data_lock:
            self._latest_data = value

    def _model_yukle(self) -> YOLO:
        try:
            logger.log("YOLO modeli yükleniyor", "INFO", {"model_path": self.config.MODEL_PATH})
            model = YOLO(self.config.MODEL_PATH, task='detect')
            logger.log("ONNX modeli yüklendi", "INFO", {
                "model": self.config.MODEL_PATH,
                "cuda_available": torch.cuda.is_available(),
                "device": 'cuda' if self.config.USE_CUDA and torch.cuda.is_available() else 'cpu'
            })
            return model
        except Exception as e:
            logger.log("Model yükleme hatası", "CRITICAL", {"error": str(e)})
            raise SystemExit("Model yüklenemedi, sistem kapatılıyor")

    def _kamera_ayarla(self) -> cv2.VideoCapture:
        try:
            logger.log("Kamera başlatılıyor", "INFO", {
                "camera_id": self.config.CAMERA_ID,
                "resolution": f"{self.config.CAMERA_WIDTH}x{self.config.CAMERA_HEIGHT}"
            })
            kamera = cv2.VideoCapture(self.config.CAMERA_ID)
            if not kamera.isOpened():
                logger.log("Kamera açılamadı", "ERROR")
                raise RuntimeError("Kamera açılamadı! Lütfen CAMERA_ID ayarını ve kamerayı kontrol edin.")
            kamera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.CAMERA_WIDTH)
            kamera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.CAMERA_HEIGHT)
            kamera.set(cv2.CAP_PROP_FPS, self.config.TARGET_FPS)
            kamera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            kamera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            actual_width = int(kamera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(kamera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = kamera.get(cv2.CAP_PROP_FPS)
            logger.log("Kamera başarıyla başlatıldı", "INFO", {
                "actual_resolution": f"{actual_width}x{actual_height}",
                "actual_fps": actual_fps,
                "target_fps": self.config.TARGET_FPS
            })
            return kamera
        except Exception as e:
            logger.log("Kamera ayarlama hatası", "CRITICAL", {"error": str(e)})
            raise SystemExit("Kamera başlatılamadı, sistem kapatılıyor")

    def _gorsellestir(self, cerceve: np.ndarray, durum: SistemDurumu,
                      detections: Optional[List] = None) -> np.ndarray:
        try:
            viz_frame = cerceve.copy()
            cv2.line(viz_frame, (self.merkez[0], 0), (self.merkez[0], self.config.CAMERA_HEIGHT), (0, 255, 0), 1)
            cv2.line(viz_frame, (0, self.merkez[1]), (self.config.CAMERA_WIDTH, self.merkez[1]), (0, 255, 0), 1)
            cv2.circle(viz_frame, self.merkez, 5, (0, 255, 0), -1)
            namlu_pos = (self.merkez[0], self.merkez[1] + self.namlu_offset_px)
            rect_size = 20
            top_left = (namlu_pos[0] - rect_size, namlu_pos[1] - rect_size)
            bottom_right = (namlu_pos[0] + rect_size, namlu_pos[1] + rect_size)
            cv2.rectangle(viz_frame, top_left, bottom_right, (0, 0, 255), 2)
            cv2.circle(viz_frame, namlu_pos, 3, (0, 0, 255), -1)
            cv2.line(viz_frame, self.merkez, namlu_pos, (0, 0, 255), 2)
            if self.latest_data and self.latest_data[2]:
                cv2.putText(viz_frame, "ATES!", (top_left[0], top_left[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            if detections:
                for detection in detections:
                    x1, y1, x2, y2, conf, cls = detection
                    is_friend = cls == 1
                    label = "DUSMAN" if is_friend else "DOST"
                    color = (255, 0, 0) if is_friend else (0, 0, 255)
                    cv2.rectangle(viz_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(viz_frame, f'{label} {conf:.2f}', (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    cv2.circle(viz_frame, center, 3, color, -1)
                    if self.hedef_balon and self._is_same_target(detection, self.hedef_balon):
                        x_offset = center[0] - self.merkez[0]
                        y_offset = center[1] - namlu_pos[1]
                        cv2.line(viz_frame, namlu_pos, center, (255, 255, 0), 2)
                        cv2.putText(viz_frame, f'X: {x_offset}px', (int(x1), int(y2) + 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                        cv2.putText(viz_frame, f'Y: {-(y_offset)}px', (int(x1), int(y2) + 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            if hasattr(self.kf, 'tahmin_gecmisi') and len(self.kf.tahmin_gecmisi) > 0:
                for i in range(1, len(self.kf.tahmin_gecmisi)):
                    cv2.line(viz_frame, self.kf.tahmin_gecmisi[i - 1], self.kf.tahmin_gecmisi[i], (0, 255, 255), 1)
                current_prediction = self.kf.tahmin_gecmisi[-1]
                cv2.circle(viz_frame, current_prediction, 5, (0, 255, 255), 2)
            status_color = self._get_status_color(durum)
            cv2.rectangle(viz_frame, (10, 10), (300, 120), (0, 0, 0), -1)
            cv2.rectangle(viz_frame, (10, 10), (300, 120), status_color, 2)
            texts = [
                f"Durum: {durum.name}",
                f"FPS: {self.fps:.1f}",
            ]
            for i, text in enumerate(texts):
                cv2.putText(viz_frame, text, (15, 30 + i * 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            if hasattr(self.kf, 'get_tracking_stats'):
                stats = self.kf.get_tracking_stats()
                cv2.rectangle(viz_frame, (310, 10), (620, 80), (0, 0, 0), -1)
                cv2.rectangle(viz_frame, (310, 10), (620, 80), (100, 100, 100), 2)
                stat_texts = [
                    f"Hız: {stats['velocity_magnitude']:.1f} px/s",
                    f"İvme: {stats['acceleration_magnitude']:.1f} px/s²",
                    f"Varyans: {stats['measurement_variance']:.2f}",
                ]
                for i, text in enumerate(stat_texts):
                    cv2.putText(viz_frame, text, (315, 30 + i * 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            return viz_frame
        except Exception as e:
            logger.log("Görselleştirme hatası", "ERROR", {"error": str(e)})
            return cerceve

    def _get_status_color(self, durum: SistemDurumu) -> Tuple[int, int, int]:
        color_map = {
            SistemDurumu.BASLATILIYOR: (0, 255, 255),
            SistemDurumu.OTONOM_MOD: (0, 255, 0),
            SistemDurumu.DOST_BALON_ARAMA: (255, 0, 0),
            SistemDurumu.HEDEF_KILITLEME: (255, 0, 0),
            SistemDurumu.BEKLEME: (0, 165, 255),
            SistemDurumu.KAPATILIYOR: (128, 128, 128)
        }
        return color_map.get(durum, (255, 255, 255))

    def _is_same_target(self, detection1, detection2, threshold: float = 50.0) -> bool:
        if detection1 is None or detection2 is None:
            return False
        x1_1, y1_1, x2_1, y2_1 = detection1[:4]
        x1_2, y1_2, x2_2, y2_2 = detection2[:4]
        center1 = ((x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2)
        center2 = ((x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2)
        distance = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
        return distance < threshold

    def _select_best_target(self, detections: List) -> Optional[List]:
        if not detections:
            return None
        friend_detections = [d for d in detections if d[5] == 1]
        if self.hedef_balon:
            best_match = None
            min_distance = float('inf')
            for detection in friend_detections:
                if self._is_same_target(detection, self.hedef_balon):
                    x1, y1, x2, y2 = detection[:4]
                    center = ((x1 + x2) / 2, (y1 + y2) / 2)
                    hx1, hy1, hx2, hy2 = self.hedef_balon[:4]
                    hedef_center = ((hx1 + hx2) / 2, (hy1 + hy2) / 2)
                    distance = np.sqrt((center[0] - hedef_center[0]) ** 2 + (center[1] - hedef_center[1]) ** 2)
                    if distance < min_distance:
                        min_distance = distance
                        best_match = detection
            if best_match:
                return best_match
        if friend_detections:
            best_detection = max(friend_detections, key=lambda x: x[4])
            return best_detection
        return None

    def _calculate_fire_decision(self, x_offset: int, y_offset: int) -> bool:
        distance_from_center = np.sqrt(x_offset ** 2 + y_offset ** 2)
        kalman_confidence = self.kf.guven_skoru
        velocity_ok = self.kf.velocity_magnitude < self.config.VELOCITY_THRESHOLD if hasattr(self.kf, 'velocity_magnitude') else True
        fire_conditions = (
                distance_from_center <= self.config.ATIS_ESIK_PIKSEL and
                kalman_confidence > self.config.MEASUREMENT_QUALITY_THRESHOLD and
                velocity_ok and
                self.kacirilan_kareler < 3
        )
        return fire_conditions

    def process_frame(self, cerceve: np.ndarray) -> np.ndarray:
        try:
            current_time = time.time()
            if current_time - self.last_frame_time > 0:
                self.fps = 1.0 / (current_time - self.last_frame_time)
            self.last_frame_time = current_time
            self.frame_count += 1
            try:
                results = self.model(cerceve, conf=self.config.CONFIDENCE, verbose=False)
                detections = []
                if results and len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = box.cls[0].cpu().numpy()
                        detections.append([x1, y1, x2, y2, conf, cls])
            except Exception as e:
                logger.log("YOLO deteksiyon hatası", "ERROR", {"error": str(e)})
                detections = []
            self._update_system_state(detections)
            x_offset, y_offset, fire_command = self._process_tracking(detections)
            if x_offset is not None and y_offset is not None:
                self.latest_data = (x_offset, y_offset, fire_command)
            viz_frame = self._gorsellestir(cerceve, self.sistem_durumu, detections)
            return viz_frame
        except Exception as e:
            logger.log("Frame işleme hatası", "ERROR", {"error": str(e)})
            return cerceve

    def _update_system_state(self, detections: List) -> None:
        current_time = time.time()
        if self.sistem_durumu == SistemDurumu.BASLATILIYOR:
            if current_time - self.start_time > 2.0:
                self.sistem_durumu = SistemDurumu.OTONOM_MOD
                logger.log("Otonom moda geçildi", "INFO")
        elif self.sistem_durumu == SistemDurumu.OTONOM_MOD:
            friend_detections = [d for d in detections if d[5] == 1]
            if friend_detections:
                self.sistem_durumu = SistemDurumu.DOST_BALON_ARAMA
                logger.log("Dost balon arama moduna geçildi", "INFO")
        elif self.sistem_durumu == SistemDurumu.DOST_BALON_ARAMA:
            if self.hedef_balon is not None:
                self.sistem_durumu = SistemDurumu.HEDEF_KILITLEME
                logger.log("Hedef kilitleme moduna geçildi", "INFO")
            elif not any(d[5] == 1 for d in detections):
                self.sistem_durumu = SistemDurumu.OTONOM_MOD
        elif self.sistem_durumu == SistemDurumu.HEDEF_KILITLEME:
            if self.kacirilan_kareler > self.config.MAX_MISSING_FRAMES:
                self.sistem_durumu = SistemDurumu.BEKLEME
                self.bekleme_zamani = current_time
                logger.log("Bekleme moduna geçildi", "WARNING")
        elif self.sistem_durumu == SistemDurumu.BEKLEME:
            if current_time - self.bekleme_zamani > self.config.BEKLEME_SURESI:
                self.sistem_durumu = SistemDurumu.OTONOM_MOD
                self.hedef_balon = None
                self.kf.reset()
                logger.log("Otonom moda geri dönüldü", "INFO")

    def _process_tracking(self, detections: List) -> Tuple[Optional[int], Optional[int], bool]:
        x_offset = None
        y_offset = None
        fire_command = False
        try:
            if detections:
                if self.hedef_balon is None:
                    self.hedef_balon = self._select_best_target(detections)
                    if self.hedef_balon:
                        logger.log("Yeni dost hedef seçildi", "INFO")
                current_target = None
                for detection in detections:
                    if self._is_same_target(detection, self.hedef_balon):
                        current_target = detection
                        break
                if current_target:
                    self.kacirilan_kareler = 0
                    self.hedef_balon = current_target
                    x1, y1, x2, y2 = current_target[:4]
                    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    predicted_center = self.kf.guncelle(center)
                    namlu_pos = (self.merkez[0], self.merkez[1] + self.namlu_offset_px)
                    x_offset = predicted_center[0] - self.merkez[0]
                    y_offset = -(predicted_center[1] - namlu_pos[1])
                    fire_command = self._calculate_fire_decision(x_offset, y_offset)
                    if fire_command:
                        logger.log("ATEŞ KOMUTU VERİLDİ", "WARNING", {
                            "x_offset": x_offset,
                            "y_offset": y_offset,
                        })
                else:
                    self.kacirilan_kareler += 1
                    if self.kacirilan_kareler <= self.config.MAX_MISSING_FRAMES:
                        predicted_center = self.kf.tahmin_et()
                        namlu_pos = (self.merkez[0], self.merkez[1] + self.namlu_offset_px)
                        x_offset = predicted_center[0] - self.merkez[0]
                        y_offset = predicted_center[1] - namlu_pos[1]
                        logger.log("Hedef kaybedildi, Kalman tahmini kullanılıyor", "WARNING", {
                            "missing_frames": self.kacirilan_kareler,
                            "predicted_center": predicted_center
                        })
            else:
                self.kacirilan_kareler += 1
                if self.hedef_balon and self.kacirilan_kareler <= self.config.MAX_MISSING_FRAMES:
                    predicted_center = self.kf.tahmin_et()
                    namlu_pos = (self.merkez[0], self.merkez[1] + self.namlu_offset_px)
                    x_offset = predicted_center[0] - self.merkez[0]
                    y_offset = predicted_center[1] - namlu_pos[1]
        except Exception as e:
            logger.log("Takip işleme hatası", "ERROR", {"error": str(e)})
        return x_offset, y_offset, fire_command

    def _cleanup(self) -> None:
        try:
            logger.log("Sistem kapatılıyor", "INFO")
            self.calistiriyor = False
            if hasattr(self, 'kamera') and self.kamera:
                self.kamera.release()
                logger.log("Kamera kapatıldı", "INFO")
        except Exception as e:
            logger.log("Temizlik hatası", "ERROR", {"error": str(e)})

# GokboruArayuz sınıfı
class GokboruArayuz(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GÖKBÖRÜ Hava Savunma Sistemi")
        self.setGeometry(100, 100, 1200, 800)
        self.config = Config()
        global logger
        logger = EnhancedLogger(self.config)
        self.balon_sistemi = BalonTakipSistemi(self.config)
        self.balon_sistemi.calistiriyor = True
        self.hedef_kilitli = False
        self.udp_receiver = UdpReceiver(listen_port=8888, target_ip='10.0.0.131')
        self.udp_receiver.imu_data_received.connect(self.guncelle_imu_verisi)
        self.udp_receiver.start()
        self.latest_imu_data = {}  # Store latest IMU data
        self.setup_ui()
        self.start_time_update()
        self.console_stream = Stream(parent=self)
        self.console_stream.newText.connect(self.append_to_log)
        sys.stdout = self.console_stream
        sys.stderr = self.console_stream

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        header_layout = QHBoxLayout()
        title = QLabel("🛡️ GÖKBÖRÜ HAVA SAVUNMA SİSTEMLERİ")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.time_label = QLabel()
        self.time_label.setFont(QFont("Arial", 12))
        self.time_label.setAlignment(Qt.AlignRight)
        header_layout.addWidget(title)
        header_layout.addWidget(self.time_label)
        main_layout.addLayout(header_layout)
        content_layout = QHBoxLayout()
        content_layout.setSpacing(10)
        main_layout.addLayout(content_layout, 1)
        left_panel = QVBoxLayout()
        left_panel.setSpacing(10)
        camera_group = QGroupBox("📹 Kamera Görüntüsü")
        camera_layout = QVBoxLayout(camera_group)
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setStyleSheet("background-color: black; border: 2px solid #444;")
        self.video_label.setAlignment(Qt.AlignCenter)
        camera_layout.addWidget(self.video_label)
        camera_status_layout = QHBoxLayout()
        self.kamera_durum = QLabel("🔴 Kamera: Bağlantı Bekleniyor...")
        self.fps_label = QLabel("FPS: --")
        camera_status_layout.addWidget(self.kamera_durum)
        camera_status_layout.addWidget(self.fps_label)
        camera_layout.addLayout(camera_status_layout)
        left_panel.addWidget(camera_group)
        content_layout.addLayout(left_panel, 2)
        right_panel = QVBoxLayout()
        right_panel.setSpacing(10)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(10)
        scroll_layout.setContentsMargins(5, 5, 5, 5)
        telemetri_group = QGroupBox("📡 Telemetri")
        telemetri_layout = QVBoxLayout(telemetri_group)
        angles_group = QGroupBox("🎯 Açı Bilgileri")
        angles_layout = QGridLayout(angles_group)
        self.yaw_label = QLabel("Yatay Açı: ---°")
        self.pitch_label = QLabel("Dikey Açı: ---°")
        self.roll_label = QLabel("Roll: ---°")
        angles_layout.addWidget(self.yaw_label, 0, 0)
        angles_layout.addWidget(self.pitch_label, 0, 1)
        angles_layout.addWidget(self.roll_label, 1, 0, 1, 2)
        telemetri_layout.addWidget(angles_group)
        sensor_group = QGroupBox("📊 Sensör Verileri")
        sensor_layout = QGridLayout(sensor_group)
        self.accel_x_label = QLabel("İvme X: ---")
        self.accel_y_label = QLabel("İvme Y: ---")
        self.accel_z_label = QLabel("İvme Z: ---")
        self.gyro_x_label = QLabel("Jiroskop X: ---")
        self.gyro_y_label = QLabel("Jiroskop Y: ---")
        self.gyro_z_label = QLabel("Jiroskop Z: ---")
        self.temp_label = QLabel("Sıcaklık: ---°C")
        sensor_layout.addWidget(self.accel_x_label, 0, 0)
        sensor_layout.addWidget(self.accel_y_label, 0, 1)
        sensor_layout.addWidget(self.accel_z_label, 1, 0)
        sensor_layout.addWidget(self.gyro_x_label, 1, 1)
        sensor_layout.addWidget(self.gyro_y_label, 2, 0)
        sensor_layout.addWidget(self.gyro_z_label, 2, 1)
        sensor_layout.addWidget(self.temp_label, 3, 0, 1, 2)
        telemetri_layout.addWidget(sensor_group)
        scroll_layout.addWidget(telemetri_group)
        tracking_group = QGroupBox("📈 Takip İstatistikleri")
        tracking_layout = QGridLayout(tracking_group)
        self.confidence_label = QLabel("Güven Skoru: ---")
        self.velocity_label = QLabel("Hız: --- px/s")
        self.acceleration_label = QLabel("İvme: --- px/s²")
        self.variance_label = QLabel("Varyans: ---")
        tracking_layout.addWidget(self.confidence_label, 0, 0)
        tracking_layout.addWidget(self.velocity_label, 0, 1)
        tracking_layout.addWidget(self.acceleration_label, 1, 0)
        tracking_layout.addWidget(self.variance_label, 1, 1)
        scroll_layout.addWidget(tracking_group)
        gorev_group = QGroupBox("🎯 Sistem Durumu")
        gorev_layout = QVBoxLayout(gorev_group)
        self.gorev_labels = []
        gorev_stages = ["AŞAMA 1: Hareketli Hedef", "AŞAMA 2: Dost/Düşman", "AŞAMA 3: Angajman"]
        for stage in gorev_stages:
            stage_layout = QHBoxLayout()
            stage_label = QLabel(stage)
            stage_status = QLabel("BEKLİYOR")
            stage_status.setStyleSheet("color: yellow; font-weight: bold;")
            self.gorev_labels.append(stage_status)
            stage_layout.addWidget(stage_label)
            stage_layout.addWidget(stage_status)
            gorev_layout.addLayout(stage_layout)
        scroll_layout.addWidget(gorev_group)
        gorev_komut_group = QGroupBox("📋 Görev Komutları")
        gorev_komut_layout = QVBoxLayout(gorev_komut_group)
        gorev_buton_layout = QGridLayout()
        self.btn_close_task = QPushButton("SİSTEMİ DURDUR")
        self.btn_close_task.clicked.connect(self.close_task)
        self.btn_close_task.setEnabled(True)
        gorev_buton_layout.addWidget(self.btn_close_task, 0, 0, 1, 2)
        self.btn_toggle_mode = QPushButton("Mod: Sensör")
        self.btn_toggle_mode.clicked.connect(self.toggle_mode)
        gorev_buton_layout.addWidget(self.btn_toggle_mode, 1, 0, 1, 2)
        gorev_komut_layout.addLayout(gorev_buton_layout)
        self.offset_label = QLabel("Offset: X=---, Y=---, Ateş: ---")
        self.offset_label.setAlignment(Qt.AlignCenter)
        gorev_komut_layout.addWidget(self.offset_label)
        mode_group = QGroupBox("🔄 Kontrol Modu")
        mode_layout = QHBoxLayout(mode_group)
        self.mode_auto = QRadioButton("Otonom")
        self.mode_manual = QRadioButton("Manuel")
        self.mode_auto.setChecked(True)
        self.mode_auto.toggled.connect(self.set_control_mode)
        mode_layout.addWidget(self.mode_auto)
        mode_layout.addWidget(self.mode_manual)
        gorev_komut_layout.addWidget(mode_group)
        self.manual_control_group = QGroupBox("🎮 Manuel Kontrol")
        self.manual_control_layout = QGridLayout(self.manual_control_group)
        self.btn_up = QPushButton("YUKARI (W)")
        self.btn_down = QPushButton("AŞAĞI (S)")
        self.btn_left = QPushButton("SOL (A)")
        self.btn_right = QPushButton("SAĞ (D)")
        self.btn_fire = QPushButton("ATEŞ (F)")  # New: Fire button
        self.btn_up.clicked.connect(lambda: self.send_control_command("UP"))
        self.btn_down.clicked.connect(lambda: self.send_control_command("DOWN"))
        self.btn_left.clicked.connect(lambda: self.send_control_command("LEFT"))
        self.btn_right.clicked.connect(lambda: self.send_control_command("RIGHT"))
        self.btn_fire.clicked.connect(lambda: self.send_control_command("FIRE"))
        self.manual_control_layout.addWidget(self.btn_up, 0, 1)
        self.manual_control_layout.addWidget(self.btn_left, 1, 0)
        self.manual_control_layout.addWidget(self.btn_right, 1, 2)
        self.manual_control_layout.addWidget(self.btn_down, 2, 1)
        self.manual_control_layout.addWidget(self.btn_fire, 3, 1)
        self.manual_control_group.setVisible(False)
        gorev_komut_layout.addWidget(self.manual_control_group)
        scroll_layout.addWidget(gorev_komut_group)
        silah_group = QGroupBox("🚀 Silah Kontrol")
        silah_layout = QVBoxLayout(silah_group)
        durum_layout = QHBoxLayout()
        self.hazir_indicator = QLabel("● HAZIR")
        self.hedef_indicator = QLabel("● HEDEF KİLİTLİ DEĞİL")
        self.hedef_indicator.setStyleSheet("color: gray; font-weight: bold;")
        durum_layout.addWidget(self.hazir_indicator)
        durum_layout.addWidget(self.hedef_indicator)
        silah_layout.addLayout(durum_layout)
        kontrol_layout = QHBoxLayout()
        self.btn_kilitle = QPushButton("HEDEF KİLİTLE")
        self.btn_atesle = QPushButton("ATEŞLE")
        self.btn_atesle.setEnabled(False)
        self.btn_kilitle.clicked.connect(self.hedef_kilitle)
        self.btn_atesle.clicked.connect(self.atesle)
        kontrol_layout.addWidget(self.btn_kilitle)
        kontrol_layout.addWidget(self.btn_atesle)
        silah_layout.addLayout(kontrol_layout)
        muhimmat_layout = QHBoxLayout()
        muhimmat_layout.addWidget(QLabel("Mühimmat:"))
        self.muhimmat_bar = QProgressBar()
        self.muhimmat_bar.setMaximum(100)
        self.muhimmat_bar.setValue(85)
        muhimmat_layout.addWidget(self.muhimmat_bar)
        silah_layout.addLayout(muhimmat_layout)
        scroll_layout.addWidget(silah_group)
        log_group = QGroupBox("📋 Sistem Kayıtları")
        log_layout = QVBoxLayout(log_group)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(150)
        self.log.append(f"[{datetime.now().strftime('%H:%M:%S')}] Sistem başlatıldı")
        log_layout.addWidget(self.log)
        scroll_layout.addWidget(log_group)
        scroll.setWidget(scroll_content)
        right_panel.addWidget(scroll)
        content_layout.addLayout(right_panel, 1)
        self.apply_styles()
        self.set_control_mode()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)

    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a1a;
                color: white;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #444;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
                background-color: #2a2a2a;
                color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: white;
            }
            QPushButton {
                background-color: #3a3a3a;
                border: 2px solid #555;
                border-radius: 5px;
                padding: 8px;
                font-weight: bold;
                min-width: 80px;
                color: white;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
            }
            QPushButton:pressed {
                background-color: #555;
            }
            QPushButton:disabled {
                background-color: #2a2a2a;
                border: 2px solid #333;
                color: #555;
            }
            QLabel {
                color: white;
            }
            QTextEdit {
                background-color: #2a2a2a;
                border: 1px solid #444;
                color: white;
                font-family: "Arial", monospace;
            }
            QProgressBar {
                border: 1px solid #444;
                border-radius: 3px;
                text-align: center;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
            QRadioButton {
                color: white;
                padding: 5px;
            }
        """)
        self.btn_close_task.setStyleSheet("background-color: #8B0000; color: white;")
        self.btn_toggle_mode.setStyleSheet("background-color: #4B0082; color: white;")
        self.btn_atesle.setStyleSheet("background-color: #8B0000; color: white;")
        self.btn_fire.setStyleSheet("background-color: #8B0000; color: white;")  # Style for fire button
        self.hazir_indicator.setStyleSheet("color: green; font-weight: bold;")
        self.hedef_indicator.setStyleSheet("color: gray; font-weight: bold;")
        self.kamera_durum.setStyleSheet("color: red; font-weight: bold;")
        self.fps_label.setStyleSheet("color: yellow; font-weight: bold;")
        self.offset_label.setStyleSheet("color: #00FF00; font-weight: bold;")

    def append_to_log(self, text):
        if not text.startswith('['):
            self.log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {text.strip()}")
        else:
            self.log.append(text.strip())
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def start_time_update(self):
        self.time_timer = QTimer()
        self.time_timer.timeout.connect(self.update_time)
        self.time_timer.start(1000)
        self.update_time()

    def update_time(self):
        now = datetime.now()
        time_str = now.strftime("%H:%M:%S")
        date_str = now.strftime("%d/%m/%Y")
        self.time_label.setText(f"🕒 {time_str}  📅 {date_str}")

    def update_frame(self):
        ret, frame = self.balon_sistemi.kamera.read()
        if ret:
            self.kamera_durum.setText("🟢 Kamera: Aktif")
            self.kamera_durum.setStyleSheet("color: green; font-weight: bold;")
            frame = cv2.resize(frame, (self.config.CAMERA_WIDTH, self.config.CAMERA_HEIGHT))
            viz_frame = self.balon_sistemi.process_frame(frame)
            if self.balon_sistemi.latest_data:
                x_offset, y_offset, fire_command = self.balon_sistemi.latest_data
                self.offset_label.setText(f"Offset: X={x_offset:.1f}, Y={y_offset:.1f}, Ateş: {'Evet' if fire_command else 'Hayır'}")
                if hasattr(self, 'control_mode') and self.control_mode == "AUTONOMOUS":
                    self.udp_receiver.send_control_command(
                        command="MOVE",
                        control_mode="AUTONOMOUS",
                        x_offset=x_offset,
                        y_offset=y_offset
                    )
            else:
                self.offset_label.setText("Offset: X=---, Y=---, Ateş: ---")
            stats = self.balon_sistemi.kf.get_tracking_stats()
            self.confidence_label.setText(f"Güven Skoru: {stats['confidence']:.2f}")
            self.velocity_label.setText(f"Hız: {stats['velocity_magnitude']:.1f} px/s")
            self.acceleration_label.setText(f"İvme: {stats['acceleration_magnitude']:.1f} px/s²")
            self.variance_label.setText(f"Varyans: {stats['measurement_variance']:.2f}")
            if self.balon_sistemi.sistem_durumu == SistemDurumu.BASLATILIYOR:
                self.gorev_labels[0].setText("AKTİF")
                self.gorev_labels[0].setStyleSheet("color: cyan; font-weight: bold;")
                self.gorev_labels[1].setText("BEKLİYOR")
                self.gorev_labels[1].setStyleSheet("color: yellow; font-weight: bold;")
                self.gorev_labels[2].setText("BEKLİYOR")
                self.gorev_labels[2].setStyleSheet("color: yellow; font-weight: bold;")
            elif self.balon_sistemi.sistem_durumu == SistemDurumu.OTONOM_MOD:
                self.gorev_labels[0].setText("TAMAMLANDI")
                self.gorev_labels[0].setStyleSheet("color: lightblue; font-weight: bold;")
                self.gorev_labels[1].setText("BEKLİYOR")
                self.gorev_labels[1].setStyleSheet("color: yellow; font-weight: bold;")
                self.gorev_labels[2].setText("BEKLİYOR")
                self.gorev_labels[2].setStyleSheet("color: yellow; font-weight: bold;")
            elif self.balon_sistemi.sistem_durumu == SistemDurumu.DOST_BALON_ARAMA:
                self.gorev_labels[0].setText("TAMAMLANDI")
                self.gorev_labels[0].setStyleSheet("color: lightblue; font-weight: bold;")
                self.gorev_labels[1].setText("AKTİF")
                self.gorev_labels[1].setStyleSheet("color: green; font-weight: bold;")
                self.gorev_labels[2].setText("BEKLİYOR")
                self.gorev_labels[2].setStyleSheet("color: yellow; font-weight: bold;")
            elif self.balon_sistemi.sistem_durumu == SistemDurumu.HEDEF_KILITLEME:
                self.gorev_labels[0].setText("TAMAMLANDI")
                self.gorev_labels[0].setStyleSheet("color: lightblue; font-weight: bold;")
                self.gorev_labels[1].setText("TAMAMLANDI")
                self.gorev_labels[1].setStyleSheet("color: lightblue; font-weight: bold;")
                self.gorev_labels[2].setText("AKTİF")
                self.gorev_labels[2].setStyleSheet("color: red; font-weight: bold;")
            elif self.balon_sistemi.sistem_durumu == SistemDurumu.BEKLEME:
                self.gorev_labels[0].setText("TAMAMLANDI")
                self.gorev_labels[0].setStyleSheet("color: lightblue; font-weight: bold;")
                self.gorev_labels[1].setText("BEKLİYOR")
                self.gorev_labels[1].setStyleSheet("color: yellow; font-weight: bold;")
                self.gorev_labels[2].setText("BEKLİYOR")
                self.gorev_labels[2].setStyleSheet("color: yellow; font-weight: bold;")
            self.fps_label.setText(f"FPS: {self.balon_sistemi.fps:.1f}")
            rgb = cv2.cvtColor(viz_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(img))
        else:
            self.kamera_durum.setText("🔴 Kamera: Bağlantı Yok")
            self.kamera_durum.setStyleSheet("color: red; font-weight: bold;")
            self.fps_label.setText("FPS: --")
            self.offset_label.setText("Offset: X=---, Y=---, Ateş: ---")

    def hedef_kilitle(self):
        if not self.hedef_kilitli:
            self.hedef_kilitli = True
            self.btn_kilitle.setText("HEDEFİ SERBEST BIRAK")
            self.btn_atesle.setEnabled(True)
            self.hedef_indicator.setText("● HEDEF KİLİTLİ")
            self.hedef_indicator.setStyleSheet("color: red; font-weight: bold;")
            self.append_to_log("Hedef kilitlendi")
        else:
            self.hedef_kilitli = False
            self.btn_kilitle.setText("HEDEF KİLİTLE")
            self.btn_atesle.setEnabled(False)
            self.hedef_indicator.setText("● HEDEF KİLİTLİ DEĞİL")
            self.hedef_indicator.setStyleSheet("color: gray; font-weight: bold;")
            self.append_to_log("Hedef serbest bırakıldı")

    def atesle(self):
        if self.hedef_kilitli:
            self.append_to_log("ATEŞ KOMUTU VERİLDİ!")
            current_ammo = self.muhimmat_bar.value()
            if current_ammo > 0:
                self.muhimmat_bar.setValue(current_ammo - 5)
            self.udp_receiver.send_control_command("FIRE", control_mode=self.control_mode)

    def guncelle_imu_verisi(self, imu_dict):
        try:
            imu = imu_dict.get('imu', {})
            self.latest_imu_data = imu  # Store latest IMU data
            if 'pitch' in imu:
                self.pitch_label.setText(f"Dikey Açı: {imu['pitch']:.1f}°")
            if 'roll' in imu:
                self.roll_label.setText(f"Roll: {imu['roll']:.1f}°")
            if 'yaw' in imu:
                self.yaw_label.setText(f"Yatay Açı: {imu['yaw']:.1f}°")
            if 'accel_x' in imu:
                self.accel_x_label.setText(f"İvme X: {imu['accel_x']:.2f}")
            if 'accel_y' in imu:
                self.accel_y_label.setText(f"İvme Y: {imu['accel_y']:.2f}")
            if 'accel_z' in imu:
                self.accel_z_label.setText(f"İvme Z: {imu['accel_z']:.2f}")
            if 'gyro_x' in imu:
                self.gyro_x_label.setText(f"Jiroskop X: {imu['gyro_x']:.2f}")
            if 'gyro_y' in imu:
                self.gyro_y_label.setText(f"Jiroskop Y: {imu['gyro_y']:.2f}")
            if 'gyro_z' in imu:
                self.gyro_z_label.setText(f"Jiroskop Z: {imu['gyro_z']:.2f}")
            if 'temp' in imu:
                self.temp_label.setText(f"Sıcaklık: {imu['temp']:.1f}°C")
        except Exception as e:
            self.append_to_log(f"IMU veri işleme hatası: {str(e)}")

    def toggle_mode(self):
        self.udp_receiver.fixed_mode = not self.udp_receiver.fixed_mode
        self.btn_toggle_mode.setText(f"Mod: {'Sabit' if self.udp_receiver.fixed_mode else 'Sensör'}")
        self.udp_receiver.send_control_command("TOGGLE_MODE")
        self.append_to_log(f"Kontrol modu: {'Sabit' if self.udp_receiver.fixed_mode else 'Sensör'}")

    def close_task(self):
        self.balon_sistemi._cleanup()
        self.timer.stop()
        self.append_to_log("Sistem durduruldu")
        self.btn_close_task.setEnabled(False)

    def set_control_mode(self):
        if self.mode_auto.isChecked():
            self.control_mode = "AUTONOMOUS"
            self.manual_control_group.setVisible(False)
            self.append_to_log("Kontrol modu: Otonom")
            self.udp_receiver.send_control_command("MODE_AUTONOMOUS")
        else:
            self.control_mode = "MANUAL"
            self.manual_control_group.setVisible(True)
            self.append_to_log("Kontrol modu: Manuel")
            self.udp_receiver.send_control_command("MODE_MANUAL")

    def send_control_command(self, command):
        if hasattr(self, 'control_mode') and self.control_mode == "MANUAL":
            yaw = self.latest_imu_data.get('yaw', None)
            pitch = self.latest_imu_data.get('pitch', None)
            self.udp_receiver.send_control_command(command, control_mode="MANUAL", yaw=yaw, pitch=pitch)
            self.append_to_log(f"Komut: {command}")

    def keyPressEvent(self, event):
        if hasattr(self, 'control_mode') and self.control_mode == "MANUAL":
            yaw = self.latest_imu_data.get('yaw', None)
            pitch = self.latest_imu_data.get('pitch', None)
            if event.key() == Qt.Key_W:
                self.send_control_command("UP")
            elif event.key() == Qt.Key_S:
                self.send_control_command("DOWN")
            elif event.key() == Qt.Key_A:
                self.send_control_command("LEFT")
            elif event.key() == Qt.Key_D:
                self.send_control_command("RIGHT")
            elif event.key() == Qt.Key_F:
                self.send_control_command("FIRE")
        super().keyPressEvent(event)

    def closeEvent(self, event):
        self.append_to_log("Sistem kapatılıyor...")
        self.balon_sistemi._cleanup()
        if hasattr(self, 'udp_receiver'):
            self.udp_receiver.stop()
        if hasattr(self, 'timer'):
            self.timer.stop()
        if hasattr(self, 'time_timer'):
            self.time_timer.stop()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        event.accept()

def main():
    app = QApplication(sys.argv)
    pencere = GokboruArayuz()
    pencere.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()