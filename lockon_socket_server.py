#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Raspberry Pi USB Kamera Stream Sunucusu
USB kamera görüntülerini TCP üzerinden client'lara gönderir
UAV tespit sistemi ile uyumlu - /dev/video0 USB kamera
"""

import socket
import cv2
import pickle
import struct
import threading
import time
import datetime
import sys
import signal
import argparse
import os

def log_message(message):
    """Zaman damgalı log mesajı yazdır"""
    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{current_time}] [RASPI_SERVER] {message}")

class USBCameraStreamer:
    """USB Kamera stream yöneticisi"""
    
    def __init__(self, camera_device="/dev/video0", resolution=(640, 480), fps=30, quality=85):
        self.camera_device = camera_device
        self.resolution = resolution
        self.fps = fps
        self.quality = quality
        self.cap = None
        
        # Stream durumu
        self.streaming = False
        self.frame_count = 0
        self.start_time = None
        self.last_frame_time = 0
        
        # Kamera başlatma
        self.init_camera()
    
    def check_camera_device(self):
        """USB kamera cihazını kontrol et"""
        if os.path.exists(self.camera_device):
            log_message(f"USB kamera cihazı bulundu: {self.camera_device}")
            return True
        else:
            log_message(f"HATA: USB kamera cihazı bulunamadı: {self.camera_device}")
            
            # Alternatif video cihazlarını ara
            for i in range(5):
                alt_device = f"/dev/video{i}"
                if os.path.exists(alt_device):
                    log_message(f"Alternatif cihaz bulundu: {alt_device}")
            
            return False
    
    def init_camera(self):
        """USB kamera sistemini başlat"""
        log_message("USB kamera sistemi başlatılıyor...")
        
        # Cihaz kontrolü
        if not self.check_camera_device():
            return False
        
        try:
            # USB kamera için OpenCV VideoCapture
            # /dev/video0 -> 0 olarak kullan
            camera_index = 0
            if self.camera_device != "/dev/video0":
                # Eğer farklı video device ise index'i çıkar
                camera_index = int(self.camera_device.split("video")[1])
            
            self.cap = cv2.VideoCapture(camera_index)
            
            if not self.cap.isOpened():
                raise Exception(f"USB kamera açılamadı: {self.camera_device}")
            
            # Kamera ayarlarını uygula
            log_message("Kamera ayarları uygulanıyor...")
            
            # Çözünürlük ayarları
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            
            # FPS ayarı
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Buffer boyutunu küçült (düşük gecikme için)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Otomatik odaklamayı kapatmaya çalış
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            
            # Pozlama ayarı (eğer destekleniyorsa)
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Manuel pozlama
            
            # Gerçek ayarları kontrol et
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            log_message(f"USB Kamera başarıyla başlatıldı:")
            log_message(f"  Cihaz: {self.camera_device}")
            log_message(f"  Çözünürlük: {actual_width}x{actual_height}")
            log_message(f"  FPS: {actual_fps}")
            log_message(f"  JPEG Kalite: {self.quality}")
            
            # Test frame al
            ret, test_frame = self.cap.read()
            if ret:
                log_message(f"Test frame başarılı: {test_frame.shape}")
            else:
                log_message("UYARI: Test frame alınamadı")
            
            return True
            
        except Exception as e:
            log_message(f"USB kamera başlatma hatası: {e}")
            return False
    
    def capture_frame(self):
        """USB kameradan frame yakala"""
        if not self.cap or not self.cap.isOpened():
            log_message("HATA: Kamera bağlantısı yok")
            return False, None
        
        try:
            # Frame yakala
            ret, frame = self.cap.read()
            
            if not ret or frame is None:
                log_message("UYARI: Frame yakalanamadı")
                return False, None
            
            # Frame kalitesi kontrolü
            if frame.size == 0:
                log_message("UYARI: Boş frame")
                return False, None
            
            return True, frame
            
        except Exception as e:
            log_message(f"Frame yakalama hatası: {e}")
            return False, None
    
    def add_overlay_info(self, frame):
        """Frame'e overlay bilgileri ekle"""
        if frame is None:
            return frame
        
        try:
            # Zaman damgası
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Frame sayısı
            cv2.putText(frame, f"Frame: {self.frame_count}", (10, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # FPS hesapla
            if self.start_time:
                elapsed = time.time() - self.start_time
                fps = self.frame_count / elapsed if elapsed > 0 else 0
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # USB kamera bilgisi
            cv2.putText(frame, f"USB Cam: {self.camera_device}", (10, 105), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Çözünürlük bilgisi
            height, width = frame.shape[:2]
            cv2.putText(frame, f"Resolution: {width}x{height}", (10, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        except Exception as e:
            log_message(f"Overlay ekleme hatası: {e}")
        
        return frame
    
    def encode_frame(self, frame):
        """Frame'i JPEG olarak encode et"""
        try:
            # JPEG formatında encode et
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.quality]
            success, buffer = cv2.imencode('.jpg', frame, encode_params)
            
            if not success:
                log_message("JPEG encode başarısız")
                return None
                
            return pickle.dumps(buffer)
            
        except Exception as e:
            log_message(f"Frame encode hatası: {e}")
            return None
    
    def start_streaming(self):
        """Streaming başlat"""
        self.streaming = True
        self.frame_count = 0
        self.start_time = time.time()
        log_message("USB kamera streaming başlatıldı")
    
    def stop_streaming(self):
        """Streaming durdur"""
        self.streaming = False
        log_message("USB kamera streaming durduruldu")
    
    def cleanup(self):
        """Temizlik işlemleri"""
        log_message("USB kamera kaynakları temizleniyor...")
        
        if self.cap:
            try:
                self.cap.release()
                log_message("USB kamera serbest bırakıldı")
            except Exception as e:
                log_message(f"USB kamera kapatma hatası: {e}")


class StreamServer:
    """TCP Stream sunucusu"""
    
    def __init__(self, host='0.0.0.0', port=8, camera_device="/dev/video0", 
                 resolution=(640, 480), fps=30, quality=85):
        self.host = host
        self.port = port
        self.server_socket = None
        self.clients = []
        self.running = False
        
        # Kamera streamer
        self.camera = USBCameraStreamer(camera_device, resolution, fps, quality)
        
        # Threading için
        self.server_thread = None
        self.streaming_thread = None
        
        # İstatistikler
        self.total_frames_sent = 0
        self.total_clients_served = 0
        
    def start_server(self):
        """Sunucuyu başlat"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            
            self.running = True
            log_message(f"Sunucu başlatıldı: {self.host}:{self.port}")
            log_message("Client bağlantıları bekleniyor...")
            
            # Streaming thread'ini başlat
            self.camera.start_streaming()
            self.streaming_thread = threading.Thread(target=self.streaming_loop, daemon=True)
            self.streaming_thread.start()
            
            # Client kabul etme döngüsü
            while self.running:
                try:
                    client_socket, client_address = self.server_socket.accept()
                    log_message(f"Yeni client bağlandı: {client_address}")
                    
                    # Client thread'ini başlat
                    client_thread = threading.Thread(
                        target=self.handle_client, 
                        args=(client_socket, client_address),
                        daemon=True
                    )
                    client_thread.start()
                    
                    self.clients.append({
                        'socket': client_socket,
                        'address': client_address,
                        'thread': client_thread,
                        'connected_time': time.time()
                    })
                    
                    self.total_clients_served += 1
                    
                except socket.error as e:
                    if self.running:
                        log_message(f"Socket hatası: {e}")
                    break
                except Exception as e:
                    log_message(f"Client kabul etme hatası: {e}")
                    
        except Exception as e:
            log_message(f"Sunucu başlatma hatası: {e}")
        finally:
            self.cleanup()
    
    def streaming_loop(self):
        """Ana streaming döngüsü"""
        log_message("Streaming döngüsü başlatıldı")
        frame_interval = 1.0 / self.camera.fps  # Frame aralığı
        
        while self.running and self.camera.streaming:
            start_time = time.time()
            
            # Frame yakala
            ret, frame = self.camera.capture_frame()
            
            if ret and frame is not None:
                # Overlay bilgileri ekle
                frame_with_overlay = self.camera.add_overlay_info(frame)
                
                # Frame'i encode et
                encoded_frame = self.camera.encode_frame(frame_with_overlay)
                
                if encoded_frame:
                    # Frame'i tüm client'lara gönder
                    self.broadcast_frame(encoded_frame)
                    self.camera.frame_count += 1
                    self.total_frames_sent += 1
                
            else:
                log_message("Frame yakalanamadı, 0.1 saniye bekleniyor...")
                time.sleep(0.1)
                continue
            
            # FPS kontrolü
            elapsed = time.time() - start_time
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def broadcast_frame(self, encoded_frame):
        """Frame'i tüm bağlı client'lara gönder"""
        if not encoded_frame:
            return
        
        # Frame boyutunu hesapla
        frame_size = len(encoded_frame)
        size_data = struct.pack("L", frame_size)
        
        # Kopya client listesi (thread safety için)
        clients_copy = self.clients[:]
        
        for client_info in clients_copy:
            try:
                client_socket = client_info['socket']
                
                # Önce boyutu gönder
                client_socket.sendall(size_data)
                # Sonra frame verisini gönder
                client_socket.sendall(encoded_frame)
                
            except (socket.error, BrokenPipeError) as e:
                log_message(f"Client {client_info['address']} bağlantısı koptu: {e}")
                self.remove_client(client_info)
            except Exception as e:
                log_message(f"Frame gönderim hatası {client_info['address']}: {e}")
                self.remove_client(client_info)
    
    def handle_client(self, client_socket, client_address):
        """Client bağlantısını yönet"""
        log_message(f"Client handler başlatıldı: {client_address}")
        
        try:
            while self.running:
                # Client'tan veri bekle (keep-alive için)
                client_socket.settimeout(30)  # 30 saniye timeout
                
                # Basit keep-alive mekanizması
                time.sleep(5)
                
        except socket.timeout:
            log_message(f"Client timeout: {client_address}")
        except (socket.error, ConnectionResetError) as e:
            log_message(f"Client bağlantı hatası {client_address}: {e}")
        except Exception as e:
            log_message(f"Client handler hatası {client_address}: {e}")
        finally:
            try:
                client_socket.close()
            except:
                pass
    
    def remove_client(self, client_info):
        """Client'ı listeden kaldır"""
        try:
            if client_info in self.clients:
                self.clients.remove(client_info)
                log_message(f"Client kaldırıldı: {client_info['address']}")
            
            client_info['socket'].close()
        except Exception as e:
            log_message(f"Client kaldırma hatası: {e}")
    
    def stop_server(self):
        """Sunucuyu durdur"""
        log_message("Sunucu durduruluyor...")
        self.running = False
        
        # Tüm client'ları kapat
        for client_info in self.clients[:]:
            self.remove_client(client_info)
        
        # Kamera streaming'i durdur
        self.camera.stop_streaming()
        
        # Server socket'i kapat
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
    
    def cleanup(self):
        """Temizlik işlemleri"""
        log_message("Sunucu temizlik işlemleri...")
        
        self.stop_server()
        self.camera.cleanup()
        
        log_message(f"Sunucu istatistikleri:")
        log_message(f"  Toplam gönderilen frame: {self.total_frames_sent}")
        log_message(f"  Toplam hizmet verilen client: {self.total_clients_served}")
        log_message("Sunucu temizlendi.")


def signal_handler(sig, frame):
    """Ctrl+C signal handler"""
    log_message("Kapatma sinyali alındı (Ctrl+C)")
    global server
    if server:
        server.stop_server()
    sys.exit(0)


def main():
    """Ana fonksiyon"""
    parser = argparse.ArgumentParser(description="Raspberry Pi USB Kamera Stream Sunucusu")
    parser.add_argument("--host", default="0.0.0.0", help="Sunucu IP adresi (varsayılan: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=12345, help="Sunucu portu (varsayılan: 12345)")
    parser.add_argument("--device", default="/dev/video0", help="USB kamera cihazı (varsayılan: /dev/video0)")
    parser.add_argument("--resolution", default="640x480", help="Kamera çözünürlüğü (varsayılan: 640x480)")
    parser.add_argument("--fps", type=int, default=30, help="FPS (varsayılan: 30)")
    parser.add_argument("--quality", type=int, default=85, help="JPEG kalitesi 0-100 (varsayılan: 85)")
    
    args = parser.parse_args()
    
    # Çözünürlüğü parse et
    try:
        width, height = map(int, args.resolution.split('x'))
        resolution = (width, height)
    except:
        log_message("Geçersiz çözünürlük formatı, 640x480 kullanılıyor")
        resolution = (640, 480)
    
    # Global server değişkeni
    global server
    
    # Signal handler kaydet
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    log_message("=== RASPBERRY PI USB KAMERA SUNUCUSU ===")
    log_message(f"Sunucu Adresi: {args.host}:{args.port}")
    log_message(f"USB Kamera: {args.device}")
    log_message(f"Çözünürlük: {resolution}")
    log_message(f"FPS: {args.fps}")
    log_message(f"JPEG Kalite: {args.quality}")
    log_message("Durdurmak için Ctrl+C tuşlarına basın")
    
    try:
        # Sunucuyu oluştur ve başlat
        server = StreamServer(
            host=args.host,
            port=args.port,
            camera_device=args.device,
            resolution=resolution,
            fps=args.fps,
            quality=args.quality
        )
        
        server.start_server()
        
    except Exception as e:
        log_message(f"Sunucu hatası: {e}")
    finally:
        if server:
            server.cleanup()


if __name__ == "__main__":
    main()
    
    def add_overlay_info(self, frame):
        """Frame'e overlay bilgileri ekle"""
        if frame is None:
            return frame
        
        # Zaman damgası
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Frame sayısı
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # FPS hesapla
        if self.start_time:
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Kamera tipi
        camera_type = "PiCamera2" if self.use_picamera else "USB Camera"
        cv2.putText(frame, camera_type, (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
    
    def encode_frame(self, frame):
        """Frame'i encode et"""
        try:
            # JPEG formatında encode et
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.quality]
            _, buffer = cv2.imencode('.jpg', frame, encode_params)
            return pickle.dumps(buffer)
        except Exception as e:
            log_message(f"Frame encode hatası: {e}")
            return None
    
    def start_streaming(self):
        """Streaming başlat"""
        self.streaming = True
        self.frame_count = 0
        self.start_time = time.time()
        log_message("Kamera streaming başlatıldı")
    
    def stop_streaming(self):
        """Streaming durdur"""
        self.streaming = False
        log_message("Kamera streaming durduruldu")
    
    def cleanup(self):
        """Temizlik işlemleri"""
        log_message("Kamera kaynakları temizleniyor...")
        
        if self.use_picamera and self.picamera2:
            try:
                self.picamera2.stop()
                log_message("PiCamera2 durduruldu")
            except Exception as e:
                log_message(f"PiCamera2 durdurma hatası: {e}")
        
        if self.cap:
            try:
                self.cap.release()
                log_message("USB kamera serbest bırakıldı")
