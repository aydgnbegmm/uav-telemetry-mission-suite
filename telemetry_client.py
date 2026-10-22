import socket
import cv2
import pickle
import struct
import datetime
import time
import threading

def log_message(message):
    """Zaman damgalı log mesajı yazdır"""
    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{current_time}] [SERVER] {message}")

def handle_client(client_socket, client_address):
    """Her client için ayrı thread'de çalışan fonksiyon"""
    log_message(f"Yeni client bağlandı: {client_address}")
    
    try:
        # Kamera başlat
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            log_message("HATA: Kamera açılamadı!")
            return
            
        # Kamera ayarları
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 25)
        
        log_message(f"Kamera başlatıldı - {client_address} için görüntü gönderimi başlıyor")
        
        frame_count = 0
        last_status_time = time.time()
        
        while True:
            # Frame oku
            ret, frame = cap.read()
            if not ret:
                log_message("HATA: Kameradan frame okunamadı!")
                break
            
            try:
                # Frame'i encode et
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                
                # Serialize et
                data = pickle.dumps(buffer)
                
                # Boyut bilgisini gönder
                size = len(data)
                client_socket.sendall(struct.pack("L", size))
                
                # Frame verisini gönder
                client_socket.sendall(data)
                
                frame_count += 1
                
                # Her 5 saniyede durum raporu
                current_time = time.time()
                if current_time - last_status_time >= 5:
                    log_message(f"Client {client_address}: {frame_count} frame gönderildi")
                    last_status_time = current_time
                
            except socket.error as e:
                log_message(f"Client {client_address} bağlantısı koptu: {e}")
                break
                
    except Exception as e:
        log_message(f"Client {client_address} işleme hatası: {e}")
        
    finally:
        if 'cap' in locals():
            cap.release()
        client_socket.close()
        log_message(f"Client {client_address} bağlantısı kapatıldı")

def start_server(host='0.0.0.0', port=8080):
    """Server'ı başlat"""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind((host, port))
        server_socket.listen(5)
        log_message(f"Server {host}:{port} adresinde dinleniyor...")
        log_message("Client bağlantıları bekleniyor...")
        
        while True:
            try:
                client_socket, client_address = server_socket.accept()
                
                # Her client için yeni thread oluştur
                client_thread = threading.Thread(
                    target=handle_client, 
                    args=(client_socket, client_address)
                )
                client_thread.daemon = True
                client_thread.start()
                
            except KeyboardInterrupt:
                log_message("Server kapatılıyor (Ctrl+C ile)...")
                break
            except Exception as e:
                log_message(f"Client kabul hatası: {e}")
                continue
                
    except Exception as e:
        log_message(f"Server başlatma hatası: {e}")
        
    finally:
        server_socket.close()
        log_message("Server kapatıldı")

if __name__ == "__main__":
    # Raspberry Pi IP'si: 10.0.04.17, Port: 8
    start_server('10.0.0.147', 12345)
