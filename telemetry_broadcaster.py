import socket
import json
import requests
from datetime import datetime

# ------------------------------
# Kullanıcı bilgileri
# ------------------------------
USERNAME = "kapsulgokturk"
PASSWORD = "K4ehtL3KLh"

# Sunucu URL (yarışma sahasında çalışacak)
BASE_URL = "http://10.0.0.10:10001"  # sahada domain çalışıyorsa bunu kullan

# ------------------------------
# Oturum açma ve session alma
# ------------------------------
def post_login():
    session = requests.Session()
    url = f"{BASE_URL}/api/giris"
    try:
        response = session.post(url, json={"kadi": USERNAME, "sifre": PASSWORD}, timeout=5)
        print("Login Durum Kodu:", response.status_code)
        print("Login Yanıtı:", response.text[:200])
        if response.status_code == 200:
            team_number = response.text
            print(f"Oturum açıldı. Takım Numarası: {team_number}")
            return session, team_number
        else:
            print(f"Hata: Oturum açılamadı. Status: {response.status_code}")
            return None, None
    except requests.exceptions.RequestException as e:
        print(f"Login hatası: {e}")
        return None, None

# ------------------------------
# Telemetri gönderme fonksiyonu
# ------------------------------
def post_telemetry(session, data):
    if not session:
        print("Önce login yapmalısınız!")
        return
    url = f"{BASE_URL}/api/telemetri_gonder"
    try:
        response = session.post(url, json=data, timeout=5)
        print("Telemetri Durum Kodu:", response.status_code)
        print("Telemetri Yanıtı:", response.text[:200])
        if response.status_code == 200:
            print("Telemetri başarıyla gönderildi ✅")
        else:
            print("Telemetri gönderilemedi ❌")
    except requests.exceptions.RequestException as e:
        print(f"Telemetri gönderme hatası: {e}")

# ------------------------------
# AUTO mod kontrol
# ------------------------------
def convert_autonomy_mode(value):
    if value == "AUTO":
        return 1
    else:
        return 0

# ------------------------------
# Jetson Nano'dan telemetri alma
# ------------------------------
def get_telemetry_from_jetson(session, team_number):
    SERVER_IP = '10.0.0.117'  # raspi IP
    SERVER_PORT = 8

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect((SERVER_IP, SERVER_PORT))
        print(f"Jetson Nano'ya bağlandı: {SERVER_IP}:{SERVER_PORT}")

        while True:
            try:
                data = sock.recv(1024)
                if not data:
                    print("Bağlantı kapandı.")
                    break
                telemetry_data = json.loads(data.decode('utf-8'))

                telemetry_dict = {
                    "takim_numarasi": team_number,
                    "iha_enlem": telemetry_data[0],
                    "iha_boylam": telemetry_data[1],
                    "iha_irtifa": telemetry_data[2],
                    "iha_yatis": telemetry_data[3],
                    "iha_dikilme": telemetry_data[4],
                    "iha_yonelme": telemetry_data[5],
                    "iha_hiz": telemetry_data[6],
                    "iha_batarya": telemetry_data[8],
                    "iha_otonom": convert_autonomy_mode(telemetry_data[9]),
                    "gps_saati": {
                        "saat": datetime.strptime(telemetry_data[10], "%Y-%m-%dT%H:%M:%S.%f%z").hour,
                        "dakika": datetime.strptime(telemetry_data[10], "%Y-%m-%dT%H:%M:%S.%f%z").minute,
                        "saniye": datetime.strptime(telemetry_data[10], "%Y-%m-%dT%H:%M:%S.%f%z").second,
                        "milisaniye": datetime.strptime(telemetry_data[10], "%Y-%m-%dT%H:%M:%S.%f%z").microsecond // 1000
                    }
                }

                post_telemetry(session, telemetry_dict)

            except json.JSONDecodeError as e:
                print(f"JSON Parse Hatası: {e}")
            except socket.error as e:
                print(f"Socket Hatası: {e}")
                break

    except Exception as e:
        print(f"Genel Hata: {e}")
    finally:
        if 'sock' in locals():
            sock.close()

# ------------------------------
# Ana program
# ------------------------------
if __name__ == "__main__":
    session, team_number = post_login()
    if session and team_number:
        get_telemetry_from_jetson(session, team_number)
    else:
        print("Login başarısız, telemetri başlatılamıyor ❌")

