import mss
import time
import cv2
import numpy as np
import os

def get_folder_size(folder):
    """Obtiene el tamaño total de la carpeta en bytes."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size

def delete_old_captures(folder):
    """Elimina todas las capturas en la carpeta."""
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"[INFO] Archivo eliminado: {file_path}")

def capture_screen(interval=10, capture_region=None,max_size_gb=1):
    """
    Captura la pantalla cada `interval` segundos y guarda la imagen en data/captures/.
    :param interval: Tiempo entre capturas en segundos.
    :param capture_region: Región de la pantalla a capturar (left, top, width, height) o None para full screen.
    """
    save_path = "chatbot_orson_judiodata/captures"
    os.makedirs(save_path, exist_ok=True)
    
    with mss.mss() as sct:
        monitor_number = 2  # Capturar el segundo monitor
        monitor = sct.monitors[monitor_number]  # Obtener el monitor secundario
        
        while True:
             # Verificar el tamaño de la carpeta y eliminar capturas si excede el límite
            if get_folder_size(save_path) > max_size_gb * 1024 * 1024 * 1024:
                print("[INFO] La carpeta ha superado 1GB, eliminando capturas...")
                delete_old_captures(save_path)
            # Capturar la pantalla del segundo monitor
            screenshot = sct.grab(monitor)
            
            # Convertir a un array de NumPy
            img = np.array(screenshot)
            
            # Convertir de BGRA a BGR para OpenCV
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            # Guardar la imagen
            filename = os.path.join(save_path, f"capture_{int(time.time())}.png")
            cv2.imwrite(filename, img)
            print(f"[INFO] Captura guardada en {filename}")
            
            # Esperar el intervalo
            time.sleep(interval)

# Ejecutar la captura de pantalla cada 10 segundos en el segundo monitor
if __name__ == "__main__":
    capture_screen()
