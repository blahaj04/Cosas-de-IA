import cv2
import time
import os
import numpy as np
import mss
import pyautogui
from processing.detect_changes import detect_change, get_latest_images

selection = None

def select_roi():
    """Permite al usuario seleccionar un área en la pantalla con el ratón sin abrir OBS."""
    global selection
    
    with mss.mss() as sct:
        monitor = sct.monitors[2]  # Capturar el segundo monitor
        screenshot = sct.grab(monitor)  # Tomar una captura de pantalla
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # Convertir a formato OpenCV
    
    selection = cv2.selectROI("Selecciona el área de captura", img, showCrosshair=True)
    cv2.destroyAllWindows()
    return selection

def capture_screenshot(capture_folder):
    """Captura una imagen de la pantalla en la zona seleccionada y la guarda."""
    global selection
    
    with mss.mss() as sct:
        monitor = sct.monitors[2]  # Capturar el segundo monitor
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        # Recortar el área seleccionada
        x, y, w, h = selection
        img_cropped = img[y:y + h, x:x + w]
        
        filename = os.path.join(capture_folder, f"capture_{int(time.time())}.png")
        cv2.imwrite(filename, img_cropped)
        print(f"[INFO] Captura guardada en {filename}")

def main():
    global selection
    
    print("[INFO] Selecciona el área de la pantalla que quieres capturar.")
    selection = select_roi()
    if selection is None or selection == (0, 0, 0, 0):
        print("[ERROR] No se seleccionó un área válida.")
        exit()
    
    capture_folder = "chatbot_orson_judio/src/captures"
    os.makedirs(capture_folder, exist_ok=True)
    
    print("[INFO] Iniciando detección de cambios en la zona seleccionada...")
    while True:
        # Capturar nueva imagen antes de analizar
        capture_screenshot(capture_folder)
        
        latest_images = get_latest_images(capture_folder)
        
        if latest_images and len(latest_images) >= 2:
            img1 = cv2.imread(latest_images[0])
            img2 = cv2.imread(latest_images[1])
            
            if img1 is None or img2 is None:
                print("[ERROR] No se pudieron cargar las imágenes correctamente.")
                time.sleep(10)
                continue
            
            x, y, w, h = selection
            img1 = img1[y:y + h, x:x + w]
            img2 = img2[y:y + h, x:x + w]
            
            # Usar la GPU si está disponible
            gpu_mat1 = cv2.cuda_GpuMat()
            gpu_mat2 = cv2.cuda_GpuMat()
            
            gpu_mat1.upload(img1)
            gpu_mat2.upload(img2)
            
            img1_gray = cv2.cuda.cvtColor(gpu_mat1, cv2.COLOR_BGR2GRAY)
            img2_gray = cv2.cuda.cvtColor(gpu_mat2, cv2.COLOR_BGR2GRAY)
            
            if detect_change(img1_gray.download(), img2_gray.download()):
                print("[ALERTA] Cambio significativo detectado en la escena!")
            else:
                print("[INFO] No hay cambios importantes.")
        else:
            print("[INFO] Esperando más capturas para análisis...")
        
        time.sleep(10)  # Analizar cada 10 segundos

if __name__ == "__main__":
    main()
