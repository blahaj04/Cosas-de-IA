import cv2
import time
import os
import numpy as np
from processing.detect_changes import detect_change, get_latest_images

selection = None


def select_roi():
    """Permite al usuario seleccionar un área en la pantalla con el ratón."""
    global selection
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Muestra la webcam temporalmente para seleccionar el área
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("[ERROR] No se pudo capturar la pantalla para la selección de ROI")
        return None
    
    selection = cv2.selectROI("Selecciona el área de captura", frame, showCrosshair=True)
    cv2.destroyAllWindows()
    return selection


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
        latest_images = get_latest_images(capture_folder)
        if latest_images and len(latest_images) >= 2:
            img1 = cv2.imread(latest_images[0])
            img2 = cv2.imread(latest_images[1])
            
            x, y, w, h = selection
            img1 = img1[y:y + h, x:x + w]
            img2 = img2[y:y + h, x:x + w]
            
            if detect_change(img1, img2):
                print("[ALERTA] Cambio significativo detectado en la escena!")
            else:
                print("[INFO] No hay cambios importantes.")
        
        time.sleep(10)  # Analizar cada 10 segundos


if __name__ == "__main__":
    main()
