import cv2
import numpy as np
import os

def get_latest_images(folder, num_images=2):
    """Obtiene las dos imágenes más recientes en la carpeta especificada."""
    images = sorted(
        [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".png")],
        key=os.path.getmtime,
        reverse=True
    )
    return images[:num_images] if len(images) >= num_images else None

def detect_change(img1, img2, threshold=1000000):
    """
    Detecta si hay un cambio significativo entre dos imágenes.
    :param img1: Primera imagen (NumPy array)
    :param img2: Segunda imagen (NumPy array)
    :param threshold: Umbral de cambio, cuanto más alto, más tolerante es.
    :return: True si hay un cambio significativo, False en caso contrario.
    """
    if img1 is None or img2 is None:
        return False
    
    # Convertir a escala de grises para comparación
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Calcular la diferencia absoluta entre las imágenes
    diff = cv2.absdiff(img1_gray, img2_gray)
    
    # Convertir la diferencia a blanco y negro para contar píxeles diferentes
    _, diff_threshold = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    change_value = np.sum(diff_threshold)
    
    print(f"[INFO] Cambio detectado: {change_value} píxeles diferentes")
    return change_value > threshold

if __name__ == "__main__":
    """ Prueba con las dos imágenes más recientes de la carpeta chatbot_orson_judio/src/captures """
    capture_folder = "chatbot_orson_judio/src/captures"
    latest_images = get_latest_images(capture_folder)
    
    if latest_images:
        img1 = cv2.imread(latest_images[0])
        img2 = cv2.imread(latest_images[1])
        
        if detect_change(img1, img2):
            print("[ALERTA] Cambio significativo detectado en la escena!")
        else:
            print("[INFO] No hay cambios importantes.")
    else:
        print("[ERROR] No se encontraron imágenes suficientes para comparar.")
