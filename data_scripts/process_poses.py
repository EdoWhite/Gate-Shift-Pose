import os
import numpy as np
import torch
from ultralytics import YOLO
import time

# Imposta il percorso del dataset
base_path = "/data/users/edbianchi/FRFS_BAK/dataset/FRFS/frames"

# Inizializza YOLO per l'estrazione delle pose
pose_detector = YOLO('/data/users/edbianchi/POSE/yolo11x-pose.pt').cuda()

# Funzione per calcolare e salvare la posa
def extract_and_save_pose(image_path, pose_save_path, conf=0.8, max_det=1):
    # Calcola le pose
    result = pose_detector.predict(image_path, conf=conf, max_det=max_det)
    for res in result:
        keypoints = res.keypoints
        if keypoints is not None and keypoints.shape[1] > 0:
            keypoints_xy = keypoints[0].xy.cpu().numpy()  # Estrai solo x, y come numpy array
            np.save(pose_save_path, keypoints_xy)
            return
    # Se non ci sono keypoints, salva un array di zeri
    np.save(pose_save_path, np.zeros((17, 2)))  # Assume 17 punti (modifica se necessario)

start_time = time.time()
# Ciclo nelle cartelle e nei frame
for root, dirs, files in os.walk(base_path):
    for file_name in files:
        if file_name.endswith('.jpg'):  # Considera solo i frame in formato .jpg
            image_path = os.path.join(root, file_name)
            pose_save_path = os.path.splitext(image_path)[0] + '.npy'  # Cambia l'estensione in .npy
            extract_and_save_pose(image_path, pose_save_path)

end_time = time.time()
total_time = end_time - start_time

print(f"Pose Extraction Completed.")
print(f"Total execution time: {total_time:.2f} seconds.")