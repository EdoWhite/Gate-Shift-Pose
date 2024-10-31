import torch
import torch.nn as nn
from ultralytics import YOLO
import torchvision.transforms as transforms
import time
from torchvision.utils import save_image
import os

class PoseModel(nn.Module):
    def __init__(self, num_joints=17, feature_dim=128, save_dir="/data/users/edbianchi/saved_frames_poses"):
        super(PoseModel, self).__init__()
        
        # Inizializza YOLO una volta sola
        #self.pose_model = YOLO('/home/clusterusers/edbianchi/POSE/yolov8m-pose.pt')
        
        # MLP per trasformare le coordinate in embedding di feature
        self.fc1 = nn.Linear(num_joints * 2, 64).cuda()  # Ingresso: x e y per ciascun joint
        self.fc2 = nn.Linear(64, feature_dim).cuda()
        
        self.feature_dim = feature_dim  # Dimensione del vettore di feature finale

        # Trasformazioni per ridimensionare e normalizzare l'input
        self.preprocess = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize a 256x256
            #transforms.ConvertImageDtype(torch.float32),  # Converte l’immagine in float32
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizzazione
        ])

        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def save_frame(self, img, idx):
        """Salva il frame corrente come immagine."""
        filename = os.path.join(self.save_dir, f"frame_{idx}.png")
        save_image(img, filename)
        print(f"Frame salvato in: {filename}")

    def normalize_0_1(self, img):
        min_val = img.min()
        max_val = img.max()
        # Normalizza tra 0 e 1 usando il min e il max
        normalized_img = (img - min_val) / (max_val - min_val)
        return normalized_img


    def forward(self, img_batch):
        start_time = time.time()

        # Preprocessa e ridimensiona ciascuna immagine nel batch
        img_batch = torch.stack([self.normalize_0_1(self.preprocess(img)) for img in img_batch]).cuda() # Applica resize e normalizzazione
        preprocess_time = time.time()
        #print(f"Tempo di preprocessamento delle immagini: {preprocess_time - start_time:.4f} sec")

        """
        for idx, img in enumerate(img_batch):
            self.save_frame(img, idx)
        """
        # Estrai keypoints per ogni immagine nel batch
        keypoints_batch = [self.extract_keypoints(img.unsqueeze(0)) for img in img_batch]
        pose_estimation_time = time.time()
        #print(f"Tempo di stima delle pose (YOLO): {pose_estimation_time - preprocess_time:.4f} sec")
        
        # Riempi con zeri per i frame che non hanno keypoints
        keypoints_batch = [kp if kp is not None else torch.zeros(self.fc1.in_features).cuda() for kp in keypoints_batch]
        
        # Concatena tutti i keypoints in un batch
        keypoints_batch = torch.stack(keypoints_batch).cuda()
        
        x = torch.relu(self.fc1(keypoints_batch))
        mlp_fc1_time = time.time()
        #print(f"Tempo MLP livello 1 (fc1): {mlp_fc1_time - pose_estimation_time:.4f} sec")

        x = self.fc2(x)
        mlp_fc2_time = time.time()
        #print(f"Tempo MLP livello 2 (fc2): {mlp_fc2_time - mlp_fc1_time:.4f} sec")

        #print(f"Tempo totale forward PoseModel: {mlp_fc2_time - start_time:.4f} sec\n")
        return x

    def extract_keypoints(self, img, conf=0.5):
        # Utilizza YOLO per ottenere i keypoints delle pose
        pose_model = YOLO('/data/users/edbianchi/POSE/yolo11n-pose.pt').cuda()
        result = pose_model.predict(img, conf=conf, half=True, max_det=1)
        
        for res in result:
            keypoints = res.keypoints
            if keypoints is not None and keypoints.shape[1] > 0:
                # Ottieni le coordinate x, y del primo individuo rilevato
                keypoints_xy = keypoints[0].xy  # Tensor di forma (num_joints, 2)
                return keypoints_xy.reshape(-1).cuda()  # Converti in un vettore di forma (num_joints * 2)
        
        return None  # Ritorna None se non ci sono keypoints rilevati