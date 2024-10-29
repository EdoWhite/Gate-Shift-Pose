import torch
import torch.nn as nn
from ultralytics import YOLO
import torchvision.transforms as transforms

class PoseModel(nn.Module):
    def __init__(self, num_joints=17, feature_dim=128):
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

    def forward(self, img_batch):
        # Normalizza valori dei pixel
        img_batch = img_batch.cuda() / 255.0
        # Preprocessa e ridimensiona ciascuna immagine nel batch
        img_batch = torch.stack([self.preprocess(img).cuda() for img in img_batch])  # Applica resize e normalizzazione
        
        # Estrai keypoints per ogni immagine nel batch
        keypoints_batch = [self.extract_keypoints(img.unsqueeze(0)) for img in img_batch]
        
        # Riempi con zeri per i frame che non hanno keypoints
        keypoints_batch = [kp if kp is not None else torch.zeros(self.fc1.in_features).cuda() for kp in keypoints_batch]
        
        # Concatena tutti i keypoints in un batch
        keypoints_batch = torch.stack(keypoints_batch).cuda()
        
        x = torch.relu(self.fc1(keypoints_batch))
        x = self.fc2(x)
        return x

    def extract_keypoints(self, img, conf=0.5):
        # Utilizza YOLO per ottenere i keypoints delle pose
        pose_model = YOLO('/home/clusterusers/edbianchi/POSE/yolov8m-pose.pt')
        result = pose_model.predict(img, conf=conf, half=True, max_det=1)
        
        for res in result:
            keypoints = res.keypoints
            if keypoints is not None and keypoints.shape[1] > 0:
                # Ottieni le coordinate x, y del primo individuo rilevato
                keypoints_xy = keypoints[0].xy  # Tensor di forma (num_joints, 2)
                return keypoints_xy.reshape(-1).cuda()  # Converti in un vettore di forma (num_joints * 2)
        
        return None  # Ritorna None se non ci sono keypoints rilevati