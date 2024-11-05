import torch
import torch.nn as nn
from ultralytics import YOLO
import torchvision.transforms as transforms
import torch.nn.functional as F
import time
from torchvision.utils import save_image
import os
import numpy as np

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
    
    import os

#TO-DO: add a Graph Convolutional Netwrok-Based approach
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, adj_matrix):
        super(GCNLayer, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.adj = torch.tensor(adj_matrix, dtype=torch.float32, requires_grad=False).cuda()

    def forward(self, x):
        # x shape: [batch_size, num_joints, in_features]
        x = torch.matmul(self.adj, x)
        x = self.fc(x)
        return F.relu(x)

class PoseGCN(nn.Module):
    def __init__(self, num_joints=17, in_features=2, hidden_dim=64, out_features=128):
        super(PoseGCN, self).__init__()

        # Adjacency matrix for YOLO pose estimation (COCO format)
        adj_matrix = np.array([
            # Nose, Eyes, Ears, Shoulders, Elbows, Wrists, Hips, Knees, Ankles
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Nose
            [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Left Eye
            [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Right Eye
            [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Left Ear
            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Right Ear
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # Left Shoulder
            [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],  # Right Shoulder
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],  # Left Elbow
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],  # Right Elbow
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # Left Wrist
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],  # Right Wrist
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],  # Left Hip
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],  # Right Hip
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],  # Left Knee
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],  # Right Knee
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],  # Left Ankle
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],  # Right Ankle
        ])

        self.gcn1 = GCNLayer(in_features, hidden_dim, adj_matrix)
        self.gcn2 = GCNLayer(hidden_dim, out_features, adj_matrix)
        self.fc = nn.Linear(num_joints * out_features, 128)

    def forward(self, pose_batch):
        # pose_batch shape: [batch_size, num_segments, num_joints, in_features]
        batch_size, num_segments, num_joints, _ = pose_batch.size()
        
        # Reshape to process joints individually
        pose_batch = pose_batch.view(-1, num_joints, 2)  # [batch_size * num_segments, num_joints, in_features]

        x = self.gcn1(pose_batch)
        x = self.gcn2(x)

        # Flatten the output for classification
        x = x.view(batch_size * num_segments, -1)
        x = self.fc(x)

        # Reshape to original batch format
        x = x.view(batch_size, num_segments, -1)
        return x

# POSES ON DISK
class PoseModelFast(nn.Module):
    def __init__(self, num_joints=17, feature_dim=128):
        super(PoseModelFast, self).__init__()

        # MLP per trasformare le coordinate in embedding di feature
        self.fc1 = nn.Linear(num_joints * 2, 64).cuda()  # Ingresso: x e y per ciascun joint
        self.fc2 = nn.Linear(64, 128).cuda()
        self.fc3 = nn.Linear(128, feature_dim).cuda()
        
        self.feature_dim = feature_dim  # Dimensione del vettore di feature finale

    def forward(self, pose_batch):
        # `pose_batch` ha dimensione [batch_size, num_segments, num_joints*2]
        batch_size, num_segments, _ = pose_batch.size()
        
        # Appiattisci i primi due assi per far passare i dati attraverso il layer lineare
        pose_batch_flat = pose_batch.view(-1, 34).cuda()
        
        # Applica l'MLP ai keypoints per ottenere il vettore di embedding
        x = torch.relu(self.fc1(pose_batch_flat))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        
        # Ripristina la dimensione originale del batch
        x = x.view(batch_size, num_segments, -1)
        
        return x
    
# CONVOLUTIUON-BASED
class PoseModelConv1D(nn.Module):
    def __init__(self, num_joints=17, feature_dim=128):
        super(PoseModelConv1D, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64 * num_joints, feature_dim)
        
        self.feature_dim = feature_dim

    def forward(self, pose_batch):
        batch_size, num_segments, _ = pose_batch.size()
        
        # Riorganizza il tensor per convoluzione (batch * segments, 2, num_joints)
        pose_batch = pose_batch.view(-1, 17, 2).permute(0, 2, 1).cuda()
        
        # Applica convoluzione 1D e attivazioni
        x = F.relu(self.conv1(pose_batch))
        x = F.relu(self.conv2(x))
        
        # Appiattisci e applica strato fully connected finale
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        x = x.view(batch_size, num_segments, -1)
        return x