from torch import nn
from ops.basic_ops import ConsensusModule
from utils.transforms import *
from torch.nn.init import normal_, constant_
from scipy.ndimage import zoom
#import cv2
import os, sys
from torch.cuda import amp
from poseModel import PoseModel, PoseModelFast, PoseModelConv1D, PoseGCN  # Importa PoseModel per la fusione delle pose
import time
from torchvision.utils import save_image
import torch.nn.functional as F

# EDITED by white

class VideoModel(nn.Module):
    def __init__(self, num_class, num_segments,
                 base_model='BNInception',
                 consensus_type='avg', before_softmax=True,
                 dropout=0.5, crop_num=1, print_spec=True,
                 gsf=True, gsf_ch_ratio=100,
                 target_transform=None, num_channels=3):
        super(VideoModel, self).__init__()
        self.num_segments = num_segments
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        self.gsf = gsf
        self.gsf_ch_ratio = gsf_ch_ratio
        self.target_transform = target_transform
        self.num_channels = num_channels

        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if print_spec:
            print(("""
    Initializing Video Model with backbone: {}.
    Model Configurations:
                        GSF:                {}
                        Channel ratio:      {}
                        num_segments:       {}
                        consensus_module:   {}
                        dropout_ratio:      {}
            """.format(base_model, self.gsf, self.gsf_ch_ratio, self.num_segments, consensus_type, self.dropout)))

        self.feature_dim = self._prepare_base_model(base_model)
       
        self.feature_dim = self._prepare_model(num_class, self.feature_dim)

        self.consensus = ConsensusModule(consensus_type)

        self.softmax = nn.Softmax()
        
        if not self.before_softmax:
            self.softmax = nn.Softmax()

    def _prepare_model(self, num_class, feature_dim):

        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            self.new_fc = nn.Linear(feature_dim, num_class)

        std = 0.001
        if self.new_fc is None:
            normal_(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
            constant_(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            normal_(self.new_fc.weight, 0, std)
            constant_(self.new_fc.bias, 0)
        return feature_dim

    def _prepare_base_model(self, base_model):
        if 'resnet' in base_model:
            if self.gsf:
                import backbones.resnetGSFModels as resnet_models
                self.base_model = getattr(resnet_models, base_model)(pretrained=True, num_segments=self.num_segments,
                                                                     gsf_ch_ratio=self.gsf_ch_ratio, num_channels=self.num_channels)
            else:
                import torchvision.models.resnet as resnet_models
                self.base_model = getattr(resnet_models, base_model)(pretrained=True)
            # print(self.base_model)
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]
            feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        elif base_model == 'bninception':
            import backbones.pytorch_load as inception
            if self.gsf:
                    self.base_model = inception.BNInception_gsf(num_segments=self.num_segments,
                                                                gsf_ch_ratio=self.gsf_ch_ratio)
            else:
                self.base_model = inception.BNInception()
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [104, 117, 128]
            self.input_std = [1, 1, 1]
            feature_dim = 1024
        elif base_model == 'inceptionv3':
            import backbones.pytorch_load as inception
            if self.gsf:
                self.base_model = inception.InceptionV3_gsf(num_segments=self.num_segments,
                                                                     gsf_ch_ratio=self.gsf_ch_ratio)
            else:
                self.base_model = inception.InceptionV3()
            self.base_model.last_layer_name = 'top_cls_fc'
            self.input_size = 229
            self.input_mean = [104, 117, 128]
            self.input_std = [1, 1, 1]
            feature_dim = 2048
        # ADDED
        elif base_model == 'inceptionv3_kinetics':
            import backbones.pytorch_load as inception
            if self.gsf:
                self.base_model = inception.InceptionV3_kinetics_gsf(num_segments=self.num_segments,
                                                                     gsf_ch_ratio=self.gsf_ch_ratio)
            else:
                self.base_model = inception.InceptionV3()
            self.base_model.last_layer_name = 'top_cls_fc'
            self.input_size = 229
            self.input_mean = [104, 117, 128]
            self.input_std = [1, 1, 1]
            feature_dim = 2048

        else:
            raise ValueError('Unknown base model: {}'.format(base_model))
        return feature_dim

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []
        linear_weight = []
        linear_bias = []

        conv_cnt = 0
        bn_cnt = 0
        for n, m in self.named_modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv3d):
                    ps = list(m.parameters())
                    conv_cnt += 1
                    if conv_cnt == 1:
                        first_conv_weight.append(ps[0])
                        if len(ps) == 2:
                            first_conv_bias.append(ps[1])
                    else:
                        normal_weight.append(ps[0])
                        if len(ps) == 2:
                            normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear): 
                ps = list(m.parameters())
                linear_weight.append(ps[0])
                if len(ps) == 2:
                    linear_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm3d):
                bn_cnt += 1
                # later BN's are frozen
                #if not self._enable_pbn or bn_cnt == 1:
                bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': linear_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "linear_weight"},
            {'params': linear_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "linear_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
        ]

    def forward(self, input, with_amp: bool=False, idx:int=0, target:int=0):
        #assert isinstance(with_amp, bool)
    
        #base_out = self.base_model(input.view((-1, 3) + input.size()[-2:]))
        base_out = self.base_model(input.view((-1, self.num_channels) + input.size()[-2:]))

        base_out_logits = base_out if self.new_fc is None else self.new_fc(base_out)
        
        #if self.dropout > 0:
        #    base_out_logits = self.new_fc(base_out)

        if not self.before_softmax:
            base_out_logits = self.softmax(base_out_logits)
            
        base_out_logits = base_out_logits.view((-1, self.num_segments) + base_out_logits.size()[1:])

        output = self.consensus(base_out_logits)

        return output

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self):
        return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                               GroupRandomHorizontalFlip(is_flow=False,
                                                                         target_transform=self.target_transform)])

class VideoModelLateFusion(nn.Module):
    def __init__(self, num_class, num_segments, base_model='BNInception',
                 consensus_type='avg', before_softmax=True,
                 dropout=0.5, crop_num=1, print_spec=True,
                 gsf=True, gsf_ch_ratio=100,
                 target_transform=None, num_channels=3):
        super(VideoModelLateFusion, self).__init__()
        
        # Inizializzazione degli attributi di configurazione
        self.num_segments = num_segments
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        self.gsf = gsf
        self.gsf_ch_ratio = gsf_ch_ratio
        self.target_transform = target_transform
        self.num_channels = num_channels

        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        # Stampa configurazione del modello
        if print_spec:
            print(("""
    VideoModelLateFusion
    Initializing Video Model with backbone: {}.
    Model Configurations:
                        GSF:                {}
                        Channel ratio:      {}
                        num_segments:       {}
                        consensus_module:   {}
                        dropout_ratio:      {}
            """.format(base_model, self.gsf, self.gsf_ch_ratio, self.num_segments, consensus_type, self.dropout)))

        # Modello backbone per le feature RGB
        self.feature_dim = self._prepare_base_model(base_model)
        self.feature_dim = self._prepare_model(num_class, self.feature_dim)

        # Inizializza il modello delle pose
        self.pose_model = PoseModel().cuda() # Inizializza PoseModel per estrarre feature dalle coordinate
        self.feature_dim += self.pose_model.feature_dim  # Aggiusta dimensione delle feature per la concatenazione

        # Fully connected layer per la classificazione finale
        self.new_fc = nn.Linear(self.feature_dim, num_class)

        # Modulo di consensus
        self.consensus = ConsensusModule(consensus_type)
        self.softmax = nn.Softmax() if not before_softmax else None

    def save_frame(self, img, idx):
        """Salva il frame corrente come immagine."""
        filename = os.path.join("/data/users/edbianchi/saved_frames_debug_1", f"frame_{idx}.png")
        save_image(img, filename)
        print(f"Frame salvato in: {filename}")

    def _prepare_model(self, num_class, feature_dim):

        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            self.new_fc = nn.Linear(feature_dim, num_class)

        std = 0.001
        if self.new_fc is None:
            normal_(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
            constant_(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            normal_(self.new_fc.weight, 0, std)
            constant_(self.new_fc.bias, 0)
        return feature_dim

    def _prepare_base_model(self, base_model):
        if 'resnet' in base_model:
            if self.gsf:
                import backbones.resnetGSFModels as resnet_models
                self.base_model = getattr(resnet_models, base_model)(pretrained=True, num_segments=self.num_segments,
                                                                     gsf_ch_ratio=self.gsf_ch_ratio, num_channels=self.num_channels)
            else:
                import torchvision.models.resnet as resnet_models
                self.base_model = getattr(resnet_models, base_model)(pretrained=True)
            # print(self.base_model)
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]
            feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        elif base_model == 'bninception':
            import backbones.pytorch_load as inception
            if self.gsf:
                    self.base_model = inception.BNInception_gsf(num_segments=self.num_segments,
                                                                gsf_ch_ratio=self.gsf_ch_ratio)
            else:
                self.base_model = inception.BNInception()
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [104, 117, 128]
            self.input_std = [1, 1, 1]
            feature_dim = 1024
        elif base_model == 'inceptionv3':
            import backbones.pytorch_load as inception
            if self.gsf:
                self.base_model = inception.InceptionV3_gsf(num_segments=self.num_segments,
                                                                     gsf_ch_ratio=self.gsf_ch_ratio)
            else:
                self.base_model = inception.InceptionV3()
            self.base_model.last_layer_name = 'top_cls_fc'
            self.input_size = 229
            self.input_mean = [104, 117, 128]
            self.input_std = [1, 1, 1]
            feature_dim = 2048
        # ADDED
        elif base_model == 'inceptionv3_kinetics':
            import backbones.pytorch_load as inception
            if self.gsf:
                self.base_model = inception.InceptionV3_kinetics_gsf(num_segments=self.num_segments,
                                                                     gsf_ch_ratio=self.gsf_ch_ratio)
            else:
                self.base_model = inception.InceptionV3()
            self.base_model.last_layer_name = 'top_cls_fc'
            self.input_size = 229
            self.input_mean = [104, 117, 128]
            self.input_std = [1, 1, 1]
            feature_dim = 2048

        else:
            raise ValueError('Unknown base model: {}'.format(base_model))
        return feature_dim

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []
        linear_weight = []
        linear_bias = []

        conv_cnt = 0
        bn_cnt = 0
        for n, m in self.named_modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv3d):
                    ps = list(m.parameters())
                    conv_cnt += 1
                    if conv_cnt == 1:
                        first_conv_weight.append(ps[0])
                        if len(ps) == 2:
                            first_conv_bias.append(ps[1])
                    else:
                        normal_weight.append(ps[0])
                        if len(ps) == 2:
                            normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear): 
                ps = list(m.parameters())
                linear_weight.append(ps[0])
                if len(ps) == 2:
                    linear_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm3d):
                bn_cnt += 1
                # later BN's are frozen
                #if not self._enable_pbn or bn_cnt == 1:
                bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': linear_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "linear_weight"},
            {'params': linear_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "linear_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
        ]

    def forward(self, input_rgb, with_amp: bool=False, idx:int=0, target:int=0):
        start_time = time.time()

        # [256, 3, 224, 224] --> [batch * num_segments, 3, 224, 224]
        input_rgb_reshaped = input_rgb.view((-1, self.num_channels) + input_rgb.size()[-2:]).cuda()
        rgb_time = time.time()
        #print(f"Tempo di preparazione dell'input RGB: {rgb_time - start_time:.4f} sec")

        """
        for idx, img in enumerate(input_rgb_reshaped):
            self.save_frame(img, idx)
        """

        base_out_rgb = self.base_model(input_rgb_reshaped)
        #print(f"base_out_rgb shape: {base_out_rgb.shape}")
        rgb_inference_time = time.time()
        #print(f"Tempo di inferenza del modello RGB: {rgb_inference_time - rgb_time:.4f} sec")
        
        base_out_pose = self.pose_model(input_rgb_reshaped)
        #print(f"base_out_pose shape: {base_out_pose.shape}")
        pose_inference_time = time.time()
        #print(f"Tempo di inferenza del modello delle pose: {pose_inference_time - rgb_inference_time:.4f} sec")
        
        # Fusione tardiva (concatenazione delle feature RGB e delle pose)
        combined_out = torch.cat((base_out_rgb, base_out_pose), dim=1).cuda()
        #print(f"combined_out shape: {combined_out.shape}")
        fusion_time = time.time()
        #print(f"Tempo di fusione delle feature RGB e pose: {fusion_time - pose_inference_time:.4f} sec")

        # Classificazione finale
        base_out_logits = self.new_fc(combined_out)
        classification_time = time.time()
        #print(f"Tempo di classificazione finale: {classification_time - fusion_time:.4f} sec")

        if not self.before_softmax:
            base_out_logits = self.softmax(base_out_logits)
            
        base_out_logits = base_out_logits.view((-1, self.num_segments) + base_out_logits.size()[1:]).cuda()
        output = self.consensus(base_out_logits).cuda()
        end_time = time.time()
        #print(f"Tempo di consenso finale: {end_time - classification_time:.4f} sec")

        # Tempo totale per la forward pass
        #print(f"Tempo totale per forward: {end_time - start_time:.4f} sec\n")

        return output
    
    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self):
        return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                               GroupRandomHorizontalFlip(is_flow=False,
                                                                         target_transform=self.target_transform)])
    
# POSES OLREADY SAVED ON DISK   
class VideoModelLateFusionFast(nn.Module):
    def __init__(self, num_class, num_segments, base_model='BNInception',
                 consensus_type='avg', before_softmax=True,
                 dropout=0.5, crop_num=1, print_spec=True,
                 gsf=True, gsf_ch_ratio=100,
                 target_transform=None, num_channels=3):
        super(VideoModelLateFusionFast, self).__init__()
        
        # Inizializzazione degli attributi di configurazione
        self.num_segments = num_segments
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        self.gsf = gsf
        self.gsf_ch_ratio = gsf_ch_ratio
        self.target_transform = target_transform
        self.num_channels = num_channels

        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        # Stampa configurazione del modello
        if print_spec:
            print(("""
    VideoModelLateFusion
    Initializing Video Model with backbone: {}.
    Model Configurations:
                        GSF:                {}
                        Channel ratio:      {}
                        num_segments:       {}
                        consensus_module:   {}
                        dropout_ratio:      {}
            """.format(base_model, self.gsf, self.gsf_ch_ratio, self.num_segments, consensus_type, self.dropout)))

        # Modello backbone per le feature RGB
        self.feature_dim = self._prepare_base_model(base_model)
        self.feature_dim = self._prepare_model(num_class, self.feature_dim)

        # Inizializza il modello delle pose
        self.pose_model = PoseModelFast().cuda()
        self.feature_dim += self.pose_model.feature_dim
        
        # Feature refinement layer - progressive learning. 
        # Introduces a sort of regularization forcing the model to compress and select relevant info
        self.feature_refinement_layer = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.BatchNorm1d(self.feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(self.feature_dim // 2, self.feature_dim // 4),
            nn.BatchNorm1d(self.feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout)
        )

        # Fully connected layer per la classificazione finale
        self.new_fc = nn.Linear(self.feature_dim // 4, num_class)

        # Modulo di consensus
        self.consensus = ConsensusModule(consensus_type)
        self.softmax = nn.Softmax() if not before_softmax else None

    def _prepare_model(self, num_class, feature_dim):

        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            self.new_fc = nn.Linear(feature_dim, num_class)

        std = 0.001
        if self.new_fc is None:
            normal_(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
            constant_(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            normal_(self.new_fc.weight, 0, std)
            constant_(self.new_fc.bias, 0)
        return feature_dim

    def _prepare_base_model(self, base_model):
        if 'resnet' in base_model:
            if self.gsf:
                import backbones.resnetGSFModels as resnet_models
                self.base_model = getattr(resnet_models, base_model)(pretrained=True, num_segments=self.num_segments,
                                                                     gsf_ch_ratio=self.gsf_ch_ratio, num_channels=self.num_channels)
            else:
                import torchvision.models.resnet as resnet_models
                self.base_model = getattr(resnet_models, base_model)(pretrained=True)
            # print(self.base_model)
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]
            feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        elif base_model == 'bninception':
            import backbones.pytorch_load as inception
            if self.gsf:
                    self.base_model = inception.BNInception_gsf(num_segments=self.num_segments,
                                                                gsf_ch_ratio=self.gsf_ch_ratio)
            else:
                self.base_model = inception.BNInception()
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [104, 117, 128]
            self.input_std = [1, 1, 1]
            feature_dim = 1024
        elif base_model == 'inceptionv3':
            import backbones.pytorch_load as inception
            if self.gsf:
                self.base_model = inception.InceptionV3_gsf(num_segments=self.num_segments,
                                                                     gsf_ch_ratio=self.gsf_ch_ratio)
            else:
                self.base_model = inception.InceptionV3()
            self.base_model.last_layer_name = 'top_cls_fc'
            self.input_size = 229
            self.input_mean = [104, 117, 128]
            self.input_std = [1, 1, 1]
            feature_dim = 2048
        # ADDED
        elif base_model == 'inceptionv3_kinetics':
            import backbones.pytorch_load as inception
            if self.gsf:
                self.base_model = inception.InceptionV3_kinetics_gsf(num_segments=self.num_segments,
                                                                     gsf_ch_ratio=self.gsf_ch_ratio)
            else:
                self.base_model = inception.InceptionV3()
            self.base_model.last_layer_name = 'top_cls_fc'
            self.input_size = 229
            self.input_mean = [104, 117, 128]
            self.input_std = [1, 1, 1]
            feature_dim = 2048

        else:
            raise ValueError('Unknown base model: {}'.format(base_model))
        return feature_dim

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []
        linear_weight = []
        linear_bias = []

        conv_cnt = 0
        bn_cnt = 0
        for n, m in self.named_modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv3d):
                    ps = list(m.parameters())
                    conv_cnt += 1
                    if conv_cnt == 1:
                        first_conv_weight.append(ps[0])
                        if len(ps) == 2:
                            first_conv_bias.append(ps[1])
                    else:
                        normal_weight.append(ps[0])
                        if len(ps) == 2:
                            normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear): 
                ps = list(m.parameters())
                linear_weight.append(ps[0])
                if len(ps) == 2:
                    linear_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm3d):
                bn_cnt += 1
                # later BN's are frozen
                #if not self._enable_pbn or bn_cnt == 1:
                bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': linear_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "linear_weight"},
            {'params': linear_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "linear_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
        ]

    def forward(self, input, with_amp: bool=False, idx:int=0, target:int=0):
        input_rgb, input_pose = input

        input_rgb_reshaped = input_rgb.view((-1, self.num_channels) + input_rgb.size()[-2:]).cuda()
        base_out_rgb = self.base_model(input_rgb_reshaped)
        
        base_out_pose = self.pose_model(input_pose)
        base_out_pose = base_out_pose.view(-1, base_out_pose.size(-1)) #from [16,16,128] to [256,128]
        
        # Optional: Normalize or balance the features before fusion
        base_out_rgb = F.normalize(base_out_rgb, p=2, dim=1)
        base_out_pose = F.normalize(base_out_pose, p=2, dim=1)
        
        # Fusione tardiva (concatenazione delle feature RGB e delle pose)
        combined_out = torch.cat((base_out_rgb, base_out_pose), dim=1).cuda()
        
        # Few learning after concatenation
        combined_out = self.feature_refinement_layer(combined_out)

        # Classificazione finale
        base_out_logits = self.new_fc(combined_out)

        if not self.before_softmax:
            base_out_logits = self.softmax(base_out_logits)
            
        base_out_logits = base_out_logits.view((-1, self.num_segments) + base_out_logits.size()[1:]).cuda()
        output = self.consensus(base_out_logits).cuda()

        return output
    
    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self):
        return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                               GroupRandomHorizontalFlip(is_flow=False,
                                                                         target_transform=self.target_transform)])
        
        
class VideoModelLateFusionAttention(nn.Module):
    def __init__(self, num_class, num_segments, base_model='BNInception',
                 consensus_type='avg', before_softmax=True,
                 dropout=0.5, crop_num=1, print_spec=True,
                 gsf=True, gsf_ch_ratio=100,
                 target_transform=None, num_channels=3, attention_heads=4):
        super(VideoModelLateFusionAttention, self).__init__()
        
        # Inizializzazione degli attributi di configurazione
        self.num_segments = num_segments
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        self.gsf = gsf
        self.gsf_ch_ratio = gsf_ch_ratio
        self.target_transform = target_transform
        self.num_channels = num_channels

        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        # Stampa configurazione del modello
        if print_spec:
            print(("""
    VideoModelLateFusion with ATTENTION
    Initializing Video Model with backbone: {}.
    Model Configurations:
                        GSF:                {}
                        Channel ratio:      {}
                        num_segments:       {}
                        consensus_module:   {}
                        dropout_ratio:      {}
            """.format(base_model, self.gsf, self.gsf_ch_ratio, self.num_segments, consensus_type, self.dropout)))

        # Modello backbone per le feature RGB
        self.feature_dim = self._prepare_base_model(base_model)
        self.feature_dim = self._prepare_model(num_class, self.feature_dim)

        # Inizializza il modello delle pose
        self.pose_model = PoseModelFast().cuda()
        self.feature_dim += self.pose_model.feature_dim
        
        # MultiheadAttention layer
        self.attention_layer = nn.MultiheadAttention(embed_dim=self.feature_dim, num_heads=attention_heads, dropout=dropout)
        
        # Feature refinement layer - progressive learning. 
        # Introduces a sort of regularization forcing the model to compress and select relevant info
        self.feature_refinement_layer = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.BatchNorm1d(self.feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(self.feature_dim // 2, self.feature_dim // 4),
            nn.BatchNorm1d(self.feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout)
        )

        # Fully connected layer per la classificazione finale
        self.new_fc = nn.Linear(self.feature_dim // 4, num_class)

        # Modulo di consensus
        self.consensus = ConsensusModule(consensus_type)
        self.softmax = nn.Softmax() if not before_softmax else None

    def _prepare_model(self, num_class, feature_dim):

        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            self.new_fc = nn.Linear(feature_dim, num_class)

        std = 0.001
        if self.new_fc is None:
            normal_(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
            constant_(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            normal_(self.new_fc.weight, 0, std)
            constant_(self.new_fc.bias, 0)
        return feature_dim

    def _prepare_base_model(self, base_model):
        if 'resnet' in base_model:
            if self.gsf:
                import backbones.resnetGSFModels as resnet_models
                self.base_model = getattr(resnet_models, base_model)(pretrained=True, num_segments=self.num_segments,
                                                                     gsf_ch_ratio=self.gsf_ch_ratio, num_channels=self.num_channels)
            else:
                import torchvision.models.resnet as resnet_models
                self.base_model = getattr(resnet_models, base_model)(pretrained=True)
            # print(self.base_model)
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]
            feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        elif base_model == 'bninception':
            import backbones.pytorch_load as inception
            if self.gsf:
                    self.base_model = inception.BNInception_gsf(num_segments=self.num_segments,
                                                                gsf_ch_ratio=self.gsf_ch_ratio)
            else:
                self.base_model = inception.BNInception()
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [104, 117, 128]
            self.input_std = [1, 1, 1]
            feature_dim = 1024
        elif base_model == 'inceptionv3':
            import backbones.pytorch_load as inception
            if self.gsf:
                self.base_model = inception.InceptionV3_gsf(num_segments=self.num_segments,
                                                                     gsf_ch_ratio=self.gsf_ch_ratio)
            else:
                self.base_model = inception.InceptionV3()
            self.base_model.last_layer_name = 'top_cls_fc'
            self.input_size = 229
            self.input_mean = [104, 117, 128]
            self.input_std = [1, 1, 1]
            feature_dim = 2048
        # ADDED
        elif base_model == 'inceptionv3_kinetics':
            import backbones.pytorch_load as inception
            if self.gsf:
                self.base_model = inception.InceptionV3_kinetics_gsf(num_segments=self.num_segments,
                                                                     gsf_ch_ratio=self.gsf_ch_ratio)
            else:
                self.base_model = inception.InceptionV3()
            self.base_model.last_layer_name = 'top_cls_fc'
            self.input_size = 229
            self.input_mean = [104, 117, 128]
            self.input_std = [1, 1, 1]
            feature_dim = 2048

        else:
            raise ValueError('Unknown base model: {}'.format(base_model))
        return feature_dim

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []
        linear_weight = []
        linear_bias = []

        conv_cnt = 0
        bn_cnt = 0
        for n, m in self.named_modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv3d):
                    ps = list(m.parameters())
                    conv_cnt += 1
                    if conv_cnt == 1:
                        first_conv_weight.append(ps[0])
                        if len(ps) == 2:
                            first_conv_bias.append(ps[1])
                    else:
                        normal_weight.append(ps[0])
                        if len(ps) == 2:
                            normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear): 
                ps = list(m.parameters())
                linear_weight.append(ps[0])
                if len(ps) == 2:
                    linear_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm3d):
                bn_cnt += 1
                # later BN's are frozen
                #if not self._enable_pbn or bn_cnt == 1:
                bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': linear_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "linear_weight"},
            {'params': linear_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "linear_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
        ]

    def forward(self, input, with_amp: bool=False, idx:int=0, target:int=0):
        input_rgb, input_pose = input

        input_rgb_reshaped = input_rgb.view((-1, self.num_channels) + input_rgb.size()[-2:]).cuda()
        base_out_rgb = self.base_model(input_rgb_reshaped)
        
        base_out_pose = self.pose_model(input_pose)
        base_out_pose = base_out_pose.view(-1, base_out_pose.size(-1)) #from [16,16,128] to [256,128]
        
        # Optional: Normalize or balance the features before fusion
        base_out_rgb = F.normalize(base_out_rgb, p=2, dim=1)
        base_out_pose = F.normalize(base_out_pose, p=2, dim=1)
        
        # Concatenate features
        combined_out = torch.cat((base_out_rgb, base_out_pose), dim=1).cuda()
        
        # MultiheadAttention requires input shape [seq_len, batch_size, embed_dim]
        combined_out = combined_out.unsqueeze(0)  # Add seq_len=1 dimension
        attention_output, _ = self.attention_layer(combined_out, combined_out, combined_out)
        attention_output = attention_output.squeeze(0)  # Remove seq_len dimension
        
        # Feature refinement
        refined_out = self.feature_refinement_layer(attention_output)

        # Final classification
        base_out_logits = self.new_fc(refined_out)

        if not self.before_softmax:
            base_out_logits = self.softmax(base_out_logits)
            
        base_out_logits = base_out_logits.view((-1, self.num_segments) + base_out_logits.size()[1:]).cuda()
        output = self.consensus(base_out_logits).cuda()

        return output
    
    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self):
        return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                               GroupRandomHorizontalFlip(is_flow=False,
                                                                         target_transform=self.target_transform)])
        
        
class VideoModelPoses(nn.Module):
    def __init__(self, num_class, num_segments, base_model='BNInception',
                 consensus_type='avg', before_softmax=True,
                 dropout=0.5, crop_num=1, print_spec=True,
                 gsf=True, gsf_ch_ratio=100,
                 target_transform=None, num_channels=3):
        super(VideoModelPoses, self).__init__()
        
        # Inizializzazione degli attributi di configurazione
        self.num_segments = num_segments
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        self.gsf = gsf
        self.gsf_ch_ratio = gsf_ch_ratio
        self.target_transform = target_transform
        self.num_channels = num_channels

        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        # Stampa configurazione del modello
        if print_spec:
            print("""\nPose-Only Version""")

      
        # Inizializza il modello delle pose
        self.pose_model = PoseModelConv1D().cuda()
        self.feature_dim = self.pose_model.feature_dim

        # Fully connected layer per la classificazione finale
        self.new_fc = nn.Linear(self.feature_dim, num_class)

        # Modulo di consensus
        self.consensus = ConsensusModule(consensus_type)
        self.softmax = nn.Softmax() if not before_softmax else None
        
        #self.base_model.last_layer_name = 'fc'
        self.input_size = 224
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]
        #self.crop_size = self.input_size
        #self.base_model = ""
        
    def _prepare_model(self, num_class, feature_dim):
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            self.new_fc = nn.Linear(feature_dim, num_class)

        std = 0.001
        if self.new_fc is None:
            normal_(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
            constant_(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            normal_(self.new_fc.weight, 0, std)
            constant_(self.new_fc.bias, 0)
        return feature_dim

    def _prepare_base_model(self, base_model):
        if 'resnet' in base_model:
            if self.gsf:
                import backbones.resnetGSFModels as resnet_models
                self.base_model = getattr(resnet_models, base_model)(pretrained=True, num_segments=self.num_segments,
                                                                     gsf_ch_ratio=self.gsf_ch_ratio, num_channels=self.num_channels)
            else:
                import torchvision.models.resnet as resnet_models
                self.base_model = getattr(resnet_models, base_model)(pretrained=True)
            # print(self.base_model)
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]
            feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        elif base_model == 'bninception':
            import backbones.pytorch_load as inception
            if self.gsf:
                    self.base_model = inception.BNInception_gsf(num_segments=self.num_segments,
                                                                gsf_ch_ratio=self.gsf_ch_ratio)
            else:
                self.base_model = inception.BNInception()
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [104, 117, 128]
            self.input_std = [1, 1, 1]
            feature_dim = 1024
        elif base_model == 'inceptionv3':
            import backbones.pytorch_load as inception
            if self.gsf:
                self.base_model = inception.InceptionV3_gsf(num_segments=self.num_segments,
                                                                     gsf_ch_ratio=self.gsf_ch_ratio)
            else:
                self.base_model = inception.InceptionV3()
            self.base_model.last_layer_name = 'top_cls_fc'
            self.input_size = 229
            self.input_mean = [104, 117, 128]
            self.input_std = [1, 1, 1]
            feature_dim = 2048
        # ADDED
        elif base_model == 'inceptionv3_kinetics':
            import backbones.pytorch_load as inception
            if self.gsf:
                self.base_model = inception.InceptionV3_kinetics_gsf(num_segments=self.num_segments,
                                                                     gsf_ch_ratio=self.gsf_ch_ratio)
            else:
                self.base_model = inception.InceptionV3()
            self.base_model.last_layer_name = 'top_cls_fc'
            self.input_size = 229
            self.input_mean = [104, 117, 128]
            self.input_std = [1, 1, 1]
            feature_dim = 2048

        else:
            raise ValueError('Unknown base model: {}'.format(base_model))
        return feature_dim

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []
        linear_weight = []
        linear_bias = []

        conv_cnt = 0
        bn_cnt = 0
        for n, m in self.named_modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv3d):
                    ps = list(m.parameters())
                    conv_cnt += 1
                    if conv_cnt == 1:
                        first_conv_weight.append(ps[0])
                        if len(ps) == 2:
                            first_conv_bias.append(ps[1])
                    else:
                        normal_weight.append(ps[0])
                        if len(ps) == 2:
                            normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear): 
                ps = list(m.parameters())
                linear_weight.append(ps[0])
                if len(ps) == 2:
                    linear_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm3d):
                bn_cnt += 1
                # later BN's are frozen
                #if not self._enable_pbn or bn_cnt == 1:
                bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': linear_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "linear_weight"},
            {'params': linear_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "linear_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
        ]


    def forward(self, input, with_amp: bool=False, idx:int=0, target:int=0):
        _, input_pose = input
        
        print("input_pose shape:")
        print(input_pose.shape)
        base_out_pose = self.pose_model(input_pose)
        base_out_pose = base_out_pose.view(-1, base_out_pose.size(-1)) #from [16,16,128] to [256,128]

        # Classificazione finale
        base_out_logits = self.new_fc(base_out_pose)

        if not self.before_softmax:
            base_out_logits = self.softmax(base_out_logits)
            
        base_out_logits = base_out_logits.view((-1, self.num_segments) + base_out_logits.size()[1:]).cuda()
        output = self.consensus(base_out_logits).cuda()
        end_time = time.time()

        return output
    
    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self):
        return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                               GroupRandomHorizontalFlip(is_flow=False,
                                                                         target_transform=self.target_transform)])