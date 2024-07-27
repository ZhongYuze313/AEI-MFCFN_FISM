import torch
from torch import nn, einsum
import torch.nn.functional as F
from argparse import Namespace
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from longformer import Longformer
from linformer import Linformer
from transformer import Transformer
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torchvision import transforms
from network import rgbD_net, conv_layer


class VTN(nn.Module):
    def __init__(self, *, frames, num_classes, img_size, patch_size, spatial_frozen, spatial_size, spatial_args, temporal_type, temporal_args, spatial_suffix=''):
        super(VTN, self).__init__()
        self.BCFM = rgbD_net()
        self.conv = conv_layer(chann_in=6, chann_out=3, k_size=3, p_size=1)
        self.frames = frames
        # print(frames, num_classes, img_size,patch_size,spatial_frozen,spatial_size,spatial_args, temporal_type, temporal_args)

        # Convert args
        spatial_args = Namespace(**spatial_args)
        temporal_args = Namespace(**temporal_args)

        self.collapse_frames = Rearrange('b f c h w -> (b f) c h w')

        #[Spatial] Transformer attention 
        self.spatial_transformer = timm.create_model(f'vit_{spatial_size}_patch{patch_size}_{img_size}{spatial_suffix}', pretrained=False, **vars(spatial_args))
        
        # Freeze spatial backbone
        self.spatial_frozen = spatial_frozen
        if spatial_frozen:
          self.spatial_transformer.eval()
        # Spatial preprocess
        self.preprocess = transforms.Compose([
          transforms.Resize(256),
          transforms.RandomCrop(img_size),
          #transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize(mean=self.spatial_transformer.default_cfg['mean'], std=self.spatial_transformer.default_cfg['std'])
        ])
        # Spatial Training preprocess
        config = resolve_data_config({}, model=self.spatial_transformer)
        self.train_preprocess = create_transform(**config, is_training=True)

       
        #Spatial to temporal rearrange
        self.spatial2temporal = Rearrange('(b f) d -> b f d', f=frames)

        #[Temporal] Transformer_attention
        assert temporal_type in ['longformer', 'linformer', 'transformer'], "Only longformer, linformer, transformer are supported"
        # Copy seq_len to frames
        temporal_args.seq_len = frames
        
        if temporal_type == 'longformer':
          self.temporal_transformer = Longformer(**vars(temporal_args))
        elif temporal_type == 'linformer':
          self.temporal_transformer = Linformer(**vars(temporal_args))
        elif temporal_type == 'transformer':
          self.temporal_transformer = Transformer(**vars(temporal_args))

        # Classifer
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(temporal_args.dim),
            nn.Linear(temporal_args.dim, num_classes)
        )
        # Random init 0.0 mean, 0.02 std
        # nn.init.normal_(self.mlp_head[1].weight, mean=0.0, std=0.02)

    def forward(self, rgbl, rgbr):
        f = rgbl.shape[1]
        # binocular froth image fusion
        for ff in range(f):
            rgbl_f = rgbl[:, ff, :, :, :]
            rgbr_f = rgbr[:, ff, :, :, :]
            fusion_feature_f, pre_FISM = self.BCFM(rgbl_f, rgbr_f)
            # fusion_feature_f = self.conv(fusion_feature_f)
            fusion_feature_f = torch.unsqueeze(fusion_feature_f, dim=1)
            pre_FISM = torch.unsqueeze(pre_FISM, dim=1)
            # print('fusion_feature_f', fusion_feature_f.shape)
            if ff == 0:
                fusion_feature = fusion_feature_f
                all_FISM = pre_FISM
            else:
                fusion_feature = torch.cat((fusion_feature, fusion_feature_f), dim=1)
                all_FISM = torch.cat((all_FISM, pre_FISM), dim=1)
        x = self.collapse_frames(fusion_feature)
        
        # Spatial Transformer
        if self.spatial_frozen:
          with torch.no_grad():
            x = self.spatial_transformer.forward_features(x)[:, 0]
        else:
          x = self.spatial_transformer.forward_features(x)[:, 0]
  
        # Spatial to temporal
        # print(x.shape)
        x = self.spatial2temporal(x)

        # Temporal Transformer
        x = self.temporal_transformer(x)

        # Classifier
        # return self.mlp_head(x), all_FISM, x1, x2, x3, fusion_feature_f
        return self.mlp_head(x), all_FISM
        # return self.mlp_head(x)
