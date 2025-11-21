import torch
import timm
import numpy as np

from einops import repeat, rearrange
from einops.layers.torch import Rearrange
from timm.layers import DropPath

from timm.models.layers import trunc_normal_, Mlp
from timm.models.vision_transformer import Block
from torch import nn
# from torchsummary import summary
import torch.nn.functional as F

from utils.utils import shuffle_and_split_mask, generate_mask


def random_indexes(size : int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes

def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))

class PatchShuffle(nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, patches : torch.Tensor):
        T, B, C = patches.shape
        remain_T = int(T * (1 - self.ratio))

        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)

        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]

        return patches, forward_indexes, backward_indexes

class PositionEncoding(nn.Module):
    def __init__(self, base_size=(28, 28), channel=128):
        super(PositionEncoding, self).__init__()
        self.base_size = base_size
        self.channel = channel
        self.position_encoding = self.generate_position_encoding(base_size)

    def generate_position_encoding(self, size):
        x, y = size
        position_encoding = np.zeros((x, y))
        for i in range(x):
            for j in range(y):
                position_encoding[i, j] = i / x + j / y

        position_encoding = np.tile(position_encoding, (self.channel, 1, 1))
        return torch.tensor(position_encoding, dtype=torch.float32).unsqueeze(0)

    def forward(self, x):
        B, C, H, W = x.size()
        position_encoding = nn.functional.interpolate(self.position_encoding.to(x.device), size=(H, W), mode='bilinear',
                                                      align_corners=True)
        return position_encoding


class MAE_Encoder(torch.nn.Module):
    def __init__(self,
                 image_size=[32, 32],
                 in_channel=512,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=12,
                 num_head=3,
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()

        self.emb_dim = emb_dim
        self.in_channel = in_channel
        self.patch_size = patch_size
        self.image_size = image_size

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))

        self.shuffle = PatchShuffle(mask_ratio)

        self.patchify = torch.nn.Conv2d(in_channel, emb_dim, patch_size, patch_size)

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)

    def forward(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')

        patches, forward_indexes, backward_indexes = self.shuffle(patches)

        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')

        return features, backward_indexes

class MAE_Decoder(torch.nn.Module):
    def __init__(self,
                 image_size=[32, 32],
                 in_channel=512,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=4,
                 num_head=3,
                 ) -> None:
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        # self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size[0] // patch_size) * (image_size[1] // patch_size) + 1, 1, emb_dim))
        self.patch_size = patch_size
        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])
        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.head = torch.nn.Linear(emb_dim, in_channel * patch_size ** 2, bias=True)
        # self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=image_size[0]//patch_size)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        # trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes, h):
        T = features.shape[0]

        backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)
        features = torch.cat([features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)], dim=0)
        features = take_indexes(features, backward_indexes)
        # features = features + self.pos_embedding

        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = self.layer_norm(features)
        features = rearrange(features, 'b t c -> t b c')
        features = features[1:] # remove global feature

        patches = self.head(features)
        mask = torch.zeros_like(patches)
        mask[T-1:] = 1
        mask = take_indexes(mask, backward_indexes[1:] - 1)
        patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=self.patch_size, p2=self.patch_size,
                              h=h // self.patch_size)
        img = patch2img(patches)
        mask = patch2img(mask)

        return img, mask

class MAE_ViT(torch.nn.Module):
    def __init__(self,
                 image_size=[32, 32],
                 in_channel=512,
                 patch_size=2,
                 emb_dim=192,
                 encoder_layer=12,
                 encoder_head=3,
                 decoder_layer=4,
                 decoder_head=3,
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()

        self.encoder = MAE_Encoder(image_size, in_channel, patch_size, emb_dim, encoder_layer, encoder_head, mask_ratio)
        self.decoder = MAE_Decoder(image_size, in_channel, patch_size, emb_dim, decoder_layer, decoder_head)

    def forward(self, img):
        features, backward_indexes = self.encoder(img)
        predicted_img, mask = self.decoder(features,  backward_indexes, img.shape[2])
        return predicted_img, mask


class ViT_Classifier_v4(torch.nn.Module):
    def __init__(self, mae: MAE_ViT, base_size=(28, 28)) -> None:
        super().__init__()
        self.encoder = mae.encoder
        self.patchify = self.encoder.patchify
        self.transformer = self.encoder.transformer
        self.layer_norm = self.encoder.layer_norm
        self.head = torch.nn.Linear(self.encoder.emb_dim, self.encoder.in_channel * self.encoder.patch_size ** 2)
        self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=self.encoder.patch_size, p2=self.encoder.patch_size, h=self.encoder.image_size[0]//self.encoder.patch_size)

        self.pos_embedding = PositionEncoding(base_size=base_size, channel=self.encoder.emb_dim)

    def forward(self, img, mask, hf_feat):
        patches = self.patchify(img)  # [B, C, N, N]
        patches = patches * mask if self.training else patches

        hf_feat = self.patchify(hf_feat)
        hf_feat = hf_feat * mask if self.training else hf_feat

        patches = patches + self.pos_embedding(patches)
        patches = rearrange(patches, 'b c h w -> (h w) b c')  # [N*N, B,C]

        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')
        patches = self.head(features)

        self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=self.encoder.patch_size,
                                   p2=self.encoder.patch_size,
                                   h=img.shape[2] // self.encoder.patch_size)
        img = self.patch2img(patches)

        return img, 0


class ViT_Classifier_v3(torch.nn.Module):
    def __init__(self, mae : MAE_ViT, num_splits=3, base_size=(28, 28)) -> None:
        super().__init__()
        self.num_splits = num_splits
        self.encoder = mae.encoder
        self.patchify = self.encoder.patchify
        self.transformer = self.encoder.transformer
        self.layer_norm = self.encoder.layer_norm
        self.head = torch.nn.Linear(self.encoder.emb_dim, self.encoder.in_channel * self.encoder.patch_size ** 2)
        self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=self.encoder.patch_size, p2=self.encoder.patch_size, h=self.encoder.image_size[0]//self.encoder.patch_size)

        self.pos_embedding = PositionEncoding(base_size=base_size, channel=self.encoder.emb_dim)

        self.weights = nn.Parameter(torch.full([self.num_splits], 1 / self.num_splits))
        self.weights.data.div_(self.weights.sum())

    def forward(self, img, mask, hf_feat):
        patches = self.patchify(img)  # [B, C, N, N]
        hf_feat = self.patchify(hf_feat)

        if self.training:
            patches_recover = []
            img_recover = []
            split_masks = shuffle_and_split_mask(mask, self.num_splits)
            for m in split_masks:
                p = patches * m
                hf = hf_feat * m
                p = p + self.pos_embedding(p) + hf
                p = rearrange(p, 'b c h w -> (h w) b c')  # [N*N, B,C]

                p = rearrange(p, 't b c -> b t c')
                features = self.layer_norm(self.transformer(p))
                patches_recover.append(rearrange(features, 'b (p1 p2) c -> b c p1 p2', p1=patches.shape[2]))
                features = rearrange(features, 'b t c -> t b c')

                p = self.head(features)
                img_recover.append(self.patch2img(p))

            loss = 0
            for i, xpi in enumerate(patches_recover):
                for j, xpj in enumerate(patches_recover):
                    if i != j:
                        sij = (1 - split_masks[i]) * (1 - split_masks[j])
                        loss += self.self_consistency_loss(xpi, xpj, sij)

            img_recover = sum([img_recover[i] * weight for i, weight in enumerate(self.weights)]) if self.num_splits != 1 else img_recover[0]

            return img_recover, loss
        else:
            patches = patches + self.pos_embedding(patches)
            patches = rearrange(patches, 'b c h w -> (h w) b c')  # [N*N, B,C]

            patches = rearrange(patches, 't b c -> b t c')
            features = self.layer_norm(self.transformer(patches))
            features = rearrange(features, 'b t c -> t b c')
            patches = self.head(features)

            self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=self.encoder.patch_size,
                                       p2=self.encoder.patch_size,
                                       h=img.shape[2] // self.encoder.patch_size)
            img = self.patch2img(patches)
            return img, 0

    def self_consistency_loss(self, xpi, xpj, sij):
        # 计算自一致性损失
        loss_1 = F.l1_loss(xpi.detach(), xpj, reduction='none')
        loss_2 = F.l1_loss(xpi, xpj.detach(), reduction='none')

        # 仅在 sij 掩码为1的位置计算损失
        masked_loss_1 = loss_1 * sij
        masked_loss_2 = loss_2 * sij

        # 最终损失为所有掩码位置的损失之和
        final_loss = masked_loss_1.mean() + masked_loss_2.mean()

        return final_loss
