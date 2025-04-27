import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from models.components import vit_seg_configs as configs
from models.components.vit_seg_modeling_resnet_skip import ResNetV2
from einops import rearrange, repeat
from .model_utils import *
from .CTM_utils import *


def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores) # b 12 256 256
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

# class Block(nn.Module):
#     def __init__(self, config, vis):
#         super(Block, self).__init__()
#         self.hidden_size = config.hidden_size
#         self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
#         self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
#         self.ffn = Mlp(config)
#         self.attn = Attention(config, vis)
#
#     def forward(self, x):
#         h = x
#         x = self.attention_norm(x)
#         x, weights = self.attn(x)
#         x = x + h
#
#         h = x
#         x = self.ffn_norm(x)
#         x = self.ffn(x)
#         x = x + h
#         return x, weights

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:   # ResNet
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])


    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x) # return [b, 512, 32, 32], [b, 256, 64, 64, ], [b, 64, 128, 128], x = [b, 1024, 16, 16]
        else:
            features = None
        x = self.patch_embeddings(x)  # (B, hidden, n_patches^(1/2), n_patches^(1/2)) # i.e., [b, 768, 16, 16]
        x = x.flatten(2) # [b, 768, 256]
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features

class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights

class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels)
        super(Conv2dReLU, self).__init__(conv, bn, relu)

class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)

class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x

class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder_CTM()

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        # encoded, _ = self.encoder(embedding_output)  # (B, n_patch, hidden)
        encoded = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, features
    
    def infere(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, ftokens, attmaps = self.encoder.infere(embedding_output)  # (B, n_patch, hidden)
        return encoded, features, ftokens, attmaps

class Transformer_DTMFormerV2(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer_DTMFormerV2, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder_DTMFormerV2()
        # self.encoder = Encoder_DTMFormerV2_changePreNorm()

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        # encoded, _ = self.encoder(embedding_output)  # (B, n_patch, hidden)
        encoded, as_out = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, features, as_out

    def infere(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, ftokens, attmaps = self.encoder.infere(embedding_output)  # (B, n_patch, hidden)
        return encoded, features, ftokens, attmaps

class Transformer_DTMFormerV2_attnloss(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer_DTMFormerV2_attnloss, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        # self.encoder = Encoder_DTMFormerV2()
        # self.encoder = Encoder_DTMFormerV2_changePreNorm()
        self.encoder = Encoder_DTMFormerV2_attnloss()

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        # encoded, _ = self.encoder(embedding_output)  # (B, n_patch, hidden)
        encoded, as_out, attns = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, features, as_out, attns

    def infere(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, ftokens, attmaps = self.encoder.infere(embedding_output)  # (B, n_patch, hidden)
        return encoded, features, ftokens, attmaps

class Transformer_DTMFormerV2_attnlossDS(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer_DTMFormerV2_attnlossDS, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        # self.encoder = Encoder_DTMFormerV2()
        # self.encoder = Encoder_DTMFormerV2_changePreNorm()
        self.encoder = Encoder_DTMFormerV2_attnlossDS()

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        # encoded, _ = self.encoder(embedding_output)  # (B, n_patch, hidden)
        encoded, as_out, attns, feature_DS = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, features, as_out, attns, feature_DS

    def infere(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, ftokens, attmaps = self.encoder.infere(embedding_output)  # (B, n_patch, hidden)
        return encoded, features, ftokens, attmaps

class Transformer_DTMFormerV2FirstStageChoose(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer_DTMFormerV2FirstStageChoose, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        # self.encoder = Encoder_DTMFormerV2()
        # self.encoder = Encoder_DTMFormerV2_changePreNorm()
        self.encoder = Encoder_DTMFormerV2FirstStageChoose(image_size=img_size)


    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        # encoded, _ = self.encoder(embedding_output)  # (B, n_patch, hidden)
        encoded, attnScores, attns, outs, variances, scores = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, features, attnScores, attns, outs, variances, scores

    def infere(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, ftokens, attmaps = self.encoder.infere(embedding_output)  # (B, n_patch, hidden)
        return encoded, features, ftokens, attmaps



class Transformer_smallCTMModule(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer_smallCTMModule, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder_smallCTMModule()

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        # encoded, _ = self.encoder(embedding_output)  # (B, n_patch, hidden)
        encoded = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, features

    def infere(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, ftokens, attmaps = self.encoder.infere(embedding_output)  # (B, n_patch, hidden)
        return encoded, features, ftokens, attmaps

class Transformer_smallCTMModule_AS(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer_smallCTMModule_AS, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder_smallCTMModule_AS(img_size=img_size)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        # encoded, _ = self.encoder(embedding_output)  # (B, n_patch, hidden)
        encoded, as_out = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, features, as_out

    def infere(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, ftokens, attmaps = self.encoder.infere(embedding_output)  # (B, n_patch, hidden)
        return encoded, features, ftokens, attmaps

class VisionTransformer_smallCTMModule(nn.Module):
    def __init__(self, config, img_size=256, num_classes=2, zero_head=False, vis=False):
        super(VisionTransformer_smallCTMModule, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer_smallCTMModule(config, img_size, vis)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=num_classes,
            kernel_size=3,
        )
        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        x, features = self.transformer(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits
    
    def infere(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        x, features, ftokens, attmaps = self.transformer.infere(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits, ftokens, attmaps

class VisionTransformer_smallCTMModule_AS(nn.Module):
    def __init__(self, config, img_size=256, num_classes=2, zero_head=False, vis=False):
        super(VisionTransformer_smallCTMModule_AS, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer_smallCTMModule_AS(config, img_size, vis)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=num_classes,
            kernel_size=3,
        )
        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, features, as_out = self.transformer(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits, as_out

    def infere(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, features, ftokens, attmaps = self.transformer.infere(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits, ftokens, attmaps

class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=256, num_classes=2, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=num_classes,
            kernel_size=3,
        )
        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, features = self.transformer(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits

    def infere(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, features, ftokens, attmaps = self.transformer.infere(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits, ftokens, attmaps

class VisionTransformer_DTMFormerV2(nn.Module):
    def __init__(self, config, img_size=256, num_classes=2, zero_head=False, vis=False):
        super(VisionTransformer_DTMFormerV2, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer_DTMFormerV2(config, img_size, vis)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=num_classes,
            kernel_size=3,
        )
        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, features, as_out = self.transformer(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits, as_out

    def infere(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, features, ftokens, attmaps = self.transformer.infere(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits, ftokens, attmaps

class VisionTransformer_DTMFormerV2_attnloss(nn.Module):
    def __init__(self, config, img_size=256, num_classes=2, zero_head=False, vis=False):
        super(VisionTransformer_DTMFormerV2_attnloss, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer_DTMFormerV2_attnloss(config, img_size, vis)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=num_classes,
            kernel_size=3,
        )
        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, features, as_out, attns = self.transformer(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits, as_out, attns

    def infere(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, features, ftokens, attmaps = self.transformer.infere(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits, ftokens, attmaps

class VisionTransformer_DTMFormerV2_attnlossDS(nn.Module):
    def __init__(self, config, img_size=256, num_classes=2, zero_head=False, vis=False):
        super(VisionTransformer_DTMFormerV2_attnlossDS, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer_DTMFormerV2_attnlossDS(config, img_size, vis)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=num_classes,
            kernel_size=3,
        )
        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, features, as_out, attns, feature_DS = self.transformer(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits, as_out, attns, feature_DS

    def infere(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, features, ftokens, attmaps = self.transformer.infere(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits, ftokens, attmaps

class VisionTransformer_DTMFormerV2FirstStageChoose(nn.Module):
    def __init__(self, config, img_size=256, num_classes=2, zero_head=False, vis=False):
        super(VisionTransformer_DTMFormerV2FirstStageChoose, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer_DTMFormerV2FirstStageChoose(config, img_size, vis)

        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=num_classes,
            kernel_size=3,
        )
        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, features, attnScores, attns, outs, variances, scores = self.transformer(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits, attnScores, attns, outs, variances, scores

    def infere(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, features, ftokens, attmaps = self.transformer.infere(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits, ftokens, attmaps




CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}

def TransUnet_Model_sparse(pretrained=False, vit_name='R50-ViT-B_16', vit_patches_size=16, num_classes=9, n_skip=3, img_size=256, **kwargs):
    #model = U2Transx(TransformerDown_SPrune, [2, 2, 2, 2], **kwargs)
    config_vit = CONFIGS[vit_name]
    config_vit.n_classes = num_classes
    config_vit.n_skip = n_skip
    if vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))
    model = VisionTransformer(config_vit, img_size=img_size, num_classes=num_classes)
    return model

def TransUnet_Model_smallCTMModule(pretrained=False, vit_name='R50-ViT-B_16', vit_patches_size=16, num_classes=9, n_skip=3, img_size=256, **kwargs):
    #model = U2Transx(TransformerDown_SPrune, [2, 2, 2, 2], **kwargs)
    config_vit = CONFIGS[vit_name]
    config_vit.n_classes = num_classes
    config_vit.n_skip = n_skip
    if vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))
    model = VisionTransformer_smallCTMModule(config_vit, img_size=img_size, num_classes=num_classes)
    return model

def TransUnet_Model_smallCTMModule_AS(pretrained=False, vit_name='R50-ViT-B_16', vit_patches_size=16, num_classes=9, n_skip=3, img_size=256, **kwargs):
    #model = U2Transx(TransformerDown_SPrune, [2, 2, 2, 2], **kwargs)
    config_vit = CONFIGS[vit_name]
    config_vit.n_classes = num_classes
    config_vit.n_skip = n_skip
    if vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))
    model = VisionTransformer_smallCTMModule_AS(config_vit, img_size=img_size, num_classes=num_classes)
    return model

def TransUnet_Model_DTMFormerV2(pretrained=False, vit_name='R50-ViT-B_16', vit_patches_size=16, num_classes=9, n_skip=3, img_size=256, **kwargs):
    #model = U2Transx(TransformerDown_SPrune, [2, 2, 2, 2], **kwargs)
    config_vit = CONFIGS[vit_name]
    config_vit.n_classes = num_classes
    config_vit.n_skip = n_skip
    if vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))
    model = VisionTransformer_DTMFormerV2(config_vit, img_size=img_size, num_classes=num_classes)
    return model

def TransUnet_Model_DTMFormerV2_attnloss(pretrained=False, vit_name='R50-ViT-B_16', vit_patches_size=16, num_classes=9, n_skip=3, img_size=256, **kwargs):
    #model = U2Transx(TransformerDown_SPrune, [2, 2, 2, 2], **kwargs)
    config_vit = CONFIGS[vit_name]
    config_vit.n_classes = num_classes
    config_vit.n_skip = n_skip
    if vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))
    model = VisionTransformer_DTMFormerV2_attnloss(config_vit, img_size=img_size, num_classes=num_classes)
    return model

def TransUnet_Model_DTMFormerV2_attnlossDS(pretrained=False, vit_name='R50-ViT-B_16', vit_patches_size=16, num_classes=9, n_skip=3, img_size=256, **kwargs):
    config_vit = CONFIGS[vit_name]
    config_vit.n_classes = num_classes
    config_vit.n_skip = n_skip
    if vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))
    model = VisionTransformer_DTMFormerV2_attnlossDS(config_vit, img_size=img_size, num_classes=num_classes)
    return model

def TransUnet_DTMFormerV2FirstStageChoose(pretrained=False, vit_name='R50-ViT-B_16', vit_patches_size=16, num_classes=9, n_skip=3, img_size=256, **kwargs):
    config_vit = CONFIGS[vit_name]
    config_vit.n_classes = num_classes
    config_vit.n_skip = n_skip
    if vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))
    model = VisionTransformer_DTMFormerV2FirstStageChoose(config_vit, img_size=img_size, num_classes=num_classes)
    return model




# change Line 358/ Line 338 to select the oversmooth method

# ======================================== some modules for the anti oversmoothing methods ========================================
from .components.anti_over_smoothing import Transformer_Vanilla, Transformer_Refiner, Transformer_Layerscale, Transformer_Reattention

class Encoder_AOS_sameconfig(nn.Module):
    def __init__(self, config, vis):
        super(Encoder_AOS_sameconfig, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        dim = 512
        heads = 8
        dim_head = 64
        dropout = config.transformer["attention_dropout_rate"]
        mlp_dim = 2048
        depth = 12
        num_patches = (256//16) * (256//16)
        self.transformer = Transformer_Vanilla(dim, depth, heads, dim_head, mlp_dim, dropout, num_patches)

    def forward(self, hidden_states):
        attn_weights = []
        hidden_states = self.transformer(hidden_states)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights
    
    def infere(self, hidden_states0):
        hidden_states, ftokens, attmaps = self.transformer.infere(hidden_states0)
        ftokens.insert(0, hidden_states0)
        encoded = self.encoder_norm(hidden_states)
        return encoded, ftokens, attmaps

from einops.layers.torch import Rearrange

class Encoder_CTM(nn.Module):
    def __init__(
            self, img_size=256, in_chans=768, out_chans=768, embed_dims=[96, 192, 384, 768],
            num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False,
            qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
            norm_layer=nn.LayerNorm, depths=[3, 3, 3, 3], sr_ratios=[8, 4, 2, 1],
            # norm_layer=nn.LayerNorm, depths=[1, 1, 1, 9], sr_ratios=[8, 4, 2, 1],
            num_stages=4, pretrained=None,
            k=5, sample_ratios=[0.5, 0.5, 0.5],
            return_map=False,):
        super().__init__()


        self.depths = depths
        self.num_stages = num_stages
        self.grid_stride = sr_ratios[0]
        self.embed_dims = embed_dims
        self.sr_ratios = sr_ratios
        self.mlp_ratios = mlp_ratios
        self.sample_ratios = sample_ratios
        self.return_map = return_map
        self.in_channs = in_chans
        self.k = k

        self.to_patch_embedding = nn.Sequential(
            nn.Linear(in_chans, embed_dims[0]),
        )
        self.from_patch_embedding = nn.Sequential(
            Rearrange('b s c -> b c s'),
        )

        # self.recover = MTA_TransUnet()
        self.recover = MTA_TransUnet_light()



        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.cur = 0

        self.stage0= nn.ModuleList([Block(
                dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[0])
                for j in range(depths[0])])
        self.norm0 = norm_layer(embed_dims[0])

        self.ctm1 = CTM(self.sample_ratios[0], self.embed_dims[0], self.embed_dims[1], self.k)
        self.stage1 = nn.ModuleList([TCBlock(
                dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[1])
                for j in range(depths[1])])
        self.norm1 = norm_layer(embed_dims[1])

        self.ctm2 = CTM(self.sample_ratios[1], self.embed_dims[1], self.embed_dims[2], self.k)
        self.stage2 = nn.ModuleList([TCBlock(
                dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[2])
                for j in range(depths[2])])
        self.norm2 = norm_layer(embed_dims[2])

        self.ctm3 = CTM(self.sample_ratios[2], self.embed_dims[2], self.embed_dims[3], self.k)
        self.stage3 = nn.ModuleList([TCBlock(
                dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[3])
                for j in range(depths[3])])
        self.norm3 = norm_layer(embed_dims[3])


    def forward(self, img):
        x = self.to_patch_embedding(img)
        outs = []
        H = 16
        W = 16

        # encoder:stage0
        # x = self.stage1(x)
        for blk in self.stage0:
            x = blk(x, H, W)
        x = self.norm0(x)
        self.cur += self.depths[0]

        # init token dict
        B, N, _ = x.shape
        idx_token = torch.arange(N)[None, :].repeat(B, 1).cuda()
        agg_weight = x.new_ones(B, N, 1)
        token_dict = {'x': x,
                      'token_num': N,
                      'map_size': [H, W],
                      'init_grid_size': [H, W],
                      'idx_token': idx_token,
                      'agg_weight': agg_weight}
        outs.append(token_dict.copy())

        # encoder:stage1
        token_dict = self.ctm1(token_dict)
        # token_dict = self.stage1(token_dict)
        for j, blk in enumerate(self.stage1):
            token_dict = blk(token_dict)
        token_dict['x'] = self.norm1(token_dict['x'])
        self.cur += self.depths[1]
        outs.append(token_dict)

        # encoder:stage2
        token_dict = self.ctm2(token_dict)
        # token_dict = self.stage2(token_dict)
        for j, blk in enumerate(self.stage2):
            token_dict = blk(token_dict)
        token_dict['x'] = self.norm2(token_dict['x'])
        self.cur += self.depths[2]
        outs.append(token_dict)

        # encoder:stage3
        token_dict = self.ctm3(token_dict)
        # token_dict = self.stage3(token_dict)
        for j, blk in enumerate(self.stage3):
            token_dict = blk(token_dict)
        token_dict['x'] = self.norm3(token_dict['x'])
        self.cur += self.depths[3]
        outs.append(token_dict)

        # MTA and recover
        x = self.recover.forward(outs)  # torch.Size([4, 1024, 128])

        return x

class Encoder_smallCTMModule(nn.Module):
    def __init__(
            self, img_size=256, patch_size=8,in_chans=768, embed_dims=[384, 768],
            num_heads=[4, 8], mlp_ratios=[4, 4], qkv_bias=False,
            qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
            norm_layer=nn.LayerNorm, depths=[2, 2], total_depth=12, sr_ratios=[8, 4],
            num_stages=4, pretrained=None,
            k=5, sample_ratios=0.125,
            return_map=False,):
        super().__init__()

        self.total_depth = total_depth
        self.depths = depths
        self.num_stages = num_stages
        self.grid_stride = sr_ratios[0]
        self.embed_dims = embed_dims
        self.sr_ratios = sr_ratios
        self.mlp_ratios = mlp_ratios
        self.sample_ratios = sample_ratios
        self.return_map = return_map
        self.in_channs = in_chans
        self.k = k


        self.to_patch_embedding = nn.Sequential(
            nn.Linear(in_chans, embed_dims[0]),
        )
        self.change_embedding_dim = nn.Sequential(
            nn.Linear(embed_dims[0], embed_dims[1]),
        )

        self.recover = MTA_TransUnet_light_smallModule()



        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.cur = 0

        self.stage0= nn.ModuleList([Block(
                dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[0])
                for j in range(depths[0])])
        self.norm0 = norm_layer(embed_dims[0])

        self.ctm1 = CTM(self.sample_ratios, self.embed_dims[0], self.embed_dims[1], self.k)
        self.stage1 = nn.ModuleList([TCBlock(
                dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[1])
                for j in range(depths[1])])
        self.norm1 = norm_layer(embed_dims[1])


    def forward(self, img):
        x = self.to_patch_embedding(img)

        for i in range(int(self.total_depth/sum(self.depths))):
            outs = []
            H = 16
            W = 16

            for blk in self.stage0:
                x = blk(x, H, W)
            x = self.norm0(x)
            self.cur += self.depths[0]

            B, N, _ = x.shape
            idx_token = torch.arange(N)[None, :].repeat(B, 1).cuda()
            agg_weight = x.new_ones(B, N, 1)
            token_dict = {'x': x,
                          'token_num': N,
                          'map_size': [H, W],
                          'init_grid_size': [H, W],
                          'idx_token': idx_token,
                          'agg_weight': agg_weight}
            outs.append(token_dict.copy())

            # encoder:stage1
            token_dict = self.ctm1(token_dict)
            # token_dict = self.stage1(token_dict)
            for j, blk in enumerate(self.stage1):
                token_dict = blk(token_dict)
            token_dict['x'] = self.norm1(token_dict['x'])
            self.cur += self.depths[1]
            outs.append(token_dict)

            x = self.recover.forward(outs)

        x = self.change_embedding_dim(x)

        return x

class Encoder_smallCTMModule_AS(nn.Module):
    def __init__(
            self, img_size=256, patch_size=8,in_chans=768, embed_dims=[384, 768],
            num_heads=[4, 8], mlp_ratios=[4, 4], qkv_bias=False,
            qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
            norm_layer=nn.LayerNorm, depths=[2, 2], as_depth=[1, 1], total_depth=12, sr_ratios=[8, 4],
            num_stages=4, pretrained=None,
            k=5, sample_ratios=0.125,
            return_map=False,):
        super().__init__()

        self.total_depth = total_depth
        self.depths = depths
        self.as_depth = as_depth
        self.num_stages = num_stages
        self.grid_stride = sr_ratios[0]
        self.embed_dims = embed_dims
        self.sr_ratios = sr_ratios
        self.mlp_ratios = mlp_ratios
        self.sample_ratios = sample_ratios
        self.return_map = return_map
        self.in_channs = in_chans
        self.k = k
        # self.patch_num = int(img_size/patch_size)
        self.patch_num = 16


        self.to_patch_embedding = nn.Sequential(
            nn.Linear(in_chans, embed_dims[0]),
        )
        self.change_embedding_dim = nn.Sequential(
            nn.Linear(embed_dims[0], embed_dims[1]),
        )

        self.recover = MTA_TransUnet_light_smallModule()



        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.cur = 0

        self.stage0_as = nn.ModuleList([Block_AS(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=1)
            for j in range(as_depth[0])])
        self.norm0_as = norm_layer(embed_dims[0])
        self.ctm_as = CTM_as(self.sample_ratios, self.embed_dims[0], self.embed_dims[1], self.k)
        self.stage1_as = nn.ModuleList([TCBlock(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for j in range(as_depth[1])])
        self.norm1_as = norm_layer(embed_dims[1])

        self.stage0 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for j in range(depths[0])])
        self.norm0 = norm_layer(embed_dims[0])
        self.ctm1 = CTM_as(self.sample_ratios, self.embed_dims[0], self.embed_dims[1], self.k)
        self.stage1 = nn.ModuleList([TCBlock(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for j in range(depths[1])])
        self.norm1 = norm_layer(embed_dims[1])


    def forward(self, img):
        x = self.to_patch_embedding(img)

        outs = []
        H = self.patch_num
        W = self.patch_num

        for blk in self.stage0_as:
            x, attn = blk(x, H, W)
        x = self.norm0_as(x)
        self.cur += self.as_depth[0]

        # attn :torch.Size([4, 4, 256, 256])    attn[:,0:2] :torch.Size([4, 2, 256, 256])
        attn_map = attn[:, 0:2].sum(dim=1).sum(dim=1)  # attn_map :torch.Size([4, 1024])

        # 归一化方法1：sigmoid-0.5 (0-0.5)
        # as_out = torch.sigmoid(attn_map) - 0.5
        # 归一化方法2：(sigmoid-0.5)*2 (0-1)
        # as_out = (torch.sigmoid(attn_map) - 0.5) * 2
        # 归一化方法3：均匀归一化 (0-1)
        b = attn_map.shape[0]
        n = attn_map.shape[1]
        min_as_out = torch.min(attn_map, dim=1)[0].repeat(n).reshape(n, b).transpose(0, 1)
        max_as_out = torch.max(attn_map, dim=1)[0].repeat(n).reshape(n, b).transpose(0, 1)
        as_out = (attn_map[:, ] - min_as_out) / (max_as_out - min_as_out)
        as_out = rearrange(as_out, 'b (h w)-> b h w', h=self.patch_num, w=self.patch_num)



        B, N, _ = x.shape
        idx_token = torch.arange(N)[None, :].repeat(B, 1).cuda()
        agg_weight = x.new_ones(B, N, 1)
        token_dict = {'x': x,
                      'token_num': N,
                      'map_size': [H, W],
                      'init_grid_size': [H, W],
                      'idx_token': idx_token,
                      'agg_weight': agg_weight}
        outs.append(token_dict.copy())

        token_dict = self.ctm_as(token_dict, as_out, ctm_stage=1)

        for j, blk in enumerate(self.stage1_as):
            token_dict = blk(token_dict)
        token_dict['x'] = self.norm1_as(token_dict['x'])
        self.cur += self.as_depth[1]
        outs.append(token_dict)

        x = self.recover.forward(outs)

        for i in range(int((self.total_depth - sum(self.as_depth)) / sum(self.depths))):
            outs = []
            H = self.patch_num
            W = self.patch_num

            for blk in self.stage0:
                x = blk(x, H, W)
            x = self.norm0(x)
            self.cur += self.depths[0]

            B, N, _ = x.shape
            idx_token = torch.arange(N)[None, :].repeat(B, 1).cuda()
            agg_weight = x.new_ones(B, N, 1)
            token_dict = {'x': x,
                          'token_num': N,
                          'map_size': [H, W],
                          'init_grid_size': [H, W],
                          'idx_token': idx_token,
                          'agg_weight': agg_weight}
            outs.append(token_dict.copy())

            # encoder:stage1
            token_dict = self.ctm1(token_dict, as_out, ctm_stage=i+2)
            # token_dict = self.stage1(token_dict)
            for j, blk in enumerate(self.stage1):
                token_dict = blk(token_dict)
            token_dict['x'] = self.norm1(token_dict['x'])
            self.cur += self.depths[1]
            outs.append(token_dict)

            x = self.recover.forward(outs)

        x = self.change_embedding_dim(x)

        return x, as_out

class Encoder_DTMFormerV2(nn.Module):
    def __init__(
            self, in_channels=768, image_size=256, out_channels=768, total_depth=12,
            patch_size=16, embed_dims=[192, 384, 768], num_heads=[2, 4, 8], mlp_ratios=[4, 4, 4],
            qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
            drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[2, 2, 4],
            # sr_ratios=[4, 2, 1], pretrained=None,
            sr_ratios=[1, 1, 1], pretrained=None,
            k=5, sample_ratios=0.25, classes=4,
            return_map=False, ):
        super().__init__()

        self.total_depth = total_depth
        self.depths = depths
        self.grid_stride = sr_ratios[0]
        self.embed_dims = embed_dims
        self.sr_ratios = sr_ratios
        self.mlp_ratios = mlp_ratios
        self.sample_ratios = sample_ratios
        self.return_map = return_map
        self.in_channels = in_channels
        self.k = k
        self.classes = classes
        self.patch_num = int(image_size/patch_size)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.to_patch_embedding = nn.Sequential(
            nn.Linear(self.in_channels, embed_dims[0]),
        )
        self.from_patch_embedding = nn.Sequential(
            nn.Linear(embed_dims[0], self.out_channels),
        )


        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.cur = 0

        self.stage0_as = nn.ModuleList([UTSTransformer(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=1)
            for j in range(self.depths[0])])
        self.norm0_as = norm_layer(embed_dims[0])

        self.ctm1 = ATM(self.sample_ratios, self.embed_dims[0], self.embed_dims[1], self.k)
        self.stage1 = nn.ModuleList([UTSBlock(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for j in range(self.depths[1])])
        self.norm1 = norm_layer(embed_dims[1])

        self.ctm2 = ATM(self.sample_ratios, self.embed_dims[1], self.embed_dims[2], self.k)
        self.stage2 = nn.ModuleList([UTSBlock(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for j in range(self.depths[2])])
        self.norm2 = norm_layer(embed_dims[2])

        self.ltr3 = FAT_PTRV2(in_channels=[embed_dims[1], embed_dims[2]], out_channels=embed_dims[1])
        self.stage3 = nn.ModuleList([UTSBlock(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for j in range(self.depths[1])])
        self.norm3 = norm_layer(embed_dims[1])

        self.ltr4 = FAT_PTRV2(in_channels=[embed_dims[0], embed_dims[1]], out_channels=embed_dims[0])
        self.stage4 = nn.ModuleList([UTSBlock(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=1)
            for j in range(self.depths[0])])
        self.norm4 = norm_layer(embed_dims[0])



    def forward(self, img):
        x = self.to_patch_embedding(img)

        outs = []
        for blk in self.stage0_as:
            x, attn = blk(x, self.patch_num, self.patch_num)
        x = self.norm0_as(x)
        self.cur += self.depths[0]

        attn_map = attn[:, 0:2].sum(dim=1).sum(dim=1)
        b = attn_map.shape[0]
        n = attn_map.shape[1]
        min_as_out = torch.min(attn_map, dim=1)[0].repeat(n).reshape(n, b).transpose(0, 1)
        max_as_out = torch.max(attn_map, dim=1)[0].repeat(n).reshape(n, b).transpose(0, 1)
        attnScore = (attn_map[:, ] - min_as_out) / (max_as_out - min_as_out)
        attnScore = rearrange(attnScore, 'b (h w)-> b h w', h=self.patch_num, w=self.patch_num)

        B, N, _ = x.shape
        idx_token = torch.arange(N)[None, :].repeat(B, 1).cuda()
        agg_weight = x.new_ones(B, N, 1)
        token_dict = {'x': x,
                      'token_num': N,
                      'map_size': [self.patch_num, self.patch_num],
                      'init_grid_size': [self.patch_num, self.patch_num],
                      'idx_token': idx_token,
                      'agg_weight': agg_weight}
        outs.append(token_dict.copy())

        token_dict = self.ctm1(token_dict, attnScore, ctm_stage=1)
        for j, blk in enumerate(self.stage1):
            token_dict = blk(token_dict)
        token_dict['x'] = self.norm1(token_dict['x'])
        self.cur += self.depths[1]
        outs.append(token_dict)

        token_dict = self.ctm2(token_dict, attnScore, ctm_stage=2)
        for blk in self.stage2:
            token_dict = blk(token_dict)
        token_dict['x'] = self.norm2(token_dict['x'])
        self.cur += self.depths[2]
        outs.append(token_dict)



        token_dict = outs[1]
        token_dict['x'] = self.ltr3([outs[1], outs[2]])
        for blk in self.stage3:
            token_dict = blk(token_dict)
        token_dict['x'] = self.norm3(token_dict['x'])
        self.cur += self.depths[1]
        outs.append(token_dict)

        token_dict = outs[0]
        token_dict['x'] = self.ltr4([outs[0], outs[3]])
        for blk in self.stage4:
            token_dict = blk(token_dict)
        x = self.norm4(token_dict['x'])


        x = self.from_patch_embedding(x)  # b c h*w


        return x, attnScore

class Encoder_DTMFormerV2_changePreNorm(nn.Module):
    def __init__(
            self, in_channels=768, image_size=256, out_channels=768, total_depth=12,
            patch_size=16, embed_dims=[192, 384, 768], num_heads=[2, 4, 8], mlp_ratios=[4, 4, 4],
            qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
            drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[2, 2, 4],
            # sr_ratios=[4, 2, 1], pretrained=None,
            sr_ratios=[1, 1, 1], pretrained=None,
            k=5, sample_ratios=0.25, classes=4,
            return_map=False, ):
        super().__init__()

        self.total_depth = total_depth
        self.depths = depths
        self.grid_stride = sr_ratios[0]
        self.embed_dims = embed_dims
        self.sr_ratios = sr_ratios
        self.mlp_ratios = mlp_ratios
        self.sample_ratios = sample_ratios
        self.return_map = return_map
        self.in_channels = in_channels
        self.k = k
        self.classes = classes
        self.patch_num = int(image_size/patch_size)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.to_patch_embedding = nn.Sequential(
            nn.Linear(self.in_channels, embed_dims[0]),
        )
        self.from_patch_embedding = nn.Sequential(
            nn.Linear(embed_dims[0], self.out_channels),
        )


        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.cur = 0

        self.stage0_as = nn.ModuleList([UTSTransformer_changePreNorm(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=1)
            for j in range(self.depths[0])])
        self.norm0_as = norm_layer(embed_dims[0])

        self.ctm1 = ATM(self.sample_ratios, self.embed_dims[0], self.embed_dims[1], self.k)
        self.stage1 = nn.ModuleList([UTSBlock_changePreNorm(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for j in range(self.depths[1])])
        self.norm1 = norm_layer(embed_dims[1])

        self.ctm2 = ATM(self.sample_ratios, self.embed_dims[1], self.embed_dims[2], self.k)
        self.stage2 = nn.ModuleList([UTSBlock_changePreNorm(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for j in range(self.depths[2])])
        self.norm2 = norm_layer(embed_dims[2])

        self.ltr3 = FAT_PTRV2(in_channels=[embed_dims[1], embed_dims[2]], out_channels=embed_dims[1])
        self.stage3 = nn.ModuleList([UTSBlock_changePreNorm(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for j in range(self.depths[1])])
        self.norm3 = norm_layer(embed_dims[1])

        self.ltr4 = FAT_PTRV2(in_channels=[embed_dims[0], embed_dims[1]], out_channels=embed_dims[0])
        self.stage4 = nn.ModuleList([UTSBlock_changePreNorm(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=1)
            for j in range(self.depths[0])])
        self.norm4 = norm_layer(embed_dims[0])



    def forward(self, img):
        x = self.to_patch_embedding(img)

        outs = []
        for blk in self.stage0_as:
            x, attn = blk(x, self.patch_num, self.patch_num)
        x = self.norm0_as(x)
        self.cur += self.depths[0]

        attn_map = attn[:, 0:2].sum(dim=1).sum(dim=1)
        b = attn_map.shape[0]
        n = attn_map.shape[1]
        min_as_out = torch.min(attn_map, dim=1)[0].repeat(n).reshape(n, b).transpose(0, 1)
        max_as_out = torch.max(attn_map, dim=1)[0].repeat(n).reshape(n, b).transpose(0, 1)
        attnScore = (attn_map[:, ] - min_as_out) / (max_as_out - min_as_out)
        attnScore = rearrange(attnScore, 'b (h w)-> b h w', h=self.patch_num, w=self.patch_num)

        B, N, _ = x.shape
        idx_token = torch.arange(N)[None, :].repeat(B, 1).cuda()
        agg_weight = x.new_ones(B, N, 1)
        token_dict = {'x': x,
                      'token_num': N,
                      'map_size': [self.patch_num, self.patch_num],
                      'init_grid_size': [self.patch_num, self.patch_num],
                      'idx_token': idx_token,
                      'agg_weight': agg_weight}
        outs.append(token_dict.copy())

        token_dict = self.ctm1(token_dict, attnScore, ctm_stage=1)
        for j, blk in enumerate(self.stage1):
            token_dict = blk(token_dict)
        token_dict['x'] = self.norm1(token_dict['x'])
        self.cur += self.depths[1]
        outs.append(token_dict)

        token_dict = self.ctm2(token_dict, attnScore, ctm_stage=2)
        for blk in self.stage2:
            token_dict = blk(token_dict)
        token_dict['x'] = self.norm2(token_dict['x'])
        self.cur += self.depths[2]
        outs.append(token_dict)



        token_dict = outs[1]
        token_dict['x'] = self.ltr3([outs[1], outs[2]])
        for blk in self.stage3:
            token_dict = blk(token_dict)
        token_dict['x'] = self.norm3(token_dict['x'])
        self.cur += self.depths[1]
        outs.append(token_dict)

        token_dict = outs[0]
        token_dict['x'] = self.ltr4([outs[0], outs[3]])
        for blk in self.stage4:
            token_dict = blk(token_dict)
        x = self.norm4(token_dict['x'])


        x = self.from_patch_embedding(x)  # b c h*w


        return x, attnScore

class Encoder_DTMFormerV2_attnloss(nn.Module):
    def __init__(
            self, in_channels=768, image_size=256, out_channels=768, total_depth=12,
            patch_size=16, embed_dims=[192, 384, 768], num_heads=[2, 4, 8], mlp_ratios=[4, 4, 4],
            qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
            drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[2, 2, 4],
            # sr_ratios=[4, 2, 1], pretrained=None,
            sr_ratios=[1, 1, 1], pretrained=None,
            k=5, sample_ratios=0.25, classes=4,
            return_map=False, ):
        super().__init__()

        self.total_depth = total_depth
        self.depths = depths
        self.grid_stride = sr_ratios[0]
        self.embed_dims = embed_dims
        self.sr_ratios = sr_ratios
        self.mlp_ratios = mlp_ratios
        self.sample_ratios = sample_ratios
        self.return_map = return_map
        self.in_channels = in_channels
        self.k = k
        self.classes = classes
        self.patch_num = int(image_size/patch_size)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.to_patch_embedding = nn.Sequential(
            nn.Linear(self.in_channels, embed_dims[0]),
        )
        self.from_patch_embedding = nn.Sequential(
            nn.Linear(embed_dims[0], self.out_channels),
        )


        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.cur = 0

        self.stage0_as = nn.ModuleList([UTSTransformer(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=1)
            for j in range(self.depths[0])])
        self.norm0_as = norm_layer(embed_dims[0])

        self.ctm1 = ATM(self.sample_ratios, self.embed_dims[0], self.embed_dims[1], self.k)
        self.stage1 = nn.ModuleList([UTSBlock_attnloss(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for j in range(self.depths[1])])
        self.norm1 = norm_layer(embed_dims[1])

        self.ctm2 = ATM(self.sample_ratios, self.embed_dims[1], self.embed_dims[2], self.k)
        self.stage2 = nn.ModuleList([UTSBlock_attnloss(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for j in range(self.depths[2])])
        self.norm2 = norm_layer(embed_dims[2])

        self.ltr3 = FAT_PTRV2(in_channels=[embed_dims[1], embed_dims[2]], out_channels=embed_dims[1])
        self.stage3 = nn.ModuleList([UTSBlock_attnloss(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for j in range(self.depths[1])])
        self.norm3 = norm_layer(embed_dims[1])

        self.ltr4 = FAT_PTRV2(in_channels=[embed_dims[0], embed_dims[1]], out_channels=embed_dims[0])
        self.stage4 = nn.ModuleList([UTSBlock_attnloss(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=1)
            for j in range(self.depths[0])])
        self.norm4 = norm_layer(embed_dims[0])



    def forward(self, img):
        x = self.to_patch_embedding(img)

        outs = []
        attns = []
        for blk in self.stage0_as:
            x, attn = blk(x, self.patch_num, self.patch_num)
            attns.append(attn)
        x = self.norm0_as(x)
        self.cur += self.depths[0]

        attn_map = attn[:, 0:2].sum(dim=1).sum(dim=1)
        b = attn_map.shape[0]
        n = attn_map.shape[1]
        min_as_out = torch.min(attn_map, dim=1)[0].repeat(n).reshape(n, b).transpose(0, 1)
        max_as_out = torch.max(attn_map, dim=1)[0].repeat(n).reshape(n, b).transpose(0, 1)
        attnScore = (attn_map[:, ] - min_as_out) / (max_as_out - min_as_out)
        attnScore = rearrange(attnScore, 'b (h w)-> b h w', h=self.patch_num, w=self.patch_num)

        B, N, _ = x.shape
        idx_token = torch.arange(N)[None, :].repeat(B, 1).cuda()
        agg_weight = x.new_ones(B, N, 1)
        token_dict = {'x': x,
                      'token_num': N,
                      'map_size': [self.patch_num, self.patch_num],
                      'init_grid_size': [self.patch_num, self.patch_num],
                      'idx_token': idx_token,
                      'agg_weight': agg_weight}
        outs.append(token_dict.copy())

        token_dict = self.ctm1(token_dict, attnScore, ctm_stage=1)
        for j, blk in enumerate(self.stage1):
            token_dict, attn = blk(token_dict)
            attns.append(attn)
        token_dict['x'] = self.norm1(token_dict['x'])
        self.cur += self.depths[1]
        outs.append(token_dict)

        token_dict = self.ctm2(token_dict, attnScore, ctm_stage=2)
        for blk in self.stage2:
            token_dict, attn = blk(token_dict)
            attns.append(attn)
        token_dict['x'] = self.norm2(token_dict['x'])
        self.cur += self.depths[2]
        outs.append(token_dict)



        token_dict = outs[1]
        token_dict['x'] = self.ltr3([outs[1], outs[2]])
        for blk in self.stage3:
            token_dict, attn = blk(token_dict)
            attns.append(attn)
        token_dict['x'] = self.norm3(token_dict['x'])
        self.cur += self.depths[1]
        outs.append(token_dict)

        token_dict = outs[0]
        token_dict['x'] = self.ltr4([outs[0], outs[3]])
        for blk in self.stage4:
            token_dict, attn = blk(token_dict)
            attns.append(attn)

        x = self.norm4(token_dict['x'])


        x = self.from_patch_embedding(x)  # b c h*w


        return x, attnScore, attns

class Encoder_DTMFormerV2_attnlossDS(nn.Module):
    def __init__(
            self, in_channels=768, image_size=256, out_channels=768, total_depth=12,
            patch_size=16, embed_dims=[192, 384, 768], num_heads=[4, 6, 8], mlp_ratios=[4, 4, 4],
            qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
            drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[2, 2, 4],
            # sr_ratios=[4, 2, 1], pretrained=None,
            sr_ratios=[1, 1, 1], pretrained=None,
            k=5, sample_ratios=0.25, classes=4,
            return_map=False, ):
        super().__init__()

        self.total_depth = total_depth
        self.depths = depths
        self.grid_stride = sr_ratios[0]
        self.embed_dims = embed_dims
        self.sr_ratios = sr_ratios
        self.mlp_ratios = mlp_ratios
        self.sample_ratios = sample_ratios
        self.return_map = return_map
        self.in_channels = in_channels
        self.k = k
        self.classes = classes
        self.patch_num = int(image_size/patch_size)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.to_patch_embedding = nn.Sequential(
            nn.Linear(self.in_channels, embed_dims[0]),
        )
        self.from_patch_embedding = nn.Sequential(
            nn.Linear(embed_dims[0], self.out_channels),
        )


        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.cur = 0

        self.stage0_as = nn.ModuleList([UTSTransformer(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=1)
            for j in range(self.depths[0])])
        self.norm0_as = norm_layer(embed_dims[0])

        self.ctm1 = ATM(self.sample_ratios, self.embed_dims[0], self.embed_dims[1], self.k)
        self.stage1 = nn.ModuleList([UTSBlock_attnloss(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for j in range(self.depths[1])])
        self.norm1 = norm_layer(embed_dims[1])

        self.ctm2 = ATM(self.sample_ratios, self.embed_dims[1], self.embed_dims[2], self.k)
        self.stage2 = nn.ModuleList([UTSBlock_attnloss(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for j in range(self.depths[2])])
        self.norm2 = norm_layer(embed_dims[2])

        self.ltr3 = FAT_PTRV2(in_channels=[embed_dims[1], embed_dims[2]], out_channels=embed_dims[1])
        self.stage3 = nn.ModuleList([UTSBlock_attnloss(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for j in range(self.depths[1])])
        self.norm3 = norm_layer(embed_dims[1])

        self.ltr4 = FAT_PTRV2(in_channels=[embed_dims[0], embed_dims[1]], out_channels=embed_dims[0])
        self.stage4 = nn.ModuleList([UTSBlock_attnloss(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=1)
            for j in range(self.depths[0])])
        self.norm4 = norm_layer(embed_dims[0])

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels // 2, kernel_size=4, stride=2, padding=1),  # 16x16 -> 32x32
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels // 2, out_channels // 4, kernel_size=4, stride=2, padding=1),
            # 32x32 -> 64x64
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels // 4, out_channels // 8, kernel_size=4, stride=2, padding=1),
            # 64x64 -> 128x128
            nn.BatchNorm2d(out_channels // 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels // 8, classes, kernel_size=4, stride=2, padding=1)  # 128x128 -> 256x256
        )


    def forward(self, img):
        x = self.to_patch_embedding(img)

        outs = []
        attns = []
        for blk in self.stage0_as:
            x, attn = blk(x, self.patch_num, self.patch_num)
            attns.append(attn)
        x = self.norm0_as(x)
        self.cur += self.depths[0]

        attn_map = attn[:, 0:2].sum(dim=1).sum(dim=1)
        b = attn_map.shape[0]
        n = attn_map.shape[1]
        min_as_out = torch.min(attn_map, dim=1)[0].repeat(n).reshape(n, b).transpose(0, 1)
        max_as_out = torch.max(attn_map, dim=1)[0].repeat(n).reshape(n, b).transpose(0, 1)
        attnScore = (attn_map[:, ] - min_as_out) / (max_as_out - min_as_out)
        attnScore = rearrange(attnScore, 'b (h w)-> b h w', h=self.patch_num, w=self.patch_num)

        B, N, _ = x.shape
        idx_token = torch.arange(N)[None, :].repeat(B, 1).cuda()
        agg_weight = x.new_ones(B, N, 1)
        token_dict = {'x': x,
                      'token_num': N,
                      'map_size': [self.patch_num, self.patch_num],
                      'init_grid_size': [self.patch_num, self.patch_num],
                      'idx_token': idx_token,
                      'agg_weight': agg_weight}
        outs.append(token_dict.copy())

        token_dict = self.ctm1(token_dict, attnScore, ctm_stage=1)
        for j, blk in enumerate(self.stage1):
            token_dict, attn = blk(token_dict)
            attns.append(attn)
        token_dict['x'] = self.norm1(token_dict['x'])
        self.cur += self.depths[1]
        outs.append(token_dict)

        token_dict = self.ctm2(token_dict, attnScore, ctm_stage=2)
        for blk in self.stage2:
            token_dict, attn = blk(token_dict)
            attns.append(attn)
        token_dict['x'] = self.norm2(token_dict['x'])
        self.cur += self.depths[2]
        outs.append(token_dict)



        token_dict = outs[1]
        token_dict['x'] = self.ltr3([outs[1], outs[2]])
        for blk in self.stage3:
            token_dict, attn = blk(token_dict)
            attns.append(attn)
        token_dict['x'] = self.norm3(token_dict['x'])
        self.cur += self.depths[1]
        outs.append(token_dict)

        token_dict = outs[0]
        token_dict['x'] = self.ltr4([outs[0], outs[3]])
        for blk in self.stage4:
            token_dict, attn = blk(token_dict)
            attns.append(attn)

        x = self.norm4(token_dict['x'])


        x = self.from_patch_embedding(x)  # b c h*w

        b, _, _ = x.size()
        x_ds = x.view(b, self.out_channels, self.patch_num, self.patch_num)
        x_ds = self.decoder(x_ds)


        return x, attnScore, attns, x_ds


class Encoder_DTMFormerV2FirstStageChoose(nn.Module):
    def __init__(
            self, in_channels=768, image_size=256, out_channels=768, total_depth=12,
            patch_size=16, embed_dims=[192, 384, 768], num_heads=[4, 6, 8], mlp_ratios=[4, 4, 4],
            qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
            drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[2, 2, 4],
            # sr_ratios=[4, 2, 1], pretrained=None,
            sr_ratios=[1, 1, 1], pretrained=None,
            k=5, sample_ratios=0.25, classes=4,
            return_map=False, ):
        super().__init__()

        self.total_depth = total_depth
        self.depths = depths
        self.grid_stride = sr_ratios[0]
        self.embed_dims = embed_dims
        self.sr_ratios = sr_ratios
        self.mlp_ratios = mlp_ratios
        self.sample_ratios = sample_ratios
        self.return_map = return_map
        self.in_channels = in_channels
        self.k = k
        self.classes = classes
        self.patch_num = int(image_size/patch_size)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.to_patch_embedding = nn.Sequential(
            nn.Linear(self.in_channels, embed_dims[0]),
        )
        self.from_patch_embedding = nn.Sequential(
            nn.Linear(embed_dims[0], self.out_channels),
        )


        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.cur = 0

        self.stage0_as = nn.ModuleList([UTSTransformer_attnloss(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=1)
            for j in range(self.depths[0])])
        self.norm0_as = norm_layer(embed_dims[0])

        self.atm1 = HTM(self.sample_ratios, self.embed_dims[0], self.embed_dims[1], self.k)
        self.stage1 = nn.ModuleList([UTSBlock_attnloss_tokenscore(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for j in range(self.depths[1])])
        self.norm1 = norm_layer(embed_dims[1])

        self.atm2 = HTM(self.sample_ratios, self.embed_dims[1], self.embed_dims[2], self.k)
        self.stage2 = nn.ModuleList([UTSBlock_attnloss_tokenscore(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for j in range(self.depths[2])])
        self.norm2 = norm_layer(embed_dims[2])

        self.ltr3 = FAT_PTRV2(in_channels=[embed_dims[1], embed_dims[2]], out_channels=embed_dims[1])
        self.stage3 = nn.ModuleList([UTSBlock_attnloss(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for j in range(self.depths[1])])
        self.norm3 = norm_layer(embed_dims[1])

        self.ltr4 = FAT_PTRV2(in_channels=[embed_dims[0], embed_dims[1]], out_channels=embed_dims[0])
        self.stage4 = nn.ModuleList([UTSBlock_attnloss(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=1)
            for j in range(self.depths[0])])
        self.norm4 = norm_layer(embed_dims[0])


    def forward(self, img):
        x = self.to_patch_embedding(img)

        outs = []
        attns = []
        attnScores = []
        variances = []
        for blk in self.stage0_as:
            x, attn = blk(x, self.patch_num, self.patch_num)
            attns.append(attn)
        x = self.norm0_as(x)
        self.cur += self.depths[0]

        attn_map = attn[:, 0:2].sum(dim=1).sum(dim=1)
        b = attn_map.shape[0]
        n = attn_map.shape[1]
        min_vals = torch.min(attn_map, dim=1, keepdim=True)[0]
        max_vals = torch.max(attn_map, dim=1, keepdim=True)[0]  # 形状为 [b, 1]
        attnScore0 = (attn_map - min_vals) / (max_vals - min_vals + 1e-8)
        attnScore = rearrange(attnScore0, 'b (h w)-> b h w', h=self.patch_num, w=self.patch_num)
        attnScores.append(attnScore)

        B, N, _ = x.shape
        idx_token = torch.arange(N)[None, :].repeat(B, 1).cuda()
        agg_weight = x.new_ones(B, N, 1)
        token_dict = {'x': x,
                      'token_num': N,
                      'map_size': [self.patch_num, self.patch_num],
                      'init_grid_size': [self.patch_num, self.patch_num],
                      'idx_token': idx_token,
                      'agg_weight': agg_weight}
        outs.append(token_dict.copy())

        token_dict, token_weight1, idx_cluster1, variance, score1 = self.atm1(token_dict, outs, attnScore, ctm_stage=1)
        for j, blk in enumerate(self.stage1):
            token_dict, attn = blk(token_dict)
            attns.append(attn)
        token_dict['x'] = self.norm1(token_dict['x'])
        self.cur += self.depths[1]
        outs.append(token_dict)
        variances.append(variance)

        attn_map = attn[:, 0:2].sum(dim=1).sum(dim=1)
        b = attn_map.shape[0]
        n = attn_map.shape[1]
        min_vals = torch.min(attn_map, dim=1, keepdim=True)[0]
        max_vals = torch.max(attn_map, dim=1, keepdim=True)[0]  # 形状为 [b, 1]
        attnScore1 = (attn_map - min_vals) / (max_vals - min_vals + 1e-8)
        attnScore = var_downup(attnScore1, outs[0], outs[1])
        attnScore = rearrange(attnScore, 'b (h w) 1-> b h w', h=self.patch_num, w=self.patch_num)
        attnScores.append(attnScore)

        token_dict, token_weight2, idx_cluster2, variance, score2 = self.atm2(token_dict, outs, attnScore1, ctm_stage=2)
        for blk in self.stage2:
            token_dict, attn = blk(token_dict)
            attns.append(attn)
        token_dict['x'] = self.norm2(token_dict['x'])
        self.cur += self.depths[2]
        outs.append(token_dict)
        variances.append(variance)

        token_dict = outs[1]
        token_dict['x'] = self.ltr3([outs[1], outs[2]])
        for blk in self.stage3:
            token_dict, attn = blk(token_dict)
            attns.append(attn)
        token_dict['x'] = self.norm3(token_dict['x'])
        self.cur += self.depths[1]
        outs.append(token_dict)

        token_dict = outs[0]
        token_dict['x'] = self.ltr4([outs[0], outs[3]])
        for blk in self.stage4:
            token_dict, attn = blk(token_dict)
            attns.append(attn)

        x = self.norm4(token_dict['x'])


        x = self.from_patch_embedding(x)  # b c h*w


        return x, attnScores, attns, outs, variances, [score1,score2]
