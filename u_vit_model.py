import torch
from torch import nn
from torch.nn import functional as F
from torch import nn, einsum
from einops import rearrange
import copy
import math
class CNNencoder_gn(nn.Module):
    """具有Group Normalization的标准卷积块"""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16, out_c, eps=1e-6),
            nn.LeakyReLU(inplace=False)
        )

    def forward(self, x):
        return self.model(x)

# Conv with LayerNorm
class CNNencoder_ln(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(4, out_c, eps=1e-6),
            nn.LeakyReLU(inplace=False)
        )

    def forward(self, x):
        out = self.model(x)
        return out


# CNN Concat with GroupNorm
class Concat_gn(nn.Module):
    """用于在解码器中将特征图拼接后进行卷积的模块"""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16, out_c, eps=1e-6),
            nn.LeakyReLU(inplace=False)
        )

    def forward(self, x, skip):
        x = torch.cat((x, skip), dim=1)
        return self.model(x)

# CNN concat with LayerNorm
class Concat_ln(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(4, out_c, eps=1e-6),
            nn.LeakyReLU(inplace=False)
        )

    def forward(self, x, skip):

        x = torch.cat((x, skip), 1)
        out = self.model(x)
        return out



# Making ViT

# Patch Embedding

class Embeddings(nn.Module):
    def __init__(self, img_size):
        super(Embeddings, self).__init__()
        down_factor = 4#缩放因子cdfx
        patch_size = (2, 2)
        n_patches = int((img_size[0]/2**down_factor// patch_size[0]) * (img_size[1]/2**down_factor// patch_size[1]))
        self.patch_embeddings = nn.Conv2d(in_channels=1024,
                                          out_channels=768,
                                          kernel_size=patch_size,
                                          stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, 768))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # input = (B, 256, 48, 32)
        x = self.patch_embeddings(x)  # (B, hidden, n_patches^(1/2), n_patches^(1/2))
        # (B, 768, 24, 16)
        x = x.flatten(2)
        # (B, 768, 384)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        # (B, 384, 768)
        position_embeddings = self.position_embeddings
        # position_embeddings = (B, 384, 768)
        embeddings = x + position_embeddings
        # (B, 384, 768)
        embeddings = self.dropout(embeddings)
        return embeddings


# Multi-head self attention (MSA) - layer norm not included

class MSA(nn.Module):
    def __init__(self):
        super(MSA, self).__init__()
        self.num_attention_heads = 12
        self.attention_head_size = int(768 / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(768, self.all_head_size)
        self.key = nn.Linear(768, self.all_head_size)
        self.value = nn.Linear(768, self.all_head_size)
        self.out = nn.Linear(768, 768)
        self.attn_dropout = nn.Dropout(0.1)
        self.proj_dropout = nn.Dropout(0.1)
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        x = x.view([x.size()[0], -1, self.num_attention_heads, self.attention_head_size])
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, q=None):
        if q is None:
            q = hidden_states

        mixed_query_layer = self.query(q)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)

        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output


# MLP - layer norm not included

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(768, 3072)
        self.fc2 = nn.Linear(3072, 768)
        self.act_fn = torch.nn.functional.gelu
        self.dropout = nn.Dropout(0.1)
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


# Block - incorporating MSA, MLP, Layer Norm

class Block(nn.Module):
    def __init__(self):
        super(Block, self).__init__()
        self.hidden_size = 768
        self.attention_norm = nn.LayerNorm(768, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(768, eps=1e-6)
        self.ffn = MLP()
        self.attn = MSA()

    def forward(self, x, q=None):
        h = x

        x = self.attention_norm(x)
        x = self.attn(x, q)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x


#  ViTencoder - ViT Encoder with Blocks

class ViTencoder(nn.Module):
    def __init__(self):
        super(ViTencoder, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(768, eps=1e-6)
        for _ in range(12):
            layer = Block()
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states, q=None):
        for layer_block in self.layer:
            hidden_states = layer_block(hidden_states, q)
        encoded = self.encoder_norm(hidden_states)
        return encoded


class ResidualBlock(nn.Module):
    """残差块，包含两次卷积操作，并在必要时使用1x1卷积调整输入的通道数以匹配输出"""
    def __init__(self, in_channels, out_channels,dropout_prob=0.2):
        super(ResidualBlock, self).__init__()
        self.conv1 = CNNencoder_gn(in_channels, out_channels)
        self.conv2 = CNNencoder_gn(out_channels, out_channels)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_prob)
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.GroupNorm(16, out_channels)  # 确保 out_channels 可以被组数整除
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        B = identity * self.sigmoid(identity)  # 使用 Sigmoid 修改的残差部分
        out += B
        out = self.dropout(out)
        return out

class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_groupnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_groupnorm),
        )
        relu = nn.LeakyReLU(inplace=True)

        gn = nn.GroupNorm(16, out_channels, eps=1e-6)

        super(Conv2dReLU, self).__init__(conv, gn, relu)


#  ViT

class ViT(nn.Module):
    def __init__(self, img_size):
        super(ViT, self).__init__()
        self.embeddings = Embeddings(img_size=img_size)
        self.encoder = ViTencoder()
        self.img_size = img_size
        self.patch_size = (2, 2)
        self.down_factor = 4
        self.conv_more = Conv2dReLU(768, 1024, kernel_size=3, padding=1, use_groupnorm=True)
        self.q_conv = nn.Conv2d(128, 1024, kernel_size=1)  # 添加这一层将 c2 的通道数调整为 1024
        self.pooling = nn.AdaptiveAvgPool2d((img_size[0] // (2**self.down_factor), img_size[1] // (2**self.down_factor)))

    def forward(self, x, q=None):
        x = self.embeddings(x)  # 原来的处理方式
        if q is not None:
            q = self.q_conv(q)  # 调整 q 的通道数
            q = self.pooling(q)  # 调整 q 的大小
            q = self.embeddings.patch_embeddings(q)
            q = q.flatten(2).transpose(-1, -2)
        x = self.encoder(x, q)  # 传入 q

        B, n_patch, hidden = x.size()
        h, w = (self.img_size[0] // 2**self.down_factor // self.patch_size[0]), (self.img_size[1] // 2**self.down_factor // self.patch_size[0])
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        return x



class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.gates = nn.Conv2d(in_channels=input_dim + hidden_dim,
                               out_channels=4 * hidden_dim,
                               kernel_size=self.kernel_size,
                               padding=self.padding,
                               bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_gates = self.gates(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_gates, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.gates.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.gates.weight.device))


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTM, self).__init__()
        self.cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size, bias)

    def forward(self, input_tensor, hidden_state=None):
        b, seq_len, _, h, w = input_tensor.size()
        if hidden_state is None:
            hidden_state = self.cell.init_hidden(b, (h, w))

        cur_state = hidden_state
        output_inner = []
        for t in range(seq_len):
            h, c = self.cell(input_tensor[:, t, :, :, :], cur_state)
            output_inner.append(h)
            cur_state = (h, c)

        layer_output = torch.stack(output_inner, dim=1)
        return layer_output, cur_state


class FusionDiffConv(nn.Module):
    def __init__(self, in_channels):
        super(FusionDiffConv, self).__init__()
        self.standard_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.diff_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)

        # 初始化为标准卷积核
        nn.init.kaiming_normal_(self.standard_conv.weight, mode='fan_out', nonlinearity='relu')

        # 初始化差分卷积核
        diff_weights = torch.tensor([[[[0., 1., 0.],
                                       [1., -4., 1.],
                                       [0., 1., 0.]]]], dtype=torch.float32)
        self.diff_conv.weight = nn.Parameter(diff_weights.repeat(in_channels, 1, 1, 1), requires_grad=False)

    def forward(self, x):
        standard_features = self.standard_conv(x)
        diff_features = self.diff_conv(x)
        return standard_features + diff_features


class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.in_channels = in_channels

    def forward(self, x):
        B, C, H, W = x.size()

        # 创建三个不同尺度的特征块
        scales = [4, 2, 1]
        blocks = []
        for scale in scales:
            size = (H // scale, W // scale)
            block = F.adaptive_avg_pool2d(x, size)
            block = self.attention(block)
            blocks.append(F.interpolate(block, size=(H, W), mode='bilinear', align_corners=False))

        # 将块重新拼接
        out = torch.stack(blocks, dim=0).sum(dim=0)
        return out

    def attention(self, x):
        # 自注意力机制
        B, C, H, W = x.size()
        q = x.view(B, C, -1).permute(0, 2, 1)  # B x HW x C
        k = x.view(B, C, -1)  # B x C x HW
        v = x.view(B, C, -1).permute(0, 2, 1)  # B x HW x C

        attention = torch.bmm(q, k)  # B x HW x HW
        attention = F.softmax(attention, dim=-1)

        out = torch.bmm(attention, v)  # B x HW x C
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return out


class BoundaryExtractionModule(nn.Module):
    def __init__(self, in_channels):
        super(BoundaryExtractionModule, self).__init__()
        self.fusion_diff_conv = FusionDiffConv(in_channels)
        self.attention_module = AttentionModule(in_channels)

    def forward(self, x):
        boundary_features = self.fusion_diff_conv(x)
        attention_features = self.attention_module(boundary_features)
        return x + attention_features

# Generator

class LearnableDilatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, max_dilation=10):
        super(LearnableDilatedConv, self).__init__()
        self.max_dilation = max_dilation
        self.dilation = nn.Parameter(torch.randint(1, max_dilation + 1, (1,)).float())
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        dilation = torch.round(self.dilation).clamp(1, self.max_dilation).item()
        padding = (self.conv.kernel_size[0] - 1) // 2 * int(dilation)
        return F.conv2d(x, self.conv.weight, self.conv.bias, stride=self.conv.stride, padding=padding, dilation=int(dilation))

class ViT_UNet(nn.Module):
    def __init__(self, img_size=(224, 224)):
        super().__init__()
        self.pooling = nn.MaxPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2)

        self.conv1_1 = ResidualBlock(3, 64)
        self.conv1_2 = ResidualBlock(64, 64)
        self.edge_detectc1 = FusionDiffConv(in_channels=64)
        self.dilated_conv_in_block = LearnableDilatedConv(128, 64)

        self.conv2_1 = ResidualBlock(64, 128)
        self.conv2_2 = ResidualBlock(128, 128)
        self.edge_detectc2 = FusionDiffConv(128)
        self.dilated_conv_down1 = LearnableDilatedConv(256, 128)

        self.conv3_1 = ResidualBlock(128, 256)
        self.conv3_2 = ResidualBlock(256, 256)
        self.edge_detectc3 = FusionDiffConv(256)
        self.dilated_conv_down2 = LearnableDilatedConv(512, 256)

        self.conv4_1 = ResidualBlock(256, 512)
        self.conv4_2 = ResidualBlock(512, 512)

        self.conv5_1 = ResidualBlock(512, 1024)
        self.conv5_2 = ResidualBlock(1024, 1024)
        self.feature_lstm = ConvLSTM(input_dim=1024, hidden_dim=1024, kernel_size=3)
        self.vit = ViT(img_size)

        self.concat1 = Concat_gn(2048, 1024)
        self.concat2 = Concat_gn(1536, 512)
        self.concat3 = Concat_gn(768, 256)
        self.edge_detectu3 = FusionDiffConv(256)
        self.dilated_conv_up_block2 = LearnableDilatedConv(512, 256)
        self.concat4 = Concat_gn(384, 128)
        self.edge_detectu4 = FusionDiffConv(128)
        self.dilated_conv_up_block3 = LearnableDilatedConv(256, 128)
        self.concat5 = Concat_gn(192, 64)
        self.edge_detectu5 = FusionDiffConv(64)
        self.dilated_conv_up_block4 = LearnableDilatedConv(128, 64)
        self.Segmentation_head = nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        c1 = self.conv1_1(x)
        c1 = self.conv1_2(c1)
        edge_features_c1 = self.edge_detectc1(c1)
        c1_combined = torch.cat([c1, edge_features_c1], dim=1)
        c1 = self.dilated_conv_in_block(c1_combined)
        p1 = self.pooling(c1)

        c2 = self.conv2_1(p1)
        c2 = self.conv2_2(c2)
        edge_features_c2 = self.edge_detectc2(c2)
        c2_combined = torch.cat([c2, edge_features_c2], dim=1)
        c2 = self.dilated_conv_down1(c2_combined)
        p2 = self.pooling(c2)

        c3 = self.conv3_1(p2)
        c3 = self.conv3_2(c3)
        edge_features_c3 = self.edge_detectc3(c3)
        c3_combined = torch.cat([c3, edge_features_c3], dim=1)
        c3 = self.dilated_conv_down2(c3_combined)
        p3 = self.pooling(c3)

        c4 = self.conv4_1(p3)
        c4 = self.conv4_2(c4)
        p4 = self.pooling(c4)

        c5 = self.conv5_1(p4)
        c5 = self.conv5_2(c5)
        c5 = self.feature_lstm(c5.unsqueeze(1))  # LSTM expects a sequence
        c5, _ = c5  # Unpack the tuple to get just the output tensor
        c5 = c5.squeeze(1)  # Remove the sequence dimension

        v = self.vit(c5, c2)  # 使用 c2 作为 q

        v1 = self.upsample(v)
        u1 = self.concat1(v1, c5)

        u1 = self.upsample(u1)

        u2 = self.concat2(u1, c4)
        u2 = self.upsample(u2)

        u3 = self.concat3(u2, c3)
        edge_features_u3 = self.edge_detectu3(u3)
        u3_combined = torch.cat([u3, edge_features_u3], dim=1)
        u3 = self.dilated_conv_up_block2(u3_combined)
        u3 = self.upsample(u3)

        u4 = self.concat4(u3, c2)
        edge_features_u4 = self.edge_detectu4(u4)
        u4_combined = torch.cat([u4, edge_features_u4], dim=1)
        u4 = self.dilated_conv_up_block3(u4_combined)
        u4 = self.upsample(u4)

        u5 = self.concat5(u4, c1)
        edge_features_u5 = self.edge_detectu5(u5)
        u5_combined = torch.cat([u5, edge_features_u5], dim=1)
        u5 = self.dilated_conv_up_block4(u5_combined)

        out = self.Segmentation_head(u5)

        return out, c1, c2, c3, c4
