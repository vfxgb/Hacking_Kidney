import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.hub

##############################################
# 1. Updated ConvLayer with GroupNorm Option #
##############################################
class ConvLayer(nn.Sequential):
    def __init__(self, ni, nf, ks=3, stride=1, padding=None, norm_type=None,
                 act_cls=nn.LeakyReLU, leaky=0.1, xtra=None):
        if padding is None:
            padding = ks // 2
        layers = [nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=padding)]
        if norm_type == "batch":
            layers.append(nn.BatchNorm2d(nf))
        elif norm_type == "group":
            layers.append(nn.GroupNorm(8, nf))
        if act_cls is not None:
            if act_cls == nn.LeakyReLU:
                layers.append(act_cls(negative_slope=leaky, inplace=True))
            else:
                layers.append(act_cls(inplace=True))
        if xtra is not None:
            layers.append(xtra)
        super().__init__(*layers)

#########################################
# 2. SelfAttention remains unchanged    #
#########################################
class SelfAttention(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.query = nn.Conv1d(n_channels, n_channels // 8, 1)
        self.key = nn.Conv1d(n_channels, n_channels // 8, 1)
        self.value = nn.Conv1d(n_channels, n_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        x_ = x.view(B, C, -1)
        q = self.query(x_).permute(0, 2, 1)
        k = self.key(x_)
        attn = torch.bmm(q, k)
        attn = F.softmax(attn, dim=-1)
        v = self.value(x_)
        out = torch.bmm(v, attn.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        return self.gamma * out + x

##############################################
# 3. CBAM: Convolutional Block Attention Module #
##############################################
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()
        # Spatial Attention
        self.conv_after_concat = nn.Conv2d(2, 1, kernel_size=kernel_size,
                                           padding=kernel_size // 2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # Channel attention
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        scale = self.sigmoid_channel(avg_out + max_out)
        x = x * scale
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid_spatial(self.conv_after_concat(concat))
        x = x * scale
        return x

#########################################
# 4. PixelShuffle_ICNR remains as is    #
#########################################
class PixelShuffle_ICNR(nn.Sequential):
    def __init__(self, ni, nf=None, scale=2, blur=False, leaky=None):
        nf = nf or ni
        layers = [nn.Conv2d(ni, nf * (scale ** 2), 1), nn.PixelShuffle(scale)]
        self.icnr(layers[0].weight)
        if blur:
            layers.append(nn.Conv2d(nf, nf, 1))
        super().__init__(*layers)
    
    def icnr(self, weight, scale=2, init=nn.init.kaiming_normal_):
        ni, nf, h, w = weight.shape
        ni2 = int(ni / (scale ** 2))
        k = init(torch.zeros([ni2, nf, h, w]))
        k = k.repeat_interleave(scale ** 2, dim=0)
        with torch.no_grad():
            weight.copy_(k)

#########################################
# 5. BiFPN: Replaces the original FPN  #
#########################################
class BiFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels, num_layers=1):
        """
        in_channels_list: list of ints corresponding to each feature map
                          (ordered from smallest to largest resolution)
        out_channels: common number of channels after lateral convolutions.
        """
        super().__init__()
        self.lateral_convs = nn.ModuleList([nn.Conv2d(in_ch, out_channels, kernel_size=1)
                                            for in_ch in in_channels_list])
        self.fusion_convs = nn.ModuleList([nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
                                           for _ in range(len(in_channels_list) - 1)])
        self.weights = nn.Parameter(torch.ones(len(in_channels_list) - 1, 2), requires_grad=True)
        self.num_layers = num_layers

    def forward(self, features):
        # Expect features ordered from lowest resolution to highest.
        feats = [l_conv(f) for l_conv, f in zip(self.lateral_convs, features)]
        for _ in range(self.num_layers):
            for i in range(1, len(feats)):
                w = F.relu(self.weights[i - 1])
                norm_w = w / (w.sum() + 1e-4)
                up_feat = F.interpolate(feats[i - 1], size=feats[i].shape[-2:], mode='bilinear', align_corners=False)
                feats[i] = self.fusion_convs[i - 1](norm_w[0] * up_feat + norm_w[1] * feats[i])
        return feats[-1]

######################################################################
# 6a. Selective State Space Module for 2D (Simple Version)
######################################################################
class SelectiveStateSpace2D(nn.Module):
    def __init__(self, in_channels, d_state=16):
        """
        A simple selective state space block.
        in_channels: number of channels of the input.
        d_state: dimension of a learned state (can be thought of as a bottleneck).
        """
        super().__init__()
        # A linear projection to represent the state dynamics
        self.linear = nn.Linear(in_channels, in_channels)
        # A gating mechanism to selectively combine the new representation and the input.
        self.gate = nn.Linear(in_channels, in_channels)
    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        # Flatten spatial dimensions: (B, H*W, C)
        x_flat = x.view(B, C, H * W).transpose(1, 2)
        out = self.linear(x_flat)
        gate = torch.sigmoid(self.gate(x_flat))
        # Weighted combination of the new features and the original input (residual)
        out = gate * out + (1 - gate) * x_flat
        out = out.transpose(1, 2).view(B, C, H, W)
        return out

######################################################################
# 6b. MambaStyleSSMDecoderBlock: Incorporating PixelShuffle, Linear Projection & SSM
######################################################################
class MambaStyleSSMDecoderBlock(nn.Module):
    def __init__(self, up_in_c, skip_in_c, out_c, norm_type="batch", use_cbam=False, d_state=16):
        """
        up_in_c   : channels from the lower-resolution decoder input.
        skip_in_c : channels from the skip connection.
        out_c     : desired output channels.
        norm_type : "batch" or "group" normalization.
        use_cbam  : whether to apply CBAM attention after fusion.
        d_state   : parameter for the selective state space module.
        """
        super().__init__()
        # Upsample using PixelShuffle_ICNR so that the output has out_c channels.
        self.upsample = PixelShuffle_ICNR(up_in_c, nf=out_c, scale=2, leaky=0.1)
        # Linear projection in flattened “linear space.”
        self.linear_proj = nn.Linear(out_c, out_c)
        # Incorporate a selective state space module.
        self.ssm = SelectiveStateSpace2D(out_c, d_state=d_state)
        # Project the skip connection (if needed).
        self.skip_proj = nn.Conv2d(skip_in_c, out_c, kernel_size=1) if skip_in_c != out_c else nn.Identity()
        # Fusion layers to combine upsampled (and SSM-processed) features with skip features.
        self.conv1 = ConvLayer(out_c * 2, out_c, norm_type=norm_type)
        self.conv2 = ConvLayer(out_c, out_c, norm_type=norm_type)
        self.cbam = CBAM(out_c) if use_cbam else None

    def forward(self, x, skip):
        # Upsample using pixel shuffle.
        x_up = self.upsample(x)  # shape: (B, out_c, H, W)
        B, C, H, W = x_up.shape
        # Reshape to linear (token) space.
        x_flat = x_up.permute(0, 2, 3, 1).reshape(B, -1, C)
        x_lin = self.linear_proj(x_flat)
        x_lin = x_lin.reshape(B, H, W, C).permute(0, 3, 1, 2)
        # Process the upsampled features through the selective state space module.
        x_ssm = self.ssm(x_lin)
        # Project skip connection features.
        skip_proj = self.skip_proj(skip)
        if x_ssm.shape[-2:] != skip_proj.shape[-2:]:
            skip_proj = F.interpolate(skip_proj, size=x_ssm.shape[-2:], mode='bilinear', align_corners=False)
        # Fuse the two sources.
        fused = torch.cat([x_ssm, skip_proj], dim=1)
        fused = self.conv1(fused)
        fused = self.conv2(fused)
        if self.cbam:
            fused = self.cbam(fused)
        return fused

##############################################
# 7. ASPP remains mostly unchanged          #
##############################################
class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, groups=1):
        super().__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation,
                                     bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self._init_weight()
    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        return self.relu(x)
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, inplanes=512, mid_c=256, dilations=[6, 12, 18, 24], out_c=None):
        super().__init__()
        self.aspps = [_ASPPModule(inplanes, mid_c, 1, padding=0, dilation=1)]
        self.aspps += [_ASPPModule(inplanes, mid_c, 3, padding=d, dilation=d, groups=4) for d in dilations]
        self.aspps = nn.ModuleList(self.aspps)
        self.global_pool = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_c, 1, stride=1, bias=False),
            nn.BatchNorm2d(mid_c), nn.ReLU())
        out_c = out_c if out_c is not None else mid_c
        self.out_conv = nn.Sequential(
            nn.Conv2d(mid_c * (2 + len(dilations)), out_c, 1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))
        self._init_weight()
    def forward(self, x):
        x0 = self.global_pool(x)
        xs = [aspp(x) for aspp in self.aspps]
        x0 = F.interpolate(x0, size=xs[0].size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x0] + xs, dim=1)
        return self.out_conv(x)
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

#########################################
# 8. SAM Feature Extractor              #
#########################################
class SAMFeatureExtractor(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(1, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.bn1(self.conv1(x)))

##############################################
# 9. Updated UneXt50: Integrating Upgrades   #
##############################################
class UneXt50(nn.Module):
    def __init__(self, in_channels=4, stride=1, norm_type="batch",
                 cbam_in_decoder=False, **kwargs):
        """
        in_channels: expecting a 4-channel input (RGB + SAM mask).
        norm_type: "batch" or "group" normalization for new layers.
        cbam_in_decoder: if True, applies CBAM in the decoder blocks.
        """
        super().__init__()
        # Load the ResNeXt backbone pretrained on RGB (using first 3 channels).
        m = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models',
                           'resnext101_32x4d_swsl')
        self.enc0 = nn.Sequential(m.conv1, m.bn1, nn.ReLU(inplace=True))
        self.enc1 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1), m.layer1)
        self.enc2 = m.layer2
        self.enc3 = m.layer3
        self.enc4 = m.layer4
        # ASPP applied on encoder output
        self.aspp = ASPP(2048, 256, out_c=512, dilations=[stride*1, stride*2, stride*3, stride*4])
        self.drop_aspp = nn.Dropout2d(0.5)
        # Use the new MambaStyle decoder blocks that incorporate PixelShuffle upsampling, linear projection and SSM.
        self.dec4 = MambaStyleSSMDecoderBlock(512, 1024, 256, norm_type=norm_type, use_cbam=cbam_in_decoder)
        self.dec3 = MambaStyleSSMDecoderBlock(256, 512, 128, norm_type=norm_type, use_cbam=cbam_in_decoder)
        self.dec2 = MambaStyleSSMDecoderBlock(128, 256, 64, norm_type=norm_type, use_cbam=cbam_in_decoder)
        self.dec1 = MambaStyleSSMDecoderBlock(64, 64, 32, norm_type=norm_type, use_cbam=cbam_in_decoder)
        # Auxiliary segmentation head (using output from dec2)
        self.aux_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )
        # BiFPN to fuse features from ASPP and all decoder outputs.
        self.bifpn = BiFPN([512, 256, 128, 64, 32], out_channels=64, num_layers=1)
        self.drop = nn.Dropout2d(0.1)
        self.final_conv = ConvLayer(64, 1, ks=1, norm_type=None, act_cls=None)
        # SAM branch: process the SAM pseudo mask (4th channel)
        self.sam_extractor = SAMFeatureExtractor(out_channels=64)
        self.final_fuse_conv = ConvLayer(128, 1, ks=1, norm_type=None, act_cls=None)

    def forward(self, x):
        """
        Expects x to be a 4-channel tensor:
          - x[:, :3, :, :] is the RGB image.
          - x[:, 3:, :, :] is the SAM pseudo mask.
        """
        rgb = x[:, :3, :, :]
        sam = x[:, 3:, :, :]
        # Backbone encoder processing on RGB input.
        enc0 = self.enc0(rgb)
        enc1 = self.enc1(enc0)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.aspp(enc4)
        # Decoder using MambaStyleSSMDecoderBlocks.
        dec4 = self.dec4(self.drop_aspp(enc5), enc3)
        dec3 = self.dec3(dec4, enc2)
        dec2 = self.dec2(dec3, enc1)
        dec1 = self.dec1(dec2, enc0)
        aux_out = self.aux_head(dec2)
        # Fuse features from the RGB branch using BiFPN.
        features = [enc5, dec4, dec3, dec2, dec1]
        fused = self.bifpn(features)
        # Process SAM pseudo mask.
        sam_features = self.sam_extractor(sam)
        if sam_features.shape[-2:] != fused.shape[-2:]:
            sam_features = F.interpolate(sam_features, size=fused.shape[-2:], mode='bilinear', align_corners=False)
        # Fuse the two representations along the channel dimension.
        combined = torch.cat([fused, sam_features], dim=1)
        out = self.final_fuse_conv(self.drop(combined))
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
        return out, aux_out
