import torch
import math
import torch.nn as nn
import torch.nn.functional as F
class SAM(torch.nn.Module):
    def __init__(self, kernel_size=7):
        super(SAM, self).__init__()
        self.conv = torch.nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(x_cat))
        return x * attention
class ECA(torch.nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECA, self).__init__()
        t = int(abs(math.log(channels, 2) + b) / gamma)
        k = t if t % 2 else t + 1
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.conv = torch.nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y
class EDAConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, expansion=0.25, reduction=16):
        super(EDAConv, self).__init__()
        mid_channels = int(in_channels * expansion)
        mid_channels = max(mid_channels, 1)
        self.pw1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            torch.nn.BatchNorm2d(mid_channels),
            torch.nn.GELU()
        )
        self.dw = torch.nn.Sequential(
            torch.nn.Conv2d(mid_channels, mid_channels, kernel_size, stride, padding=kernel_size // 2,
                            groups=mid_channels,
                            bias=False),
            torch.nn.BatchNorm2d(mid_channels),
            torch.nn.GELU()
        )
        self.eca = ECA(mid_channels)
        self.pw2 = torch.nn.Sequential(
            torch.nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            torch.nn.BatchNorm2d(out_channels)
        )
        self.sam = SAM(kernel_size=7)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                torch.nn.BatchNorm2d(out_channels)
            )
        self.act = torch.nn.GELU()
    def forward(self, x):
        identity = x
        out = self.pw1(x)
        out = self.dw(out)
        out = self.eca(out)
        out = self.pw2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.act(out)
        out = self.sam(out)
        return out
class ChannelSqueezeFusion(torch.nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(ChannelSqueezeFusion, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = max(1, in_channels // reduction)
        self.attention = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(in_channels, self.mid_channels, 1, bias=False),
            torch.nn.GELU(),
            torch.nn.Conv2d(self.mid_channels, in_channels, 1, bias=False),
            torch.nn.Sigmoid()
        )
        self.conv = torch.nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.norm = torch.nn.BatchNorm2d(in_channels)
        self.act = torch.nn.GELU()
    def forward(self, x):
        if x.size(1) == 0:
            return x
        attn = self.attention(x)
        out = x * attn
        out = self.conv(out)
        out = self.norm(out)
        out = self.act(out)
        return out
class MKC(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=0.5, branches=3, reduction=8):
        super(MKC, self).__init__()
        self.branches = min(branches, in_channels)
        mid_channels = max(self.branches, int(in_channels * expansion))
        mid_channels = ((mid_channels + self.branches - 1) // self.branches) * self.branches
        branch_channels = mid_channels // self.branches
        self.pw1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            torch.nn.BatchNorm2d(mid_channels),
            torch.nn.GELU()
        )
        self.branch_convs = torch.nn.ModuleList()
        for i in range(self.branches):
            kernel_size = 2 * i + 3
            self.branch_convs.append(
                EDAConv(branch_channels, branch_channels, kernel_size=kernel_size, stride=stride)
            )
        self.fusion = ChannelSqueezeFusion(mid_channels, reduction=reduction)
        self.pw2 = torch.nn.Sequential(
            torch.nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            torch.nn.BatchNorm2d(out_channels)
        )
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                torch.nn.BatchNorm2d(out_channels)
            )
        self.act = torch.nn.GELU()
        self.dropout = torch.nn.Dropout2d(0.1)
    def forward(self, x):
        identity = x
        x = self.pw1(x)
        branch_size = x.size(1) // self.branches
        outs = []
        for i in range(self.branches):
            start_idx = i * branch_size
            end_idx = (i + 1) * branch_size
            branch_input = x[:, start_idx:end_idx, :, :]
            outs.append(self.branch_convs[i](branch_input))
        out = torch.cat(outs, dim=1)
        out = self.fusion(out)
        out = self.pw2(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = self.dropout(out)
        out += identity
        out = self.act(out)
        return out
class MKCBlock(torch.nn.Module):
    def __init__(self, c1, c2, N=3, shortcut=True, g=1, e=0.25):
        super().__init__()
        self.mkc = MKC(c1, c2, expansion=e, branches=N)
        self.add = shortcut and c1 == c2
    def forward(self, x):
        return x + self.mkc(x) if self.add else self.mkc(x)
class MKC_Layer(torch.nn.Module):
    def __init__(self, in_chnl, out_chnl, r=3, dropout_rate=0.1):
        super().__init__()
        self.mkc = MKCBlock(in_chnl, out_chnl, N=r, shortcut=True, e=0.25)
        self.bn = torch.nn.BatchNorm2d(out_chnl)
        self.act = torch.nn.LeakyReLU(0.2)
        self.dropout = torch.nn.Dropout2d(dropout_rate)
    def forward(self, inp):
        x = self.mkc(inp)
        x = self.bn(x)
        x = self.act(x)
        x = self.dropout(x)
        return x
class PhaseRotation(nn.Module):
    def __init__(self, channels):
        super(PhaseRotation, self).__init__()
        self.channels = channels
        self.phase_weights = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, x):
        phase_weights = self.phase_weights(x) * math.pi
        cos_theta = torch.cos(phase_weights)
        sin_theta = torch.sin(phase_weights)
        half_ch = self.channels // 2
        if self.channels % 2 != 0:
            half_ch = self.channels // 2 + 1
        real_input = x[:, :half_ch]
        imag_input = x[:, half_ch:] if half_ch < self.channels else torch.zeros_like(real_input)

        real_output = real_input * cos_theta[:, :half_ch] - imag_input * sin_theta[:,
                                                               half_ch:] if half_ch < self.channels else real_input * cos_theta
        imag_output = real_input * sin_theta[:, :half_ch] + imag_input * cos_theta[:,
                                                               half_ch:] if half_ch < self.channels else real_input * sin_theta
        if half_ch < self.channels:
            out = torch.cat([real_output, imag_output], dim=1)
        else:
            out = real_output
        return out
class NoisyLaplacianConv(nn.Module):
    def __init__(self, channels):
        super(NoisyLaplacianConv, self).__init__()
        self.depth_conv = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1,
            groups=channels, bias=False
        )
        self.project = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
        with torch.no_grad():
            kernel = torch.zeros(channels, 1, 3, 3)  #
            laplacian_kernel = torch.tensor([
                [0.5, 1.0, 0.5],
                [1.0, -6.0, 1.0],
                [0.5, 1.0, 0.5]
            ]) * 0.1
            for i in range(channels):
                noise = torch.randn(3, 3) * 0.01
                kernel[i, 0] = laplacian_kernel + noise
            self.depth_conv.weight.data = kernel
    def forward(self, x):
        flow = self.depth_conv(x)
        out = self.project(flow + x)
        return out
class LaplacianBoostBlock(nn.Module):
    def __init__(self, channels, iterations=3):
        super(LaplacianBoostBlock, self).__init__()
        self.iterations = iterations
        self.channels = channels
        self.coef_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
        self.register_buffer('laplacian_kernel', torch.tensor([
            [0.25, 0.5, 0.25],
            [0.5, -3.0, 0.5],
            [0.25, 0.5, 0.25]
        ]).reshape(1, 1, 3, 3).repeat(channels, 1, 1, 1))
    def forward(self, x):
        coef = self.coef_predictor(x) * 0.1
        out = x
        for _ in range(self.iterations):
            laplacian = F.conv2d(
                out, self.laplacian_kernel, padding=1, groups=self.channels
            )
            out = out + coef * laplacian
        return out
class PGLC(nn.Module):
    def __init__(self, channels, n_levels, dropout_rate=0.1):
        super(PGLC, self).__init__()
        self.channels = channels
        self.n_levels = n_levels
        self.phase_rotation = PhaseRotation(channels)
        self.flows = nn.ModuleList([
            NoisyLaplacianConv(channels) for _ in range(n_levels)
        ])
        self.laplacianboost = LaplacianBoostBlock(channels, iterations=2)
        self.integration = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
        self.dropout = nn.Dropout2d(dropout_rate)
    def forward(self, x):
        identity = x
        x = self.phase_rotation(x)
        for i in range(self.n_levels):
            x = self.flows[i](x)
            if i < self.n_levels - 1:
                x = self.dropout(x)
        x = self.laplacianboost(x)
        x = self.integration(x)
        return x + identity
class ChannelSELayer(torch.nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super(ChannelSELayer, self).__init__()
        self.gp_avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.reduction_ratio = reduction_ratio
        num_channels_reduced = max(1, num_channels // self.reduction_ratio)
        self.fc1 = torch.nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = torch.nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.act = torch.nn.LeakyReLU(0.2)
        self.sigmoid = torch.nn.Sigmoid()
        self.bn = torch.nn.BatchNorm2d(num_channels)
        self.dropout = torch.nn.Dropout(0.2)
    def forward(self, inp):
        batch_size, num_channels, H, W = inp.size()
        out = self.act(self.fc1(self.gp_avg_pool(inp).view(batch_size, num_channels)))
        out = self.dropout(out)
        out = self.sigmoid(self.fc2(out))
        out = torch.mul(inp, out.view(batch_size, num_channels, 1, 1))
        out = self.bn(out)
        out = self.act(out)
        return out
class SEConvBlock(torch.nn.Module):
    def __init__(self, num_in_filters, num_out_filters, kernel_size, stride=(1, 1), activation="LeakyReLU",
                 dropout_rate=0.2):
        super().__init__()
        self.activation = torch.nn.LeakyReLU(0.2)
        self.conv1 = torch.nn.Conv2d(in_channels=num_in_filters, out_channels=num_out_filters, kernel_size=kernel_size,
                                     stride=stride, padding="same")
        self.batchnorm = torch.nn.BatchNorm2d(num_out_filters, momentum=0.01)
        self.sqe = ChannelSELayer(num_out_filters)
        self.dropout = torch.nn.Dropout2d(dropout_rate)
    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.sqe(x)
class Conv2d_channel(torch.nn.Module):
    def __init__(self, num_in_filters, num_out_filters, dropout_rate=0.2):
        super().__init__()
        self.activation = torch.nn.LeakyReLU(0.2)
        self.conv1 = torch.nn.Conv2d(in_channels=num_in_filters, out_channels=num_out_filters, kernel_size=(1, 1),
                                     padding="same")
        self.batchnorm = torch.nn.BatchNorm2d(num_out_filters, momentum=0.01)
        self.sqe = ChannelSELayer(num_out_filters)
        self.dropout = torch.nn.Dropout2d(dropout_rate)
    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.sqe(x)
class MLCA(torch.nn.Module):
    def __init__(self, in_filters1, in_filters2, in_filters3, in_filters4, dropout_rate=0.2):
        super().__init__()
        self.transform1 = SEConvBlock(in_filters1, in_filters1, (1, 1), dropout_rate=dropout_rate)
        self.transform2 = SEConvBlock(in_filters2, in_filters2, (1, 1), dropout_rate=dropout_rate)
        self.transform3 = SEConvBlock(in_filters3, in_filters3, (1, 1), dropout_rate=dropout_rate)
        self.transform4 = SEConvBlock(in_filters4, in_filters4, (1, 1), dropout_rate=dropout_rate)
        self.weight_predictor = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(in_filters1 + in_filters2 + in_filters3 + in_filters4, 4, kernel_size=1),
            torch.nn.Softmax(dim=1)
        )
        self.scale_attention = torch.nn.ModuleList([
            ChannelSELayer(in_filters1),
            ChannelSELayer(in_filters2),
            ChannelSELayer(in_filters3),
            ChannelSELayer(in_filters4)
        ])
        self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.down = torch.nn.AvgPool2d(2)
    def forward(self, x1, x2, x3, x4):
        t1 = self.transform1(x1)
        t2 = self.transform2(x2)
        t3 = self.transform3(x3)
        t4 = self.transform4(x4)
        t2_up = self.up(t2)
        t3_up = self.up(self.up(t3))
        t4_up = self.up(self.up(self.up(t4)))
        combined = torch.cat([t1, t2_up, t3_up, t4_up], dim=1)
        weights = self.weight_predictor(combined)
        out1 = t1 * weights[:, 0:1, :, :]
        out2 = t2 * weights[:, 1:2, :, :]
        out3 = t3 * weights[:, 2:3, :, :]
        out4 = t4 * weights[:, 3:4, :, :]
        out1 = self.scale_attention[0](out1 + x1)
        out2 = self.scale_attention[1](out2 + x2)
        out3 = self.scale_attention[2](out3 + x3)
        out4 = self.scale_attention[3](out4 + x4)
        return out1, out2, out3, out4
class MLCA_block(torch.nn.Module):
    def __init__(self, in_filters1, in_filters2, in_filters3, in_filters4, dropout_rate=0.2):
        super().__init__()
        self.mlca = MLCA(in_filters1, in_filters2, in_filters3, in_filters4, dropout_rate)
        self.g1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_filters1, in_filters1, kernel_size=1),
            torch.nn.BatchNorm2d(in_filters1, momentum=0.01),
            torch.nn.Sigmoid()
        )
        self.g2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_filters2, in_filters2, kernel_size=1),
            torch.nn.BatchNorm2d(in_filters2, momentum=0.01),
            torch.nn.Sigmoid()
        )
        self.g3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_filters3, in_filters3, kernel_size=1),
            torch.nn.BatchNorm2d(in_filters3, momentum=0.01),
            torch.nn.Sigmoid()
        )
        self.g4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_filters4, in_filters4, kernel_size=1),
            torch.nn.BatchNorm2d(in_filters4, momentum=0.01),
            torch.nn.Sigmoid()
        )
    def forward(self, x1, x2, x3, x4, t1=None, t2=None, t3=None, t4=None):
        if t1 is not None:
            x1 = x1 + t1 * self.g1(x1)
        if t2 is not None:
            x2 = x2 + t2 * self.g2(x2)
        if t3 is not None:
            x3 = x3 + t3 * self.g3(x3)
        if t4 is not None:
            x4 = x4 + t4 * self.g4(x4)
        return self.mlca(x1, x2, x3, x4)
class MMP_Net(torch.nn.Module):
    def __init__(self, n_channels, n_classes, n_filts=32, dropout_rate=0.2):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.pool = torch.nn.MaxPool2d(2)
        self.cnv1 = MKC_Layer(n_channels, n_filts, r=3, dropout_rate=dropout_rate)
        self.cnv2 = MKC_Layer(n_filts, n_filts * 2, r=3, dropout_rate=dropout_rate)
        self.cnv3 = MKC_Layer(n_filts * 2, n_filts * 4, r=3, dropout_rate=dropout_rate)
        self.cnv4 = MKC_Layer(n_filts * 4, n_filts * 8, r=3, dropout_rate=dropout_rate)
        self.cnv5 = MKC_Layer(n_filts * 8, n_filts * 16, r=3, dropout_rate=dropout_rate)
        self.pglc1 = PGLC(n_filts, 4, dropout_rate=dropout_rate)
        self.pglc2 = PGLC(n_filts * 2, 3, dropout_rate=dropout_rate)
        self.pglc3 = PGLC(n_filts * 4, 2, dropout_rate=dropout_rate)
        self.pglc4 = PGLC(n_filts * 8, 1, dropout_rate=dropout_rate)
        self.mlca1 = MLCA_block(n_filts, n_filts * 2, n_filts * 4, n_filts * 8, dropout_rate=dropout_rate)
        self.mlca2 = MLCA_block(n_filts, n_filts * 2, n_filts * 4, n_filts * 8, dropout_rate=dropout_rate)
        self.mlca3 = MLCA_block(n_filts, n_filts * 2, n_filts * 4, n_filts * 8, dropout_rate=dropout_rate)
        self.up6 = torch.nn.ConvTranspose2d(n_filts * 16, n_filts * 8, kernel_size=2, stride=2)
        self.cnv6 = MKC_Layer(n_filts * 8 + n_filts * 8, n_filts * 8, r=3, dropout_rate=dropout_rate)
        self.up7 = torch.nn.ConvTranspose2d(n_filts * 8, n_filts * 4, kernel_size=2, stride=2)
        self.cnv7 = MKC_Layer(n_filts * 4 + n_filts * 4, n_filts * 4, r=3, dropout_rate=dropout_rate)
        self.up8 = torch.nn.ConvTranspose2d(n_filts * 4, n_filts * 2, kernel_size=2, stride=2)
        self.cnv8 = MKC_Layer(n_filts * 2 + n_filts * 2, n_filts * 2, r=3, dropout_rate=dropout_rate)
        self.up9 = torch.nn.ConvTranspose2d(n_filts * 2, n_filts, kernel_size=2, stride=2)
        self.cnv9 = MKC_Layer(n_filts + n_filts, n_filts, r=3, dropout_rate=dropout_rate)
        if n_classes == 1:
            self.out = torch.nn.Conv2d(n_filts, n_classes, kernel_size=1)
            self.last_activation = torch.nn.Sigmoid()
        else:
            self.out = torch.nn.Conv2d(n_filts, n_classes + 1, kernel_size=1)
            self.last_activation = None
    def forward(self, x):
        x1 = x
        x2 = self.cnv1(x1)
        x2 = self.pglc1(x2)
        x2p = self.pool(x2)
        x3 = self.cnv2(x2p)
        x3 = self.pglc2(x3)
        x3p = self.pool(x3)
        x4 = self.cnv3(x3p)
        x4 = self.pglc3(x4)
        x4p = self.pool(x4)
        x5 = self.cnv4(x4p)
        x5 = self.pglc4(x5)
        x5p = self.pool(x5)
        x2, x3, x4, x5 = self.mlca1(x2, x3, x4, x5)
        x2, x3, x4, x5 = self.mlca2(x2, x3, x4, x5)
        x2, x3, x4, x5 = self.mlca3(x2, x3, x4, x5)
        x6 = self.cnv5(x5p)
        x7 = self.up6(x6)
        x7 = self.cnv6(torch.cat([x7, x5], dim=1))
        x8 = self.up7(x7)
        x8 = self.cnv7(torch.cat([x8, x4], dim=1))
        x9 = self.up8(x8)
        x9 = self.cnv8(torch.cat([x9, x3], dim=1))
        x10 = self.up9(x9)
        x10 = self.cnv9(torch.cat([x10, x2], dim=1))
        if self.last_activation is not None:
            logits = self.last_activation(self.out(x10))
        else:
            logits = self.out(x10)
        return logits
