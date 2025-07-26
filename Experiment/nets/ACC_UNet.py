"""
ACC-UNet architecture using PyTorch
"""

import torch


# Use the original ChannelSELayer but add dropout
class ChannelSELayer(torch.nn.Module):
    def __init__(self, num_channels, dropout_rate=0.2):  # Add dropout parameter
        super(ChannelSELayer, self).__init__()

        self.gp_avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.reduction_ratio = 8
        num_channels_reduced = num_channels // self.reduction_ratio

        self.fc1 = torch.nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = torch.nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.act = torch.nn.LeakyReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.bn = torch.nn.BatchNorm2d(num_channels)
        self.dropout = torch.nn.Dropout(dropout_rate)  # Add dropout layer

    def forward(self, inp):
        batch_size, num_channels, H, W = inp.size()

        out = self.gp_avg_pool(inp).view(batch_size, num_channels)
        out = self.act(self.fc1(out))
        out = self.dropout(out)  # Apply dropout
        out = self.sigmoid(self.fc2(out))

        out = torch.mul(inp, out.view(batch_size, num_channels, 1, 1))
        out = self.bn(out)
        out = self.act(out)

        return out


# Use the original HANCLayer - NOT the one from your modification
class HANCLayer(torch.nn.Module):
    def __init__(self, in_chnl, out_chnl, k):
        super(HANCLayer, self).__init__()

        self.k = k

        self.cnv = torch.nn.Conv2d((2 * k - 1) * in_chnl, out_chnl, kernel_size=(1, 1))
        self.act = torch.nn.LeakyReLU()
        self.bn = torch.nn.BatchNorm2d(out_chnl)

    def forward(self, inp):
        batch_size, num_channels, H, W = inp.size()

        x = inp

        if self.k == 1:
            x = inp

        elif self.k == 2:
            x = torch.concat(
                [
                    x,
                    torch.nn.Upsample(scale_factor=2)(torch.nn.AvgPool2d(2)(x)),
                    torch.nn.Upsample(scale_factor=2)(torch.nn.MaxPool2d(2)(x)),
                ],
                dim=2,
            )

        elif self.k == 3:
            x = torch.concat(
                [
                    x,
                    torch.nn.Upsample(scale_factor=2)(torch.nn.AvgPool2d(2)(x)),
                    torch.nn.Upsample(scale_factor=4)(torch.nn.AvgPool2d(4)(x)),
                    torch.nn.Upsample(scale_factor=2)(torch.nn.MaxPool2d(2)(x)),
                    torch.nn.Upsample(scale_factor=4)(torch.nn.MaxPool2d(4)(x)),
                ],
                dim=2,
            )

        elif self.k == 4:
            x = torch.concat(
                [
                    x,
                    torch.nn.Upsample(scale_factor=2)(torch.nn.AvgPool2d(2)(x)),
                    torch.nn.Upsample(scale_factor=4)(torch.nn.AvgPool2d(4)(x)),
                    torch.nn.Upsample(scale_factor=8)(torch.nn.AvgPool2d(8)(x)),
                    torch.nn.Upsample(scale_factor=2)(torch.nn.MaxPool2d(2)(x)),
                    torch.nn.Upsample(scale_factor=4)(torch.nn.MaxPool2d(4)(x)),
                    torch.nn.Upsample(scale_factor=8)(torch.nn.MaxPool2d(8)(x)),
                ],
                dim=2,
            )

        elif self.k == 5:
            x = torch.concat(
                [
                    x,
                    torch.nn.Upsample(scale_factor=2)(torch.nn.AvgPool2d(2)(x)),
                    torch.nn.Upsample(scale_factor=4)(torch.nn.AvgPool2d(4)(x)),
                    torch.nn.Upsample(scale_factor=8)(torch.nn.AvgPool2d(8)(x)),
                    torch.nn.Upsample(scale_factor=16)(torch.nn.AvgPool2d(16)(x)),
                    torch.nn.Upsample(scale_factor=2)(torch.nn.MaxPool2d(2)(x)),
                    torch.nn.Upsample(scale_factor=4)(torch.nn.MaxPool2d(4)(x)),
                    torch.nn.Upsample(scale_factor=8)(torch.nn.MaxPool2d(8)(x)),
                    torch.nn.Upsample(scale_factor=16)(torch.nn.MaxPool2d(16)(x)),
                ],
                dim=2,
            )

        x = x.view(batch_size, num_channels * (2 * self.k - 1), H, W)

        x = self.act(self.bn(self.cnv(x)))

        return x


# Modified HANCBlock with dropout and regularization
class HANCBlock(torch.nn.Module):
    def __init__(self, n_filts, out_channels, k=3, inv_fctr=3, dropout_rate=0.1):
        super().__init__()

        # Add L2 regularization through weight initialization
        def weight_init(m):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

        self.conv1 = torch.nn.Conv2d(n_filts, n_filts * inv_fctr, kernel_size=1)
        self.norm1 = torch.nn.BatchNorm2d(n_filts * inv_fctr)
        self.dropout1 = torch.nn.Dropout2d(dropout_rate)  # Add dropout

        self.conv2 = torch.nn.Conv2d(
            n_filts * inv_fctr,
            n_filts * inv_fctr,
            kernel_size=3,
            padding=1,
            groups=n_filts * inv_fctr,
        )
        self.norm2 = torch.nn.BatchNorm2d(n_filts * inv_fctr)
        self.dropout2 = torch.nn.Dropout2d(dropout_rate)  # Add dropout

        self.hnc = HANCLayer(n_filts * inv_fctr, n_filts, k)

        self.norm = torch.nn.BatchNorm2d(n_filts)

        self.conv3 = torch.nn.Conv2d(n_filts, out_channels, kernel_size=1)
        self.norm3 = torch.nn.BatchNorm2d(out_channels)
        self.dropout3 = torch.nn.Dropout2d(dropout_rate)  # Add dropout

        self.sqe = ChannelSELayer(out_channels, dropout_rate=dropout_rate)

        self.activation = torch.nn.LeakyReLU()

        # Apply weight initialization
        self.apply(weight_init)

    def forward(self, inp):
        x = self.conv1(inp)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.dropout1(x)  # Apply dropout

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)
        x = self.dropout2(x)  # Apply dropout

        x = self.hnc(x)

        x = self.norm(x + inp)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.activation(x)
        x = self.dropout3(x)  # Apply dropout
        x = self.sqe(x)

        return x


# Modified ResPath with dropout
class ResPath(torch.nn.Module):
    def __init__(self, in_chnls, n_lvl, dropout_rate=0.1):
        super(ResPath, self).__init__()

        self.convs = torch.nn.ModuleList([])
        self.bns = torch.nn.ModuleList([])
        self.sqes = torch.nn.ModuleList([])
        self.dropouts = torch.nn.ModuleList([])  # Add dropout list

        self.bn = torch.nn.BatchNorm2d(in_chnls)
        self.act = torch.nn.LeakyReLU()
        self.sqe = torch.nn.BatchNorm2d(in_chnls)

        for i in range(n_lvl):
            self.convs.append(
                torch.nn.Conv2d(in_chnls, in_chnls, kernel_size=(3, 3), padding=1)
            )
            self.bns.append(torch.nn.BatchNorm2d(in_chnls))
            self.sqes.append(ChannelSELayer(in_chnls, dropout_rate=dropout_rate))
            self.dropouts.append(torch.nn.Dropout2d(dropout_rate))  # Add dropout for each level

    def forward(self, x):
        for i in range(len(self.convs)):
            residual = x
            x_processed = self.convs[i](x)
            x_processed = self.bns[i](x_processed)
            x_processed = self.act(x_processed)
            x_processed = self.dropouts[i](x_processed)  # Apply dropout
            x_processed = self.sqes[i](x_processed)
            x = x + x_processed  # Residual connection

        return self.sqe(self.act(self.bn(x)))


# Reuse original MLFC class but add dropout
class MLFC(torch.nn.Module):
    def __init__(self, in_filters1, in_filters2, in_filters3, in_filters4, lenn=1, dropout_rate=0.1):
        super().__init__()

        self.in_filters1 = in_filters1
        self.in_filters2 = in_filters2
        self.in_filters3 = in_filters3
        self.in_filters4 = in_filters4
        self.in_filters = (
                in_filters1 + in_filters2 + in_filters3 + in_filters4
        )

        self.no_param_up = torch.nn.Upsample(scale_factor=2)
        self.no_param_down = torch.nn.AvgPool2d(2)

        # Add dropout to the original implementation
        self.dropout = torch.nn.Dropout2d(dropout_rate)

        self.cnv_blks1 = torch.nn.ModuleList([])
        self.cnv_blks2 = torch.nn.ModuleList([])
        self.cnv_blks3 = torch.nn.ModuleList([])
        self.cnv_blks4 = torch.nn.ModuleList([])

        self.cnv_mrg1 = torch.nn.ModuleList([])
        self.cnv_mrg2 = torch.nn.ModuleList([])
        self.cnv_mrg3 = torch.nn.ModuleList([])
        self.cnv_mrg4 = torch.nn.ModuleList([])

        self.bns1 = torch.nn.ModuleList([])
        self.bns2 = torch.nn.ModuleList([])
        self.bns3 = torch.nn.ModuleList([])
        self.bns4 = torch.nn.ModuleList([])

        self.bns_mrg1 = torch.nn.ModuleList([])
        self.bns_mrg2 = torch.nn.ModuleList([])
        self.bns_mrg3 = torch.nn.ModuleList([])
        self.bns_mrg4 = torch.nn.ModuleList([])

        for i in range(lenn):
            self.cnv_blks1.append(
                Conv2d_batchnorm(self.in_filters, in_filters1, (1, 1))
            )
            self.cnv_mrg1.append(Conv2d_batchnorm(2 * in_filters1, in_filters1, (1, 1)))
            self.bns1.append(torch.nn.BatchNorm2d(in_filters1))
            self.bns_mrg1.append(torch.nn.BatchNorm2d(in_filters1))

            self.cnv_blks2.append(
                Conv2d_batchnorm(self.in_filters, in_filters2, (1, 1))
            )
            self.cnv_mrg2.append(Conv2d_batchnorm(2 * in_filters2, in_filters2, (1, 1)))
            self.bns2.append(torch.nn.BatchNorm2d(in_filters2))
            self.bns_mrg2.append(torch.nn.BatchNorm2d(in_filters2))

            self.cnv_blks3.append(
                Conv2d_batchnorm(self.in_filters, in_filters3, (1, 1))
            )
            self.cnv_mrg3.append(Conv2d_batchnorm(2 * in_filters3, in_filters3, (1, 1)))
            self.bns3.append(torch.nn.BatchNorm2d(in_filters3))
            self.bns_mrg3.append(torch.nn.BatchNorm2d(in_filters3))

            self.cnv_blks4.append(
                Conv2d_batchnorm(self.in_filters, in_filters4, (1, 1))
            )
            self.cnv_mrg4.append(Conv2d_batchnorm(2 * in_filters4, in_filters4, (1, 1)))
            self.bns4.append(torch.nn.BatchNorm2d(in_filters4))
            self.bns_mrg4.append(torch.nn.BatchNorm2d(in_filters4))

        self.act = torch.nn.LeakyReLU()

        self.sqe1 = ChannelSELayer(in_filters1, dropout_rate=dropout_rate)
        self.sqe2 = ChannelSELayer(in_filters2, dropout_rate=dropout_rate)
        self.sqe3 = ChannelSELayer(in_filters3, dropout_rate=dropout_rate)
        self.sqe4 = ChannelSELayer(in_filters4, dropout_rate=dropout_rate)

    def forward(self, x1, x2, x3, x4):
        batch_size, _, h1, w1 = x1.shape
        _, _, h2, w2 = x2.shape
        _, _, h3, w3 = x3.shape
        _, _, h4, w4 = x4.shape

        for i in range(len(self.cnv_blks1)):
            x_c1 = self.act(
                self.bns1[i](
                    self.cnv_blks1[i](
                        torch.cat(
                            [
                                x1,
                                self.no_param_up(x2),
                                self.no_param_up(self.no_param_up(x3)),
                                self.no_param_up(self.no_param_up(self.no_param_up(x4))),
                            ],
                            dim=1,
                        )
                    )
                )
            )
            # Apply dropout after activation
            x_c1 = self.dropout(x_c1)

            x_c2 = self.act(
                self.bns2[i](
                    self.cnv_blks2[i](
                        torch.cat(
                            [
                                self.no_param_down(x1),
                                (x2),
                                (self.no_param_up(x3)),
                                (self.no_param_up(self.no_param_up(x4))),
                            ],
                            dim=1,
                        )
                    )
                )
            )
            # Apply dropout
            x_c2 = self.dropout(x_c2)

            x_c3 = self.act(
                self.bns3[i](
                    self.cnv_blks3[i](
                        torch.cat(
                            [
                                self.no_param_down(self.no_param_down(x1)),
                                self.no_param_down(x2),
                                (x3),
                                (self.no_param_up(x4)),
                            ],
                            dim=1,
                        )
                    )
                )
            )
            # Apply dropout
            x_c3 = self.dropout(x_c3)

            x_c4 = self.act(
                self.bns4[i](
                    self.cnv_blks4[i](
                        torch.cat(
                            [
                                self.no_param_down(self.no_param_down(self.no_param_down(x1))),
                                self.no_param_down(self.no_param_down(x2)),
                                self.no_param_down(x3),
                                x4,
                            ],
                            dim=1,
                        )
                    )
                )
            )
            # Apply dropout
            x_c4 = self.dropout(x_c4)

            x_c1 = self.act(
                self.bns_mrg1[i](
                    self.cnv_mrg1[i](
                        torch.cat([x_c1, x1], dim=2).view(batch_size, 2 * self.in_filters1, h1, w1)
                    )
                    + x1
                )
            )

            x_c2 = self.act(
                self.bns_mrg2[i](
                    self.cnv_mrg2[i](
                        torch.cat([x_c2, x2], dim=2).view(batch_size, 2 * self.in_filters2, h2, w2)
                    )
                    + x2
                )
            )

            x_c3 = self.act(
                self.bns_mrg3[i](
                    self.cnv_mrg3[i](
                        torch.cat([x_c3, x3], dim=2).view(batch_size, 2 * self.in_filters3, h3, w3)
                    )
                    + x3
                )
            )

            x_c4 = self.act(
                self.bns_mrg4[i](
                    self.cnv_mrg4[i](
                        torch.cat([x_c4, x4], dim=2).view(batch_size, 2 * self.in_filters4, h4, w4)
                    )
                    + x4
                )
            )

        x1 = self.sqe1(x_c1)
        x2 = self.sqe2(x_c2)
        x3 = self.sqe3(x_c3)
        x4 = self.sqe4(x_c4)

        return x1, x2, x3, x4


# Ensure you define Conv2d_batchnorm correctly
class Conv2d_batchnorm(torch.nn.Module):
    def __init__(
            self,
            num_in_filters,
            num_out_filters,
            kernel_size,
            stride=(1, 1),
            activation="LeakyReLU",
            dropout_rate=0.1
    ):
        super().__init__()
        self.activation = torch.nn.LeakyReLU()
        self.conv1 = torch.nn.Conv2d(
            in_channels=num_in_filters,
            out_channels=num_out_filters,
            kernel_size=kernel_size,
            stride=stride,
            padding="same",
        )
        self.batchnorm = torch.nn.BatchNorm2d(num_out_filters)
        self.sqe = ChannelSELayer(num_out_filters, dropout_rate=dropout_rate)
        self.dropout = torch.nn.Dropout2d(dropout_rate)  # Add dropout

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)
        x = self.activation(x)
        x = self.dropout(x)  # Apply dropout
        x = self.sqe(x)
        return x


# Modified ACC_UNet class with dropout and complexity control
class ACC_UNet(torch.nn.Module):
    def __init__(self, n_channels, n_classes, n_filts=32, dropout_rate=0.2, reduce_complexity=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Optionally reduce model complexity
        if reduce_complexity:
            k_values = [2, 2, 1, 1, 1]  # Lower k parameters for HANC
            inv_fctr = 2  # Lower inflation factor
        else:
            k_values = [3, 3, 3, 2, 1]  # Original k values
            inv_fctr = 3  # Original inflation factor

        self.pool = torch.nn.MaxPool2d(2)

        self.cnv11 = HANCBlock(n_channels, n_filts, k=k_values[0], inv_fctr=inv_fctr, dropout_rate=dropout_rate)
        self.cnv12 = HANCBlock(n_filts, n_filts, k=k_values[0], inv_fctr=inv_fctr, dropout_rate=dropout_rate)

        self.cnv21 = HANCBlock(n_filts, n_filts * 2, k=k_values[1], inv_fctr=inv_fctr, dropout_rate=dropout_rate)
        self.cnv22 = HANCBlock(n_filts * 2, n_filts * 2, k=k_values[1], inv_fctr=inv_fctr, dropout_rate=dropout_rate)

        self.cnv31 = HANCBlock(n_filts * 2, n_filts * 4, k=k_values[2], inv_fctr=inv_fctr, dropout_rate=dropout_rate)
        self.cnv32 = HANCBlock(n_filts * 4, n_filts * 4, k=k_values[2], inv_fctr=inv_fctr, dropout_rate=dropout_rate)

        self.cnv41 = HANCBlock(n_filts * 4, n_filts * 8, k=k_values[3], inv_fctr=inv_fctr, dropout_rate=dropout_rate)
        self.cnv42 = HANCBlock(n_filts * 8, n_filts * 8, k=k_values[3], inv_fctr=inv_fctr, dropout_rate=dropout_rate)

        self.cnv51 = HANCBlock(n_filts * 8, n_filts * 16, k=k_values[4], inv_fctr=inv_fctr, dropout_rate=dropout_rate)
        self.cnv52 = HANCBlock(n_filts * 16, n_filts * 16, k=k_values[4], inv_fctr=inv_fctr, dropout_rate=dropout_rate)

        # Keep ResPath but add dropout
        self.rspth1 = ResPath(n_filts, 4, dropout_rate=dropout_rate)
        self.rspth2 = ResPath(n_filts * 2, 3, dropout_rate=dropout_rate)
        self.rspth3 = ResPath(n_filts * 4, 2, dropout_rate=dropout_rate)
        self.rspth4 = ResPath(n_filts * 8, 1, dropout_rate=dropout_rate)

        # Reduce MLFC layers count optionally
        if reduce_complexity:
            self.mlfc = MLFC(n_filts, n_filts * 2, n_filts * 4, n_filts * 8, lenn=1, dropout_rate=dropout_rate)
        else:
            self.mlfc1 = MLFC(n_filts, n_filts * 2, n_filts * 4, n_filts * 8, lenn=1, dropout_rate=dropout_rate)
            self.mlfc2 = MLFC(n_filts, n_filts * 2, n_filts * 4, n_filts * 8, lenn=1, dropout_rate=dropout_rate)
            self.mlfc3 = MLFC(n_filts, n_filts * 2, n_filts * 4, n_filts * 8, lenn=1, dropout_rate=dropout_rate)

        # Decoder part
        self.up6 = torch.nn.ConvTranspose2d(n_filts * 16, n_filts * 8, kernel_size=(2, 2), stride=2)
        self.cnv61 = HANCBlock(n_filts * 8 + n_filts * 8, n_filts * 8, k=k_values[3], inv_fctr=inv_fctr,
                               dropout_rate=dropout_rate)
        self.cnv62 = HANCBlock(n_filts * 8, n_filts * 8, k=k_values[3], inv_fctr=inv_fctr, dropout_rate=dropout_rate)

        self.up7 = torch.nn.ConvTranspose2d(n_filts * 8, n_filts * 4, kernel_size=(2, 2), stride=2)
        self.cnv71 = HANCBlock(n_filts * 4 + n_filts * 4, n_filts * 4, k=k_values[2], inv_fctr=inv_fctr,
                               dropout_rate=dropout_rate)
        self.cnv72 = HANCBlock(n_filts * 4, n_filts * 4, k=k_values[2], inv_fctr=inv_fctr, dropout_rate=dropout_rate)

        self.up8 = torch.nn.ConvTranspose2d(n_filts * 4, n_filts * 2, kernel_size=(2, 2), stride=2)
        self.cnv81 = HANCBlock(n_filts * 2 + n_filts * 2, n_filts * 2, k=k_values[1], inv_fctr=inv_fctr,
                               dropout_rate=dropout_rate)
        self.cnv82 = HANCBlock(n_filts * 2, n_filts * 2, k=k_values[1], inv_fctr=inv_fctr, dropout_rate=dropout_rate)

        self.up9 = torch.nn.ConvTranspose2d(n_filts * 2, n_filts, kernel_size=(2, 2), stride=2)
        self.cnv91 = HANCBlock(n_filts + n_filts, n_filts, k=k_values[0], inv_fctr=inv_fctr, dropout_rate=dropout_rate)
        self.cnv92 = HANCBlock(n_filts, n_filts, k=k_values[0], inv_fctr=inv_fctr, dropout_rate=dropout_rate)

        self.reduce_complexity = reduce_complexity
        self.dropout = torch.nn.Dropout2d(dropout_rate)

        if n_classes == 1:
           self.out = torch.nn.Conv2d(n_filts, n_classes, kernel_size=(1, 1))
           self.last_activation = torch.nn.Sigmoid()
        else:
           self.out = torch.nn.Conv2d(n_filts, n_classes, kernel_size=(1, 1))  # 改为n_classes而不是n_classes+1
           self.last_activation = torch.nn.Softmax(dim=1)  # 添加Softmax激活

    def forward(self, x):
        x1 = x

        x2 = self.cnv11(x1)
        x2 = self.cnv12(x2)

        x2p = self.pool(x2)

        x3 = self.cnv21(x2p)
        x3 = self.cnv22(x3)

        x3p = self.pool(x3)

        x4 = self.cnv31(x3p)
        x4 = self.cnv32(x4)

        x4p = self.pool(x4)

        x5 = self.cnv41(x4p)
        x5 = self.cnv42(x5)

        x5p = self.pool(x5)

        x6 = self.cnv51(x5p)
        x6 = self.cnv52(x6)

        x2 = self.rspth1(x2)
        x3 = self.rspth2(x3)
        x4 = self.rspth3(x4)
        x5 = self.rspth4(x5)

        # Adjust MLFC usage based on complexity setting
        if self.reduce_complexity:
            x2, x3, x4, x5 = self.mlfc(x2, x3, x4, x5)
        else:
            x2, x3, x4, x5 = self.mlfc1(x2, x3, x4, x5)
            x2, x3, x4, x5 = self.mlfc2(x2, x3, x4, x5)
            x2, x3, x4, x5 = self.mlfc3(x2, x3, x4, x5)

        x7 = self.up6(x6)
        x7 = self.cnv61(torch.cat([x7, x5], dim=1))
        x7 = self.cnv62(x7)

        x8 = self.up7(x7)
        x8 = self.cnv71(torch.cat([x8, x4], dim=1))
        x8 = self.cnv72(x8)

        x9 = self.up8(x8)
        x9 = self.cnv81(torch.cat([x9, x3], dim=1))

        x9 = self.cnv82(x9)

        x10 = self.up9(x9)
        x10 = self.cnv91(torch.cat([x10, x2], dim=1))
        x10 = self.cnv92(x10)

        if self.last_activation is not None:
            logits = self.last_activation(self.out(x10))

        else:
            logits = self.out(x10)

        return logits
