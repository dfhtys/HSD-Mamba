import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from mamba_ssm import Mamba
except ImportError:
    raise ImportError("pip install mamba-ssm")


class StructureEnhancingSkip(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.edge_extractor = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        edge_weight = self.edge_extractor(x)
        out = x + (x * edge_weight)
        return self.proj(out)


def eight_way_scan(x):

    B, C, H, W = x.shape
    xs1 = x.flatten(2, 3)
    xs2 = x.transpose(2, 3).flatten(2, 3)
    xs3 = x.flip([2, 3]).flatten(2, 3)
    xs4 = x.transpose(2, 3).flip([2, 3]).flatten(2, 3)
    xs5 = x.flip([3]).flatten(2, 3)
    xs6 = x.transpose(2, 3).flip([2]).flatten(2, 3)
    xs7 = x.flip([2]).flatten(2, 3)
    xs8 = x.transpose(2, 3).flip([3]).flatten(2, 3)
    return torch.stack([xs1, xs2, xs3, xs4, xs5, xs6, xs7, xs8], dim=1)


def eight_way_merge(xs, H, W):

    B, _, C, _ = xs.shape
    y1 = xs[:, 0].view(B, C, H, W)
    y2 = xs[:, 1].view(B, C, W, H).transpose(2, 3)
    y3 = xs[:, 2].view(B, C, H, W).flip([2, 3])
    y4 = xs[:, 3].view(B, C, W, H).flip([2, 3]).transpose(2, 3)
    y5 = xs[:, 4].view(B, C, H, W).flip([3])
    y6 = xs[:, 5].view(B, C, W, H).flip([2]).transpose(2, 3)
    y7 = xs[:, 6].view(B, C, H, W).flip([2])
    y8 = xs[:, 7].view(B, C, W, H).flip([3]).transpose(2, 3)
    return torch.stack([y1, y2, y3, y4, y5, y6, y7, y8], dim=1)



class EightWayMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=3, expand=2):
        super().__init__()
        self.d_model = d_model


        self.conv_local = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=d_conv, padding=(d_conv - 1) // 2, groups=d_model, bias=False),
            nn.BatchNorm2d(d_model),
            nn.SiLU()
        )


        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=expand)


        self.attention_fusion = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(d_model, max(d_model // 4, 8), 1, bias=False),
            nn.BatchNorm2d(max(d_model // 4, 8)),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(d_model // 4, 8), 8, 1, bias=False),
            nn.Softmax(dim=1)
        )

        self.out_proj = nn.Sequential(
            nn.Conv2d(d_model, d_model, 1, bias=False),
            nn.BatchNorm2d(d_model)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x


        x_local = self.conv_local(x)


        xs = eight_way_scan(x_local)


        xs_flat = xs.permute(0, 1, 3, 2).contiguous().view(B * 8, H * W, C)
        xs_mamba = self.mamba(xs_flat)


        xs_mamba = xs_mamba.view(B, 8, H * W, C).permute(0, 1, 3, 2).contiguous()
        y_multi_dir = eight_way_merge(xs_mamba, H, W)


        attn_weights = self.attention_fusion(x_local)
        attn_weights = attn_weights.unsqueeze(2)
        # attn_weights = attn_weights.view(B, 8, 1, 1, 1)


        out = torch.sum(y_multi_dir * attn_weights, dim=1)

        return residual + self.out_proj(out)



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.01, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride), nn.BatchNorm2d(out_channels))

    def forward(self, x):
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + self.shortcut(x))



class Hybrid_EightWayMamba(nn.Module):
    def __init__(self, in_channels=1, channel=32):
        super().__init__()



        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, channel, 3, 1, 1),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU(0.01)
        )


        self.enc1 = ResidualBlock(channel, channel)
        self.p1 = nn.MaxPool2d(2)

        self.enc2 = ResidualBlock(channel, channel * 2)
        self.p2 = nn.MaxPool2d(2)


        self.ex3 = nn.Conv2d(channel * 2, channel * 4, 1)
        self.enc3 = EightWayMambaBlock(channel * 4)
        self.p3 = nn.MaxPool2d(2)

        self.ex4 = nn.Conv2d(channel * 4, channel * 8, 1)
        self.enc4 = EightWayMambaBlock(channel * 8)
        self.p4 = nn.MaxPool2d(2)

        self.b_conv = nn.Conv2d(channel * 8, channel * 16, 1)
        self.bridge = EightWayMambaBlock(channel * 16)


        self.up4 = nn.ConvTranspose2d(channel * 16, channel * 8, 2, 2)
        self.d4_fuse = nn.Conv2d(channel * 16, channel * 8, 1)
        self.dec4 = EightWayMambaBlock(channel * 8)
        self.skip4 = StructureEnhancingSkip(channel * 8)

        self.up3 = nn.ConvTranspose2d(channel * 8, channel * 4, 2, 2)
        self.d3_fuse = nn.Conv2d(channel * 8, channel * 4, 1)
        self.dec3 = EightWayMambaBlock(channel * 4)
        self.skip3 = StructureEnhancingSkip(channel * 4)

        self.up2 = nn.ConvTranspose2d(channel * 4, channel * 2, 2, 2)
        self.dec2 = ResidualBlock(channel * 4, channel * 2)
        self.skip2 = StructureEnhancingSkip(channel * 2)

        self.up1 = nn.ConvTranspose2d(channel * 2, channel, 2, 2)
        self.dec1 = ResidualBlock(channel * 2, channel)
        self.skip1 = StructureEnhancingSkip(channel)

        self.final = nn.Sequential(
            nn.Conv2d(channel, in_channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.input_conv(x)


        c1 = self.enc1(x)
        p1 = self.p1(c1)

        c2 = self.enc2(p1)
        p2 = self.p2(c2)

        c3 = self.enc3(self.ex3(p2))
        p3 = self.p3(c3)

        c4 = self.enc4(self.ex4(p3))
        p4 = self.p4(c4)

        b = self.bridge(self.b_conv(p4))


        d4 = self.up4(b)
        if d4.size() != c4.size(): d4 = F.interpolate(d4, c4.shape[2:])
        c4_skip = self.skip4(c4)
        d4 = self.dec4(self.d4_fuse(torch.cat([c4_skip, d4], 1)))

        d3 = self.up3(d4)
        if d3.size() != c3.size(): d3 = F.interpolate(d3, c3.shape[2:])
        c3_skip = self.skip3(c3)
        d3 = self.dec3(self.d3_fuse(torch.cat([c3_skip, d3], 1)))

        d2 = self.up2(d3)
        if d2.size() != c2.size(): d2 = F.interpolate(d2, c2.shape[2:])
        c2_skip = self.skip2(c2)
        d2 = self.dec2(torch.cat([c2_skip, d2], 1))

        d1 = self.up1(d2)
        if d1.size() != c1.size(): d1 = F.interpolate(d1, c1.shape[2:])
        c1_skip = self.skip1(c1)
        d1 = self.dec1(torch.cat([c1_skip, d1], 1))

        return self.final(d1)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")


    model =  Hybrid_EightWayMamba(in_channels=1, channel=32).to(device)
    print("Model initialized. Testing forward pass with multiple resolutions...")

    try:
        y_512 = model(torch.randn(1, 1, 512, 512).to(device))
        print(f"✅ Pass! Full-size testing output shape: {y_512.shape}")

        y_256 = model(torch.randn(1, 1, 256, 256).to(device))
        print(f"✅ Pass! Cropped training output shape: {y_256.shape}")

        print("🎉 Network refactored! 8-Way Mamba with Self-Attention is working perfectly.")
    except Exception as e:
        print(f"❌ Error: {e}")