# -*- coding: utf-8 -*-
"""DeepKANSeg-Swin-B 模型定义（自包含版本，无任何环境锁定与本地路径依赖）

来源：Wang et al., "Kolmogorov-Arnold Network for Remote Sensing Image Semantic
Segmentation", IEEE TGRS 2026（开源代码 github.com/sstary/SSRS）。

本队改动（相对论文）：
  - 编码器由 ResNet-18 / ViT 换为 Swin-B（ImageNet 公开权重）
  - DFR 通道由 512 降为 256（Swin 末层是 1024 维，防参数爆炸）
  - GLKAN 解码器 decode_channels=128（1024px 大图需要更宽）

KAN 基函数保持论文默认最优点：grid_size=5, spline_order=3。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from timm.layers import DropPath, trunc_normal_

NUM_CLASSES = 9


class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3,
                 scale_noise=0.1, scale_base=1.0, scale_spline=1.0,
                 enable_standalone_scale_spline=True,
                 base_activation=nn.SiLU, grid_eps=0.02, grid_range=(-1, 1)):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]).expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order))
        self.spline_scaler = nn.Parameter(torch.Tensor(out_features, in_features))
        self.scale_noise, self.scale_base, self.scale_spline = scale_noise, scale_base, scale_spline
        self.base_activation = base_activation()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = ((torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5)
                     * self.scale_noise / self.grid_size)
            self.spline_weight.data.copy_(self.curve2coeff(
                self.grid.T[self.spline_order:-self.spline_order], noise))
            nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x):
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = ((x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)])
                     * bases[:, :, :-1]) + \
                    ((grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)])
                     * bases[:, :, 1:])
        return bases.contiguous()

    def curve2coeff(self, x, y):
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        return solution.permute(2, 0, 1).contiguous()

    def forward(self, x):
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            (self.spline_weight * self.spline_scaler.unsqueeze(-1)).view(self.out_features, -1))
        return base_output + spline_output


# ================== KAN 模块 (移植自 deepKAN_ResNet.py, einops 改原生实现) ==================
class DW_bn_relu(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.relu(self.bn(self.dwconv(x)))
        return x.flatten(2).transpose(1, 2)


class KANLayer(nn.Module):
    """3 个 KANLinear + DwConv 交错 (论文 Fig.3(b) KAN block x3); no_kan 时退化为 Linear (论文自带消融开关)"""
    def __init__(self, dim, no_kan=False):
        super().__init__()
        Lin = (lambda: nn.Linear(dim, dim)) if no_kan else (lambda: KANLinear(dim, dim))
        self.fc1, self.fc2, self.fc3 = Lin(), Lin(), Lin()
        self.dw1 = DW_bn_relu(dim)
        self.dw2 = DW_bn_relu(dim)
        self.dw3 = DW_bn_relu(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.fc1(x.reshape(B * N, C)).reshape(B, N, C).contiguous()
        x = self.dw1(x, H, W)
        x = self.fc2(x.reshape(B * N, C)).reshape(B, N, C).contiguous()
        x = self.dw2(x, H, W)
        x = self.fc3(x.reshape(B * N, C)).reshape(B, N, C).contiguous()
        x = self.dw3(x, H, W)
        return x


class KANBlock(nn.Module):
    def __init__(self, dim, drop_path=0., no_kan=False):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm = nn.LayerNorm(dim)
        self.layer = KANLayer(dim, no_kan=no_kan)

    def forward(self, x, H, W):
        return x + self.drop_path(self.layer(self.norm(x), H, W))


class ConvBN(nn.Sequential):
    def __init__(self, cin, cout, k=3, stride=1):
        super().__init__(nn.Conv2d(cin, cout, k, stride, (k - 1) // 2, bias=False),
                         nn.BatchNorm2d(cout))


class ConvBNReLU(nn.Sequential):
    def __init__(self, cin, cout, k=3, stride=1):
        super().__init__(nn.Conv2d(cin, cout, k, stride, (k - 1) // 2, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU6())


class SeparableConvBN(nn.Sequential):
    def __init__(self, cin, cout, k=3, stride=1):
        super().__init__(nn.Conv2d(cin, cin, k, stride, (k - 1) // 2, groups=cin, bias=False),
                         nn.BatchNorm2d(cin),
                         nn.Conv2d(cin, cout, 1, bias=False))


class GlobalLocalAttention(nn.Module):
    """论文 Fig.4 的 GLAttn: 窗口自注意力(全局) + 卷积(局部), 原生 torch 实现"""
    def __init__(self, dim=128, num_heads=8, window_size=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.ws = window_size
        self.qkv = nn.Conv2d(dim, 3 * dim, 1, bias=False)
        self.local1 = ConvBN(dim, dim, 3)
        self.local2 = ConvBN(dim, dim, 1)
        self.proj = SeparableConvBN(dim, dim, k=window_size)
        self.attn_x = nn.AvgPool2d((window_size, 1), stride=1, padding=(window_size // 2 - 1, 0))
        self.attn_y = nn.AvgPool2d((1, window_size), stride=1, padding=(0, window_size // 2 - 1))
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads))
        coords = torch.stack(torch.meshgrid(torch.arange(window_size),
                                            torch.arange(window_size), indexing="ij"))
        cf = torch.flatten(coords, 1)
        rc = cf[:, :, None] - cf[:, None, :]
        rc = rc.permute(1, 2, 0).contiguous()
        rc[:, :, 0] += window_size - 1
        rc[:, :, 1] += window_size - 1
        rc[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("relative_position_index", rc.sum(-1))
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x):
        B, C, H, W = x.shape
        ws = self.ws
        local = self.local2(x) + self.local1(x)
        # pad 到窗口整数倍
        ph = (ws - H % ws) % ws; pw = (ws - W % ws) % ws
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        Hp, Wp = H + ph, W + pw
        hh, ww = Hp // ws, Wp // ws
        qkv = self.qkv(x)  # B,3C,Hp,Wp
        qkv = qkv.view(B, 3, self.num_heads, C // self.num_heads, hh, ws, ww, ws)
        qkv = qkv.permute(1, 0, 4, 6, 2, 5, 7, 3)  # 3,B,hh,ww,h,ws,ws,d
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q.reshape(B * hh * ww, self.num_heads, ws * ws, C // self.num_heads)
        k = k.reshape(B * hh * ww, self.num_heads, ws * ws, C // self.num_heads)
        v = v.reshape(B * hh * ww, self.num_heads, ws * ws, C // self.num_heads)
        dots = (q @ k.transpose(-2, -1)) * self.scale
        rpb = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        rpb = rpb.view(ws * ws, ws * ws, -1).permute(2, 0, 1).contiguous()
        dots = dots + rpb.unsqueeze(0)
        attn = dots.softmax(dim=-1) @ v  # B*hh*ww, h, ws*ws, d
        attn = attn.view(B, hh, ww, self.num_heads, ws, ws, C // self.num_heads)
        attn = attn.permute(0, 3, 6, 1, 4, 2, 5).reshape(B, C, Hp, Wp)
        attn = attn[:, :, :H, :W]
        out = self.attn_x(F.pad(attn, (0, 0, 0, 1), mode="reflect")) + \
              self.attn_y(F.pad(attn, (0, 1, 0, 0), mode="reflect"))
        out = out + local
        out = F.pad(out, (0, 1, 0, 1), mode="reflect")
        out = self.proj(out)
        return out[:, :, :H, :W]


class GLKANBlock(nn.Module):
    """论文 Fig.4: LN/BN + GLAttn + 残差, 然后 KANBlock; no_kan 消融为 Linear"""
    def __init__(self, dim, num_heads=8, window_size=8, no_kan=False):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = GlobalLocalAttention(dim, num_heads, window_size)
        self.kan = KANBlock(dim, no_kan=no_kan)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        B, _, H, W = x.shape
        z = x.flatten(2).transpose(1, 2)
        z = self.kan(z, H, W)
        return z.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()


class WF(nn.Module):
    """加权跳跃融合 (论文 weighted-sum skip)"""
    def __init__(self, in_channels, decode_channels, eps=1e-8):
        super().__init__()
        self.pre_conv = nn.Conv2d(in_channels, decode_channels, 1, bias=False)
        self.weights = nn.Parameter(torch.ones(2))
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, 3)

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        w = F.relu(self.weights)
        w = w / (w.sum() + self.eps)
        return self.post_conv(w[0] * self.pre_conv(res) + w[1] * x)


class FeatureRefinementHead(nn.Module):
    def __init__(self, in_channels, decode_channels):
        super().__init__()
        self.pre_conv = nn.Conv2d(in_channels, decode_channels, 1, bias=False)
        self.weights = nn.Parameter(torch.ones(2))
        self.eps = 1e-8
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, 3)
        self.pa = nn.Sequential(nn.Conv2d(decode_channels, decode_channels, 3, 1, 1,
                                          groups=decode_channels), nn.Sigmoid())
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                nn.Conv2d(decode_channels, decode_channels // 16, 1),
                                nn.ReLU6(),
                                nn.Conv2d(decode_channels // 16, decode_channels, 1),
                                nn.Sigmoid())
        self.shortcut = ConvBN(decode_channels, decode_channels, 1)
        self.proj = SeparableConvBN(decode_channels, decode_channels, 3)
        self.act = nn.ReLU6()

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        w = F.relu(self.weights)
        w = w / (w.sum() + self.eps)
        x = self.post_conv(w[0] * self.pre_conv(res) + w[1] * x)
        shortcut = self.shortcut(x)
        x = self.pa(x) * x + self.ca(x) * x
        return self.act(self.proj(x) + shortcut)


class KANDecoder(nn.Module):
    def __init__(self, encoder_channels=(128, 256, 512, 1024), decode_channels=128,
                 num_classes=9, window_size=8, b2_kan=False):
        super().__init__()
        dc = decode_channels
        self.pre_conv = ConvBN(encoder_channels[-1], dc, 1)
        self.b4 = GLKANBlock(dc, 8, window_size)
        self.p3 = WF(encoder_channels[-2], dc)
        self.b3 = GLKANBlock(dc, 8, window_size)
        self.p2 = WF(encoder_channels[-3], dc)
        # b2_kan=True 恢复论文完整形态 (梯度检查点防分页); False 为 6GB 显存消融版 Linear
        self.b2 = GLKANBlock(dc, 8, window_size, no_kan=not b2_kan)
        self.p1 = FeatureRefinementHead(encoder_channels[-4], dc)
        self.segmentation_head = nn.Sequential(
            ConvBNReLU(dc, dc), nn.Dropout2d(0.1),
            nn.Conv2d(dc, num_classes, 1))

    def forward(self, res1, res2, res3, res4, h, w):
        if self.training and torch.is_grad_enabled():
            # 梯度检查点: KAN 中间激活是显存大头, 重计算换显存防驱动分页
            x = _ckpt(self.b4, self.pre_conv(res4), use_reentrant=False)
            x = _ckpt(self.b3, self.p3(x, res3), use_reentrant=False)
            x = _ckpt(self.b2, self.p2(x, res2), use_reentrant=False)
        else:
            x = self.b4(self.pre_conv(res4))
            x = self.b3(self.p3(x, res3))
            x = self.b2(self.p2(x, res2))
        x = self.p1(x, res1)
        x = self.segmentation_head(x)
        return F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)


class DeepKANSegSwin(nn.Module):
    """Swin 编码器 + DFR(降维KAN精炼) + GLKAN 解码器"""
    ENC_CFG = {"swin_b": ("tu-swin_base_patch4_window7_224", (128, 256, 512, 1024)),
               "swin_t": ("tu-swin_tiny_patch4_window7_224", (96, 192, 384, 768))}

    def __init__(self, num_classes=9, decode_channels=128, dfr_dim=256, b2_kan=False,
                 enc="swin_b"):
        super().__init__()
        name, ch = self.ENC_CFG[enc]
        tmp = smp.UPerNet(encoder_name=name,
                          encoder_weights="imagenet", in_channels=3, classes=num_classes,
                          strict_img_size=False)
        self.encoder = tmp.encoder  # NCHW: [3,0,*ch] 取后4级
        del tmp
        self.dfr_in = ConvBN(ch[-1], dfr_dim, 1)
        self.dfr_blocks = nn.ModuleList([KANBlock(dfr_dim) for _ in range(4)])
        self.dfr_out = nn.Conv2d(dfr_dim, ch[-1], 1, bias=False)
        nn.init.zeros_(self.dfr_out.weight)  # 零初始化: 热启动时 DFR 恒等, 稳定起步
        self.decoder = KANDecoder(ch, decode_channels, num_classes,
                                  b2_kan=b2_kan)

    def forward(self, x):
        h, w = x.size()[-2:]
        fs = self.encoder(x)
        res1, res2, res3, res4 = fs[2], fs[3], fs[4], fs[5]
        B, _, H4, W4 = res4.shape
        z = self.dfr_in(res4)
        z = z.flatten(2).transpose(1, 2)
        if self.training and torch.is_grad_enabled():
            for blk in self.dfr_blocks:
                z = _ckpt(blk, z, H4, W4, use_reentrant=False)
        else:
            for blk in self.dfr_blocks:
                z = blk(z, H4, W4)
        z = z.transpose(1, 2).reshape(B, -1, H4, W4)
        res4 = res4 + self.dfr_out(z)
        return self.decoder(res1, res2, res3, res4, h, w)


# ================== 训练/评估/推理 (沿用 seg_swin 配方) ==================
def ckpt_for(b2_kan=False, pl=False, enc="swin_b"):
    if enc == "swin_t":
        return str(P.assert_run_path(os.path.join(ARTIFACT_ROOT, "ckpt_kan_t" + ("_pl" if pl else "")
                            + ("_b2" if b2_kan else "") + ".pth")
                            ))
    name = "ckpt_kan_pl.pth" if pl else ("ckpt_kan_b2.pth" if b2_kan else "ckpt_kan_swin.pth")
    return str(P.assert_run_path(os.path.join(ARTIFACT_ROOT, name)))


def build_pl_masks(tau, cache="pl_cache", tag=""):
    """pl_cache(npz) -> pl_masks{t}_{tag}_tXX/(png): conf<tau 的像素置 Ignore(0); 目录名随 tau/来源变化防陈旧"""
    out = str(P.assert_run_path(os.path.join(
        ARTIFACT_ROOT, f"pl_masks{'_' + tag if tag else ''}_t{int(round(tau * 100))}")))
    if os.path.isdir(out) and len([f for f in os.listdir(out) if f.endswith(".png")]) == 500:
        return out
    os.makedirs(out, exist_ok=True)
    src = str(P.assert_run_path(os.path.join(ARTIFACT_ROOT, cache)))
    for fn in sorted(f for f in os.listdir(src) if f.endswith(".npz")):
        z = np.load(os.path.join(src, fn))
        m = np.where(z["conf"].astype(np.float32) >= tau, z["pred"], 0).astype(np.uint8)
        Image.fromarray(m, mode="L").save(os.path.join(out, fn.replace(".npz", ".png")))
    print(f"[pl] 伪标签掩码已生成 τ={tau} -> {out}")
    return out


def build_model(b2_kan=False, enc="swin_b"):
    P.verify_inputs()
    return DeepKANSegSwin(NUM_CLASSES, b2_kan=b2_kan, enc=enc).to(DEVICE)

