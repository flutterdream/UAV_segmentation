# -*- coding: utf-8 -*-
"""
v5: DeepKANSeg 框架移植到 Swin-B (单模型, 竞赛合规, 仅官方数据+ImageNet公开权重)
- 框架来源: Wang et al., "Kolmogorov-Arnold Network for Remote Sensing Image
  Semantic Segmentation", IEEE TGRS 2026 (代码 github.com/sstary/SSRS)
- 适配: 编码器换为我们已训练的 Swin-B (从 v4.1 EMA 热启动);
  DFR 降维 1024->256 防参数爆炸 (论文 ResNet18 为 512 维, Swin 为 1024 维);
  GLKAN 解码器 decode_channels=128 (论文 64, 我们 1024px 大图加宽)
- 训练配方与 v4.1 相同: 强增强+边界加权CE+Dice+0.5*Lovász+EMA+温和采样
用法:
  python kan_swin.py --train --crop 640 --target-iters 8000
  python kan_swin.py --eval / --eval-ms / --calibrate / --infer
"""
import argparse, json, math, os, random, time
import numpy as np
from PIL import Image

import strict_env as P
P.configure_process()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _ckpt
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from timm.layers import DropPath, trunc_normal_

import seg_swin as S  # 复用数据集/预处理/推理/评估

ROOT = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_ROOT = str(P.RUN_ROOT)
CKPT = os.path.join(ARTIFACT_ROOT, "ckpt_kan_swin.pth")
BIAS = os.path.join(ARTIFACT_ROOT, "bias_kan_swin.npy")
LOG = os.path.join(ARTIFACT_ROOT, "log_kan_swin.json")
NUM_CLASSES, IGNORE = 9, 0
DEVICE = S.DEVICE
CE_WEIGHTS = S.CE_WEIGHTS


# ================== KAN 层 (移植自 sstary/SSRS DeepKANSeg/model/kan.py) ==================
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


def train(a):
    _sd = getattr(a, "seed", 42)
    random.seed(_sd); np.random.seed(_sd); torch.manual_seed(_sd)
    val_ids = json.load(open(os.path.join(ROOT, "split.json"), encoding="utf-8"))["val_ids"]
    all_ids = list(range(6996))
    random.Random(123).shuffle(all_ids)
    if val_ids != all_ids[:500]:
        raise RuntimeError("fixed split does not match seed123 shuffled first 500 IDs")
    train_ids = all_ids[500:]  # 历史顺序：禁止排序
    forbidden_legacy = [name for name in ("distill", "bn", "exclude_noise", "weak_fix", "b2_kan")
                        if getattr(a, name, False)]
    if forbidden_legacy:
        raise RuntimeError("strict chain disables legacy experiment flags: " + ", ".join(forbidden_legacy))

    enc = getattr(a, "enc", "swin_b")
    model = build_model(getattr(a, "b2_kan", False), enc)
    if getattr(a, "ckpt_name", None):  # 自定义存档名 (新管线分支, 最高优先级)
        if os.path.basename(a.ckpt_name) != a.ckpt_name or not a.ckpt_name.endswith(".pth"):
            raise ValueError("--ckpt-name must be a plain .pth filename")
        CKPT = str(P.assert_run_path(os.path.join(ARTIFACT_ROOT, a.ckpt_name)))
    elif getattr(a, "anchor", False):
        CKPT = str(P.assert_run_path(os.path.join(ARTIFACT_ROOT, "ckpt_kan_anchor.pth")))
    else:
        CKPT = ckpt_for(getattr(a, "b2_kan", False), getattr(a, "pl", False), enc)
    if a.crop >= 640:
        _inner = getattr(model.encoder, "model", None)
        if _inner is not None and hasattr(_inner, "set_grad_checkpointing"):
            _inner.set_grad_checkpointing(True)
            print("[grad-ckpt] 已启用 (crop>=640)")
    dice = smp.losses.DiceLoss(mode="multiclass", ignore_index=IGNORE)
    _ce = list(CE_WEIGHTS)
    w = torch.tensor([1.0] + _ce, dtype=torch.float32).to(DEVICE)
    lovasz = smp.losses.LovaszLoss(mode="multiclass", ignore_index=IGNORE)

    # 参数分组: 编码器低 lr 保护, 新 DFR/解码器高 lr
    enc_params = list(model.encoder.parameters())
    new_params = [p for m_ in (model.dfr_in, model.dfr_blocks, model.dfr_out, model.decoder)
                  for p in m_.parameters()]
    opt = torch.optim.AdamW([
        {"params": enc_params, "lr": 3e-5},
        {"params": new_params, "lr": 1.5e-4}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.target_iters, eta_min=3e-6)

    start, hist, ema = 0, [], None
    parent_names = ["public_swin_b"]
    if os.path.exists(CKPT):
        sd = P.load_checkpoint(CKPT, map_location=DEVICE)
        model.load_state_dict(sd["model"]); opt.load_state_dict(sd["opt"])
        sched.load_state_dict(sd["sched"]); start = sd["iter"]; hist = sd.get("hist", [])
        ema = sd.get("ema")
        parent_names = list(sd["strict_repro"].get("parents", parent_names))
        print(f"[resume] from iter {start}")
        if a.restart_lr > 0:  # 热重启: 余弦已收敛则按剩余步数开新周期
            for g, base in zip(opt.param_groups, (3e-5, 1.5e-4)):
                g["lr"] = base * a.restart_lr
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=a.target_iters - start, eta_min=3e-6)
            print(f"[restart-lr] x{a.restart_lr} 新余弦 T_max={a.target_iters - start}")
    elif (getattr(a, "pl", False) or getattr(a, "anchor", False)) and enc == "swin_b":
        if not a.warm_base:
            raise RuntimeError("PL/anchor stage requires an explicit --warm-base from strict run root")
        warm_src = str(P.assert_run_path(os.path.join(ARTIFACT_ROOT, a.warm_base), must_exist=True))
        sd = P.load_checkpoint(warm_src, map_location=DEVICE)
        if a.warm_parent_stage:
            meta = sd.get("strict_repro", {})
            if (meta.get("stage") != a.warm_parent_stage or
                    (a.warm_parent_iter is not None and sd.get("iter") != a.warm_parent_iter)):
                raise RuntimeError("PL/anchor warm parent does not match declared lineage")
        model.load_state_dict(sd.get("ema") or sd["model"])
        parent_names = [os.path.basename(warm_src)]
        print(f"[warm-{'anchor' if getattr(a, 'anchor', False) else 'pl'}] "
              f"{os.path.basename(warm_src)} EMA iter={sd['iter']}")
        if a.restart_lr > 0:
            for g, base in zip(opt.param_groups, (3e-5, 1.5e-4)):
                g["lr"] = base * a.restart_lr
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=a.target_iters, eta_min=3e-6)
            print(f"[restart-lr] x{a.restart_lr} 自暖启动新余弦 T_max={a.target_iters}")
    elif enc == "swin_b":  # 从 v4.1 (640) EMA 热启动编码器 (仅 swin_b 谱系)
        src_name = a.encoder_warm or os.path.basename(S.ckpt_path(640))
        src = str(P.assert_run_path(os.path.join(ARTIFACT_ROOT, src_name), must_exist=True))
        sd = P.load_checkpoint(src, map_location=DEVICE)
        if a.encoder_parent_stage:
            meta = sd.get("strict_repro", {})
            if (sd.get("iter") != 4000 or meta.get("stage") != a.encoder_parent_stage or
                    meta.get("seed") != 42):
                raise RuntimeError("KAN encoder parent does not match declared v4.1 lineage")
        enc_sd = {k[8:]: v for k, v in (sd.get("ema") or sd["model"]).items()
                  if k.startswith("encoder.")}
        missing, unexpected = model.encoder.load_state_dict(enc_sd, strict=False)
        parent_names = [os.path.basename(src)]
        print(f"[warm] v4.1 encoder EMA iter={sd['iter']} "
              f"(missing={len(missing)} unexpected={len(unexpected)})")

    stats = np.load(os.path.join(ROOT, "train_mask_stats.npy"))
    boost = np.array([0, 1.2, 1.0, 1.0, 1.0, 1.5, 1.0, 1.0, 1.5], dtype=np.float32)
    wi = (stats[train_ids] * boost).sum(axis=1)
    wi = wi / wi.mean()
    ds = S.UAV(train_ids, True, a.crop, extra=True)
    pl_dir = None
    if getattr(a, "anchor", False):
        if not a.anchor_dir:
            raise RuntimeError("anchor stage requires --anchor-dir")
        pl_dir = str(P.assert_run_path(os.path.join(ARTIFACT_ROOT, a.anchor_dir)))
        if (not os.path.isdir(pl_dir) or
                len([f for f in os.listdir(pl_dir) if f.endswith(".png")]) != 500):
            raise RuntimeError(f"independent-teacher anchor directory must contain 500 files: {pl_dir}")
    elif getattr(a, "pl", False):  # P1: test1 伪标签回灌, 队友配方 3x 采样
        pl_dir = build_pl_masks(getattr(a, "pl_tau", 0.85),
                                cache=(getattr(a, "pl_cache", None) or "pl_cache"),
                                tag=(getattr(a, "pl_tag", None) or ""))
    if pl_dir:
        pl_ids = [f[:-4] for f in sorted(os.listdir(pl_dir))]
        ds_pl = S.UAV(pl_ids, True, a.crop, extra=True,
                      img_dir=S.TEST_DIR, msk_dir=pl_dir, name_fmt="{k}.png")
        ds = torch.utils.data.ConcatDataset([ds, ds_pl])
        wi = np.concatenate([wi, np.full(len(pl_ids), 3.0, np.float32)])
        kind = "本次独立教师锚标签" if getattr(a, "anchor", False) else f"τ={a.pl_tau} 伪标签"
        print(f"[pl] {kind} {len(pl_ids)} 张 @3x, "
              f"批次占比≈{len(pl_ids) * 3.0 / wi.sum():.0%}")
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.from_numpy(wi).double(), num_samples=len(train_ids) * 4, replacement=True)
    dl = DataLoader(ds, batch_size=a.batch_size, sampler=sampler,
                    num_workers=a.num_workers, persistent_workers=(a.num_workers > 0), drop_last=True)

    if ema is None:
        ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print("[kan] EMA 已初始化 (decay=0.999)")
    ema_pairs = []
    msd = model.state_dict()
    for k, et in ema.items():
        if et.dtype.is_floating_point and k in msd:
            ema_pairs.append((et, msd[k]))
    ema_list = [p[0] for p in ema_pairs]
    src_list = [p[1] for p in ema_pairs]
    print(f"[EMA] 绑定 {len(ema_pairs)} 个张量")

    def save_ck(it, hist):
        tmp = CKPT + ".tmp"  # 原子写: 防超时掐断损坏存档
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(),
                        sched=sched.state_dict(), iter=it, hist=hist, ema=ema,
                        strict_repro=P.strict_metadata(a.stage, _sd, parent_names)), tmp)
        os.replace(tmp, CKPT)
        if it % 1000 == 0:  # 峰值狩猎: 每1000步留轻量快照(仅推理所需), 防峰值被覆盖
            if getattr(a, "ckpt_name", None):
                tag = a.ckpt_name[:-4] if a.ckpt_name.endswith(".pth") else a.ckpt_name
            else:
                tag = ("ckpt_kan" + ("t" if enc == "swin_t" else "")
                       + ("anchor" if getattr(a, "anchor", False)
                          else "pl" if getattr(a, "pl", False) else "")
                       + ("b2" if getattr(a, "b2_kan", False) else ""))
            torch.save(dict(model=model.state_dict(), iter=it, ema=ema,
                            strict_repro=P.strict_metadata(a.stage, _sd, parent_names)),
                       str(P.assert_run_path(os.path.join(ARTIFACT_ROOT, f"{tag}_i{it}.pth"))))

    it = start; t0 = time.time(); model.train(); done = False
    opt.zero_grad()
    while it < a.target_iters and not done:
        for x, y, wmap in dl:
            if it >= a.target_iters or time.time() - t0 > a.max_seconds:
                done = True; break
            x, y, wmap = x.to(DEVICE), y.to(DEVICE), wmap.to(DEVICE)
            with torch.autocast("cuda", torch.float16, enabled=DEVICE == "cuda"):
                out = model(x)
                ce_none = F.cross_entropy(out, y, weight=w, ignore_index=IGNORE, reduction="none")
                out_ds = F.interpolate(out, scale_factor=0.5, mode="bilinear", align_corners=False)
                y_ds = F.interpolate(y.unsqueeze(1).float(), scale_factor=0.5,
                                     mode="nearest").squeeze(1).long()
                loss = ((ce_none * wmap).sum() / wmap.sum() + dice(out, y)
                        + 0.5 * lovasz(out_ds, y_ds))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            with torch.no_grad():
                torch._foreach_mul_(ema_list, 0.999)
                torch._foreach_add_(ema_list, src_list, alpha=0.001)
            hist.append(round(loss.item(), 4)); it += 1
            if it % 50 == 0:
                print(f"iter {it}/{a.target_iters} loss={np.mean(hist[-50:]):.4f} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
            if it % 200 == 0:
                save_ck(it, hist)
    save_ck(it, hist)
    print(f"[saved] iter={it}, peak={torch.cuda.max_memory_allocated()/2**30:.2f}GB")
    print("[done]" if it >= a.target_iters else f"[partial] {it}/{a.target_iters}")


def load_trained(use_ema=True, path=None, b2_kan=False, pl=False, enc="swin_b"):
    model = DeepKANSegSwin(NUM_CLASSES, b2_kan=b2_kan, enc=enc)
    model.encoder = model.encoder  # 结构已建
    p = path or ckpt_for(b2_kan, pl, enc)
    p = str(P.assert_run_path(p, must_exist=True))
    sd = P.load_checkpoint(p, map_location="cpu")
    model.load_state_dict(sd["ema"] if (use_ema and sd.get("ema") is not None) else sd["model"])
    model = model.to(DEVICE)
    print(f"[load] {os.path.basename(p)} {'EMA ' if use_ema else ''}iter={sd['iter']}")
    return model, sd["iter"]


def load_bias():
    if os.path.exists(BIAS):
        b = torch.from_numpy(np.load(BIAS).astype(np.float32))
        print(f"[bias:bias_kan_swin.npy] {np.round(b.numpy(), 3).tolist()}")
        return b
    return None


def run_eval(tta, a, use_ema=True):
    model, iters = load_trained(use_ema, path=getattr(a, "ckpt", None),
                                b2_kan=getattr(a, "b2_kan", False),
                                pl=getattr(a, "pl", False),
                                enc=getattr(a, "enc", "swin_b"))
    val_ids = json.load(open(os.path.join(ROOT, "split.json")))["val_ids"]
    bias = load_bias()
    miou, ious = S.evaluate(model, val_ids[:a.n], tta, bias)
    res = dict(iters=iters, tta=tta, n=a.n, ema=use_ema, bias=(bias is not None),
               val_mIoU=round(miou, 4),
               val_IoU={n: (None if np.isnan(v) else round(v, 4))
                        for n, v in zip(S.CLASS_NAMES, ious)})
    key = "kan_" + ("plain" if not tta else (f"{tta}_n{a.n}" if tta in ("ms", "ms2") else f"tta_n{a.n}"))
    if use_ema:
        key += "_ema"
    if bias is not None:
        key += "_bias"
    log = json.load(open(LOG)) if os.path.exists(LOG) else {}
    log[key] = res
    json.dump(log, open(LOG, "w"), ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False, indent=2))


@torch.no_grad()
def calibrate(a):
    model, iters = load_trained(use_ema=not a.raw, b2_kan=getattr(a, "b2_kan", False),
                                pl=getattr(a, "pl", False), enc=getattr(a, "enc", "swin_b")); model.eval()
    n = min(a.n, 100)
    val_ids = json.load(open(os.path.join(ROOT, "split.json")))["val_ids"][:n]
    cache_ms, cache_tta, cache_plain = [], [], []
    for i, k in enumerate(val_ids):
        img = np.array(Image.open(os.path.join(S.IMG_DIR, f"{k:04d}.png")).convert("RGB"))
        msk = np.array(Image.open(os.path.join(S.MSK_DIR, f"{k:04d}.png")))
        cache_ms.append((S.predict_logits(model, img, tta="ms").half(), msk))
        cache_tta.append((S.predict_logits(model, img, tta=True).half(), msk))
        cache_plain.append((S.predict_logits(model, img, tta=False).half(), msk))
        if (i + 1) % 25 == 0:
            print(f"  cached {i+1}/{len(val_ids)}", flush=True)

    def score(b, cache):
        conf = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
        bt = torch.from_numpy(b).view(-1, 1, 1)
        for lg, msk in cache:
            pred = (lg[:, ::2, ::2].float() + bt).argmax(0).numpy().ravel()
            t = msk[::2, ::2].ravel()
            m = t != IGNORE
            conf += np.bincount(NUM_CLASSES * t[m] + pred[m],
                                minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)
        inter = np.diag(conf).astype(np.float64)
        union = conf.sum(1) + conf.sum(0) - inter
        iou = [inter[c] / union[c] if union[c] > 0 else float("nan") for c in range(1, NUM_CLASSES)]
        return float(np.nanmean(iou)), iou

    bias = np.zeros(NUM_CLASSES, np.float32)
    base_ms, _ = score(bias, cache_ms)
    base_tta, _ = score(bias, cache_tta)
    base_plain, _ = score(bias, cache_plain)
    routes = {"ms": (base_ms, cache_ms), "tta": (base_tta, cache_tta),
              "plain": (base_plain, cache_plain)}
    route = max(routes, key=lambda r: routes[r][0])
    best, cache = routes[route]
    print(f"[calib] iters={iters} n={n} base: ms={base_ms:.4f} tta={base_tta:.4f} "
          f"plain={base_plain:.4f} -> 校准 {route} 路", flush=True)
    for rd in range(3):
        improved = False
        for c in range(1, NUM_CLASSES):
            for d in (0.05, -0.05):
                b2 = bias.copy(); b2[c] += d
                m2, _ = score(b2, cache)
                if m2 > best + 1e-5:
                    bias = b2; best = m2; improved = True
                    print(f"[calib] rd{rd} cls{c}{d:+.2f} -> {m2:.4f}", flush=True)
        if not improved:
            break
    np.save(BIAS, bias)
    _, ious = score(bias, cache)
    print(f"[calib] final={best:.4f} route={route} bias={np.round(bias, 3).tolist()}")


@torch.no_grad()
def infer(a):
    b2 = getattr(a, "b2_kan", False)
    pl = getattr(a, "pl", False)
    enc = getattr(a, "enc", "swin_b")
    model, iters = load_trained(b2_kan=b2, pl=pl, enc=enc, path=getattr(a, "ckpt", None)); model.eval()
    bias = load_bias()
    route = "ms2"  # 8路TTA已在两个验证切片上稳定优于ms (+0.35/+0.40)
    out_dir = os.path.join(ARTIFACT_ROOT, "submission_test1_kan" + ("_t" if enc == "swin_t" else "")
                           + ("_anchor" if getattr(a, "anchor", False)
                              else "_pl" if pl else "") + ("_b2" if b2 else "") + "_" + route
                           + ("_cal" if bias is not None else ""))
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(os.listdir(S.TEST_DIR))
    for n, fn in enumerate(files):
        out_path = os.path.join(out_dir, fn)
        if os.path.exists(out_path):
            continue
        img = np.array(Image.open(os.path.join(S.TEST_DIR, fn)).convert("RGB"))
        pred = S.predict_logits(model, img, tta=route, bias=bias).argmax(0).numpy().astype(np.uint8)
        Image.fromarray(pred, mode="L").save(out_path)
        if (n + 1) % 25 == 0:
            print(f"infer {n+1}/{len(files)}", flush=True)
    for fn in files[:5] + files[-5:]:
        im = Image.open(os.path.join(out_dir, fn))
        assert im.mode == "L" and im.size == (1024, 1024), (fn, im.mode, im.size)
        assert set(np.unique(im)) <= set(range(9)), fn
    import shutil
    shutil.make_archive(out_dir, "zip", out_dir)
    print(f"[infer done] {len(files)} 张, iters={iters}, tta={route}, zip={out_dir}.zip 已生成并通过格式自检")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--eval-tta", action="store_true")
    ap.add_argument("--eval-ms", action="store_true")
    ap.add_argument("--eval-ms2", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--infer", action="store_true")
    ap.add_argument("--target-iters", type=int, default=8000)
    ap.add_argument("--restart-lr", type=float, default=0.0)
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--b2-kan", action="store_true", dest="b2_kan")
    ap.add_argument("--pl", action="store_true")
    ap.add_argument("--anchor", action="store_true",
                    help="使用本次独立教师生成的500张硬锚标签")
    ap.add_argument("--anchor-dir", type=str, default=None)
    ap.add_argument("--distill", action="store_true",
                    help="已禁用的历史队友教师入口；严格复现会拒绝")
    ap.add_argument("--pl-tau", type=float, default=0.85, dest="pl_tau")
    ap.add_argument("--enc", choices=["swin_b", "swin_t"], default="swin_b")
    ap.add_argument("--bn", action="store_true",
                    help="A3 实验: 逐图亮度归一化 (训练/评估/推理全程开启)")
    ap.add_argument("--exclude-noise", action="store_true", dest="exclude_noise",
                    help="A4 实验: 剔除噪声审计 top-10 疑似错标图")
    ap.add_argument("--weak-fix", action="store_true", dest="weak_fix",
                    help="N2 弱类修正: Agricultural boost1.0->1.5 + CE0.90->1.20, Background CE0.40->0.60")
    ap.add_argument("--ckpt-name", type=str, default=None,
                    help="自定义存档名 (新管线分支; 快照名同步派生)")
    ap.add_argument("--warm-base", type=str, default=None,
                    help="PL/anchor 暖启动源；必须是 strict run root 内文件名")
    ap.add_argument("--warm-parent-stage", type=str, default=None)
    ap.add_argument("--warm-parent-iter", type=int, default=None)
    ap.add_argument("--pl-cache", type=str, default=None,
                    help="伪标签缓存目录 (默认 pl_cache)")
    ap.add_argument("--pl-tag", type=str, default=None,
                    help="伪标签掩码目录后缀 (默认空)")
    ap.add_argument("--max-seconds", type=int, default=240)
    ap.add_argument("--crop", type=int, default=640)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=3)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42, help="随机种子(默认42)")
    ap.add_argument("--stage", type=str, default="main_kan_base")
    ap.add_argument("--encoder-warm", type=str, default=None,
                    help="KAN基座编码器初始化；strict run root 内v4.1文件名")
    ap.add_argument("--encoder-parent-stage", type=str, default=None,
                    help="KAN编码器父断点必须匹配的v4.1 stage")
    a = ap.parse_args()
    if getattr(a, "bn", False):  # A3: 全程开启亮度归一化 (train/eval/infer 一致)
        S.BN = True
        print("[bn] 逐图亮度归一化已开启")
    if a.train: train(a)
    elif a.calibrate: calibrate(a)
    elif a.eval: run_eval(False, a, use_ema=not a.raw)
    elif a.eval_tta: run_eval(True, a, use_ema=not a.raw)
    elif a.eval_ms: run_eval("ms", a, use_ema=not a.raw)
    elif a.eval_ms2: run_eval("ms2", a, use_ema=not a.raw)
    elif a.infer: infer(a)
