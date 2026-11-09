"""Standalone unet--FOM inference. Only float32 seven-field NPYs are saved.

Dependencies: PyTorch >=2.3, NumPy, and a working local CuPy/CUDA environment.
Training is NOT imported, executed or required. Current defaults run Run_00
on the local computer; Windows paths must use raw strings (r'...').

Examples:
 python 02_unet_inference.py --model /kaggle/input/WEIGHTS/best.pt \
   --data-root /kaggle/input/DATASET --output /kaggle/working/unet_ch16_train03_pred
 python 02_unet_inference.py --model /path/best.pt \
   --initial-dir /path/Run_31 --run-id 31 --output /path/unet_Run31

The single-run defaults below evaluate Run_00 as a development trajectory.
For batch evaluation set INITIAL_DIR=None and DATA_ROOT to the parent of
Run_31--Run_35. Only ten exact initial filenames 002000..020000 are ever opened.
No future reference fields, training, local PDE correction, plotting or paper
error metrics. Best checkpoint's config/norm/protocol are mandatory and strict.

The solver class below is a VERBATIM copy of PhaseFieldSimulator in run03.py;
its float64 CuPy state and numerical equations are unchanged. FOM failures abort
instead of silently falling back to a PDE-free trajectory as run03 can do.
Each burst injects this network's current seven-field prediction and executes
exactly 2000 _single_step calls. An accepted post-burst endpoint REPLACES the
pre-burst endpoint in the ten-frame FIFO and in output, matching run03's output
cadence. Thus the last FIFO gap is 8000 at a burst, otherwise 6000. Actual lags
enter the same time MLP used in training; no invented uniform spacing.

Output per Run: 90 frame_XXXXXX.npy files, shape (7,512,512), float32,
channel order c,eta1,...,eta6; ten initial frames plus 80 accepted endpoints.
Learning step40 saves frame_262000.npy; step80 saves frame_504000.npy.
The pre-burst states 260000 and 502000 are not separately saved, just as in
run03. Pair evaluation by filename's actual fine step, never by frame ordinal.
Warmup 20000 FOM steps are supplied by the archived initial histories; this
process makes 4000 additional fine-step calls, giving the shared trajectory
budget 24000 including archived warmup. No extra FOM regeneration is claimed.
Mean/range/residual handling is identical to both training files; no local map.
Runtime prints parameter count, NN and FOM wall time, IO time and GPU memory;
these are costs, not additional prediction-quality evaluation metrics.
Existing frame outputs are never deleted or overwritten. Use a fresh directory.
Channel-scale revision: default encoder 16/32/64, bottleneck 64, three pools.
The saved config reconstructs the exact width/depth; inference has no independent
architecture override. Load schema 3 best.pt trained by the revised trainer;
schema 1/2 weights from earlier architecture/training protocols are rejected.
"""
from __future__ import annotations
import argparse
import math
import os
from pathlib import Path
import re
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

KIND = 'unet'

# ==================== 路径配置：直接修改这里 ====================
# 本地推理：下载训练生成的best.pt，可重命名为best_UNET.pt；勿使用latest.pt。
MODEL_PATH = r'D:\vscode work\xiangchang\LSTM\model03\投稿改\基线方法\best_UNET.pt'
OUTPUT_DIR = r'D:\vscode work\xiangchang\LSTM\model03\投稿改\基线方法\UNET_npy'
# 方式一：批量预测 Run31--35。填其父目录，INITIAL_DIR 保持 None。
DATA_ROOT = None  # 例如 '/kaggle/input/测试数据集'
RUN_IDS = [31, 32, 33, 34, 35]  # 可加入 0，作为额外开发轨迹
# 方式二：只预测一条。填 Run 目录及 RUN_ID，DATA_ROOT 保持 None。
INITIAL_DIR = r'D:\vscode work\xiangchang\LSTM\model03\投稿改\基线方法\Run_00'
RUN_ID = 0  # 例如 31；允许 0 或 31--35
# 本地单条预测示例（按实际位置修改，取消对应行注释即可）：
# MODEL_PATH = str(Path(__file__).resolve().parent / 'weights_unet_ch16_train03' / 'best.pt')
# INITIAL_DIR = r'D:\vscode work\xiangchang\LSTM\model03\run\Run_31'
# RUN_ID = 31
# OUTPUT_DIR = str(Path(__file__).resolve().parent / 'output_unet_ch16_Run31')
# 命令行参数优先；批量和单条初始数据配置只能选择一种。
# =============================================================

SCHEMA = 3
PROTOCOL = dict(grid=512, fields=7, history=10, save_interval=2000,
                lead=6000, burst=2000, learned_steps=80, burst_at=[40, 80],
                initial_end=20000, terminal=504000, fom_budget=24000,
                loss_weights=[1., .03, .03, .02], mean_control='initial',
                geometry='dy(nx)+dx(ny); eps=1e-8', history_rule='replace-at-burst-FIFO',
                train_history='dense-train03', validation='curriculum-AR-train03')


def geometry(x):
    """Normalized (...,7,H,W) -> (...,9,H,W), exact train03 kappa_impl."""
    x = x.float()
    eta = (x[..., 1:7, :, :] + 1.) / 2.
    phi = (2. * (1. - eta.square().sum(-3, keepdim=True))).clamp(0., 1.)
    gy = (torch.roll(phi, -1, -2) - torch.roll(phi, 1, -2)) / 2.
    gx = (torch.roll(phi, -1, -1) - torch.roll(phi, 1, -1)) / 2.
    gm = (gx.square() + gy.square() + 1e-8).sqrt()
    nx = torch.where(gm > 1e-6, gx / gm, 0.)
    ny = torch.where(gm > 1e-6, gy / gm, 0.)
    kap = ((torch.roll(nx, -1, -2) - torch.roll(nx, 1, -2)) +
           (torch.roll(ny, -1, -1) - torch.roll(ny, 1, -1))) / 2.
    kap = torch.where(gm > 1e-6, kap, 0.)
    return torch.cat((x, kap, gm), dim=-3)


def normalize(a, norm):
    a = np.array(a, dtype=np.float32, copy=True)
    a[0] = (a[0] - norm['c_min']) / (norm['c_max'] - norm['c_min']) * 2. - 1.
    a[1:7] = a[1:7] * 2. - 1.
    return torch.from_numpy(a)


def denormalize(x, norm):
    x = x.float().clone()
    x[..., 0, :, :] = (x[..., 0, :, :] + 1.) / 2. * (norm['c_max'] - norm['c_min']) + norm['c_min']
    x[..., 1:7, :, :] = (x[..., 1:7, :, :] + 1.) / 2.
    return x


def time_features(times):
    """Relative history lags and forecast lead; no absolute-time extrapolation."""
    lags = (times - times[:, -1:]).float() / 6000.
    return torch.cat((lags, torch.ones_like(lags[:, :1])), dim=1)


def advance_history(history, times, state, step):
    """Append one accepted state; at bursts call ONCE with post-burst time."""
    return (torch.cat((history[:, 1:], geometry(state.detach()).unsqueeze(1)), 1),
            torch.cat((times[:, 1:], step.reshape(-1, 1)), 1))


def project(raw_delta, latest, mean_target, norm):
    """Common FP32 bounded residual and mean/range control; NO local corrector."""
    delta = .35 * torch.tanh(raw_delta.float() / .35)
    dc = delta[:, :1] - delta[:, :1].mean((-2, -1), keepdim=True)
    delta = torch.cat((dc, delta[:, 1:]), 1)
    pred = (latest[:, :7].float() + delta).clamp(-1., 1.)
    c = pred[:, :1]
    c = c + mean_target - c.mean((-2, -1), keepdim=True)
    lo = (.08 - norm['c_min']) / (norm['c_max'] - norm['c_min']) * 2. - 1.
    hi = (.24 - norm['c_min']) / (norm['c_max'] - norm['c_min']) * 2. - 1.
    return torch.cat((c.clamp(lo, hi), pred[:, 1:7]), 1)


class Sine(nn.Module):
    def forward(self, x):
        return x.sin()


def time_mlp():
    return nn.Sequential(nn.Linear(11, 128), Sine(), nn.Linear(128, 128), Sine())


def group_count(ch):
    return math.gcd(8, ch)


class ConvBlock(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1, padding_mode='circular'),
            nn.GroupNorm(group_count(co), co), nn.GELU(),
            nn.Conv2d(co, co, 3, padding=1, padding_mode='circular'),
            nn.GroupNorm(group_count(co), co), nn.GELU())

    def forward(self, x):
        return self.net(x)


class Baseline(nn.Module):
    """Time-conditioned U-Net; default channels 16/32/64, bottleneck 64.

    Cap channels at 4*base_width rather than doubling to 128/256. Three
    downsamplings retain the original Hybrid's 64x64 bottleneck resolution.
    History stacking, double convolutions and temporal skips remain intact.
    """
    def __init__(self, cfg):
        super().__init__()
        pools = cfg['layers']
        if pools not in (3, 4) or cfg['width'] < 1:
            raise ValueError('U-Net requires positive width and 3 or 4 pooling levels')
        self.divisor = 2**pools
        widths = [cfg['width'] * min(2**i, 4) for i in range(pools+1)]
        self.time = time_mlp()
        self.enc = nn.ModuleList([ConvBlock(90 if i == 0 else widths[i-1], c)
                                  for i, c in enumerate(widths)])
        self.mod = nn.ModuleList([nn.Linear(128, c) for c in widths])
        self.up = nn.ModuleList([nn.ConvTranspose2d(widths[i], widths[i-1], 2, 2)
                                for i in range(pools, 0, -1)])
        self.dec = nn.ModuleList([ConvBlock(2 * widths[i-1], widths[i-1])
                                 for i in range(pools, 0, -1)])
        self.head = nn.Conv2d(widths[0], 7, 1)
        # Small nonzero head/modulation: stable residual start without blocking
        # early feature gradients or imposing an identity-only initial model.
        for layer in self.mod:
            nn.init.normal_(layer.weight, std=.01)
            nn.init.zeros_(layer.bias)
        nn.init.normal_(self.head.weight, std=.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, history, temporal):
        b, _, _, h, w = history.shape
        if h % self.divisor or w % self.divisor:
            raise ValueError(f'U-Net grid must be divisible by {self.divisor}')
        x = history.reshape(b, 90, h, w)
        t = self.time(temporal)
        skips = []
        for i, (enc, mod) in enumerate(zip(self.enc, self.mod)):
            if i:
                x = F.max_pool2d(x, 2)
            x = enc(x)
            skips.append(x * (1. + mod(t)[:, :, None, None]))
        x = skips[-1]
        for i, (up, dec) in enumerate(zip(self.up, self.dec)):
            x = dec(torch.cat((up(x), skips[-2-i]), 1))
        return self.head(x).float()


def step_number(path):
    match = re.fullmatch(r'frame_(\d+)\.npy', Path(path).name)
    if not match:
        raise ValueError(f'Invalid physical-step filename: {path}')
    return int(match.group(1))

def checked_array(path, grid=512):
    a = np.load(path, mmap_mode='r', allow_pickle=False)
    if a.shape != (7, grid, grid) or a.dtype.kind != 'f':
        raise ValueError(f'{path}: expected floating (7,{grid},{grid}), got {a.shape}/{a.dtype}')
    if not np.isfinite(a).all():
        raise ValueError(f'Nonfinite input: {path}')
    if a[0].min() < .08-1e-5 or a[0].max() > .24+1e-5:
        raise ValueError(f'Concentration outside FOM [0.08,0.24], check channel order: {path}')
    if a[1:].min() < -1e-5 or a[1:].max() > 1.00001:
        raise ValueError(f'Order parameters outside [0,1]: {path}')
    return a

class PhaseFieldSimulator:
    def __init__(self, nx=512, ny=512, run_id=1):
        self.run_id = run_id
        self.nx, self.ny = nx, ny
        self.dt = 5e-4
        self.save_interval = 2000
        self.c_gamma, self.c_gprime, self.c_matrix = 0.10, 0.22, 0.13
        self.gamma_int, self.gamma_gb, self.delta = 0.091, 1.02, 0.0072
        self.C11, self.C12, self.C44 = 174.9e9, 117.6e9, 71.6e9
        self.f0 = 1.27e9
        self.lam, self.dc = 5.0, 0.12
        self.koA = (self.lam * self.dc / 4) ** 2
        self.gid = self.gamma_int / (self.f0 * 2e-9)
        self.ktA = (self.gid * 6 / self.dc ** 3) ** 2
        self.A_well = np.sqrt(self.ktA / self.koA)
        self.kappa_c = self.koA * self.A_well
        self.D_bulk, self.D_gb = 10.0, 30.0
        self.S_stab = self.D_bulk * self.A_well
        self.elastic_scale, self.GB_penalty = 150.0, 50.0
        self.alpha_pinning, self.beta_zener = 25.0, 1500.0
        self.W_gb, self.kappa_eta, self.L_mobility = 1000.0, 160.0, 0.4
        self.num_grains = 6
        self.grain_seeds = [(100,150),(300,100),(450,220),(380,420),(180,400),(80,280)]
        self.grain_angles = [0.0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6]
        self._init_spectral()
        self._init_grains()
        self._init_elastic_kernels()
        self.lhs_eta = 1.0 + self.dt * self.L_mobility * self.kappa_eta * self.K2
        self.lhs_c   = 1.0 + self.dt * self.S_stab * (self.K2 ** 2)
        self.step_count = self.frame_count = 0

    def _init_spectral(self):
        kx = cp.fft.fftfreq(self.nx) * 2 * np.pi
        ky = cp.fft.fftfreq(self.ny) * 2 * np.pi
        self.KX, self.KY = cp.meshgrid(kx, ky)
        self.K2 = self.KX ** 2 + self.KY ** 2
        self.K2_safe = cp.where(self.K2 == 0, 1.0, self.K2)

    def _init_grains(self):
        yg, xg = cp.indices((self.ny, self.nx))
        cp.random.seed(777 + self.run_id)
        _nx = (cp.random.randn(self.ny, self.nx)+1j*cp.random.randn(self.ny, self.nx))
        _nx = _nx * cp.exp(-self.K2 * 60**2 / 2)
        _fx = cp.real(cp.fft.ifft2(_nx)); _fx *= 25.0 / cp.std(_fx)
        cp.random.seed(888 + self.run_id)
        _ny = (cp.random.randn(self.ny, self.nx)+1j*cp.random.randn(self.ny, self.nx))
        _ny = _ny * cp.exp(-self.K2 * 60**2 / 2)
        _fy = cp.real(cp.fft.ifft2(_ny)); _fy *= 25.0 / cp.std(_fy)
        xgw, ygw = xg + _fx, yg + _fy
        ds = []
        for sx, sy in self.grain_seeds:
            dx = xgw - sx; dx -= cp.round(dx / self.nx) * self.nx
            dy = ygw - sy; dy -= cp.round(dy / self.ny) * self.ny
            ds.append(dx**2 + dy**2)
        self.grain_id = cp.argmin(cp.array(ds), axis=0)
        self.etas = cp.zeros((self.num_grains, self.ny, self.nx))
        for i in range(self.num_grains):
            self.etas[i] = cp.where(self.grain_id == i, 1.0, 0.0)
        for i in range(self.num_grains):
            self.etas[i] = cp.real(cp.fft.ifft2(cp.fft.fft2(self.etas[i])*cp.exp(-self.K2*1.0**2/2)))
        cp.random.seed(42 + self.run_id * 7)
        self.c = cp.full((self.ny, self.nx), self.c_matrix)
        gbx = (self.grain_id != cp.roll(self.grain_id, 1, axis=1))
        gby = (self.grain_id != cp.roll(self.grain_id, 1, axis=0))
        gbm = gbx | gby; gbi = cp.where(gbm); ng = len(gbi[0])
        cp.random.seed(99 + self.run_id * 10)
        ch = cp.random.choice(cp.arange(ng), size=60, replace=False)
        for k in ch:
            iy, ix = int(gbi[0][k]), int(gbi[1][k]); ow = int(self.grain_id[iy, ix])
            offs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(-1,1),(1,1)]
            bd = bx = by = 0
            for dx, dy in offs:
                ct = sum(1 for s in range(1,8) if 0<=iy+s*dy<self.ny and 0<=ix+s*dx<self.nx and int(self.grain_id[iy+s*dy,ix+s*dx])==ow)
                if ct>bd: bd, bx, by = ct, dx, dy
            cx, cy = ix+bx*6, iy+by*6
            YY, XX = cp.meshgrid(cp.clip(cp.arange(self.ny)-cy,-12,12), cp.clip(cp.arange(self.nx)-cx,-12,12), indexing='ij')
            self.c = cp.where((self.grain_id==ow)&(XX**2+YY**2<16), self.c_gprime, self.c)
        cp.random.seed(123 + self.run_id * 13)
        for g in range(self.num_grains):
            mg = (self.grain_id == g); gyi, gxi = cp.where(mg)
            for _ in range(60):
                idx = int(cp.random.randint(0, len(gxi), 1)[0])
                sx, sy = int(gxi[idx]), int(gyi[idx])
                self.c = cp.where((self.grain_id==g)&((xg-sx)**2+(yg-sy)**2<4), self.c_gprime, self.c)
        self.phi = self._get_phi(self.etas)
        self.Deff = self.D_bulk * (1 - self.phi) + self.D_gb * self.phi

    def _init_elastic_kernels(self):
        K4r = (self.KX**4 + self.KY**4) / self.K2_safe**2
        AZ = 2 * self.C44 / (self.C11 - self.C12)
        an = (AZ - 1) / (AZ + 1)
        B0 = self.elastic_scale * (self.C11 / self.f0) * self.delta**2
        self.B_kernels = cp.zeros((self.num_grains, self.ny, self.nx))
        for idx, th in enumerate(self.grain_angles):
            KXr = self.KX * cp.cos(th) + self.KY * cp.sin(th)
            KYr = -self.KX * cp.sin(th) + self.KY * cp.cos(th)
            K4rr = (KXr**4 + KYr**4) / self.K2_safe**2
            Bk = -B0 * (1.0 + an * K4rr); Bk[0, 0] = 0.0
            self.B_kernels[idx] = Bk

    def _get_phi(self, etas_in):
        return cp.clip((1.0 - cp.sum(etas_in**2, axis=0)) * 2.0, 0.0, 1.0)

    def _laplacian(self, f):
        return cp.real(cp.fft.ifft2(-self.K2 * cp.fft.fft2(f)))

    def _single_step(self):
        fc = self.A_well*(self.c-self.c_gamma)*(self.c-self.c_gprime)*(2*self.c-self.c_gamma-self.c_gprime)
        cf = cp.fft.fft2(self.c)
        dE = cp.zeros((self.ny, self.nx))
        for i in range(self.num_grains):
            dE += cp.real(cp.fft.ifft2(self.B_kernels[i]*cf))*self.etas[i]
        mu = fc - self.kappa_c*self._laplacian(self.c) + dE + self.GB_penalty*self.phi*(self.c-self.c_gamma)
        Dx = 0.5*(self.Deff+cp.roll(self.Deff,-1,1)); fx = Dx*(cp.roll(mu,-1,1)-mu)
        Dy = 0.5*(self.Deff+cp.roll(self.Deff,-1,0)); fy = Dy*(cp.roll(mu,-1,0)-mu)
        df = (fx-cp.roll(fx,1,1)) + (fy-cp.roll(fy,1,0))
        cf = cp.fft.fft2(self.c)
        rc = cf + self.dt*cp.fft.fft2(df) + self.dt*self.S_stab*(self.K2**2)*cf
        self.c = cp.real(cp.fft.ifft2(rc/self.lhs_c))
        self.c += (self.c_matrix - cp.mean(self.c)); self.c = cp.clip(self.c, 0.08, 0.24)
        Wgv = self.W_gb*(1.0+self.alpha_pinning*cp.clip(self.c-self.c_gamma,0,1))
        zf = self.beta_zener*cp.clip(self.c-self.c_gamma,0,1)**2
        ss = cp.sum(self.etas**2, axis=0)
        for i in range(self.num_grains):
            sj = ss - self.etas[i]**2
            sw = self.etas[i]*(self.etas[i]**2-1) + 2*self.etas[i]*sj
            sz = zf*self.etas[i]*(1-self.etas[i])*(self.etas[i]-0.5)
            ef = cp.fft.fft2(self.etas[i])
            sf = cp.fft.fft2(-self.L_mobility*(Wgv*sw+sz))
            self.etas[i] = cp.real(cp.fft.ifft2((ef+self.dt*sf)/self.lhs_eta))
            self.etas[i] = cp.clip(self.etas[i], 0.0, 1.0)
        self.phi = self._get_phi(self.etas)
        self.Deff = self.D_bulk*(1-self.phi) + self.D_gb*self.phi
        self.step_count += 1

    def inject_state(self, frame):
        """7通道直接注入PDE: c + 6etas (不再走旧2通道比例缩放)"""
        self.c = cp.clip(cp.asarray(frame[0], dtype=cp.float64), 0.08, 0.24)
        for i in range(self.num_grains):
            self.etas[i] = cp.clip(cp.asarray(frame[1+i], dtype=cp.float64), 0.0, 1.0)
        self.phi = self._get_phi(self.etas)
        self.Deff = self.D_bulk * (1 - self.phi) + self.D_gb * self.phi

    def advance_one_frame(self):
        for _ in range(self.save_interval):
            self._single_step()
        self.frame_count += 1
        return np.stack([cp.asnumpy(self.c).astype(np.float32),
                         cp.asnumpy(self.phi).astype(np.float32)], axis=0)


def atomic_npy(path, array):
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite prediction: {path}')
    tmp = path.with_name(path.name + '.tmp')
    try:
        with open(tmp, 'wb') as f:
            np.save(f, array, allow_pickle=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def load_best(path, device):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    if ck.get('schema') != SCHEMA or ck.get('kind') != KIND or ck.get('protocol') != PROTOCOL:
        raise ValueError('Wrong checkpoint kind/schema/physics protocol')
    if not (25 <= ck.get('epoch', 0) <= ck['config']['epochs']):
        raise ValueError('Use best.pt selected in the U=40 training phase')
    if 'optimizer_state_dict' in ck:
        raise ValueError('Use independent best.pt, not latest.pt (resume-only).')
    norm = {k: float(ck['norm_params'][k]) for k in ('c_min', 'c_max')}
    if not (.08-1e-5 <= norm['c_min'] < norm['c_max'] <= .24+1e-5):
        raise ValueError('Invalid normalization in checkpoint')
    model = Baseline(ck['config']).to(device)
    model.load_state_dict(ck['model_state_dict'], strict=True)
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('Checkpoint contains nonfinite parameters')
    parameters = sum(p.numel() for p in model.parameters())
    if parameters != ck['parameters']:
        raise ValueError('Parameter count disagrees with checkpoint')
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f'Loaded {KIND} epoch={ck["epoch"]}, parameters={parameters:,}, '
          f'validation-selected loss={ck["best_loss"]:.7g}, config={ck["config"]}', flush=True)
    return model, norm


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    cp.cuda.get_current_stream().synchronize()


@torch.inference_mode()
def predict_run(model, norm, initial_dir, output_dir, rid, device):
    initial_dir, output_dir = Path(initial_dir), Path(output_dir)
    # Open exact past files only: do not glob/read the future reference folder.
    times_list = list(range(2000, 20001, 2000))
    initial = [np.array(checked_array(initial_dir/f'frame_{t:06d}.npy'),
                        dtype=np.float32, copy=True) for t in times_list]
    if output_dir.exists() and list(output_dir.glob('frame_*.npy')):
        raise FileExistsError(f'Existing frames in {output_dir}; select a fresh output directory')
    if output_dir.resolve() == initial_dir.resolve():
        raise ValueError('Output must be distinct from initial/reference data')
    x = torch.stack([normalize(a, norm) for a in initial]).unsqueeze(0).to(device)
    history = geometry(x)
    times = torch.tensor([times_list], dtype=torch.int64, device=device)
    mean = x[:, -1, :1].mean((-2, -1), keepdim=True)
    # Fail before writing output if full-order advancement is unavailable.
    synchronize(device)
    setup_start = time.perf_counter()
    solver = PhaseFieldSimulator(nx=512, ny=512, run_id=rid)
    solver.inject_state(initial[-1])
    synchronize(device)
    setup_seconds = time.perf_counter()-setup_start
    output_dir.mkdir(parents=True, exist_ok=True)
    io_start = time.perf_counter()
    for t, a in zip(times_list, initial):
        atomic_npy(output_dir/f'frame_{t:06d}.npy', a)
    io_seconds = time.perf_counter()-io_start
    nn_seconds = fom_seconds = 0.
    fine_calls = bursts = 0
    accepted_steps = times_list.copy()
    cur = 20000
    trajectory_start = time.perf_counter()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    # FP32 inference for both networks. Training AMP only covers neural kernels;
    # Fourier FFTs, losses and state feedback were also FP32 during training.
    for learned in range(1, 81):
        synchronize(device)
        start = time.perf_counter()
        delta = model(history, time_features(times))
        pred = project(delta, history[:, -1], mean, norm)
        if not torch.isfinite(pred).all():
            raise RuntimeError(f'Run{rid:02d} learned step{learned}: nonfinite network output')
        cur += 6000
        synchronize(device)
        nn_seconds += time.perf_counter()-start
        if learned in (40, 80):
            start = time.perf_counter()
            state = denormalize(pred[0], norm).cpu().numpy().astype(np.float32, copy=False)
            solver.inject_state(state)
            before = solver.step_count
            for _ in range(2000):
                solver._single_step()
            if solver.step_count-before != 2000:
                raise RuntimeError('Incorrect FOM fine-step count')
            state = np.stack([cp.asnumpy(solver.c).astype(np.float32)] +
                             [cp.asnumpy(solver.etas[i]).astype(np.float32) for i in range(6)])
            synchronize(device)
            fom_seconds += time.perf_counter()-start
            if not np.isfinite(state).all():
                raise RuntimeError('Nonfinite FOM burst; refusing to continue')
            cur += 2000
            fine_calls += 2000
            bursts += 1
            feedback = normalize(state, norm).unsqueeze(0).to(device)
        else:
            feedback = pred
            state = denormalize(pred[0], norm).cpu().numpy().astype(np.float32, copy=False)
        # Exactly one history insertion, with the accepted state's true timestamp.
        start = time.perf_counter()
        history, times = advance_history(history, times, feedback, times.new_tensor([cur]))
        synchronize(device)
        nn_seconds += time.perf_counter()-start
        expected = 20000 + 6000*learned + 2000*(learned//40)
        if cur != expected or int(times[0, -1]) != cur or cur <= accepted_steps[-1]:
            raise RuntimeError('Physical time/history misalignment')
        io_start = time.perf_counter()
        atomic_npy(output_dir/f'frame_{cur:06d}.npy', np.ascontiguousarray(state, dtype=np.float32))
        io_seconds += time.perf_counter()-io_start
        accepted_steps.append(cur)
        if learned % 10 == 0 or learned in (1, 40, 80):
            print(f'Run_{rid:02d} learned={learned}/80 fine_step={cur} '
                  f'bursts={bursts} NN={nn_seconds:.3f}s FOM={fom_seconds:.3f}s', flush=True)
    if cur != 504000 or fine_calls != 4000 or bursts != 2 or len(accepted_steps) != 90:
        raise RuntimeError('Terminal trajectory/budget invariant failed')
    expected_names = {f'frame_{t:06d}.npy' for t in accepted_steps}
    if {p.name for p in output_dir.glob('frame_*.npy')} != expected_names:
        raise RuntimeError('Unexpected output frame inventory')
    peak = torch.cuda.max_memory_allocated(device)/2**30 if device.type == 'cuda' else 0.
    print(f'Run_{rid:02d} DONE -> {output_dir}; 90 float32 (7,512,512) NPYs; '
          f'NN={nn_seconds:.3f}s ({nn_seconds/80:.4f}s/learned step incl. feedback), '
          f'FOM={fom_seconds:.3f}s, solver-setup={setup_seconds:.3f}s, '
          f'NPY-IO={io_seconds:.3f}s, rollout-wall={time.perf_counter()-trajectory_start:.3f}s, '
          f'torch-peak={peak:.3f}GiB; FOM budget=20000 archived+{fine_calls} new=24000.', flush=True)
    del solver
    cp.get_default_memory_pool().free_all_blocks()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default=MODEL_PATH, help='Training-selected best.pt')
    p.add_argument('--data-root', default=DATA_ROOT, help='Parent containing Run_31..Run_35')
    p.add_argument('--runs', nargs='+', type=int, default=RUN_IDS)
    p.add_argument('--initial-dir', default=INITIAL_DIR, help='Single Run folder, only initial ten files used')
    p.add_argument('--run-id', type=int, default=RUN_ID, help='Required with --initial-dir')
    p.add_argument('--output', default=OUTPUT_DIR)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--fom-device', type=int, default=0, help='CuPy GPU if network runs on CPU')
    args = p.parse_args()
    if not args.model:
        p.error('Set MODEL_PATH in the path configuration or supply --model')
    if args.initial_dir:
        if args.data_root or args.run_id is None or args.run_id not in [0, 31, 32, 33, 34, 35]:
            p.error('--initial-dir requires --run-id 0 or 31..35 and no --data-root')
        jobs = [(args.run_id, Path(args.initial_dir), Path(args.output))]
    else:
        if not args.data_root:
            p.error('Supply --data-root or --initial-dir with --run-id')
        if len(set(args.runs)) != len(args.runs) or any(r not in [0,31,32,33,34,35] for r in args.runs):
            p.error('Independent runs are 31..35; 0 is an optional development trajectory')
        jobs = [(r, Path(args.data_root)/f'Run_{r:02d}', Path(args.output)/f'Run_{r:02d}') for r in args.runs]
    device = torch.device(args.device)
    if device.type not in ('cpu', 'cuda'):
        p.error('Use cpu or cuda:N (FOM always requires CuPy/CUDA)')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; FOM cannot be silently omitted')
        torch.cuda.set_device(device)
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    global cp
    try:
        import cupy as cp
        fom_device = (device.index if device.index is not None else torch.cuda.current_device()) if device.type == 'cuda' else args.fom_device
        cp.cuda.Device(fom_device).use()
        cp.zeros(1)  # Verify a working solver GPU before opening the trajectory.
    except Exception as exc:
        raise RuntimeError('A working CuPy/CUDA solver is mandatory for the common FOM budget') from exc
    model, norm = load_best(args.model, device)
    for rid, initial_dir, output in jobs:
        predict_run(model, norm, initial_dir, output, rid, device)


if __name__ == '__main__':
    main()
