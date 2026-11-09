"""以 train03.py 为底稿的时间条件 U-Net训练基线。

AutoDL / 本地服务器：在终端直接运行本文件，文件名保留以兼容已有四文件交付。
将本文件与 Run_01 ... Run_30 放在同一目录，无需填写服务器绝对路径：
  python -u 01_train_unet_kaggle.py
  python -u 01_train_unet_kaggle.py --gpus 1
  python -u 01_train_unet_kaggle.py --gpus 2
默认以脚本所在目录为数据根目录（不受终端当前工作目录影响），权重保存在
同目录的 unet_training/ 下；再次执行同一命令自动读取 latest.pt 完整续训。
可用 --data-root / --output / --norm-from / --resume 显式指定其他位置；
这些参数若填相对路径，也以脚本所在目录为基准。
原版 add_curvature / compute_curvature / pred_to_input / PhaseFieldDataset /
show_ram 为底稿；仅在原归一化扫描中补进度打印；数据划分/读取/计算不变。
面向单张RTX 4090 24GB，默认全局batch=8；可用--batch显式覆盖。
其余训练设置沿用train03：Adam lr=1e-3、40 epochs、FP32、渐进U8->40、TF线性衰减，
同阶段纯AR验证；每200批打印进度。自动GPU/CPU和DataParallel，不加GPU硬拦截。

架构差异：非递归网络每步读取十帧FIFO和真实相对时间，不使用ConvLSTM或PDE局部修正。
两基线保持共同的残差/范围/初始均值处理，与对应推理程序一致。
为可靠续训，仅封装原数据集：固定每样本增强种子、冻结已给定归一化。
没有此前的非均匀历史采样增强。验证回归train03的当前U，正式best仅选U40阶段。
保留周期原子checkpoint、模型/Adam/调度器/AMP/随机状态和中途batch恢复。
SCHEMA=3：此前不同训练协议的latest不得接续；使用新的输出目录。
依赖：PyTorch>=2.3、NumPy、psutil。训练不生成预测NPY、图像或评价文件。
"""
from __future__ import annotations
import argparse, copy, ctypes, datetime, gc, glob, hashlib, itertools, math, os
from pathlib import Path
import random, re, signal, sys, time
import numpy as np
import psutil
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

KIND = 'unet'
# ==================== AutoDL：代码与30个Run目录放在一起 ====================
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = str(SCRIPT_DIR)
SAVE_PATH = str(SCRIPT_DIR / 'unet_training' / 'best.pt')
RESUME_READ = 'auto'  # 自动恢复本模型目录的latest.pt；也可填其他checkpoint路径。
NORM_FROM = None  # 可填原Hybrid best_model.pth，共享归一化；不是初始化网络权重。
# 若把原Hybrid权重也放在此目录，可设 NORM_FROM = 'best_model.pth'。
# None表示仍按train03仅扫描Run_01~Run_24计算归一化；不会自动加载其他模型。
# =============================================================
FRAME_STEP = 3
SEQ_LEN = 10
TRUNC = 20  # 无recurrent state，不需保留跨步计算图；逐步反向等价且节省显存。
BATCH_SIZE = 8  # 全局batch（所有GPU合计），accum=1；续训须与checkpoint保持一致。
GPU_COUNT = 0  # 自动使用可见GPU；单卡直接训练，多卡DataParallel，batch须>=卡数。
MODEL_WIDTH = 16
MODEL_LAYERS = 3
SCHEMA = 3
PROTOCOL = dict(grid=512, fields=7, history=10, save_interval=2000,
                lead=6000, burst=2000, learned_steps=80, burst_at=[40, 80],
                initial_end=20000, terminal=504000, fom_budget=24000,
                loss_weights=[1., .03, .03, .02], mean_control='initial',
                geometry='dy(nx)+dx(ny); eps=1e-8', history_rule='replace-at-burst-FIFO',
                train_history='dense-train03', validation='curriculum-AR-train03')

def local_path(value):
    """相对路径固定以本文件所在目录为基准，便于整体上传或搬移项目。"""
    path = Path(os.path.expandvars(str(value))).expanduser()
    return str((path if path.is_absolute() else SCRIPT_DIR / path).resolve())


def check_data_directory(root):
    """只识别规定的30个Run，保持原来的训练/验证划分与文件读取规则。"""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f'数据根目录不存在: {root}；默认应将Run_01~Run_30放在脚本旁。')
    runs = [root / f'Run_{rid:02d}' for rid in range(1,31)]
    missing = [p.name for p in runs if not p.is_dir()]
    if missing:
        raise FileNotFoundError(
            f'数据目录 {root} 缺少: {", ".join(missing)}。'
            '请将30个Run目录直接放在脚本旁，或用--data-root指定其共同父目录。')
    empty = [p.name for p in runs if next(p.glob('frame_*.npy'),None) is None]
    if empty:
        raise FileNotFoundError(f'这些Run目录没有frame_*.npy: {", ".join(empty)}')
    print(f'已识别30个Run目录 | 训练: Run_01~Run_24 (24个) | 验证: Run_25~Run_30 (6个)',
          flush=True)
    return runs


def add_curvature(frame7):
    """7通道帧 → 9通道: 追加 界面曲率κ 和 梯度幅值|∇φ| (物理驱动量, 从etas计算)"""
    etas = frame7[1:7]
    phi = np.clip((1.0 - np.sum(etas ** 2, axis=0)) * 2.0, 0, 1)
    p = np.pad(phi, 1, mode='wrap')
    gy = (p[2:, 1:-1] - p[:-2, 1:-1]) / 2.0
    gx = (p[1:-1, 2:] - p[1:-1, :-2]) / 2.0
    gm = np.sqrt(gx ** 2 + gy ** 2 + 1e-8)
    nx = np.where(gm > 1e-6, gx / gm, 0.0)
    ny = np.where(gm > 1e-6, gy / gm, 0.0)
    nxp = np.pad(nx, 1, mode='wrap'); nyp = np.pad(ny, 1, mode='wrap')
    kx = (nxp[2:, 1:-1] - nxp[:-2, 1:-1]) / 2.0
    ky = (nyp[1:-1, 2:] - nyp[1:-1, :-2]) / 2.0
    kappa = np.where(gm > 1e-6, kx + ky, 0.0)
    return np.concatenate([frame7, kappa[None].astype(frame7.dtype),
                           gm[None].astype(frame7.dtype)], axis=0)


def compute_curvature(phi):
    """界面曲率 κ = ∇·(∇φ/|∇φ|); phi: (B,1,1,H,W) → 同形状 (可微, 周期边界)"""
    p = phi[:, :, 0]
    p = F.pad(p, (1, 1, 1, 1), mode='circular')
    gy = (p[:, :, 2:, 1:-1] - p[:, :, :-2, 1:-1]) / 2.0
    gx = (p[:, :, 1:-1, 2:] - p[:, :, 1:-1, :-2]) / 2.0
    gm = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
    nx = torch.where(gm > 1e-6, gx / gm, torch.zeros_like(gm))
    ny = torch.where(gm > 1e-6, gy / gm, torch.zeros_like(gm))
    nx = F.pad(nx, (1, 1, 1, 1), mode='circular')
    ny = F.pad(ny, (1, 1, 1, 1), mode='circular')
    kx = (nx[:, :, 2:, 1:-1] - nx[:, :, :-2, 1:-1]) / 2.0
    ky = (ny[:, :, 1:-1, 2:] - ny[:, :, 1:-1, :-2]) / 2.0
    kappa = torch.where(gm > 1e-6, kx + ky, torch.zeros_like(gm))
    return kappa.unsqueeze(2)


def pred_to_input(pred7):
    """7通道归一化预测 → 9通道输入 (补κ和|∇φ|, 与add_curvature公式一致)
    AR推演时每步预测都要补曲率通道, 否则输入通道数与训练不一致"""
    eta_raw = (pred7[:, :, 1:] + 1) / 2                              # 归一化etas → [0,1]
    phi = torch.clamp((1.0 - eta_raw.pow(2).sum(dim=2, keepdim=True)) * 2.0, 0, 1)
    gy_p = F.pad(phi[:, :, 0], (1, 1, 1, 1), mode='circular')
    gx_p = F.pad(phi[:, :, 0], (1, 1, 1, 1), mode='circular')
    gyy = (gy_p[:, :, 2:, 1:-1] - gy_p[:, :, :-2, 1:-1]) / 2.0
    gxx = (gx_p[:, :, 1:-1, 2:] - gx_p[:, :, 1:-1, :-2]) / 2.0
    gm = torch.sqrt(gyy ** 2 + gxx ** 2 + 1e-8)
    kap = compute_curvature(phi)
    return torch.cat([pred7, kap, gm.unsqueeze(2)], dim=2)


def show_ram(tag=""):
    """打印当前进程RAM占用(GB)。"""
    process = psutil.Process(os.getpid())
    ram = process.memory_info().rss / 1024**3
    print(f"[{tag}] RAM = {ram:.2f} GB")


class PhaseFieldDataset(Dataset):
    """
    原始7通道: ch0=c(浓度), ch1~6=etas(晶粒序参数); 读入后补κ/|∇φ|成9通道
    示例(FRAME_STEP=2, UNROLL=8, seq_len=10):
      X = [f0..f9]                  ← 10帧连续输入
      Y = [f12, f14, ..., f26]     ← 8个目标帧, 间距2帧=4000 PDE步
    """
    def __init__(self, root_dir, seq_len=10, unroll=8, step=5,
                 mode='train', norm_params=None, stride=1):
        self.seq_len = seq_len
        self.unroll = unroll
        self.step = step
        self.samples = []
        self.norm_params = norm_params
        self.mode = mode
        self.stride = stride

        # Run-level split is explicit.  Run_00 is reserved for the independent
        # test trajectory and must never enter normalization, training, or
        # validation statistics, even when it is present under root_dir.
        train_ids = list(range(1, 25))
        val_ids = list(range(25, 31))
        ids = train_ids if mode == 'train' else val_ids
        run_folders = [os.path.join(root_dir, f"Run_{i:02d}")
                       for i in ids
                       if os.path.isdir(os.path.join(root_dir, f"Run_{i:02d}"))]
        missing = [i for i in ids
                   if not os.path.isdir(os.path.join(root_dir, f"Run_{i:02d}"))]
        if missing:
            raise FileNotFoundError(f"缺少{mode} Run目录: {missing}")

        st = self.stride if mode == 'train' else 1   # 训练窗口采样步长: 1=全量(结果无影响), 2=时间减半(信息损失极小)
        for rd in run_folders:
            frames = sorted(glob.glob(os.path.join(rd, "frame_*.npy")))
            max_window = seq_len + unroll * step
            for i in range(0, len(frames) - max_window + 1, st):
                X_files = frames[i : i + seq_len]
                Y_files = [frames[i + seq_len + (t+1)*step - 1]
                           for t in range(unroll)]
                self.samples.append((X_files, Y_files))

        if mode == 'train':
            self._compute_stats(run_folders)

    def _compute_stats(self, run_folders):
        """扫描训练集c通道min/max, etas范围固定[0,1]"""
        print(f"扫描 {len(run_folders)} 个Run, 计算c通道归一化参数 (512²)...", flush=True)
        c_min, c_max = float('inf'), float('-inf')
        started = time.perf_counter()
        completed = 0
        for ri, rd in enumerate(run_folders, 1):
            files = glob.glob(os.path.join(rd, "frame_*.npy"))[::5]
            print(f"  [归一化 {ri}/{len(run_folders)}] {Path(rd).name}: 待扫描 {len(files)} 帧", flush=True)
            last_report = time.perf_counter()
            for fi, f in enumerate(files, 1):
                arr = np.load(f, mmap_mode='r')
                c_min = min(c_min, float(arr[0].min()))
                c_max = max(c_max, float(arr[0].max()))
                completed += 1
                now = time.perf_counter()
                if fi == 1 or fi % 10 == 0 or fi == len(files) or now-last_report >= 15:
                    print(f"    {Path(rd).name} [{fi}/{len(files)}] 已完成 {Path(f).name}; "
                          f"累计{completed}帧，用时{now-started:.1f}s", flush=True)
                    last_report = now
        self.norm_params = {'c_min': float(c_min), 'c_max': float(c_max)}
        print(f"c=[{c_min:.4f},{c_max:.4f}], etas=[0,1]; 扫描耗时 {time.perf_counter()-started:.1f}s", flush=True)

    def _load_frame(self, path):
        """mmap读取 → 7通道+2曲率通道(κ,|∇φ|) → 归一化 → tensor"""
        arr = np.load(path, mmap_mode='r')
        data = np.array(arr, dtype=np.float32)             # (7, 512, 512)
        data = add_curvature(data)                         # (9, 512, 512)
        data[0] = (data[0] - self.norm_params['c_min']) / \
                  (self.norm_params['c_max'] - self.norm_params['c_min']) * 2 - 1
        for ch in range(1, 7):
            data[ch] = data[ch] * 2 - 1                    # etas: [0,1]→[-1,1]
        # ch7=κ, ch8=|∇φ| 保持原值 (BN会归一化)
        return torch.from_numpy(data)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X_files, Y_files = self.samples[idx]
        X = torch.stack([self._load_frame(f) for f in X_files])
        Y = torch.stack([self._load_frame(f) for f in Y_files])
        if self.mode == 'train':
            # 周期平移增强: PDE为周期边界, 整幅循环平移是物理严格等价的状态
            dy = np.random.randint(0, X.shape[-2])
            dx = np.random.randint(0, X.shape[-1])
            X = torch.roll(X, shifts=(dy, dx), dims=(-2, -1))
            Y = torch.roll(Y, shifts=(dy, dx), dims=(-2, -1))
        return X, Y


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


def physical_loss(pred, target, latest, norm):
    """Per-example [MSE, L_phi, L_rep, L_Z], identical spatial definitions."""
    pred, target, latest = pred.float(), target.float(), latest.float()
    phi_p = (2. * (1. - ((pred[:, 1:7] + 1.) / 2.).square().sum(1, keepdim=True))).clamp(0, 1)
    phi_t = (2. * (1. - ((target[:, 1:7] + 1.) / 2.).square().sum(1, keepdim=True))).clamp(0, 1)
    threshold = (.18 - norm['c_min']) / (norm['c_max'] - norm['c_min']) * 2. - 1.
    dims = (1, 2, 3)
    mse = (pred - target).square().mean(dims)
    phi = (phi_p - phi_t).abs().mean(dims)
    mask = F.max_pool2d(phi_t, 3, stride=1, padding=1)  # original ZERO padding
    rep = (mask * (pred[:, :1] - threshold).clamp_min(0.)).mean(dims)
    zener = ((pred[:, :1] > threshold).float() *
             (pred[:, 1:7] - latest[:, 1:7]).abs()).mean(dims)
    return torch.stack((mse, phi, rep, zener), 1)


def total_loss(components):
    return components @ components.new_tensor([1., .03, .03, .02])


def step_number(path):
    match = re.fullmatch(r'frame_(\d+)\.npy', Path(path).name)
    if not match:
        raise ValueError(f'Invalid physical-step filename: {path}')
    return int(match.group(1))


def cpu_tree(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: cpu_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [cpu_tree(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(cpu_tree(v) for v in obj)
    return copy.deepcopy(obj)


def rng_state(device):
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
                cuda_all=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None)


def restore_rng(state, device):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if device.type == 'cuda' and state.get('cuda_all') is not None:
        torch.cuda.set_rng_state_all(state['cuda_all'])
    elif device.type == 'cuda' and state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'], device)


def atomic_save(obj, path, rotate=False):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    try:
        with open(tmp, 'wb') as f:
            torch.save(obj, f)
            f.flush()
            os.fsync(f.fileno())
        if rotate and path.exists():
            os.replace(path, path.with_name(path.stem + '.prev.pt'))
        os.replace(tmp, path)
        if os.name != 'nt':
            fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    finally:
        if tmp.exists():
            tmp.unlink()


def memory_report(device):
    if device.type != 'cuda':
        return 'CPU'
    free, total = torch.cuda.mem_get_info(device)
    return (f'{torch.cuda.get_device_name(device)} peak_alloc='
            f'{torch.cuda.max_memory_allocated(device)/2**30:.2f}GiB '
            f'reserved={torch.cuda.memory_reserved(device)/2**30:.2f}GiB '
            f'free={free/2**30:.2f}/{total/2**30:.2f}GiB')



# =====================================================================
# train03 数据集只加可重复增强/已给定归一化的封装；不改变样本和读取方式。
# =====================================================================
class TrainingDataset(PhaseFieldDataset):
    def __init__(self, *args, seed=20261022, epoch=0, **kwargs):
        self.seed, self.epoch = seed, epoch
        super().__init__(*args, **kwargs)

    def _compute_stats(self, run_folders):
        # train03 原版会重复扫描并覆盖外部 norm。这里保留给定/续训 norm，
        # 未提供时仍调用原版 glob(...)[::5] 统计；不使用测试集。
        if self.norm_params is None:
            super()._compute_stats(run_folders)

    def __getitem__(self, idx):
        # 原版的周期平移增强不变。按样本/epoch固定其随机种子，使多 worker
        # 预取不影响中途恢复；无需保存或依赖 worker 内部随机进度。
        previous = np.random.get_state()
        try:
            seed = int(np.random.SeedSequence([self.seed, self.epoch, int(idx)]).generate_state(1)[0])
            np.random.seed(seed)
            return super().__getitem__(idx)
        finally:
            np.random.set_state(previous)


class ResumeBatchSampler:
    """与 shuffle=True 相同的 RandomSampler；跳过索引，不重读已完成 batch。"""
    def __init__(self, dataset, batch_size, seed, epoch, start=0):
        generator = torch.Generator().manual_seed(seed+epoch)
        sampler = torch.utils.data.RandomSampler(dataset, generator=generator)
        self.base = torch.utils.data.BatchSampler(sampler, batch_size, drop_last=False)
        self.start = start

    def __iter__(self):
        return itertools.islice(iter(self.base), self.start, None)

    def __len__(self):
        return max(0, len(self.base)-self.start)


def make_loader(ds, cfg, device, *, train, epoch, start=0):
    kwargs = dict(num_workers=cfg['workers'], pin_memory=device.type == 'cuda',
                  generator=torch.Generator().manual_seed(cfg['seed']+epoch+(0 if train else 1000003)))
    if cfg['workers']:
        kwargs['prefetch_factor'] = 1
    if train:
        return DataLoader(ds, batch_sampler=ResumeBatchSampler(ds,cfg['batch'],cfg['seed'],epoch,start), **kwargs)
    return DataLoader(ds,batch_size=cfg['batch'],shuffle=False,**kwargs)


def dataset_for(cfg, norm, unroll, mode, epoch=0):
    return TrainingDataset(cfg['data_root'],seq_len=cfg['seq_len'],unroll=unroll,
                           step=cfg['frame_step'],mode=mode,norm_params=norm,
                           stride=cfg['stride'],seed=cfg['seed'],epoch=epoch)


def dataset_fingerprint(root):
    digest = hashlib.sha256()
    for rid in range(1,31):
        paths = sorted(glob.glob(os.path.join(root,f'Run_{rid:02d}','frame_*.npy')))
        if not paths:
            raise FileNotFoundError(f'缺少 Run_{rid:02d} 训练/验证数据')
        times = [step_number(p) for p in paths]
        if any(b-a != 2000 for a,b in zip(times,times[1:])):
            raise ValueError(f'Run_{rid:02d}: 数据必须按2000个FOM细步保存')
        for p in paths:
            digest.update(f'{rid}/{Path(p).name}/{os.path.getsize(p)}'.encode())
        if rid%5==0:
            print(f'  文件列表: Run_{rid:02d}/Run_30 已核对',flush=True)
    return digest.hexdigest()


def read_norm(path):
    if not path:
        return None
    ck = torch.load(path,map_location='cpu',weights_only=True)
    norm = {k:float(ck['norm_params'][k]) for k in ('c_min','c_max')}
    if not (.08-1e-5 <= norm['c_min'] < norm['c_max'] <= .24+1e-5):
        raise ValueError('权重中的归一化参数不合法')
    return norm


def load_resume(cfg):
    output = Path(cfg['output'])
    if cfg['resume'] == 'none':
        if any((output/n).exists() for n in ('latest.pt','latest.prev.pt',cfg['best_name'])):
            raise FileExistsError('新训练请使用新输出目录，已有权重不会覆盖')
        return None
    candidates = [Path(cfg['resume'])] if cfg['resume'] != 'auto' else [output/'latest.pt',output/'latest.prev.pt']
    for p in candidates:
        if p.is_file():
            try:
                return torch.load(p,map_location='cpu',weights_only=False)
            except Exception as error:
                if cfg['resume'] != 'auto':
                    raise
                print(f'读取 {p} 失败: {error}; 尝试备份',flush=True)
    if cfg['resume'] != 'auto':
        raise FileNotFoundError(cfg['resume'])
    if any((output/n).exists() for n in ('latest.pt','latest.prev.pt',cfg['best_name'])):
        raise RuntimeError('输出目录已有权重但无可恢复的 latest.pt；请使用新目录')
    return None


def rollout(model, X, Y_full, unroll, tf, norm, cfg, device, *, train=False,
            scaler=None, denominator=None):
    """仅替换 train03 的 recurrent 前向：十帧 FIFO + 实际相对时间条件。

    X/Y_full 直接由原版 PhaseFieldDataset 返回，都是9通道。
    所有损失和范围/均值处理为FP32。无 recurrent state，无 PDE local corrector。
    """
    history = X.to(device,non_blocking=True)
    batch = history.shape[0]
    times = torch.arange(SEQ_LEN,device=device).unsqueeze(0).expand(batch,-1)*2000
    mean = history[:,-1,:1].mean((-2,-1),keepdim=True)
    sums = torch.zeros(4,device=device,dtype=torch.float64)
    for t in range(unroll):
        with torch.autocast(device_type=device.type,dtype=torch.float16,
                            enabled=device.type=='cuda' and cfg['amp']):
            delta = model(history,time_features(times))
        pred = project(delta,history[:,-1],mean,norm)
        target9 = Y_full[:,t].to(device,non_blocking=True)
        components = physical_loss(pred,target9[:,:7],history[:,-1],norm)
        if not torch.isfinite(components).all():
            raise RuntimeError(f'非有限损失: rollout t={t}; 保留最近有效 checkpoint')
        if train:
            loss = total_loss(components).sum()/(denominator*unroll)
            scaler.scale(loss).backward()
        sums += components.detach().sum(0).double()/unroll
        if t < unroll-1:
            # 与 train03 相同：整个 batch 共用一次 teacher-forcing 抽样。
            if train and torch.rand(1).item() < tf:
                nxt = target9.detach()
            else:
                nxt = pred_to_input(pred.unsqueeze(1)).squeeze(1).detach()
            history = torch.cat((history[:,1:],nxt.unsqueeze(1)),dim=1)
            times = torch.cat((times[:,1:],times[:,-1:]+6000),dim=1)
    return sums,batch


@torch.no_grad()
def validate(model, loader, unroll, norm, cfg, device):
    model.eval()
    sums = torch.zeros(4,dtype=torch.float64,device=device)
    count = 0
    for X,Y_full in loader:
        comp,n = rollout(model,X,Y_full,unroll,0.,norm,cfg,device)
        sums += comp
        count += n
    if not count:
        raise RuntimeError('验证集为空')
    components = (sums/count).cpu().tolist()
    return sum(a*b for a,b in zip(components,PROTOCOL['loss_weights'])),components


def reset_peaks(device,gpu_ids):
    if device.type == 'cuda':
        for i in gpu_ids:
            torch.cuda.reset_peak_memory_stats(i)


def print_memory(device,gpu_ids,tag):
    if device.type == 'cuda':
        for i in gpu_ids:
            print(f'  [{tag}] GPU{i}: {memory_report(torch.device(f"cuda:{i}"))}',flush=True)
    show_ram(tag)


def preflight(model,raw,optimizer,scaler,norm,cfg,device,gpu_ids):
    """原版数据路径/样本，测完整U8/24/40训练批；探测后恢复全部状态。"""
    saved=(cpu_tree(raw.state_dict()),cpu_tree(optimizer.state_dict()),
           copy.deepcopy(scaler.state_dict()),rng_state(device))
    times_by_u={}
    started=time.perf_counter()
    try:
        for unroll in (8,24,40):
            ds=dataset_for(cfg,norm,unroll,'train')
            if not len(ds):
                raise RuntimeError(f'U={unroll} 训练集为空')
            samples=[ds[i%len(ds)] for i in range(cfg['batch'])]
            X=torch.stack([v[0] for v in samples]); Y=torch.stack([v[1] for v in samples])
            durations=[]
            reset_peaks(device,gpu_ids)
            for probe in range(cfg['profile_steps']+1):
                model.train(); optimizer.zero_grad(set_to_none=True)
                if device.type=='cuda': torch.cuda.synchronize()
                t0=time.perf_counter()
                rollout(model,X,Y,unroll,0.,norm,cfg,device,train=True,scaler=scaler,denominator=X.shape[0])
                scaler.unscale_(optimizer)
                grad=torch.nn.utils.clip_grad_norm_(raw.parameters(),1.)
                if not torch.isfinite(grad): raise RuntimeError('预检梯度非有限')
                scaler.step(optimizer); scaler.update()
                if device.type=='cuda': torch.cuda.synchronize()
                if probe: durations.append(time.perf_counter()-t0)
            t0=time.perf_counter()
            model.eval()
            with torch.no_grad(): rollout(model,X,Y,unroll,0.,norm,cfg,device)
            if device.type=='cuda': torch.cuda.synchronize()
            val_seconds=time.perf_counter()-t0
            times_by_u[unroll]=(float(np.median(durations)),val_seconds,len(ds))
            print(f'预检 U={unroll}: 训练batch={times_by_u[unroll][0]:.3f}s; '
                  f'纯AR验证batch={val_seconds:.3f}s; samples={len(ds)}',flush=True)
            print_memory(device,gpu_ids,f'预检 U={unroll}')
            del X,Y,samples,ds
        total=0.
        for epoch in range(1,cfg['epochs']+1):
            u=min(8+4*((epoch-1)//3),40)
            train_s=float(np.interp(u,[8,24,40],[times_by_u[v][0] for v in (8,24,40)]))
            val_s=float(np.interp(u,[8,24,40],[times_by_u[v][1] for v in (8,24,40)]))
            n=len(dataset_for(cfg,norm,u,'train'))
            nv=len(dataset_for(cfg,norm,u,'val'))
            total+=math.ceil(n/cfg['batch'])*train_s+math.ceil(nv/cfg['batch'])*val_s
        print(f'预计 {cfg["epochs"]} epochs 训练+验证计算约 {total/3600:.1f}h。'
              '这是当前设备短测外推；另有数据加载、checkpoint和GPU频率变化开销。'
              '不会按40小时自动停止或减少数据。',flush=True)
    finally:
        raw.load_state_dict(saved[0],strict=True)
        optimizer.load_state_dict(saved[1]); scaler.load_state_dict(saved[2])
        optimizer.zero_grad(set_to_none=True); restore_rng(saved[3],device)
        if device.type=='cuda': torch.cuda.empty_cache()
    return time.perf_counter()-started


def arguments(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=int,default=BATCH_SIZE,
                        help='全局batch，默认8（面向RTX 4090 24GB）；断点续训须保持原值')
    parser.add_argument('--epochs',type=int,default=40)
    parser.add_argument('--lr',type=float,default=1e-3)
    parser.add_argument('--unroll-start',type=int,default=8)
    parser.add_argument('--unroll-max',type=int,default=40)
    parser.add_argument('--unroll-step',type=int,default=4)
    parser.add_argument('--seq-len',type=int,default=SEQ_LEN)
    parser.add_argument('--frame-step',type=int,default=FRAME_STEP)
    parser.add_argument('--stride',type=int,default=1)
    parser.add_argument('--data-root',default=DATA_ROOT,help='Run_01~Run_30的共同父目录；默认本文件所在目录')
    parser.add_argument('--output',default=str(Path(SAVE_PATH).parent),help='模型保存目录；默认本文件旁的unet_training')
    parser.add_argument('--norm-from',default=NORM_FROM,help='可选：原Hybrid权重中的归一化参数')
    parser.add_argument('--resume',default=RESUME_READ,help='auto=自动续训；none=新训练；或指定latest.pt')
    parser.add_argument('--width',type=int,default=MODEL_WIDTH)
    parser.add_argument('--layers',type=int,default=MODEL_LAYERS)
    parser.add_argument('--modes',type=int,default=32)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--seed',type=int,default=20261022)
    parser.add_argument('--gpus',type=int,default=GPU_COUNT,help='0=自动使用可见GPU；正整数=指定使用数量')
    parser.add_argument('--amp',action=argparse.BooleanOptionalAction,default=False,
                        help='默认与train03相同FP32；可选AMP仅作用于网络，物理损失仍FP32')
    parser.add_argument('--accum',type=int,default=1)
    parser.add_argument('--preflight',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--profile-steps',type=int,default=1)
    parser.add_argument('--checkpoint-seconds',type=float,default=600.)
    parser.add_argument('--checkpoint-groups',type=int,default=50)
    parser.add_argument('--session-hours',type=float,default=0.)
    parser.add_argument('--print-every',type=int,default=200)
    parsed=parser.parse_args(argv)  # 服务器终端启动；拼错的参数直接报错。
    cfg=vars(parsed)
    cfg['best_name']=Path(SAVE_PATH).name
    cfg['data_root']=local_path(cfg['data_root'])
    cfg['output']=local_path(cfg['output'])
    norm_from=cfg['norm_from']
    cfg['norm_from']=None if norm_from is None or str(norm_from).strip().lower() in ('','none') else local_path(norm_from)
    resume=str(cfg['resume']).strip()
    cfg['resume']=resume.lower() if resume.lower() in ('auto','none') else local_path(resume)
    if (cfg['seq_len'],cfg['frame_step'],cfg['unroll_start'],cfg['unroll_max'],cfg['unroll_step']) != (10,3,8,40,4):
        parser.error('正式协议固定 seq_len=10, frame_step=3, curriculum=8/4/40')
    if cfg['epochs']<40 or min(cfg[k] for k in ('batch','accum','stride','width','layers','modes','profile_steps','checkpoint_groups','print_every'))<1:
        parser.error('epochs至少40，其余训练/网络参数必须为正')
    if cfg['workers']<0 or cfg['lr']<=0 or cfg['checkpoint_seconds']<=0 or cfg['gpus']<0:
        parser.error('无效 workers/lr/checkpoint/gpus 设置')
    if cfg['preflight_only']: cfg['preflight']=True
    return cfg


def main():
    args=argparse.Namespace(**arguments())
    cfg=vars(args)
    print(f'脚本目录: {SCRIPT_DIR}\n数据目录: {args.data_root}\n权重目录: {args.output}',flush=True)
    print(f'归一化来源: {args.norm_from or "仅训练集统计（续训时使用checkpoint内参数）"} | 续训: {args.resume}',flush=True)
    check_data_directory(args.data_root)
    # 直接沿用train03：没有CUDA时回退CPU，不增加额外GPU硬报错。
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    visible=torch.cuda.device_count()
    print(f'设备: {device} | GPU数量: {visible}',flush=True)
    if int(os.environ.get('WORLD_SIZE','1'))>1:
        raise RuntimeError('本文件内置DataParallel，请直接python运行，无需torchrun')
    used=visible if args.gpus==0 else args.gpus
    if device.type=='cpu': used=0
    if used>visible: raise ValueError(f'请求{used}张GPU，但只检测到{visible}张')
    gpu_ids=list(range(used))
    if len(gpu_ids)>args.batch:
        raise ValueError('batch须至少等于使用的GPU数量')
    for gpu_id in gpu_ids:
        prop=torch.cuda.get_device_properties(gpu_id)
        print(f'  GPU {gpu_id}: {prop.name} | 独立显存 {prop.total_memory/1024**3:.1f} GiB',flush=True)
    print(f'使用GPU数: {len(gpu_ids)} | 全局batch={args.batch} | 数据读取worker={args.workers} | '
          f'网络精度: {"AMP" if args.amp and device.type=="cuda" else "FP32"}',flush=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if device.type=='cuda':
        torch.cuda.set_device(0); torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.allow_tf32=False; torch.backends.cuda.matmul.allow_tf32=False
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    ck=load_resume(cfg)
    if ck is not None and (ck.get('schema')!=SCHEMA or ck.get('kind')!=KIND or ck.get('protocol')!=PROTOCOL):
        raise ValueError('旧版基线的训练协议不同，请用新的输出目录；不要覆盖旧权重')
    norm=ck['norm_params'] if ck else read_norm(args.norm_from)
    # 原版PhaseFieldDataset：连续10帧，目标每3帧，mmap读入即补几何通道。
    ds=dataset_for(cfg,norm,args.unroll_start,'train')
    norm=ds.norm_params
    c_min,c_max=norm['c_min'],norm['c_max']
    if not len(ds): raise RuntimeError('训练集为空')
    sample_file=ds.samples[0][0][0]
    sample=np.load(sample_file,mmap_mode='r',allow_pickle=False)
    if sample.shape!=(7,512,512): raise ValueError(f'{sample_file}: 要求(7,512,512)，实际{sample.shape}')
    print(f'数据集: {sample.shape[0]}通道 (c+6etas), 训练对数量: {len(ds)}',flush=True)
    c_precip_norm=(.18-c_min)/(c_max-c_min)*2-1
    print(f'析出相阈值物理0.18 → 归一化 {c_precip_norm:.4f}',flush=True)
    show_ram('数据就绪')
    print('核对训练/验证文件列表，用于断点续训...',flush=True)
    fingerprint_started=time.perf_counter()
    fingerprint=dataset_fingerprint(args.data_root)
    print(f'文件列表核对完成，耗时{time.perf_counter()-fingerprint_started:.1f}s',flush=True)
    model=Baseline(cfg).to(device)
    if len(gpu_ids)>1:
        model=nn.DataParallel(model,device_ids=gpu_ids)
        print(f'  DataParallel ON (双卡/多卡并行): GPU {gpu_ids} | 全局batch={args.batch}',flush=True)
    raw=model.module if hasattr(model,'module') else model
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda' and args.amp)
    print(f'模型参数量: {sum(p.numel() for p in model.parameters()):,}',flush=True)
    show_ram('训练开始')
    epoch,next_batch,last_unroll=1,0,0
    best_loss,stage_best=float('inf'),float('inf')
    best_payload=None
    train_sums=[0.,0.,0.,0.,0.]
    cumulative,epoch_seconds=0.,0.
    if ck is not None:
        immutable=('batch','epochs','lr','seq_len','frame_step','stride','width','layers','modes','accum','seed','amp')
        if ck['fingerprint']!=fingerprint or ck['device_count']!=len(gpu_ids):
            raise ValueError('数据或设备数量改变，无法精确续训')
        for key in immutable:
            if ck['config'][key]!=cfg[key]: raise ValueError(f'续训配置改变: {key}')
        raw.load_state_dict(ck['model_state_dict'],strict=True)
        optimizer.load_state_dict(ck['optimizer_state_dict']); scheduler.load_state_dict(ck['scheduler_state_dict'])
        scaler.load_state_dict(ck['scaler_state_dict'])
        epoch,next_batch,last_unroll=ck['epoch'],ck['next_batch'],ck['unroll']
        best_loss,stage_best,best_payload=ck['best_loss'],ck['stage_best'],ck['best_payload']
        train_sums=ck['train_sums']; cumulative=ck['effective_seconds']; epoch_seconds=ck['epoch_seconds']
        restore_rng(ck['rng'],device)
        if best_payload is not None: atomic_save(best_payload,output/args.best_name)
        print(f'续训: epoch {epoch}, batch {next_batch}, best_U40={best_loss:.6g}',flush=True)
    if args.preflight and (device.type=='cuda' or args.preflight_only):
        seconds=preflight(model,raw,optimizer,scaler,norm,cfg,device,gpu_ids)
        print(f'预检用时 {seconds:.1f}s；权重、优化器、AMP和随机状态均已恢复',flush=True)
    if args.preflight_only: return
    stopped=[False]
    def mark_stop(signum,frame): stopped[0]=True
    for sig in (signal.SIGINT,signal.SIGTERM): signal.signal(sig,mark_stop)
    session_start=clock_start=time.perf_counter()
    checkpoint_clock=clock_start
    def effective_now(): return cumulative+time.perf_counter()-clock_start
    def save_checkpoint(current_epoch,batch_index,current_unroll,sums,seconds):
        nonlocal cumulative,clock_start,checkpoint_clock
        if device.type=='cuda': torch.cuda.synchronize()
        cumulative=effective_now()
        payload=dict(schema=SCHEMA,kind=KIND,config=dict(cfg),protocol=PROTOCOL,norm_params=norm,
                     epoch=current_epoch,next_batch=batch_index,unroll=current_unroll,
                     best_loss=best_loss,stage_best=stage_best,best_payload=best_payload,
                     train_sums=list(sums),effective_seconds=cumulative,epoch_seconds=seconds,
                     effective_gpu_hours=cumulative*len(gpu_ids)/3600,device_count=len(gpu_ids),
                     fingerprint=fingerprint,rng=rng_state(device),
                     model_state_dict=raw.state_dict(),optimizer_state_dict=optimizer.state_dict(),
                     scheduler_state_dict=scheduler.state_dict(),scaler_state_dict=scaler.state_dict())
        atomic_save(payload,output/'latest.pt',rotate=True)
        clock_start=checkpoint_clock=time.perf_counter()
    while epoch<=args.epochs:
        unroll=min(args.unroll_start+(epoch-1)//3*args.unroll_step,args.unroll_max)
        tf=max(0.,1.-epoch/args.epochs)
        if unroll!=last_unroll:
            stage_best=float('inf')
            print(f'  --> UNROLL → {unroll}, 当前阶段best重置',flush=True)
            last_unroll=unroll
        ds=dataset_for(cfg,norm,unroll,'train',epoch)
        val_ds=dataset_for(cfg,norm,unroll,'val',epoch)
        n_batches=math.ceil(len(ds)/args.batch)
        loader=iter(make_loader(ds,cfg,device,train=True,epoch=epoch,start=next_batch))
        val_loader=make_loader(val_ds,cfg,device,train=False,epoch=epoch)
        model.train(); reset_peaks(device,gpu_ids)
        epoch_clock=effective_now(); epoch_session=time.perf_counter(); start_batch=next_batch
        sums=torch.tensor(train_sums,dtype=torch.float64,device=device)
        for group_start in range(next_batch,n_batches,args.accum):
            optimizer.zero_grad(set_to_none=True)
            actual=min(len(ds)-group_start*args.batch,args.batch*args.accum)
            batch_sums=torch.zeros(4,dtype=torch.float64,device=device); seen=0
            for bi in range(group_start,min(group_start+args.accum,n_batches)):
                X,Y_full=next(loader)
                comp,n=rollout(model,X,Y_full,unroll,tf,norm,cfg,device,train=True,scaler=scaler,denominator=actual)
                batch_sums+=comp; seen+=n
                del X,Y_full
            scaler.unscale_(optimizer)
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(grad): raise RuntimeError('梯度非有限，保留上一有效checkpoint')
            scaler.step(optimizer); scaler.update()
            if any(not torch.isfinite(p).all() for p in raw.parameters()):
                raise RuntimeError('参数非有限，保留上一有效checkpoint')
            sums[:4]+=batch_sums; sums[4]+=seen
            batch_loss=total_loss((batch_sums/seen).float()).item()
            next_batch=min(group_start+args.accum,n_batches)
            if group_start%args.print_every==0 or next_batch==n_batches:
                elapsed=time.perf_counter()-epoch_session
                eta=elapsed*(n_batches-next_batch)/max(1,next_batch-start_batch)
                print(f'  Epoch {epoch} [{group_start}/{n_batches}] Loss={batch_loss:.4f} '
                      f'ETA={datetime.timedelta(seconds=int(eta))}',flush=True)
            now=time.perf_counter()
            pause=stopped[0] or (args.session_hours>0 and now-session_start>=args.session_hours*3600)
            if pause or now-checkpoint_clock>=args.checkpoint_seconds or next_batch%args.checkpoint_groups==0:
                save_checkpoint(epoch,next_batch,unroll,sums.cpu().tolist(),epoch_seconds+effective_now()-epoch_clock)
            if pause:
                print(f'已在优化器更新边界保存 {output/"latest.pt"}；下次可完整续训',flush=True)
                return
        avg_train=sum(v*w for v,w in zip((sums[:4]/sums[4]).cpu().tolist(),PROTOCOL['loss_weights']))
        avg_val,val_comp=validate(model,val_loader,unroll,norm,cfg,device)
        if not math.isfinite(avg_train+avg_val): raise RuntimeError('训练/验证非有限，不覆盖有效权重')
        scheduler.step(); stage_best=min(stage_best,avg_val)
        print(f'Epoch {epoch:>3} | U={unroll} TF={tf:.2f} | Train: {avg_train:.5f} '
              f'Val: {avg_val:.5f} | LR: {optimizer.param_groups[0]["lr"]:.6f}',flush=True)
        comp=(sums[:4]/sums[4]).cpu().tolist()
        print('  Loss分解(加权/步): '+'  '.join(f'{k}={v*w:.5f}' for k,v,w in
              zip(('mse','phi','gb','zener'),comp,PROTOCOL['loss_weights'])),flush=True)
        total_epoch=epoch_seconds+effective_now()-epoch_clock
        print(f'  本轮有效耗时: {total_epoch:.1f}s | 累计训练+验证: {effective_now()/3600:.3f}h',flush=True)
        print_memory(device,gpu_ids,f'Epoch {epoch} 结束')
        # 原版最终也在U40阶段竞争。独立best.pt只保留此阶段最佳，避免新权重覆盖最优。
        if unroll==40 and avg_val<best_loss:
            best_loss=avg_val
            best_payload=dict(schema=SCHEMA,kind=KIND,config=dict(cfg),protocol=PROTOCOL,
                              norm_params=norm,model_state_dict=cpu_tree(raw.state_dict()),epoch=epoch,
                              unroll=unroll,best_loss=best_loss,validation_components=val_comp,
                              parameters=sum(p.numel() for p in raw.parameters()),
                              effective_seconds=effective_now(),effective_gpu_hours=effective_now()*len(gpu_ids)/3600,
                              device_count=len(gpu_ids),fingerprint=fingerprint)
            atomic_save(best_payload,output/args.best_name)
            print(f'  -> 最佳模型已保存 (epoch {epoch})',flush=True)
        epoch+=1; next_batch=0; train_sums=[0.,0.,0.,0.,0.]; epoch_seconds=0.
        save_checkpoint(epoch,0,unroll,train_sums,0.)
        if device.type=='cuda': torch.cuda.empty_cache()
        gc.collect()
        if sys.platform=='linux': ctypes.CDLL('libc.so.6').malloc_trim(0)
    print(f'训练完成. 最佳U40 Val Loss: {best_loss:.5f}; 权重: {output/args.best_name}',flush=True)


if __name__=='__main__':
    main()
