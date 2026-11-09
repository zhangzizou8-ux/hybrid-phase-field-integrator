"""仅取消局部PDE修正、保留平均浓度控制的ConvLSTM训练消融。
单文件独立运行，不依赖原train03.py或基线脚本。将Run_01~Run_30与本脚本放同一目录：
    python -u 01_train_no_pde_keep_mean.py
默认实际batch=2，每个batch更新一次参数，不使用梯度累积，与train03一致。
    python -u 01_train_no_pde_keep_mean.py --print-every 10
显存预检：python -u 01_train_no_pde_keep_mean.py --preflight-only
正式协议：原模型9入7出、FP32、Adam1e-3、40轮cosine、U8每3轮+4至40、
TF=1-epoch/40、TBPTT20、训练Run01–24/验证Run25–30、stride1。
只移除晶界浓度松弛与局部序参量PDE更新；保留几何输入、范围投影、
解码浓度残差零均值、每步归一化均值控制及原四项损失。
均值控制目标保持train03的当前递归输入：TF时为当前参考输入，其余为反馈状态。
下游推理也必须取消同一局部PDE更新，并保留初始历史目标均值和原FOM burst。
默认只生成/覆盖同目录best_no_pde_mean.pt：model_state_dict为当前U阶段最佳，
40轮完成后为U40最佳；同文件嵌入最近完整epoch状态自动续训，不生成第二份latest。
预检不更新权重，恢复BN和RNG，不生成pt；不输出NPY、图片、CSV或日志文件。
相对路径均以脚本目录为基准，可选--data-root、--output、--norm-from。
为与既有Hybrid严格共享归一化，可用--norm-from best_model.pth，仅读norm_params。
默认内置已核验的原Hybrid归一化常数，避免服务器glob顺序改变消融口径；
--recompute-norm可显式改为原train03训练集glob[::5]扫描。
依赖PyTorch>=2.3、NumPy。单卡4090不使用AMP，不改变递归状态顺序。
"""
from pathlib import Path
import os, glob, argparse, random, math, gc
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
SCRIPT_DIR = Path(__file__).resolve().parent
KIND = 'convlstm_no_pde_keep_mean_batch2_v2'
TRUNC = 20
# 已从原Hybrid best_model.pth核验；仅来自原训练集，便于服务器独立使用。
DEFAULT_NORM = {'c_min': 0.09222412109375, 'c_max': 0.231201171875}

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

class ConvLSTMCell(nn.Module):
    """ConvLSTM单步单元: 用卷积门控在特征空间传递时序记忆"""
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        # 单次卷积输出4个门: input/forget/output/cell
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim,
                              kernel_size, padding=kernel_size // 2,
                              padding_mode='circular')

    def forward(self, x, cur_state):
        h_cur, c_cur = cur_state               # h=短时隐藏, c=长时记忆
        combined = torch.cat([x, h_cur], dim=1)
        cc_i, cc_f, cc_o, cc_g = torch.split(
            self.conv(combined), self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i); f = torch.sigmoid(cc_f)    # 输入门, 遗忘门
        o = torch.sigmoid(cc_o); g = torch.tanh(cc_g)       # 输出门, 候选记忆
        return f * c_cur + i * g, o * torch.tanh(f * c_cur + i * g)

    def init_hidden(self, batch_size, image_size, device):
        h, w = image_size
        return (torch.zeros(batch_size, self.hidden_dim, h, w, device=device),
                torch.zeros(batch_size, self.hidden_dim, h, w, device=device))

class ConvLSTMUNet(nn.Module):
    """UNet空间编码 → ConvLSTM时序瓶颈 → UNet空间解码 + 残差输出
    训练/推理均支持状态传递, 消灭train/inference gap"""
    def __init__(self, in_channels=7, out_channels=7, delta_scale=0.35):
        super().__init__()
        self.delta_scale = delta_scale   # 0.35: 折中——移动较到位减少重影, 又不放大噪声 (原0.25)
        # 编码器: 512→256→128→64 (每级2倍下采样)
        # 编码器: pad_mode='circular' 匹配PDE周期边界, 消除边缘假象
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, padding_mode='circular'),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2))
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1, padding_mode='circular'),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, padding_mode='circular'),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        # ConvLSTM瓶颈: 在64×64特征图上处理时空信息
        self.convlstm = ConvLSTMCell(input_dim=64, hidden_dim=64)
        # 解码器: 64→128→256→512 + 跳跃连接
        self.up3 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(64 + 32, 32, 3, padding=1, padding_mode='circular'),
            nn.BatchNorm2d(32), nn.ReLU())
        self.up2 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(32 + 16, 16, 3, padding=1, padding_mode='circular'),
            nn.BatchNorm2d(16), nn.ReLU())
        self.up1 = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
        self.dec1 = nn.Conv2d(16 + in_channels, out_channels, 3, padding=1,
                               padding_mode='circular')

    def encode(self, x_batch):
        """Batch编码: (B*T, C, H, W) → 3级下采样特征"""
        e1 = self.enc1(x_batch)   # 256² × 16ch
        e2 = self.enc2(e1)        # 128² × 32ch
        e3 = self.enc3(e2)        # 64²  × 64ch
        return e1, e2, e3

    def decode(self, h_t, e2_last, e1_last, x_last):
        """解码: ConvLSTM隐状态 + 跳跃连接 → 残差输出(delta) + 上一帧 = 预测帧"""
        d3 = self.up3(h_t)
        d3_out = self.dec3(torch.cat([d3, e2_last], dim=1))  # 跳跃连接: e2
        d2 = self.up2(d3_out)
        d2_out = self.dec2(torch.cat([d2, e1_last], dim=1))  # 跳跃连接: e1
        d1 = self.up1(d2_out)
        delta_raw = self.dec1(torch.cat([d1, x_last], dim=1))
        # 残差输出: delta_scale限制单步变化幅度, 保持数值稳定
        delta = self.delta_scale * torch.tanh(delta_raw / self.delta_scale)
        # ★ 质量守恒(第一道): c通道delta强制零均值(Δc空间积分=0);
        #   范围投影之后仍按当前输入帧均值调整(第二道；最终clip可留残差) ★
        delta[:, 0:1] = delta[:, 0:1] - delta[:, 0:1].mean(dim=[-2,-1], keepdim=True)
        pred = torch.clamp(x_last[:, :7] + delta, -1, 1)   # 只对前7通道做残差(后2通道是曲率,由etas导出)
        return pred.unsqueeze(1)

    def forward(self, x):
        """多帧前向: seq_len帧 → ConvLSTM逐帧消化 → 预测下一帧 + (h,c)状态
        返回 (pred, (h, c)) —— 训练和推理共用此接口初始化"""
        b, seq_len, c, h, w = x.size()
        x_flat = x.view(b * seq_len, c, h, w)
        e1, e2, e3 = self.encode(x_flat)
        _, c3, h3, w3 = e3.size()
        e3_seq = e3.view(b, seq_len, c3, h3, w3)
        # 初始化ConvLSTM状态, 逐帧输入时序特征
        h_t, c_t = self.convlstm.init_hidden(b, (h3, w3), x.device)
        for t in range(seq_len):
            # ConvLSTM在fp32下计算: 长链递归中fp16累加易溢出(实测U=32时发散为NaN), 特征图仅64²开销可忽略
            with torch.cuda.amp.autocast(enabled=False):
                h_t, c_t = self.convlstm(e3_seq[:, t].float(), [h_t.float(), c_t.float()])
        # 跳跃连接取输入序列最后一帧的编码器特征
        e2_last = e2.view(b, seq_len, -1, h // 4, w // 4)[:, -1]
        e1_last = e1.view(b, seq_len, -1, h // 2, w // 2)[:, -1]
        x_last = x[:, -1]
        pred = self.decode(h_t, e2_last, e1_last, x_last)
        return pred, (h_t, c_t)

    def forward_step(self, x_single, h_prev, c_prev):
        """单帧推演: 编码1帧 → ConvLSTM用上一步状态推进一步 → 解码
        h_prev/c_prev跨UNROLL步骤保持, 不丢失长程记忆"""
        e1, e2, e3 = self.encode(x_single)
        with torch.cuda.amp.autocast(enabled=False):
            h_t, c_t = self.convlstm(e3.float(), [h_prev.float(), c_prev.float()])
        pred = self.decode(h_t, e2, e1, x_single)
        return pred, h_t, c_t

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

        if mode == 'train' and self.norm_params is None:
            self._compute_stats(run_folders)

    def _compute_stats(self, run_folders):
        """扫描训练集c通道min/max, etas范围固定[0,1]"""
        print(f"扫描 {len(run_folders)} 个Run, 计算c通道归一化参数 (512²)...")
        c_min, c_max = float('inf'), float('-inf')
        for rd in run_folders:
            for f in glob.glob(os.path.join(rd, "frame_*.npy"))[::5]:
                arr = np.load(f, mmap_mode='r')
                c_min = min(c_min, float(arr[0].min()))
                c_max = max(c_max, float(arr[0].max()))
        self.norm_params = {'c_min': float(c_min), 'c_max': float(c_max)}
        print(f"c=[{c_min:.4f},{c_max:.4f}], etas=[0,1]")

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

def range_project(pred7, norm):
    """仅保留原 physics_correct 末尾的物理范围投影；不执行任何局部PDE更新。"""
    lo, hi = norm['c_min'], norm['c_max']
    c = ((pred7[:, :, :1] + 1) / 2 * (hi-lo) + lo).clamp(.08, .24)
    eta = ((pred7[:, :, 1:] + 1) / 2).clamp(0, 1)
    return torch.cat([(c-lo)/(hi-lo)*2-1, eta*2-1], 2).clamp(-1, 1)


def mean_control(pred7, current_input, norm):
    """保留train03每步均值调整：目标是当前递归输入的均值，包含TF输入。"""
    c = pred7[:, :, :1]
    target = current_input[:, :1].unsqueeze(1).float().mean((-2, -1), keepdim=True)
    c = c + (target - c.mean((-2, -1), keepdim=True))
    lo, hi = norm['c_min'], norm['c_max']
    c = c.clamp((.08-lo)/(hi-lo)*2-1, (.24-lo)/(hi-lo)*2-1)
    # 与原版相同：最终clip后可能留有小的均值偏差，不声称严格守恒。
    return torch.cat([c, pred7[:, :, 1:]], 2)


def loss_components(pred, gt9, current_input, norm):
    eta_p, eta_t = pred[:, :, 1:], gt9[:, :, 1:7]
    phi_p = (2*(1-((eta_p+1)/2).square().sum(2, keepdim=True))).clamp(0, 1)
    phi_t = (2*(1-((eta_t+1)/2).square().sum(2, keepdim=True))).clamp(0, 1)
    c = pred[:, :, :1]
    threshold = (.18-norm['c_min'])/(norm['c_max']-norm['c_min'])*2-1
    gb = F.max_pool2d(phi_t.squeeze(2), 3, 1, 1).unsqueeze(2)
    return (F.mse_loss(pred, gt9[:, :, :7]), F.l1_loss(phi_p, phi_t),
            (gb*(c-threshold).clamp_min(0)).mean(),
            ((c>threshold).float()*(eta_p-current_input[:, 1:7].unsqueeze(1)).abs()).mean())


def rollout(model, X, Y, norm, tf, backward=False):
    """原版递归和TBPTT20：显式反馈detach，段末反向，内部状态跨段保留数值。"""
    unroll = Y.shape[1]
    state = nxt = None
    total = np.zeros(4, dtype=np.float64)
    for start in range(0, unroll, TRUNC):
        segment = 0.
        for t in range(start, min(start+TRUNC, unroll)):
            current = X[:, -1] if t == 0 else nxt.squeeze(1)
            if t == 0:
                pred, state = model(X)
            else:
                pred, h, c = model.forward_step(current, *state)
                state = (h, c)
            pred = mean_control(range_project(pred, norm), current, norm)
            components = loss_components(pred, Y[:, t:t+1], current, norm)
            loss = components[0] + .03*components[2] + .02*components[3] + .03*components[1]
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at rollout step {t}; checkpoint unchanged.')
            total += np.array([v.detach().item() for v in components])/unroll
            segment = segment + loss
            if t < unroll-1:
                # 和train03一致：每一步整批共用一次TF抽样。
                nxt = Y[:, t:t+1] if backward and torch.rand(1).item() < tf else pred_to_input(pred).detach()
        if backward:
            (segment/unroll).backward()
        state = tuple(s.detach() for s in state)
        del segment, pred, components, loss
    return total


def resolve_path(value):
    p = Path(value).expanduser()
    return (p if p.is_absolute() else SCRIPT_DIR/p).resolve()


def detect_data(root):
    """直接同目录Run，或同目录下至多三层的唯一数据根；不读取Run00/31–35。"""
    root = resolve_path(root)
    def complete(p):
        return all((p/f'Run_{i:02d}').is_dir() for i in range(1, 31))
    if complete(root):
        candidates = [root]
    else:
        candidates = []
        for parent, dirs, _ in os.walk(root):
            p = Path(parent)
            depth = len(p.relative_to(root).parts)
            if complete(p):
                candidates.append(p); dirs[:] = []
            elif depth >= 3:
                dirs[:] = []
            else:
                dirs[:] = [d for d in dirs if not d.startswith('.') and not d.startswith('Run_')]
    if len(candidates) != 1:
        raise RuntimeError(f'需要唯一的Run_01~Run_30共同父目录，发现{candidates}；请使用--data-root指定。')
    root = candidates[0]
    for rid in range(1, 31):
        files = sorted((root/f'Run_{rid:02d}').glob('frame_*.npy'))
        if len(files) < 130:
            raise ValueError(f'{root}/Run_{rid:02d}: 至少需130帧以训练U40，实际{len(files)}帧。')
        steps = [int(p.stem.split('_')[-1]) for p in files]
        if steps[0] != 2000: raise ValueError(f'Run_{rid:02d}: 首帧必须为frame_002000.npy')
        if any(b-a != 2000 for a,b in zip(steps,steps[1:])):
            raise ValueError(f'Run_{rid:02d}: 帧间隔必须为2000步，禁止跨缺帧构造窗口。')
        for p in (files[0], files[-1]):
            if np.load(p, mmap_mode='r').shape != (7, 512, 512):
                raise ValueError(f'{p}: 需要(7,512,512)的NPY物理场。')
    print(f'数据: {root}\n训练Run01–24，验证Run25–30；Run00/31–35不参与训练或归一化。', flush=True)
    return root


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}


def atomic_save(payload, path):
    """训练中原子替换同一个pt；不保存latest、备份、预测或日志文件。"""
    temp = path.with_name(path.name+'.tmp')
    try:
        with open(temp, 'wb') as f:
            torch.save(payload, f); f.flush(); os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists(): temp.unlink()


def loader_for(ds, args, epoch, train, device):
    generator = torch.Generator().manual_seed(args.seed+epoch)
    return DataLoader(ds, batch_size=args.batch, shuffle=train,
                      num_workers=args.workers, pin_memory=device.type=='cuda',
                      generator=generator, worker_init_fn=seed_worker)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed); np.random.seed(seed)


def finite_model(model):
    return all(torch.isfinite(v).all().item() for v in model.state_dict().values())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default=str(SCRIPT_DIR))
    parser.add_argument('--output', default=str(SCRIPT_DIR/'best_no_pde_mean.pt'), help='唯一pt文件路径')
    parser.add_argument('--batch', type=int, default=2, help='实际batch，默认2；每批更新一次参数，无梯度累积')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20261028)
    parser.add_argument('--norm-from', default=None, help='可选原Hybrid checkpoint；只读取norm_params，不加载网络')
    parser.add_argument('--recompute-norm', action='store_true', help='显式重新扫描训练集；默认共享原Hybrid归一化')
    parser.add_argument('--resume', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--preflight', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--preflight-only', action='store_true', help='只运行真实U40前后向显存检查，不生成pt')
    parser.add_argument('--print-every', type=int, default=200)
    args = parser.parse_args(argv)
    if args.batch < 1 or args.workers < 0 or args.print_every < 1:
        parser.error('batch>0，workers>=0，print-every>0。')
    output = resolve_path(args.output)
    if output.suffix != '.pt': parser.error('--output必须为.pt文件路径')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if device.type=='cuda':
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    root = detect_data(args.data_root)
    print(f'{device}: {torch.cuda.get_device_name(0) if device.type=="cuda" else "CPU"} | FP32 | '
          f'batch={args.batch}, gradient accumulation=OFF (one update per batch)', flush=True)
    print('仅取消局部PDE更新；保留范围投影、均值控制、零均值浓度残差、9通道及全部原损失。',flush=True)
    print('BatchNorm与优化器均使用实际batch；默认batch=2，与train03一致。',flush=True)
    config = dict(batch=args.batch, gradient_accumulation=1, seed=args.seed,
                  epochs=40, lr=.001, seq_len=10, frame_step=3, trunc=20, stride=1,
                  unroll_start=8, unroll_step=4, unroll_max=40, fp32=True,
                  local_pde_correction=False, normalized_mean_control=True)
    ck = None
    if output.exists() and args.resume and not args.preflight_only:
        ck = torch.load(output, map_location='cpu', weights_only=False)
        if ck.get('kind') != KIND or ck.get('training_config') != config:
            raise ValueError('现有pt的消融类型/训练配置不匹配；batch8旧训练不能接续为batch2。请另选--output开始新训练，不能混用No-local或原Hybrid权重。')
    elif output.exists() and not args.resume:
        raise FileExistsError(f'{output}已存在。新训练请指定其他--output或自行移走原文件。')
    norm = ck['norm_params'] if ck else (None if args.recompute_norm else dict(DEFAULT_NORM))
    if args.norm_from:
        requested = torch.load(resolve_path(args.norm_from), map_location='cpu', weights_only=False)['norm_params']
        if ck and requested != norm: raise ValueError('续训归一化与--norm-from不一致')
        norm = requested
    print(f'归一化参数: {norm if norm else "扫描训练集"}', flush=True)
    probe = PhaseFieldDataset(str(root), seq_len=10, unroll=40, step=3, mode='train',norm_params=norm)
    norm = probe.norm_params
    if not np.isfinite([norm['c_min'],norm['c_max']]).all() or norm['c_max'] <= norm['c_min']:
        raise ValueError('归一化参数无效')
    model = ConvLSTMUNet(in_channels=9, out_channels=7).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
    start_epoch, best_loss, best_state, best_epoch, last_unroll = 1, float('inf'), None, None, 0
    if ck:
        model.load_state_dict(ck['resume_state_dict'])
        optimizer.load_state_dict(ck['optimizer_state_dict'])
        scheduler.load_state_dict(ck['scheduler_state_dict'])
        start_epoch=ck['completed_epoch']+1; last_unroll=ck['unroll']
        best_loss=ck['best_loss']; best_state=ck['model_state_dict']; best_epoch=ck['epoch']
        random.setstate(ck['rng_python']);np.random.set_state(ck['rng_numpy']);torch.set_rng_state(ck['rng_torch'])
        if device.type=='cuda' and ck['rng_cuda'] is not None: torch.cuda.set_rng_state(ck['rng_cuda'],0)
    if args.preflight or args.preflight_only:
        # 最大U40完整前/后向；不更新参数，恢复BN缓冲与随机状态，不污染正式训练。
        before=cpu_state(model)
        rng=(random.getstate(), np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state(0) if device.type=='cuda' else None)
        X,Y=next(iter(loader_for(probe,args,0,False,device)))
        try:
            if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
            model.train();optimizer.zero_grad(set_to_none=True)
            rollout(model,X.to(device),Y.to(device),norm,0.,True)
            if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise FloatingPointError('预检梯度非有限')
            print('U40前后向预检通过'+(f'，GPU峰值 {torch.cuda.max_memory_allocated()/1024**3:.2f} GiB' if device.type=='cuda' else ''),flush=True)
        except torch.OutOfMemoryError as e:
            raise RuntimeError('显存不足：请检查GPU可用显存和其他占用进程。正式训练默认batch=2；未自动改变实验参数。') from e
        finally:
            model.load_state_dict(before);optimizer.zero_grad(set_to_none=True)
            random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2])
            if device.type=='cuda':torch.cuda.set_rng_state(rng[3],0);torch.cuda.empty_cache()
            del before,X,Y
        if args.preflight_only:return
    del probe, ck
    output.parent.mkdir(parents=True,exist_ok=True)
    for epoch in range(start_epoch,41):
        u=min(8+(epoch-1)//3*4,40);tf=max(0.,1-epoch/40)
        if u!=last_unroll:best_loss=float('inf');best_state=None;best_epoch=None
        train=PhaseFieldDataset(str(root),seq_len=10,unroll=u,step=3,mode='train',norm_params=norm)
        val=PhaseFieldDataset(str(root),seq_len=10,unroll=u,step=3,mode='val',norm_params=norm)
        tl=loader_for(train,args,epoch,True,device);vl=loader_for(val,args,epoch,False,device)
        model.train();metrics=np.zeros(4);count=0
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
        for bi,(X,Y) in enumerate(tl):
            optimizer.zero_grad(set_to_none=True)
            score=rollout(model,X.to(device),Y.to(device),norm,tf,True)
            metrics+=score;count+=len(X)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            if bi%args.print_every==0:
                batch_loss=score[0]+.03*score[2]+.02*score[3]+.03*score[1]
                print(f'Epoch {epoch}/40 U={u} TF={tf:.3f} batch={bi+1}/{len(tl)} '
                      f'samples={count}/{len(train)} loss={batch_loss:.6g}',flush=True)
        if not finite_model(model):raise FloatingPointError('模型/BN缓冲非有限，未覆盖pt')
        model.eval();validation=np.zeros(4)
        with torch.no_grad():
            for X,Y in vl:validation+=rollout(model,X.to(device),Y.to(device),norm,0.)
        val_loss=float(validation@np.array([1.,.03,.03,.02])/len(vl))
        if not math.isfinite(val_loss):raise FloatingPointError('验证非有限，未覆盖pt')
        if val_loss<best_loss:best_loss=val_loss;best_state=cpu_state(model);best_epoch=epoch
        scheduler.step();last_unroll=u
        payload=dict(kind=KIND, model_state_dict=best_state,norm_params=norm,
                     epoch=best_epoch,best_loss=best_loss,unroll=u,seq_len=10,frame_step=3,
                     local_pde_correction=False,normalized_mean_control=True,
                     formal_best=(u==40),training_config=config,
                     # 同一个pt嵌入最近epoch状态用于续训；推理读取model_state_dict即可。
                     resume_state_dict=cpu_state(model),optimizer_state_dict=optimizer.state_dict(),
                     scheduler_state_dict=scheduler.state_dict(),completed_epoch=epoch,
                     rng_python=random.getstate(),rng_numpy=np.random.get_state(),
                     rng_torch=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state(0) if device.type=='cuda' else None)
        atomic_save(payload,output)
        print(f'Epoch {epoch}/40 U={u} train={metrics@np.array([1.,.03,.03,.02])/len(tl):.6g} '
              f'val={val_loss:.6g} stage_best={best_loss:.6g} best_epoch={best_epoch} '
              f'peak={(torch.cuda.max_memory_allocated()/1024**3) if device.type=="cuda" else 0:.2f}GiB',flush=True)
        del payload,tl,vl,train,val
        gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()
    print(f'完成：唯一权重文件 {output}；model_state_dict为U40阶段最优模型，含归一化及续训状态。',flush=True)


if __name__=='__main__':
    main()
