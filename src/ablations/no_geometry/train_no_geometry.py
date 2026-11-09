"""
Figure 9b ablation — ConvLSTM without curvature-enhanced input descriptors
                       (512² native, 7 physical input channels)
================================================================================
  设计思路 (model01骨架 + 曲率输入消融 + 完整训练过程):
   教训1 (model02): 堆Vf/Lgb/计数等"总量统计loss"会把模型训成"凑统计量", 结构崩坏(LSW K偏差201%).
            → 回归简洁配方: MSE主导, 最终loss仅4项, 约束总权重0.08 (远小于MSE 1.0).
  教训2 (model02): TF没衰减到0就停训 → 模型从未学过纯AR推演 → 推理时输入是自己的预测就糊化.
            → epochs=40, TF线性衰减到0(epoch40), U=40自epoch25起覆盖最后16轮, 与推理段长一致.
  教训3 (v2): 推理段长(60步)远超训练unroll(24) → 外推区退化(融合/尺寸跳跃).
            → UNROLL_MAX=40 = 推理段长40步, 无外推.
   最终loss设计(精简4项, 界面剖面由修正层硬接管):
     - mse(1.0)主拟合 + phi(0.03)锁GB场位置 + gb(0.03,GT掩膜)防跨晶界融合 + zener(0.02)钉扎
     - 砍掉bin/grad/curv/lsw: bin/grad被AC修正层覆盖, curv/lsw被GT蕴含(MSE已对齐GT)
   物理驱动输入(老师要求):
     - 曲率通道: 输入7→9通道(追加界面曲率κ和梯度幅值|∇φ|)
     - κ/|∇φ|是η的导出量, 不含新信息; 作用是省去网络隐式计算 + 显式体现"曲率驱动晶界"
   物理驱动硬约束(老师要求的"物理法则驱动", 不是loss软约束):
     - 每步解码输出后执行 physics_correct()(与PDE同源算子):
       GB排斥 + AC界面弛豫(锁宽度) + Zener钉扎 + 物理钳制
     - 质量守恒双保险: decode中Δc强制零均值(第一道) + 修正层之后按输入帧均值重钉(第二道)
       (顺序与PDE一致: 先物理更新, 再 c += c_matrix−mean(c))
     - 修正强度极小, 只在"违反物理"处起作用, 不覆盖模型学到的动力学
       (守恒/排斥/宽度属于GT每帧恒成立的定律, 可安全强制)
   过程工程(保留):
    - 全模型fp32训练(消除fp16溢出) + 截断BPTT(20步, 分段立即反向, 峰值显存=20步的图)
    - NaN保护(跳批/逐步诊断/5连停/不覆盖checkpoint) + 周期平移增强
    - 每轮loss分量分解 + GPU峰值显存监控

  本消融移除由 η 导出的 κ 和 |∇φ| 两个输入描述子，保留物理修正层和四项loss。
  用法: python Figure9b_train_no_curvature.py
"""
import os, glob, argparse, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import psutil, gc, ctypes

# =====================================================================
# 曲率增强输入函数保留作与主模型对照；本消融不调用它，直接使用7个物理通道。
# =====================================================================
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
    """7通道归一化预测 → 7通道下一步输入 (无曲率增强描述子)。"""
    return pred7

# =====================================================================
# 物理修正层 (神经算子 + 物理修正): 每步解码输出后强制执行PDE的"硬约束/局部算子"
#   对应多晶3.py / run03.py 的物理项 (全部同源, 参数一致):
#     ① GB排斥算子  : c ← c − γ·φ·(c−c_gamma)  → PDE的 GB_penalty·φ·(c−c_gamma)
#                      晶界带内析出相组分每步被压回 → 治"晶界两侧析出相融合/半径跳变"
#     ② AC界面弛豫   : η ← 显式AC一步(固定W,κ)  → PDE的 Allen-Cahn 更新
#                      锁界面宽度ξ=√(κ/W)恒定 → 治"晶界忽粗忽细/Lgb统计怪异"
#     ③ Zener钉扎    : AC步内含 zener_force·η(1−η)(η−0.5) → 析出相拖住晶界
#     ④ 物理钳制     : c∈[0.08,0.24], η∈[0,1] → 与PDE每步clip完全一致
#   注意: 修正强度远小于LSTM单步变化量, 只在违反物理处起作用, 不覆盖模型学到的动力学;
#         可微 → 训练时参与反向传播, 模型学的是"物理修正后的残差", 训练/推理完全一致
# =====================================================================
PHY_C_GAMMA    = 0.10      # 基体平衡组分 (与多晶3.py一致)
PHY_GB_GAMMA   = 0.05      # GB排斥强度: 每步把晶界带内c向c_gamma压 (PDE的GB_penalty项)
PHY_AC_DT      = 3.0e-5    # AC弛豫有效时间步: ≈0.06个PDE步, 只锁宽度不移动晶界
PHY_W_GB       = 1000.0
PHY_KAPPA_ETA  = 160.0
PHY_L_MOB      = 0.4
PHY_ALPHA      = 25.0      # 溶质拖曳: W_gb(c)=W_gb(1+α·clip(c−c_gamma,0,1))
PHY_BETA_ZENER = 1500.0
PHY_C_MIN, PHY_C_MAX = 0.08, 0.24


def _lap3x3(x):
    """周期边界3×3 Laplacian (显式AC弛豫用, ∇²的离散近似)
    先F.pad(circular)再普通卷积: 兼容所有torch版本的conv2d"""
    b, _, ch, h, w = x.shape
    xr = x.reshape(b * ch, 1, h, w)
    xp = F.pad(xr, (1, 1, 1, 1), mode='circular')
    k = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]],
                     dtype=x.dtype, device=x.device)
    y = F.conv2d(xp, k)
    return y.reshape(b, 1, ch, h, w)


def physics_correct(pred7, c_min, c_max, gb_gamma=PHY_GB_GAMMA, ac_dt=PHY_AC_DT):
    """7通道归一化预测 (B,1,7,H,W) → 物理算子修正 → 同形状
    与PDE每帧恒成立的定律完全一致, 作为硬约束而非loss软约束"""
    c_n = pred7[:, :, 0:1]                        # (B,1,1,H,W) 归一化c
    e_n = pred7[:, :, 1:7]                        # (B,1,6,H,W) 归一化etas
    c = (c_n + 1) / 2 * (c_max - c_min) + c_min   # → 物理c
    e = (e_n + 1) / 2                             # → 物理eta [0,1]
    # φ = clip(2(1−Ση²),0,1) (与PDE一致)
    phi = torch.clamp((1.0 - e.pow(2).sum(dim=2, keepdim=True)) * 2.0, 0.0, 1.0)
    # ① GB排斥算子: 晶界带(φ>0)内c向c_gamma弛豫 (PDE: μ += GB_penalty·φ·(c−c_gamma))
    c = c - gb_gamma * phi * (c - PHY_C_GAMMA)
    # ② AC界面弛豫一步: ∂η/∂t = −L[W(c)·s_well + s_zener − κ∇²η]
    sum_sq = e.pow(2).sum(dim=2, keepdim=True)
    s_well = e * (e.pow(2) - 1.0) + 2.0 * e * (sum_sq - e.pow(2))   # 多阱势(含其它晶粒)
    zf = PHY_BETA_ZENER * torch.clamp(c - PHY_C_GAMMA, 0.0, None).pow(2)  # Zener力幅
    s_zener = zf * e * (1.0 - e) * (e - 0.5)                          # Zener钉扎形状项
    W_gb_var = PHY_W_GB * (1.0 + PHY_ALPHA * torch.clamp(c - PHY_C_GAMMA, 0.0, None))
    # 界面指示掩膜 4η(1−η): 晶内=0, 界面中心=1 → 修正只重塑过渡区(锁宽度),
    # 晶内/多晶重叠区的强多阱力被抑制, 不会误推η
    m_iface = 4.0 * e * (1.0 - e)
    e = e - ac_dt * PHY_L_MOB * m_iface * (W_gb_var * s_well + s_zener
                                           - PHY_KAPPA_ETA * _lap3x3(e))
    # ③ 物理钳制 (与PDE每步一致)
    c = c.clamp(PHY_C_MIN, PHY_C_MAX)
    e = e.clamp(0.0, 1.0)
    # ④ 归一化回训练空间
    c_n2 = (c - c_min) / (c_max - c_min) * 2.0 - 1.0
    e_n2 = e * 2.0 - 1.0
    return torch.cat([c_n2, e_n2], dim=2).clamp(-1.0, 1.0)

def show_ram(tag=""):
    """打印当前进程RAM占用(GB)——用于监控Kaggle 30GB限制"""
    process = psutil.Process(os.getpid())
    ram = process.memory_info().rss / 1024**3
    print(f"[{tag}] RAM = {ram:.2f} GB")

# ==================== 路径配置 ====================
# 本地
# DATA_ROOT   = r"D:\vscode work\xiangchang\数据集"
# SAVE_PATH   = r"D:\vscode work\xiangchang\LSTM\best_model.pth"
# RESUME_READ = r"D:\vscode work\xiangchang\LSTM\latest.pth"
# Kaggle (取消本地注释, 注释下面三行)
DATA_ROOT   = "/kaggle/input/datasets/y2hbtudssjfd/shujujishuangtongdao"
SAVE_PATH   = "/kaggle/working/best_model.pth"
RESUME_READ = "/kaggle/input/datasets/y2hbtudssjfd/conv-latest/latest.pth"
# ================================================

# ★ 核心参数 ★
FRAME_STEP = 3     # 3帧=6000 PDE步/步, 80步覆盖全模拟 → 修正点260000/500000(物理点), 段长40=UNROLL
SEQ_LEN    = 10    # 10帧窗口=8000 PDE步 → 充分捕捉晶界迁移趋势
TRUNC      = 20    # 截断BPTT: 每20步一段(梯度视野120k PDE步), 段末立即反向释放计算图


# =====================================================================
# 模型: ConvLSTM + UNet 编解码器 + 残差输出
# =====================================================================
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
        #   修正层之后还会按输入帧均值重钉(第二道), 双保险 ★
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


# =====================================================================
# 数据集: 连续短帧(抓速度) + 大步长跳跃(目标帧) + 7通道mmap即时读取
# =====================================================================
class PhaseFieldDataset(Dataset):
    """
    原始7通道: ch0=c(浓度), ch1~6=etas(晶粒序参数); 本消融不补κ/|∇φ|
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
        """mmap读取 7 个物理通道 → 归一化 → tensor"""
        arr = np.load(path, mmap_mode='r')
        data = np.array(arr, dtype=np.float32)             # (7, 512, 512)
        data[0] = (data[0] - self.norm_params['c_min']) / \
                  (self.norm_params['c_max'] - self.norm_params['c_min']) * 2 - 1
        for ch in range(1, 7):
            data[ch] = data[ch] * 2 - 1                    # etas: [0,1]→[-1,1]
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


# =====================================================================
# 训练循环: 渐进UNROLL + TF衰减 + 状态传递 + 独立GT + 断点续训
# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch",      type=int, default=2)   # 实测B=4@U=16已10.4GB, U=40需~24GB会爆; B=2全程安全(时间与B=4相同)
    parser.add_argument("--epochs",     type=int, default=40)   # TF在epoch40衰减到0, 含纯AR微调阶段
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--unroll-start", type=int, default=8)
    parser.add_argument("--unroll-max",   type=int, default=40)   # v3: 24→40, 让推理段长=训练UNROLL
    parser.add_argument("--unroll-step",  type=int, default=4)
    parser.add_argument("--seq-len",    type=int, default=SEQ_LEN)
    parser.add_argument("--frame-step", type=int, default=FRAME_STEP)
    parser.add_argument("--stride", type=int, default=1,
                        help="训练窗口采样步长: 1=全量(结果无影响), 2=训练时间减半(信息损失极小)")
    args, _ = parser.parse_known_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device} | GPU数量: {torch.cuda.device_count()}")

    # 数据: 从最小UNROLL开始获取归一化参数, 同时用mmap即时读取(不预加载RAM)
    ds = PhaseFieldDataset(DATA_ROOT, seq_len=args.seq_len,
                           unroll=args.unroll_start, step=args.frame_step,
                           mode='train', stride=args.stride)
    norm = ds.norm_params
    c_min, c_max = norm['c_min'], norm['c_max']   # 物理修正层归一化换算用
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=4, pin_memory=True)
    # 检测第一帧通道数
    sample_file = sorted(glob.glob(os.path.join(DATA_ROOT, "Run_*", "frame_*.npy")))[0]
    nch = np.load(sample_file, mmap_mode='r').shape[0]
    print(f"数据集: {nch}通道 ({'c+phi' if nch==2 else 'c+6etas' if nch==7 else '未知'}), "
          f"训练对数量: {len(ds)}")
    # 析出相判定阈值 c>0.18 (与compare.py/test.py一致) 换算到归一化[-1,1]空间
    # 注意: 不能用c_gamma=0.10, 那会让约束全图生效导致晶界不敢动
    c_precip_norm = (0.18 - norm['c_min']) / (norm['c_max'] - norm['c_min']) * 2 - 1
    print(f"析出相阈值物理0.18 → 归一化 {c_precip_norm:.4f}")
    show_ram("数据就绪")

# 模型: ConvLSTMUNet + (多卡时自动DataParallel) + 全fp32训练 + Cosine调度
    model = ConvLSTMUNet(in_channels=7, out_channels=7).to(device)   # 仅7个物理场通道
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print("  DataParallel ON (双卡并行)")
        if args.batch % torch.cuda.device_count() != 0:
            print(f"  警告: batch={args.batch} 不能被GPU数{torch.cuda.device_count()}整除, DataParallel会报错; 请调整")

    criterion = nn.MSELoss()                    # 纯像素MSE —— 大步长+独立GT下不会恒等
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)            # LR cos衰减: 平滑收敛
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    show_ram("训练开始")

    # 断点续训: 检测latest.pth → 恢复model+optimizer+scheduler+epoch
    start_epoch, best_loss = 1, float('inf')
    resume_save = os.path.join(os.path.dirname(SAVE_PATH), "latest.pth")
    resume_load = RESUME_READ if os.path.exists(RESUME_READ) else resume_save
    if os.path.exists(resume_load):
        ck = torch.load(resume_load, map_location='cpu')
        raw = model.module if hasattr(model, 'module') else model
        raw.load_state_dict(ck['model_state_dict'])
        optimizer.load_state_dict(ck['optimizer_state_dict'])
        scheduler.load_state_dict(ck['scheduler_state_dict'])
        start_epoch = ck['epoch'] + 1
        best_loss = ck['best_loss']
        # 恢复UNROLL小本本: 新格式直接读, 旧格式按epoch推算
        main._last_unroll = ck.get('unroll', 0)
        if main._last_unroll == 0:
            main._last_unroll = min(8 + (ck['epoch'] - 1) // 3 * 4, 40)
            print(f"  旧格式checkpoint: 按epoch {ck['epoch']} 推算UNROLL={main._last_unroll}")
        print(f"续训: 从epoch {ck['epoch']}恢复, best_loss={best_loss:.5f}")
        del ck

    for epoch in range(start_epoch, args.epochs + 1):
        # 渐进式UNROLL: 8→...→40, 每3epoch +4 (v3: 更快进入高UNROLL, 覆盖推理段长)
        unroll = min(args.unroll_start + (epoch - 1) // 3 * args.unroll_step,
                     args.unroll_max)
        # Teacher Forcing: 1.0 → 0 线性衰减, 后期纯AR与推理一致
        tf = max(0.0, 1.0 - epoch / args.epochs)

        # UNROLL变化时重建数据集(样本窗口不同), 重置best_loss进入新阶段竞争
        if epoch == start_epoch or unroll != getattr(main, '_last_unroll', 0):
            if unroll != getattr(main, '_last_unroll', 0):
                best_loss = float('inf')
                print(f"  --> UNROLL → {unroll}, best_loss 重置(新难度阶段)")
            main._last_unroll = unroll
            ds = PhaseFieldDataset(DATA_ROOT, seq_len=args.seq_len,
                                   unroll=unroll, step=args.frame_step,
                                   mode='train', norm_params=norm, stride=args.stride)
            val_ds = PhaseFieldDataset(DATA_ROOT, seq_len=args.seq_len,
                                       unroll=unroll, step=args.frame_step,
                                       mode='val', norm_params=norm, stride=args.stride)
            loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                                num_workers=4, pin_memory=True)
            val_loader = DataLoader(val_ds, batch_size=args.batch,
                                    shuffle=False, num_workers=4,
                                    pin_memory=True)

        raw_model = model.module if hasattr(model, 'module') else model
        raw_model.train()
        train_loss, n_batches = 0.0, len(loader)
        nan_batches = 0
        # 每轮loss分量分解(用于权重校准: 各分量既不能近乎为0, 也不能压过MSE)
        comp_keys = ('mse', 'gb', 'zener', 'phi')
        comp_w = (1.0, 0.03, 0.02, 0.03)
        comp = {k: 0.0 for k in comp_keys}
        torch.cuda.reset_peak_memory_stats()   # 每轮重置, 峰值=本轮实际占用

        # 分段截断BPTT: 一段=TRUNC步, 段末立即backward并释放该段计算图
        # (函数作用域隔离本段所有中间量, 返回后自动释放 → 峰值显存=一段的图)
        def run_segment(t0, t1, h_state, c_state, nxt):
            seg_total = 0.0
            nan_t = None
            for t in range(t0, t1):
                with torch.cuda.amp.autocast(enabled=False):   # 全fp32: 消除fp16溢出
                    # t=0: 用完整10帧forward初始化ConvLSTM状态
                    if t == 0:
                        pred_t, (h_state, c_state) = model(X)
                    # t>0: 用forward_step单帧推演, 保持状态连续性
                    else:
                        pred_t, h_state, c_state = raw_model.forward_step(
                            nxt.squeeze(1), h_state, c_state)
                    # ★ 物理修正层: 每步输出后强制执行PDE硬约束
                    #   (GB排斥/AC界面锁宽/Zener钉扎/物理钳制, 可微, 与推理一致)
                    pred_t = physics_correct(pred_t, c_min, c_max)
                    # 质量守恒(修正后重钉均值): 与PDE每步 c += c_matrix−mean(c) 同序,
                    # 修正层局部压c不会破坏总溶质守恒
                    c_ref = (X[:, -1][:, 0:1] if t == 0 else nxt.squeeze(1)[:, 0:1]
                             ).float().unsqueeze(1)      # (B,1,1,H,W) 与pred的c通道同形
                    c_pred2 = pred_t[:, :, 0:1]
                    c_new = c_pred2 + (c_ref.mean(dim=[-2, -1], keepdim=True)
                                       - c_pred2.mean(dim=[-2, -1], keepdim=True))
                    # 钳制用PDE物理范围[0.08,0.24]换算(远宽于观测范围): 边界截断不会破坏守恒
                    c_new = c_new.clamp((PHY_C_MIN - c_min) / (c_max - c_min) * 2 - 1,
                                        (PHY_C_MAX - c_min) / (c_max - c_min) * 2 - 1)
                    pred_t = torch.cat([c_new, pred_t[:, :, 1:]], dim=2)
                    # ★ 关键: 每步loss用自己的独立GT, 不是V8的同一帧 ★
                    mse = criterion(pred_t, Y_full[:, t:t+1, :7])   # 目标只取前7通道(后2是曲率,由etas导出)
                    eta_p = pred_t[:, :, 1:]                 # (B,1,6,H,W) 预测etas
                    eta_t = Y_full[:, t:t+1, 1:7]            # 目标6个etas
                    # ★ 物理约束Loss: GB排斥 + Zener钉扎 (PDE物理项的软约束) ★
                    # 当前预测的输入帧 (归一化空间)
                    if t == 0:
                        input_last = X[:, -1]                # 初始: 10帧输入最后一帧
                    else:
                        input_last = nxt.squeeze(1)          # 后续: 上一步预测帧
                    # 从预测etas算phi: φ = clip(2(1-Ση²), 0, 1), η还原到[0,1]
                    eta_sq = ((eta_p + 1) / 2) ** 2          # (B,1,6,H,W)
                    phi_pred = torch.clamp((1.0 - eta_sq.sum(dim=2, keepdim=True)) * 2.0, 0, 1)
                    c_pred = pred_t[:, :, 0:1]               # (B,1,1,H,W)
                    # GT phi (目标etas还原到[0,1]) — 供GB排斥/lgb/phi loss复用
                    eta_sq_t = ((eta_t + 1) / 2) ** 2
                    phi_t = torch.clamp((1.0 - eta_sq_t.sum(dim=2, keepdim=True)) * 2.0, 0, 1)
                    # GB排斥(GT晶界掩膜+膨胀): 用GT的φ(max_pool膨胀1px)做掩膜, 不依赖预测φ
                    # 治跨晶粒融合: GT晶界带处的析出相(高浓度)被罚, 膨胀覆盖融合团块跨越晶界的部分
                    gb_mask = F.max_pool2d(phi_t.squeeze(2), kernel_size=3, stride=1, padding=1).unsqueeze(2)
                    loss_gb_repel = (gb_mask * torch.clamp(c_pred - c_precip_norm, 0, None)).mean()
                    # Zener钉扎: 析出相处(c高)晶界移动应被抑制 → 治晶界穿过析出相
                    precip_mask = (c_pred > c_precip_norm).float()     # (B,1,1,H,W)
                    eta_change = (eta_p - input_last[:, 1:7].unsqueeze(1)).abs()  # 只取6个etas(9通道时1:7)
                    loss_zener = (precip_mask * eta_change).mean()
                    # ★ phi场一致性Loss: 像素级对齐晶界场(锁宽度/位置/深浅), 治宽窄漂移 ★
                    loss_phi = F.l1_loss(phi_pred, phi_t)
                    comp['mse'] += mse.detach().item()
                    comp['gb'] += loss_gb_repel.detach().item()
                    comp['zener'] += loss_zener.detach().item()
                    comp['phi'] += loss_phi.detach().item()
                    seg_total += (mse + 0.03 * loss_gb_repel + 0.02 * loss_zener
                                  + 0.03 * loss_phi)
                    if not torch.isfinite(seg_total):
                        # 逐t定位NaN来源(外层guard只查最后一步, 这里能定位到具体步)
                        print(f"  [epoch {epoch} batch {bi} t={t}] seg_total=NaN | "
                              f"mse={float(mse.detach()):.3e} gb={float(loss_gb_repel.detach()):.3e} "
                              f"zener={float(loss_zener.detach()):.3e} "
                              f"phi={float(loss_phi.detach()):.3e} | pred有限={bool(torch.isfinite(pred_t).all())}")
                        nan_t = t
                        break

                    # 准备下一步输入: TF概率用真值, 否则用自己的预测
                    if t < unroll - 1:
                        nxt = Y_full[:, t:t+1] if torch.rand(1).item() < tf \
                              else pred_to_input(pred_t).detach()   # AR: 预测补曲率通道→9通道
            return seg_total, h_state, c_state, nxt, nan_t

        for bi, (X, Y_full) in enumerate(loader):
            X, Y_full = X.to(device), Y_full.to(device)
            optimizer.zero_grad()
            h_state, c_state, nxt = None, None, None
            batch_loss = 0.0
            nan_batch = False
            for seg in range(0, unroll, TRUNC):
                t1 = min(seg + TRUNC, unroll)
                seg_total, h_state, c_state, nxt, nan_t = run_segment(seg, t1, h_state, c_state, nxt)
                if nan_t is not None:
                    nan_batch = True
                    break
                (seg_total / unroll).backward()   # 段末立即反向并释放本段计算图
                batch_loss += float(seg_total.detach()) / unroll
                h_state, c_state = h_state.detach(), c_state.detach()   # 截断: 断梯度, 状态值继续前传
                del seg_total
            if nan_batch:
                nan_batches += 1
                print(f"  [epoch {epoch} batch {bi}] 段内NaN(t={nan_t}), 跳过该batch更新 (连续{nan_batches}批)")
                if nan_batches >= 5:
                    raise RuntimeError("训练发散(loss连续NaN), 请从最近的有效checkpoint恢复")
                del X, Y_full, nxt
                if h_state is not None:
                    del h_state, c_state
                torch.cuda.empty_cache()
                continue
            nan_batches = 0
            # 梯度完整性检查(替代GradScaler的inf/nan防护): 非有限梯度跳过更新
            bad_grad = [n for n, p in raw_model.named_parameters()
                        if p.grad is not None and not torch.isfinite(p.grad).all()]
            if bad_grad:
                nan_batches += 1
                print(f"  [epoch {epoch} batch {bi}] 梯度非有限: {bad_grad[:3]} (连续{nan_batches}批), 跳过该batch更新")
                if nan_batches >= 5:
                    raise RuntimeError("训练发散(梯度连续NaN), 请从最近的有效checkpoint恢复")
                del X, Y_full, nxt
                torch.cuda.empty_cache()
                continue
            nan_batches = 0
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 防梯度爆炸
            optimizer.step()
            # 权重/缓冲完整性检查: 任一参数或BN统计非有限立即报错, 定位污染发生的batch
            bad = [n for n, p in raw_model.named_parameters() if not torch.isfinite(p).all()]
            bad += [n for n, b in raw_model.named_buffers() if not torch.isfinite(b).all()]
            if bad:
                raise RuntimeError(f"[epoch {epoch} batch {bi}] 参数/缓冲出现非有限值: {bad[:5]}")
            train_loss += batch_loss
            del X, Y_full, nxt
            if bi % 200 == 0:   # 进度打印: 每200个batch一条, 减少日志量
                print(f"  Epoch {epoch} [{bi}/{n_batches}] Loss={batch_loss:.4f}")

        avg_train = train_loss / n_batches

        # 验证: 纯AR, 无TF
        raw_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, Y_full in val_loader:
                X = X.to(device)
                h_v, c_v = None, None
                for t in range(unroll):
                    if t == 0:
                        pred_t, (h_v, c_v) = model(X)
                    else:
                        pred_t, h_v, c_v = raw_model.forward_step(
                            nxt.squeeze(1), h_v, c_v)
                    # 物理修正层: 与训练完全一致
                    pred_t = physics_correct(pred_t, c_min, c_max)
                    # 质量守恒(修正后重钉均值, 与训练/PDE同序)
                    c_ref = (X[:, -1][:, 0:1] if t == 0 else nxt.squeeze(1)[:, 0:1]
                             ).float().unsqueeze(1)      # (B,1,1,H,W) 与pred的c通道同形
                    c_pred2 = pred_t[:, :, 0:1]
                    c_new = c_pred2 + (c_ref.mean(dim=[-2, -1], keepdim=True)
                                       - c_pred2.mean(dim=[-2, -1], keepdim=True))
                    c_new = c_new.clamp((PHY_C_MIN - c_min) / (c_max - c_min) * 2 - 1,
                                        (PHY_C_MAX - c_min) / (c_max - c_min) * 2 - 1)
                    pred_t = torch.cat([c_new, pred_t[:, :, 1:]], dim=2)
                    val_mse = criterion(pred_t, Y_full[:, t:t+1, :7].to(device)).item()
                    eta_p = pred_t[:, :, 1:]
                    eta_t = Y_full[:, t:t+1, 1:7].to(device)
                    # 与训练一致: GB排斥 + Zener钉扎
                    if t == 0:
                        input_last = X[:, -1]
                    else:
                        input_last = nxt.squeeze(1)
                    eta_sq = ((eta_p + 1) / 2) ** 2
                    phi_pred = torch.clamp((1.0 - eta_sq.sum(dim=2, keepdim=True)) * 2.0, 0, 1)
                    c_pred = pred_t[:, :, 0:1]
                    eta_sq_t = ((eta_t + 1) / 2) ** 2
                    phi_t = torch.clamp((1.0 - eta_sq_t.sum(dim=2, keepdim=True)) * 2.0, 0, 1)
                    val_gb = (F.max_pool2d(phi_t.squeeze(2), kernel_size=3, stride=1, padding=1).unsqueeze(2)
                              * torch.clamp(c_pred - c_precip_norm, 0, None)).mean().item()
                    precip_mask = (c_pred > c_precip_norm).float()
                    eta_change = (eta_p - input_last[:, 1:7].unsqueeze(1)).abs()
                    val_zn = (precip_mask * eta_change).mean().item()
                    # phi场一致性: 与训练循环的loss_phi一致
                    val_phi = F.l1_loss(phi_pred, phi_t).item()
                    val_loss += (val_mse + 0.03 * val_gb + 0.02 * val_zn
                                 + 0.03 * val_phi)
                    nxt = pred_to_input(pred_t)
        avg_val = val_loss / (len(val_loader) * unroll)

        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:>3} | U={unroll} TF={tf:.2f} | "
              f"Train: {avg_train:.5f} Val: {avg_val:.5f} | LR: {lr:.6f}")
        n_steps = max(n_batches * unroll, 1)
        print("  Loss分解(加权/步): " + "  ".join(
            f"{k}={w * comp[k] / n_steps:.5f}" for k, w in zip(comp_keys, comp_w)))
        print(f"  GPU峰值显存: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

        # 保存: latest.pth每轮覆盖, best_model.pth仅val创新低时更新
        improved = avg_val < best_loss
        if improved:
            best_loss = avg_val
        raw = model.module if hasattr(model, 'module') else model
        data = {
            'model_state_dict': raw.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch, 'best_loss': best_loss,
            'norm_params': norm, 'frame_step': args.frame_step,
            'seq_len': args.seq_len, 'unroll': unroll,
        }
        if np.isfinite(float(avg_train)) and np.isfinite(float(avg_val)):
            torch.save(data, resume_save)      # 每轮覆盖, 防Kaggle 12h超时
            if improved:
                torch.save(data, SAVE_PATH)    # val创新低时更新best
                print(f"  -> 最佳模型已保存 (epoch {epoch})")
        else:
            print("  警告: 本轮Train/Val含NaN, 跳过保存, 保留上一轮有效checkpoint")
        del data

        torch.cuda.empty_cache()
        gc.collect()
        if sys.platform == 'linux':
            ctypes.CDLL('libc.so.6').malloc_trim(0)  # Linux强制归还堆内存
        show_ram(f"Epoch {epoch} 结束")

    print(f"训练完成. 最佳Val Loss: {best_loss:.5f}")


if __name__ == "__main__":
    main()
