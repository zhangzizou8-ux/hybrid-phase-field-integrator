"""
run03.py — ConvLSTM大步长推理引擎 (512²原生, 7物理通道+2曲率输入, PDE间歇精修)
  用法: python run03.py

  与train03.py严格同构 (model03 = model01骨架 + 物理修正层 + 精简4项loss):
    1. FRAME_STEP=3 → 每步6000 PDE步；初始帧为20000步，80步AI推进并在第40、80步各精修2000步，最终到达504000步
    2. 每步AI输出后执行 physics_correct() 物理修正层(与训练完全一致):
       GB排斥 + AC界面锁宽 + Zener钉扎 + 物理钳制 → 修正后按初始c均值重钉(守恒)
    3. PDE_CORRECT=40 → 每40步PDE精修2000步(共2次), AI段长40步 = 训练UNROLL(40), 无外推
       ★ 推理模式开关 USE_PDE_CORRECT: True=带PDE修正(默认) / False=纯AI推演(不调用PDE求解器)
    4. 修正帧按真实物理步号保存, 每次运行自动清理旧帧

  工作流:
    输入10帧512²×7ch(+2曲率) → model()初始化ConvLSTM → forward_step逐帧推演80步
    → 每步: AI输出 → 物理修正层 → 守恒重钉 → 保存/作为下一步输入
    → 每40步: PDE精修2000步 → 加速比 ≈ 120×
"""
import os, argparse, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cupy as cp
from tqdm import tqdm

# 与train03一致: 7通道帧 → 9通道 (追加界面曲率κ和梯度幅值|∇φ|)
def add_curvature(frame7):
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

# =====================================================================
# 物理修正层 (与train03.py的physics_correct完全一致, 同源PDE算子):
#   每步AI输出后强制执行: GB排斥 + AC界面弛豫(锁宽度) + Zener钉扎 + 物理钳制
#   → 对应多晶3.py的 GB_penalty/W_gb/kappa_eta/zener/clip 项
#   训练和推理共用同一算子, 模型学的是"物理修正后的残差"
# =====================================================================
PHY_C_GAMMA    = 0.10
PHY_GB_GAMMA   = 0.05
PHY_AC_DT      = 3.0e-5
PHY_W_GB       = 1000.0
PHY_KAPPA_ETA  = 160.0
PHY_L_MOB      = 0.4
PHY_ALPHA      = 25.0
PHY_BETA_ZENER = 1500.0
PHY_C_MIN, PHY_C_MAX = 0.08, 0.24


def _lap3x3(x):
    """周期边界3×3 Laplacian (先F.pad(circular)再卷积, 兼容所有torch版本)"""
    b, _, ch, h, w = x.shape
    xr = x.reshape(b * ch, 1, h, w)
    xp = F.pad(xr, (1, 1, 1, 1), mode='circular')
    k = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]],
                     dtype=x.dtype, device=x.device)
    y = F.conv2d(xp, k)
    return y.reshape(b, 1, ch, h, w)


def physics_correct(pred7, c_min, c_max, gb_gamma=PHY_GB_GAMMA, ac_dt=PHY_AC_DT):
    """7通道归一化预测 (B,1,7,H,W) → 物理算子修正 → 同形状"""
    c_n = pred7[:, :, 0:1]
    e_n = pred7[:, :, 1:7]
    c = (c_n + 1) / 2 * (c_max - c_min) + c_min
    e = (e_n + 1) / 2
    phi = torch.clamp((1.0 - e.pow(2).sum(dim=2, keepdim=True)) * 2.0, 0.0, 1.0)
    # ① GB排斥算子: 晶界带内c向c_gamma弛豫 (治跨晶界融合)
    c = c - gb_gamma * phi * (c - PHY_C_GAMMA)
    # ② AC界面弛豫一步 (锁界面宽度)
    sum_sq = e.pow(2).sum(dim=2, keepdim=True)
    s_well = e * (e.pow(2) - 1.0) + 2.0 * e * (sum_sq - e.pow(2))
    zf = PHY_BETA_ZENER * torch.clamp(c - PHY_C_GAMMA, 0.0, None).pow(2)
    s_zener = zf * e * (1.0 - e) * (e - 0.5)
    W_gb_var = PHY_W_GB * (1.0 + PHY_ALPHA * torch.clamp(c - PHY_C_GAMMA, 0.0, None))
    # 界面指示掩膜: 只重塑过渡区(锁宽度), 晶内/重叠区不误推
    m_iface = 4.0 * e * (1.0 - e)
    e = e - ac_dt * PHY_L_MOB * m_iface * (W_gb_var * s_well + s_zener
                                           - PHY_KAPPA_ETA * _lap3x3(e))
    # ③ 物理钳制
    c = c.clamp(PHY_C_MIN, PHY_C_MAX)
    e = e.clamp(0.0, 1.0)
    # ④ 归一化回训练空间
    c_n2 = (c - c_min) / (c_max - c_min) * 2.0 - 1.0
    e_n2 = e * 2.0 - 1.0
    return torch.cat([c_n2, e_n2], dim=2).clamp(-1.0, 1.0)


# =====================================================================
# 模型定义 (与train03.py完全一致: ConvLSTMUNet + BatchNorm + 残差)
# =====================================================================
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim,
                              kernel_size, padding=kernel_size // 2,
                              padding_mode='circular')
    def forward(self, x, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([x, h_cur], dim=1)
        cc_i, cc_f, cc_o, cc_g = torch.split(
            self.conv(combined), self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i); f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o); g = torch.tanh(cc_g)
        return f * c_cur + i * g, o * torch.tanh(f * c_cur + i * g)
    def init_hidden(self, batch_size, image_size, device):
        h, w = image_size
        return (torch.zeros(batch_size, self.hidden_dim, h, w, device=device),
                torch.zeros(batch_size, self.hidden_dim, h, w, device=device))


class ConvLSTMUNet(nn.Module):
    """UNet编码(3级下采样, BatchNorm) → ConvLSTM瓶颈 → UNet解码(跳跃连接) → 残差"""
    def __init__(self, in_channels=7, out_channels=7, delta_scale=0.35):
        super().__init__()
        self.delta_scale = delta_scale   # 0.35: 与train_01.py一致
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, padding_mode='circular'),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2))
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1, padding_mode='circular'),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, padding_mode='circular'),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.convlstm = ConvLSTMCell(input_dim=64, hidden_dim=64)
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
        e1 = self.enc1(x_batch)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        return e1, e2, e3

    def decode(self, h_t, e2_last, e1_last, x_last):
        d3 = self.up3(h_t)
        d3_out = self.dec3(torch.cat([d3, e2_last], dim=1))
        d2 = self.up2(d3_out)
        d2_out = self.dec2(torch.cat([d2, e1_last], dim=1))
        d1 = self.up1(d2_out)
        delta_raw = self.dec1(torch.cat([d1, x_last], dim=1))
        delta = self.delta_scale * torch.tanh(delta_raw / self.delta_scale)
        delta[:, 0:1] = delta[:, 0:1] - delta[:, 0:1].mean(dim=[-2,-1], keepdim=True)
        pred = torch.clamp(x_last[:, :7] + delta, -1, 1)   # 只对前7通道做残差(后2是曲率)
        return pred.unsqueeze(1)

    def forward(self, x):
        """seq_len帧 → ConvLSTM逐帧消化 → (pred, (h, c))"""
        b, seq_len, c, h, w = x.size()
        x_flat = x.view(b * seq_len, c, h, w)
        e1, e2, e3 = self.encode(x_flat)
        _, c3, h3, w3 = e3.size()
        e3_seq = e3.view(b, seq_len, c3, h3, w3)
        h_t, c_t = self.convlstm.init_hidden(b, (h3, w3), x.device)
        for t in range(seq_len):
            h_t, c_t = self.convlstm(e3_seq[:, t], [h_t, c_t])
        e2_last = e2.view(b, seq_len, -1, h // 4, w // 4)[:, -1]
        e1_last = e1.view(b, seq_len, -1, h // 2, w // 2)[:, -1]
        x_last = x[:, -1]
        pred = self.decode(h_t, e2_last, e1_last, x_last)
        return pred, (h_t, c_t)

    def forward_step(self, x_single, h_prev, c_prev):
        """单帧推演: 编码1帧 → ConvLSTM一步 → 解码, 跨步保持状态"""
        e1, e2, e3 = self.encode(x_single)
        h_t, c_t = self.convlstm(e3, [h_prev, c_prev])
        pred = self.decode(h_t, e2, e1, x_single)
        return pred, h_t, c_t


# =====================================================================
# PDE 求解器: 6晶粒 CH+AC+Zener钉扎+弹性, 谱方法512²(cupy)
# =====================================================================
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


# ==================== 路径配置 (自己改这里) ====================
# 训练好的模型: 从Kaggle下载 best_model.pth 后放到你选的位置
MODEL_PATH = r"D:\vscode work\xiangchang\LSTM\model03\best_model.pth"
# 推理输出目录: 自己选位置
OUTPUT_DIR = r"D:\vscode work\xiangchang\LSTM\model03\output_AI"

# 10帧初始输入 (GT数据集 Run_00 前10帧, 与训练数据同源)
FRAME_1  = r"D:\vscode work\xiangchang\LSTM\model03\run\Run_00\frame_002000.npy"
FRAME_2  = r"D:\vscode work\xiangchang\LSTM\model03\run\Run_00\frame_004000.npy"
FRAME_3  = r"D:\vscode work\xiangchang\LSTM\model03\run\Run_00\frame_006000.npy"
FRAME_4  = r"D:\vscode work\xiangchang\LSTM\model03\run\Run_00\frame_008000.npy"
FRAME_5  = r"D:\vscode work\xiangchang\LSTM\model03\run\Run_00\frame_010000.npy"
FRAME_6  = r"D:\vscode work\xiangchang\LSTM\model03\run\Run_00\frame_012000.npy"
FRAME_7  = r"D:\vscode work\xiangchang\LSTM\model03\run\Run_00\frame_014000.npy"
FRAME_8  = r"D:\vscode work\xiangchang\LSTM\model03\run\Run_00\frame_016000.npy"
FRAME_9  = r"D:\vscode work\xiangchang\LSTM\model03\run\Run_00\frame_018000.npy"
FRAME_10 = r"D:\vscode work\xiangchang\LSTM\model03\run\Run_00\frame_020000.npy"

TOTAL_STEPS  = 80          # 80步 × 3帧 × 2000步 = 480,000；加初始20,000步和两次PDE精修各2,000步，共504,000步
# ★ 推理模式开关 ★
#   True  = 带PDE修正(当前方案): 每 PDE_CORRECT 步调用PDE精修2000步, 精度高, 全程约1~2分钟
#   False = 纯AI推演: 不实例化PDE求解器, 全程约2~4秒, 但长程误差(晶界拓扑/Lgb)会累积
# The manuscript's primary trajectory is the Hybrid run.  Use
# ``--pde-correct 0`` explicitly when generating the PDE-free control.
USE_PDE_CORRECT = True
PDE_CORRECT  = 40          # 仅 USE_PDE_CORRECT=True 时生效: 每40步修正(共2次), AI段长=训练UNROLL(40)
RUN_ID       = 0
# ===============================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", nargs=10, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--pde-correct", type=int, default=None)
    parser.add_argument("--run-id", type=int, default=None)
    args, _ = parser.parse_known_args()

    frame_paths = args.frames or [FRAME_1, FRAME_2, FRAME_3, FRAME_4,
                                   FRAME_5, FRAME_6, FRAME_7, FRAME_8,
                                   FRAME_9, FRAME_10]
    output_dir  = args.output if args.output is not None else OUTPUT_DIR
    model_path  = args.model if args.model is not None else MODEL_PATH
    total_steps = args.total_steps if args.total_steps is not None else TOTAL_STEPS
    # 推理模式: 开关关闭时 pde_correct=0 → 不调用PDE求解器(纯AI); 命令行 --pde-correct 可覆盖开关
    pde_correct = args.pde_correct if args.pde_correct is not None \
        else (PDE_CORRECT if USE_PDE_CORRECT else 0)
    run_id      = args.run_id if args.run_id is not None else RUN_ID

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode_txt = f"PDE修正 (每{pde_correct}步精修2000步)" if pde_correct > 0 else "纯AI推演 (无PDE修正)"
    print(f"设备: {device} | 推理模式: {mode_txt}")

    # 加载模型
    ck = torch.load(model_path, map_location=device, weights_only=True)
    model = ConvLSTMUNet(in_channels=9, out_channels=7).to(device)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    norm = ck['norm_params']
    c_min, c_max = norm['c_min'], norm['c_max']
    frame_step = ck.get('frame_step', 5)
    seq_len = ck.get('seq_len', 10)
    print(f"模型已加载 (epoch {ck['epoch']}, loss={ck['best_loss']:.5f}, "
          f"frame_step={frame_step}, seq_len={seq_len})")
    if frame_step != 3:
        print(f"  警告: 模型frame_step={frame_step} ≠ 3, 与默认TOTAL_STEPS/PDE_CORRECT不匹配, 请检查!")

    # 加载10帧初始输入 (原生512²)
    init_512_raw = [np.load(p).astype(np.float32) for p in frame_paths]   # 7通道原始帧(保存用)
    init_512 = [add_curvature(f) for f in init_512_raw]                   # 9通道模型输入
    c_init_mean = float(init_512[-1][0].mean())
    c_init_norm = (c_init_mean - c_min) / (c_max - c_min) * 2 - 1    # 匹配[-1,1]归一化区间
    print(f"初始帧加载完成 ({len(init_512)}帧, 512²). c均值={c_init_mean:.4f}")

    def to_norm_gpu(f_cpu):
        t = torch.from_numpy(f_cpu).to(device)
        t[0] = (t[0] - c_min) / (c_max - c_min) * 2 - 1          # c
        for ch in range(1, 7):
            t[ch] = t[ch] * 2 - 1                                   # etas:[0,1]→[-1,1]
        return t

    def denorm_gpu(t):
        out = t.clone()
        out[0] = (out[0] + 1) / 2 * (c_max - c_min) + c_min       # c
        for ch in range(1, 7):
            out[ch] = (out[ch] + 1) / 2                              # etas:[-1,1]→[0,1]
        return out

    frames = list(init_512_raw)
    sources = ["INIT"] * len(init_512)

    # PDE修正器 (可选)
    pde = None
    if pde_correct > 0:
        try:
            pde = PhaseFieldSimulator(run_id=run_id)
            pde.inject_state(init_512[-1])
            print("PDE修正器已就绪")
        except Exception as e:
            print(f"PDE修正器不可用 ({e})")

    # =====================================================================
    # 推理循环: 10帧初始化ConvLSTM → forward_step逐帧推演 → PDE间歇修正
    # =====================================================================
    t = torch.stack([to_norm_gpu(f) for f in init_512]).unsqueeze(0).to(device)

    init_steps = [int(os.path.basename(p).replace("frame_", "").replace(".npy", ""))
                  for p in frame_paths]
    ai_step_size = frame_step * 2000        # 每步AI推进 = 3帧×2000 = 6000 PDE步
    step_nums = list(init_steps)            # 每帧真实物理步号 (初始帧用实际步号)
    cur_step = init_steps[-1]               # 当前物理步号 (随AI步/修正步推进)

    pbar = tqdm(range(total_steps), desc="推演", unit="步")
    n_pde = 0
    for step in pbar:
        with torch.no_grad():                                 # 纯fp32推理(与fp32训练一致)
            if step == 0:
                pred, (h_st, c_st) = model(t)            # 首步: 完整10帧forward
            else:
                pred, h_st, c_st = model.forward_step(    # 后续: 单帧推演
                    last_norm.squeeze(1), h_st, c_st)

        pred = torch.clamp(pred, -1, 1)                      # 保护A: 数值截断
        # ★ 物理修正层: 与train03训练一致 (GB排斥/AC界面锁宽/Zener/钳制)
        pred = physics_correct(pred, c_min, c_max)
        # 质量守恒(修正后重钉均值): 与PDE每步 c += c_matrix−mean(c) 同序,
        # 修正层的GB局部压c不影响总溶质守恒
        c_pred = pred[:, :, 0]
        drift = c_init_norm - c_pred.mean().item()
        # 钳制用PDE物理范围[0.08,0.24]换算: 远宽于观测范围, 不破坏守恒
        pred[:, :, 0] = torch.clamp(c_pred + drift,
                                    (PHY_C_MIN - c_min) / (c_max - c_min) * 2 - 1,
                                    (PHY_C_MAX - c_min) / (c_max - c_min) * 2 - 1)

        pred_cpu = denorm_gpu(pred[0, 0]).cpu().numpy()

        cur_step += ai_step_size                             # AI一步推进

        if pde is not None and (step + 1) % pde_correct == 0:   # 80步: 修step39/79
            pde.inject_state(pred_cpu)
            for _ in range(2000):
                pde._single_step()
            pred_cpu = np.stack([cp.asnumpy(pde.c).astype(np.float32)] +
                                [cp.asnumpy(pde.etas[i]).astype(np.float32) for i in range(6)], axis=0)
            # 注意: 不重置LSTM状态! 训练时状态是"10帧前向"连续累积的,
            # 修正后冷启动(0状态+1帧)会让状态完全脱离训练分布, 后续AR立即发散(实测Lgb 4步暴涨5倍)
            cur_step += 2000                                   # PDE精修推进2000步(真实物理步号)
            sources.append("PDE")
            n_pde += 1
        else:
            sources.append("AI")
        # 下一帧输入 = 7通道预测 + 2曲率通道(从预测etas导出), 与训练一致
        last_norm = to_norm_gpu(add_curvature(pred_cpu)).unsqueeze(0).unsqueeze(0)

        # 时间步由初始帧、AI推进和PDE精修严格累加，不再截断为500000步
        if cur_step > 504000:
            raise RuntimeError(f"物理步数超过预期终点504000：{cur_step}")
        step_nums.append(cur_step)
        frames.append(pred_cpu)
        pbar.set_postfix({"c_mean": f"{pred_cpu[0].mean():.4f}", "step": cur_step})

    pbar.close()

    os.makedirs(output_dir, exist_ok=True)
    # 清理旧帧: 多次运行的残留帧会污染对比(实测149帧=多轮run混用)
    old_frames = glob.glob(os.path.join(output_dir, "frame_*.npy"))
    for f in old_frames:
        os.remove(f)
    if old_frames:
        print(f"已清理 {len(old_frames)} 个旧帧文件")
    for i, f in enumerate(frames):
        np.save(os.path.join(output_dir, f"frame_{step_nums[i]:06d}.npy"),
                f.astype(np.float32))

    n_ai = sources.count("AI")
    print(f"\n完成: {len(frames)}帧 (原生512²) → {output_dir}")
    print(f"  INIT:{len(init_512)}  AI:{n_ai}  PDE:{n_pde}")
    # 双口径加速比: 替换口径(AI替代步/PDE修正步) vs 含初始暖机(全程/PDE修正步)
    if n_pde > 0:
        speedup_pure = total_steps * ai_step_size / (n_pde * 2000)
        speedup_all  = (total_steps * ai_step_size + init_steps[-1]) / (n_pde * 2000)
        print(f"  加速比(替换口径):   ~{speedup_pure:.0f}×  (AI替代{total_steps*ai_step_size}步/PDE修正{n_pde*2000}步)")
        print(f"  加速比(含初始暖机): ~{speedup_all:.0f}×  (全程{total_steps*ai_step_size+init_steps[-1]}步/PDE修正步)")
    else:
        print("  无PDE修正 (纯AI)")


if __name__ == "__main__":
    main()
