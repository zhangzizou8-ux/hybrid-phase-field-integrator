"""Fig. 9a: no-physics-operator 推理脚本。

独立复用 run03.py 的模型、PDE模拟器和数据处理定义，但不修改 run03.py。
仍使用9通道输入；每个AI步后不调用 physics_correct()，也不执行浓度均值重锚定；
默认保留每40个AI步、2000个PDE步的间歇PDE锚定。
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
import run03  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Fig. 9a no-physics-operator inference")
    parser.add_argument("--model", default=str(PROJECT_ROOT / "best_model_no_physics_operator.pth"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "output_F9a_no_physics"))
    parser.add_argument("--total-steps", type=int, default=run03.TOTAL_STEPS)
    parser.add_argument("--pde-correct", type=int, default=40)
    parser.add_argument("--run-id", type=int, default=run03.RUN_ID)
    parser.add_argument("--frames", nargs=10, default=None,
                        help="按时间顺序提供Run00的10个初始frame_*.npy")
    args = parser.parse_args()

    frame_paths = args.frames or [
        run03.FRAME_1, run03.FRAME_2, run03.FRAME_3, run03.FRAME_4, run03.FRAME_5,
        run03.FRAME_6, run03.FRAME_7, run03.FRAME_8, run03.FRAME_9, run03.FRAME_10,
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.model, map_location=device, weights_only=True)
    model = run03.ConvLSTMUNet(in_channels=9, out_channels=7).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    norm = ck["norm_params"]
    c_min, c_max = norm["c_min"], norm["c_max"]
    frame_step = ck.get("frame_step", 3)
    if ck.get("seq_len", 10) != 10:
        raise ValueError("Fig. 9要求checkpoint使用10帧输入")

    init_raw = [np.load(p).astype(np.float32) for p in frame_paths]
    init_inputs = [run03.add_curvature(f) for f in init_raw]

    def to_norm_gpu(frame):
        t = torch.from_numpy(frame).to(device)
        t[0] = (t[0] - c_min) / (c_max - c_min) * 2 - 1
        for ch in range(1, 7):
            t[ch] = t[ch] * 2 - 1
        return t

    def denorm_gpu(t):
        out = t.clone()
        out[0] = (out[0] + 1) / 2 * (c_max - c_min) + c_min
        for ch in range(1, 7):
            out[ch] = (out[ch] + 1) / 2
        return out

    pde = None
    if args.pde_correct > 0:
        pde = run03.PhaseFieldSimulator(run_id=args.run_id)
        pde.inject_state(init_inputs[-1])

    history = torch.stack([to_norm_gpu(f) for f in init_inputs]).unsqueeze(0)
    last_norm = None
    init_steps = [int(Path(p).stem.replace("frame_", "")) for p in frame_paths]
    step_nums, frames = list(init_steps), list(init_raw)
    sources = ["INIT"] * len(init_raw)
    ai_step_size, cur_step = frame_step * 2000, init_steps[-1]
    n_pde = 0

    for step in tqdm(range(args.total_steps), desc="F9a no-physics", unit="AI-step"):
        with torch.no_grad():
            if step == 0:
                pred, (h_st, c_st) = model(history)
            else:
                pred, h_st, c_st = model.forward_step(last_norm.squeeze(1), h_st, c_st)
        # 仅保留通用数值截断；不调用局部物理算子，不重锚定浓度均值。
        pred_cpu = denorm_gpu(torch.clamp(pred, -1, 1)[0, 0]).cpu().numpy()
        cur_step += ai_step_size

        if pde is not None and (step + 1) % args.pde_correct == 0:
            pde.inject_state(pred_cpu)
            for _ in range(2000):
                pde._single_step()
            pred_cpu = np.stack(
                [run03.cp.asnumpy(pde.c).astype(np.float32)]
                + [run03.cp.asnumpy(pde.etas[i]).astype(np.float32) for i in range(6)], axis=0
            )
            cur_step += 2000
            sources.append("PDE")
            n_pde += 1
        else:
            sources.append("AI")

        last_norm = to_norm_gpu(run03.add_curvature(pred_cpu)).unsqueeze(0).unsqueeze(0)
        if cur_step > 504000:
            raise RuntimeError(f"物理步数超过预期终点504000: {cur_step}")
        step_nums.append(cur_step)
        frames.append(pred_cpu)

    os.makedirs(args.output, exist_ok=True)
    for old in glob.glob(os.path.join(args.output, "frame_*.npy")):
        os.remove(old)
    for n, frame in zip(step_nums, frames):
        np.save(os.path.join(args.output, f"frame_{n:06d}.npy"), frame.astype(np.float32))
    print(f"完成: {len(frames)}帧 → {args.output}; AI={sources.count('AI')}, PDE={n_pde}")


if __name__ == "__main__":
    main()
