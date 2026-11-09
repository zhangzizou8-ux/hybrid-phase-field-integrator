"""将七物理场NPY转换成与phasefield.py/save_plot一致的PNG。

直接运行：python 03_npy_to_png.py
默认读取脚本旁的UNET_npy，输出到UNET_png，文件命名gb_XXXXXX.png。
PNG处理完成后自动生成同目录的evolution.gif，按本次选定的物理步数排序。
默认按真实时间间隔播放：2000细步=100ms；--gif-uniform改为每帧固定时长。
--gif-duration可调整上述100ms基准，--no-gif则仅生成PNG。
GIF默认约20帧/秒，使用相邻PNG的线性图像渐变补帧，使播放更连续。
补帧仅用于显示，不代表真实中间物理状态，也不修改NPY或原PNG。
--gif-fps 0可关闭补帧，恢复只播放原始采样图像的模式。
单帧示例：python 03_npy_to_png.py --input UNET_npy/frame_504000.npy
选定时刻：python 03_npy_to_png.py --steps 20000 262000 504000
FNO图片与动画：python 03_npy_to_png.py --input FNO_npy --output FNO_PNG
已有PNG会直接复用，仍会重新生成FNO_PNG/evolution.gif；不要加--no-gif。

输入必须为物理量而非归一化数据，形状(7,512,512)，通道为c,eta1,...,eta6。
沿用参考函数的jet色图、浓度范围[0.08,0.24]、origin='lower'、灰度晶界
叠加(phi>0.05, alpha=0.6)、100 nm比例尺、6x6英寸和150 dpi。
参考注释提到裁剪，但save_plot实际使用完整512x512，本程序同样不裁剪。
仅依赖NumPy、Matplotlib、Pillow；不导入或执行phasefield.py，不运行FOM/网络。
不修改NPY；已有PNG默认跳过，--overwrite才允许覆盖PNG。GIF每次重新生成。
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PATH = SCRIPT_DIR / 'UNET_npy'
OUTPUT_DIR = SCRIPT_DIR / 'UNET_png'
DPI = 150


def local_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else SCRIPT_DIR / path).resolve()


def frame_step(path):
    match = re.fullmatch(r'frame_(\d+)\.npy', path.name)
    if match is None:
        raise ValueError(f'文件名应为frame_物理步数.npy：{path}')
    return int(match.group(1))


def collect_frames(source, selected_steps):
    if source.is_file():
        paths = [source]
    elif source.is_dir():
        paths = list(source.glob('frame_*.npy'))
    else:
        raise FileNotFoundError(f'输入路径不存在：{source}')
    if not paths:
        raise FileNotFoundError(f'没有找到frame_*.npy：{source}')
    indexed = {}
    for path in paths:
        step = frame_step(path)
        if step in indexed:
            raise ValueError(f'同一物理步数存在两个文件：{indexed[step]} 和 {path}')
        indexed[step] = path
    if selected_steps is not None:
        missing = sorted(set(selected_steps) - set(indexed))
        if missing:
            raise FileNotFoundError(f'输入中没有这些物理时刻：{missing}')
        indexed = {step: indexed[step] for step in set(selected_steps)}
    return [(step, indexed[step]) for step in sorted(indexed)]


def read_fields(path):
    array = np.load(path, mmap_mode='r', allow_pickle=False)
    if array.shape != (7, 512, 512) or array.dtype.kind != 'f':
        raise ValueError(f'{path}：要求浮点(7,512,512)，实际{array.shape}/{array.dtype}')
    if not np.isfinite(array).all():
        raise ValueError(f'{path}：含NaN或Inf')
    if array[0].min() < 0.08-1e-4 or array[0].max() > 0.24+1e-4:
        raise ValueError(f'{path}：c不在[0.08,0.24]附近，请确认输入是物理场而非归一化数据')
    if array[1:].min() < -1e-4 or array[1:].max() > 1.0+1e-4:
        raise ValueError(f'{path}：eta不在[0,1]附近，请检查通道顺序和归一化')
    c = np.asarray(array[0], dtype=np.float32)
    # 原FOM内部用float64；这里只计算相同的绘图描述符，不改变输入数据。
    etas = np.asarray(array[1:7], dtype=np.float64)
    phi = np.clip((1.0 - np.sum(etas**2, axis=0))*2.0, 0.0, 1.0)
    return c, phi


def save_plot(c_np, phi_np, output, dpi):
    """对应phasefield.py的save_plot；保留其完整灰度叠加方式。"""
    fig, ax = plt.subplots(figsize=(6, 6))
    try:
        ax.imshow(c_np, cmap='jet', origin='lower', vmin=0.08, vmax=0.24)
        gb = np.where(phi_np > 0.05, 1.0, 0.0)
        ax.imshow(gb, cmap='gray', origin='lower', alpha=0.6, vmin=0, vmax=1)
        ax.axis('off')
        # 网格间距2 nm，50像素对应100 nm，与参考代码一致。
        ax.plot([440, 490], [20, 20], color='white', lw=4)
        ax.text(465, 28, '100 nm', color='white', fontsize=10,
                ha='center', va='bottom', weight='bold')
        fig.savefig(output, dpi=dpi, bbox_inches='tight', pad_inches=0)
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', default=str(INPUT_PATH), help='NPY目录或单个frame_*.npy')
    parser.add_argument('--output', default=str(OUTPUT_DIR), help='PNG输出目录')
    parser.add_argument('--steps', nargs='+', type=int, help='只转换指定物理步数')
    parser.add_argument('--dpi', type=int, default=DPI)
    parser.add_argument('--overwrite', action='store_true', help='允许覆盖已有PNG；不会修改NPY')
    parser.add_argument('--no-gif', action='store_true', help='只转换PNG，不生成GIF')
    parser.add_argument('--gif-duration', type=int, default=100,
                        help='GIF每2000物理细步的播放毫秒数，默认100；固定时长模式则为每帧时长')
    parser.add_argument('--gif-uniform', action='store_true', help='每帧固定时长，不按物理时间间隔调整')
    parser.add_argument('--gif-fps', type=int, default=20,
                        help='GIF视觉渐变目标帧率，默认20；0=不补帧，仅播放原始采样图像')
    args = parser.parse_args()
    if args.dpi <= 0 or (args.steps is not None and any(t < 0 for t in args.steps)):
        parser.error('dpi必须为正，物理步数不能为负')
    if args.gif_duration < 10:
        parser.error('GIF基准时长至少10毫秒（GIF时间精度为10毫秒）')
    if not 0 <= args.gif_fps <= 100:
        parser.error('gif-fps须为0~100；0表示关闭视觉补帧')
    source, output = local_path(args.input), local_path(args.output)
    frames = collect_frames(source, args.steps)
    output.mkdir(parents=True, exist_ok=True)
    print(f'输入：{source}\n输出：{output}\n共{len(frames)}帧；采用phasefield.py绘图样式。', flush=True)
    written = skipped = 0
    for index, (step, path) in enumerate(frames, 1):
        target = output / f'gb_{step:06d}.png'
        if target.exists() and not args.overwrite:
            skipped += 1
            print(f'[{index}/{len(frames)}] 已存在，跳过：{target.name}', flush=True)
            continue
        c, phi = read_fields(path)
        save_plot(c, phi, target, args.dpi)
        written += 1
        print(f'[{index}/{len(frames)}] {path.name} -> {target.name}', flush=True)
    print(f'完成：新增{written}张，跳过{skipped}张；目录：{output}', flush=True)
    # PNG生成结束后，再将同一批按物理时间排序的图像合成为GIF。
    if not args.no_gif:
        save_gif(frames, output, args.gif_duration, args.gif_uniform, args.gif_fps)


def save_gif(frames, output, duration_ms=100, uniform=False, fps=20):
    """共享调色板、无限循环；可做RGB渐变补帧，仅用于显示而非物理预测。"""
    images = []
    delays = []
    palette = None
    current = following = None
    size = None
    temporary = output / 'evolution.gif.tmp'
    target = output / 'evolution.gif'
    times = [step for step, _ in frames]
    gaps = [b-a for a, b in zip(times, times[1:])] + [2000]
    durations = [max(10, int(round(duration_ms*(1 if uniform else gap/2000)/10))*10)
                 for gap in gaps]
    if any(value > 655350 for value in durations):
        raise ValueError('GIF单帧时间超过格式上限，请减小--gif-duration或使用--gif-uniform')
    try:
        with Image.open(output / f'gb_{times[0]:06d}.png') as source:
            current = source.convert('RGB')
        size = current.size
        palette = current.quantize(colors=256, dither=Image.Dither.NONE)
        for index, step in enumerate(times):
            if index+1 < len(times):
                with Image.open(output / f'gb_{times[index+1]:06d}.png') as source:
                    following = source.convert('RGB')
                if following.size != size:
                    raise ValueError('PNG尺寸不一致，请使用同一个dpi并加--overwrite重新生成')
            # 用10ms整数单位划分时长，使补帧前后的物理播放时间完全相同。
            units = durations[index]//10
            parts = min(units, max(1, round(durations[index]*fps/1000))) if fps and following is not None else 1
            boundaries = [round(j*units/parts) for j in range(parts+1)]
            for j in range(parts):
                visual = (current.copy() if j == 0 else
                          Image.blend(current, following, boundaries[j]/units))
                try:
                    images.append(visual.quantize(palette=palette, dither=Image.Dither.NONE))
                finally:
                    visual.close()
                delays.append((boundaries[j+1]-boundaries[j])*10)
            current.close()
            current, following = following, None
            if index == 0 or (index+1) % 20 == 0 or index+1 == len(times):
                print(f'[GIF {index+1}/{len(times)}] gb_{step:06d}.png；已准备{len(images)}个动画帧', flush=True)
        if sum(delays) != sum(durations):
            raise RuntimeError('GIF补帧前后总时长不一致')
        comment = (b'Visualization-only RGB cross-fade between saved states; not additional physical predictions.'
                   if fps else b'Original sampled states; no visual interpolation.')
        images[0].save(temporary, format='GIF', save_all=True, append_images=images[1:],
                       duration=delays, loop=0, disposal=2, optimize=False, comment=comment)
        os.replace(temporary, target)
        mode = '每帧固定时长' if uniform else '按真实物理时间间隔'
        print(f'GIF完成：{target}；{len(times)}个原始时刻 -> {len(images)}个动画帧，'
              f'循环播放，{mode}，单次循环约{sum(delays)/1000:.2f}秒。', flush=True)
        if fps:
            print('说明：新增帧为视觉渐变，不是新增的真实物理预测。', flush=True)
    finally:
        for item in images:
            item.close()
        if palette is not None:
            palette.close()
        for item in (current, following):
            if item is not None:
                item.close()
        if temporary.exists():
            temporary.unlink()


if __name__ == '__main__':
    main()
