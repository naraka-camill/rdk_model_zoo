# Copyright (c) 2025 D-Robotics Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""通过 arecord (ALSA) 实现麦克风实时语音识别。

本脚本通过 ``arecord`` 命令实时采集麦克风音频，对固定长度的音频块
执行 ASR 推理，并实时打印识别出的文字。

工作流程：
    1) 解析命令行参数。
    2) 加载词表 JSON -> 构建 id2token 映射。
    3) 创建 ASRConfig 并初始化 ASR 运行时封装。
    4) 以子进程方式启动 ``arecord``，通过管道读取原始 PCM 数据。
    5) 在环形缓冲区中累积音频采样点；当积满一个完整块时，
       执行预处理 -> 推理 -> 解码 -> 打印文字。
    6) 持续运行直到用户按下 Ctrl+C。

无需安装任何第三方 Python 音频库——仅需系统安装 ``arecord``
（属于 ``alsa-utils`` 包）。

示例：
    python realtime_asr.py \\
        --vocab-file ../../test_data/vocab.json

注意事项：
    - 本脚本适用于搭载 BPU 量化 ASR 模型 (.hbm) 的 RDK S100/S600 平台。
    - 麦克风采样率设置为 ``--new-rate``（默认 16000 Hz）。
    - 默认使用 ``hw:0,0`` 作为 ALSA 设备，立体声输入（模型内部自动转为单声道）。
    - 按 Ctrl+C 停止录音。
"""

import argparse
import json
import os
import struct
import subprocess
import sys
import threading

import numpy as np

# 将项目根目录加入 sys.path，以便导入工具模块
sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.inspect as inspect
import utils.py_utils.file_io as file_io
import utils.py_utils.nn_math as nn_math

from asr import ASR, ASRConfig


def main() -> None:
    """运行实时麦克风语音识别。

    从默认麦克风采集音频块，依次经过 ASR 流水线处理，
    并将逐步识别的文字打印到控制台。
    """
    soc = inspect.get_soc_name().lower()
    model_download_url = (f"https://archive.d-robotics.cc/downloads/rdk_model_zoo/"
                          f"rdk_{soc}/asr/asr.hbm")

    parser = argparse.ArgumentParser(
        description="通过麦克风实时语音识别。")

    parser.add_argument('--model-path', type=str,
                        default=f'/opt/hobot/model/{soc}/basic/asr.hbm',
                        help='BPU 量化模型文件路径（*.hbm）')
    parser.add_argument('--priority', type=int, default=0,
                        help='推理优先级（0~255），0 最低，255 最高')
    parser.add_argument('--bpu-cores', nargs='+', type=int, default=[0],
                        help='BPU 核心索引列表，例如 --bpu-cores 0 1')
    parser.add_argument('--vocab-file', type=str, default='../../test_data/vocab.json',
                        help='词表 JSON 文件路径（token -> id 映射）')
    parser.add_argument('--audio-maxlen', type=int, default=30000,
                        help='每次推理的音频采样点数（在 new_rate Hz 下）')
    parser.add_argument('--new-rate', type=int, default=16000,
                        help='目标采样率（Hz）')
    parser.add_argument('--alsa-device', type=str, default='hw:0,0',
                        help='ALSA 设备名，例如 hw:0,0 或 plughw:1,0')
    parser.add_argument('--channels', type=int, default=2,
                        help='麦克风通道数（默认 2，自动平均为单声道）')
    parser.add_argument('--continuous', action='store_true', default=False,
                        help='持续模式：仅打印新文字，不重复输出')

    opt = parser.parse_args()

    # 若模型文件不存在则自动下载
    file_io.download_model_if_needed(opt.model_path, model_download_url)

    # 加载词表 JSON
    with open(opt.vocab_file, 'r', encoding='utf-8') as f:
        token2id = json.load(f)
    id2token = {v: k for k, v in token2id.items()}

    # 初始化 ASR 模型
    config = ASRConfig(
        model_path=opt.model_path,
        audio_maxlen=opt.audio_maxlen,
        new_rate=opt.new_rate,
    )
    model = ASR(config)
    model.set_scheduling_params(priority=opt.priority, bpu_cores=opt.bpu_cores)
    inspect.print_model_info(model.model)

    byte_depth = 2  # S16_LE 每个采样点 2 字节
    frame_size = byte_depth * opt.channels  # 每帧（多声道）的字节数
    bytes_per_chunk = opt.audio_maxlen * frame_size  # 每次读取的目标字节数

    print(f"\n{'='*60}")
    print(f"实时语音识别（arecord）")
    print(f"  设备    : {opt.alsa_device}")
    print(f"  采样率  : {opt.new_rate} Hz")
    print(f"  声道数  : {opt.channels}")
    print(f"  音频块  : {opt.audio_maxlen} 采样点（{opt.audio_maxlen / opt.new_rate:.1f} 秒）")
    print(f"按 Ctrl+C 停止。")
    print(f"{'='*60}\n")

    # 构建 arecord 命令——持续向 stdout 输出原始 PCM 数据流
    arecord_cmd = [
        'arecord',
        '-D', opt.alsa_device,
        '-c', str(opt.channels),
        '-r', str(opt.new_rate),
        '-f', 'S16_LE',
        '-t', 'raw',          # 原始 PCM（不含 WAV 文件头）
    ]

    # 共享状态：环形缓冲区、去重缓存、停止事件
    ring_buffer = np.zeros(opt.audio_maxlen, dtype=np.float32)
    buffer_pos = 0
    last_text = ""
    accumulated_text = ""
    stop_event = threading.Event()

    def process_chunk(chunk: np.ndarray) -> str:
        """对单个音频块执行预处理 -> 推理 -> 解码。"""
        data = nn_math.zscore_normalize_lastdim(chunk)
        tensor = data[None, :].astype(np.float32)
        input_tensor = {model.model_name: {model.input_names[0]: tensor}}
        outputs = model.forward(input_tensor)
        return model.post_process(outputs, id2token)

    def feed_audio(raw_bytes: bytes) -> None:
        """将原始 PCM 字节解析为 float32 单声道数据并送入环形缓冲区。"""
        nonlocal buffer_pos, last_text, accumulated_text

        # 解包 S16_LE 交织采样点
        num_frames = len(raw_bytes) // frame_size
        fmt = f'<{num_frames * opt.channels}h'
        samples = struct.unpack(fmt, raw_bytes[:num_frames * frame_size])

        # 转为 float32 numpy 数组，交织形状: (num_frames, channels)
        arr = np.array(samples, dtype=np.float32).reshape(-1, opt.channels)

        # 多声道平均 -> 单声道
        mono = np.mean(arr, axis=1)

        # 填充环形缓冲区
        remaining = len(mono)
        src = 0
        while remaining > 0:
            space = opt.audio_maxlen - buffer_pos
            to_copy = min(space, remaining)
            ring_buffer[buffer_pos:buffer_pos + to_copy] = mono[src:src + to_copy]
            buffer_pos += to_copy
            src += to_copy
            remaining -= to_copy

            if buffer_pos >= opt.audio_maxlen:
                text = process_chunk(ring_buffer.copy())
                buffer_pos = 0

                if opt.continuous:
                    new_part = text
                    if new_part and new_part != last_text:
                        print(new_part, end='', flush=True)
                        last_text = new_part
                        accumulated_text += new_part
                else:
                    if text.strip():
                        print(f"[chunk] {text}", flush=True)

    # --------------------------------------------------------------
    # 启动 arecord 并通过管道读取数据
    # --------------------------------------------------------------
    try:
        proc = subprocess.Popen(
            arecord_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
    except FileNotFoundError:
        print("错误：未找到 'arecord' 命令。请安装 alsa-utils：")
        print("    sudo apt install alsa-utils")
        sys.exit(1)

    print("正在聆听 ...（按 Ctrl+C 停止）")

    try:
        # 按块读取原始 PCM，批量处理以获得更高效率
        while not stop_event.is_set():
            raw = proc.stdout.read(bytes_per_chunk)
            if not raw:
                break
            feed_audio(raw)
    except KeyboardInterrupt:
        print("\n\n已由用户停止。")
    finally:
        stop_event.set()
        proc.terminate()
        proc.wait(timeout=2)

    # 连续模式下打印最终完整识别结果
    if opt.continuous and accumulated_text:
        print(f"\n\n{'='*60}")
        print("完整识别结果：")
        print(f"{'='*60}")
        print(accumulated_text)

    print("完成。")


if __name__ == "__main__":
    main()
