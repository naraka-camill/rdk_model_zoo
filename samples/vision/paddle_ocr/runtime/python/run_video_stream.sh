#!/bin/bash

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

# PaddleOCR 实时视频流演示一键运行脚本
#
# 用法:
#   ./run_video_stream.sh                    # 默认 USB 摄像头
#   ./run_video_stream.sh --video-source 0   # 指定摄像头设备号
#   ./run_video_stream.sh --video-source /path/to/video.mp4 --video-type file
#   ./run_video_stream.sh --video-type mipi  # RDK 开发板 MIPI 摄像头
#
# 完整参数见: python3 video_stream.py --help

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# --- 依赖安装 ---
MISSING=""
python3 -c "import numpy"       2>/dev/null || MISSING="$MISSING numpy"
python3 -c "import cv2"         2>/dev/null || MISSING="$MISSING opencv-python"
python3 -c "import pyclipper"   2>/dev/null || MISSING="$MISSING pyclipper"
python3 -c "import PIL"         2>/dev/null || MISSING="$MISSING Pillow"

if [ -n "$MISSING" ]; then
    echo "[Setup] Installing missing packages:$MISSING"
    pip install $MISSING
fi

# --- SOC 探测与模型路径 ---
SOC_RAW=$(cat /sys/class/boardinfo/soc_name 2>/dev/null | tr 'A-Z' 'a-z' | tr -d '()' | xargs)
SOC="${SOC_RAW:-s100}"
case "$SOC" in
  s600) MODEL_SOC="s600" ;;
  *)    MODEL_SOC="s100" ;;
esac
echo "[Setup] SOC: $SOC, Model variant: rdk_${MODEL_SOC}"

MODEL_BASE_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_${MODEL_SOC}/paddle_ocr"

DET_MODEL_PATH="/opt/hobot/model/${MODEL_SOC}/basic/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm"
REC_MODEL_PATH="/opt/hobot/model/${MODEL_SOC}/basic/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm"

# --- 下载模型（如缺失） ---
if command -v wget &>/dev/null; then
  DL_CMD="wget -q -O"
elif command -v curl &>/dev/null; then
  DL_CMD="curl -fL -o"
else
  echo "ERROR: neither wget nor curl found" >&2
  exit 1
fi

echo "[Setup] Det model  : $DET_MODEL_PATH"
if [[ ! -f "$DET_MODEL_PATH" ]]; then
  echo "[Setup] Detection model not found, downloading..."
  mkdir -p "$(dirname "$DET_MODEL_PATH")"
  $DL_CMD "$DET_MODEL_PATH" \
    "${MODEL_BASE_URL}/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm"
  echo "[Setup] Detection model downloaded."
else
  echo "[Setup] Detection model already exists."
fi

echo "[Setup] Rec model  : $REC_MODEL_PATH"
if [[ ! -f "$REC_MODEL_PATH" ]]; then
  echo "[Setup] Recognition model not found, downloading..."
  mkdir -p "$(dirname "$REC_MODEL_PATH")"
  $DL_CMD "$REC_MODEL_PATH" \
    "${MODEL_BASE_URL}/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm"
  echo "[Setup] Recognition model downloaded."
else
  echo "[Setup] Recognition model already exists."
fi

# --- 启动视频流 ---
echo "[Start] PaddleOCR real-time video stream demo"
python3 video_stream.py \
    --det-model-path "$DET_MODEL_PATH" \
    --rec-model-path "$REC_MODEL_PATH" \
    "$@"
