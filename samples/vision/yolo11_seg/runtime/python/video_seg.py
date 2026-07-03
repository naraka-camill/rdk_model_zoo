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

"""
YOLO11-Seg 实例分割视频流实时推理脚本

本脚本支持从摄像头或视频文件读取视频流，对每一帧进行 BPU 推理，
实时绘制检测框、实例掩码和轮廓，并在窗口中显示结果。

工作流程:
    1) 解析命令行参数（视频来源、模型路径、阈值等）。
    2) 若模型文件缺失，自动根据 SoC 类型下载。
    3) 创建 YoloV11SegConfig 并初始化 YoloV11Seg 运行时封装。
    4) 逐帧读取 -> 预处理 -> BPU 推理 -> 后处理（检测框 + 掩码）。
    5) 绘制检测框、掩码、轮廓，并叠加 FPS 信息。
    6) 实时显示结果，并可选保存输出视频。

用法示例:
    # 使用默认摄像头（设备 ID 0）
    python video_seg.py

    # 使用视频文件
    python video_seg.py --video-path ../../test_data/test_video.mp4

    # 保存输出视频
    python video_seg.py --video-path ../../test_data/test_video.mp4 --save-video output_result.mp4
"""

import os
import cv2
import sys
import time
import argparse
import numpy as np

sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.file_io as file_io
import utils.py_utils.inspect as inspect
import utils.py_utils.visualize as visualize
from yolo11seg import YoloV11Seg, YoloV11SegConfig


# 显示窗口的固定尺寸（按模型输入分辨率显示）
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 640


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        解析后的参数对象。
    """
    soc = inspect.get_soc_name().lower()
    model_suffix = "nashp" if soc == "s600" else "nashe"

    parser = argparse.ArgumentParser(
        description="YOLO11-Seg 视频流实时实例分割")

    parser.add_argument('--model-path', type=str,
                        default=f'/opt/hobot/model/{soc}/basic/yolo11n_seg_{model_suffix}_640x640_nv12.hbm',
                        help='BPU 量化模型 (*.hbm) 路径')
    parser.add_argument('--priority', type=int, default=0,
                        help='模型优先级 (0~255)，0 最低，255 最高')
    parser.add_argument('--bpu-cores', nargs='+', type=int, default=[0],
                        help='BPU 核心索引列表，例如 --bpu-cores 0 1')
    parser.add_argument('--video-path', type=str, default=None,
                        help='视频文件路径；若不指定则使用默认摄像头')
    parser.add_argument('--camera-id', type=int, default=0,
                        help='摄像头设备 ID（仅 video-path 为空时生效）')
    parser.add_argument('--label-file', type=str,
                        default='../../test_data/coco_classes.names',
                        help='COCO 类别标签文件路径')
    parser.add_argument('--save-video', type=str, default=None,
                        help='输出视频保存路径（可选），例如 result.mp4')
    parser.add_argument('--nms-thres', type=float, default=0.7,
                        help='NMS 的 IoU 阈值')
    parser.add_argument('--score-thres', type=float, default=0.25,
                        help='检测置信度阈值')
    parser.add_argument('--no-morph', action='store_true',
                        help='禁用掩码后处理的形态学开运算')
    parser.add_argument('--no-contour', action='store_true',
                        help='禁用结果图上绘制轮廓线')
    parser.add_argument('--display-fps', type=int, default=30,
                        help='目标显示帧率（fps），默认 30')
    parser.add_argument('--loop', action='store_true',
                        help='视频文件播放结束后循环播放')

    return parser.parse_args()


def init_video_capture(video_path: str, camera_id: int) -> cv2.VideoCapture:
    """初始化视频捕获对象。

    优先打开视频文件；若未指定视频文件，则打开摄像头。

    Args:
        video_path: 视频文件路径。若为 None 则使用摄像头。
        camera_id: 摄像头设备 ID。

    Returns:
        配置好的 VideoCapture 对象。

    Raises:
        RuntimeError: 无法打开视频源时抛出。
    """
    if video_path is not None:
        cap = cv2.VideoCapture(video_path)
        source_desc = f"视频文件: {video_path}"
    else:
        cap = cv2.VideoCapture(camera_id)
        source_desc = f"摄像头 (ID: {camera_id})"

    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频源: {source_desc}")

    # 获取视频原始信息
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if video_path else -1

    print(f"[视频源] {source_desc}")
    print(f"[分辨率] {orig_w} x {orig_h}")
    print(f"[帧率  ] {orig_fps:.2f} fps")
    if total_frames > 0:
        print(f"[总帧数] {total_frames}")

    return cap


def init_video_writer(save_path: str, fps: float) -> cv2.VideoWriter | None:
    """初始化视频写入器。

    Args:
        save_path: 输出视频保存路径。若为 None 则不保存。
        fps: 输出视频的帧率。

    Returns:
        配置好的 VideoWriter 对象，若不保存则返回 None。
    """
    if save_path is None:
        return None

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(save_path, fourcc, fps,
                             (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    if not writer.isOpened():
        print(f"[警告] 无法创建输出视频: {save_path}")
        return None

    print(f"[保存  ] 输出视频将保存至: {save_path}")
    return writer


def run_inference_loop(
    yolo11_seg: YoloV11Seg,
    cap: cv2.VideoCapture,
    coco_names: list,
    video_writer: cv2.VideoWriter | None,
    no_contour: bool,
    target_fps: int,
    loop: bool,
) -> None:
    """执行实时推理主循环。

    逐帧读取 -> 预处理 -> 推理 -> 后处理 -> 可视化 -> 显示/保存。

    Args:
        yolo11_seg: YOLO11-Seg 模型实例。
        cap: 视频捕获对象。
        coco_names: COCO 类别名称列表。
        video_writer: 视频写入器（可选）。
        no_contour: 是否跳过轮廓绘制。
        target_fps: 目标显示帧率。
        loop: 是否循环播放视频文件。

    Returns:
        None
    """
    window_name = "YOLO11-Seg 实时实例分割"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, DISPLAY_WIDTH, DISPLAY_HEIGHT)

    # FPS 计算相关变量
    frame_count = 0
    fps_avg = 0.0
    alpha = 0.1  # 滑动平均平滑系数

    # 帧间隔时间（秒）
    frame_interval = 1.0 / target_fps

    print("\n[开始] 实时推理中，按 'q' 或 ESC 退出...\n")

    while True:
        loop_start = time.time()

        # ----- 读取一帧 -----
        ret, frame = cap.read()
        if not ret:
            if loop and video_writer is None:
                # 循环播放：重置到第一帧
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                print("[结束] 视频读取完毕。")
                break

        frame_h, frame_w = frame.shape[:2]

        # ----- 预处理 -> BPU 推理 -> 后处理 -----
        t_pre_start = time.perf_counter()

        input_array = yolo11_seg.pre_process(frame)
        outputs = yolo11_seg.forward(input_array)
        boxes, scores, cls_ids, masks = yolo11_seg.post_process(
            outputs, frame_w, frame_h)

        t_infer_end = time.perf_counter()
        infer_ms = (t_infer_end - t_pre_start) * 1000

        # ----- 可视化 -----
        # 将帧缩放到 640x640 用于统一显示（不改变推理精度，仅用于显示）
        display_img = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

        # 由于显示的图像被缩放，需要将检测框坐标缩放到显示尺寸
        scale_x = DISPLAY_WIDTH / frame_w
        scale_y = DISPLAY_HEIGHT / frame_h
        display_boxes = boxes.copy()
        display_boxes[:, 0] = boxes[:, 0] * scale_x  # x1
        display_boxes[:, 1] = boxes[:, 1] * scale_y  # y1
        display_boxes[:, 2] = boxes[:, 2] * scale_x  # x2
        display_boxes[:, 3] = boxes[:, 3] * scale_y  # y2

        # 将掩码缩放到显示尺寸（掩码在原图坐标下，需要与缩放后的显示框匹配）
        # 注意：必须与 draw_masks 中 crop 区域的方式一致（int 截断），否则尺寸不匹配
        display_masks = []
        for box_disp, mask in zip(display_boxes, masks):
            if mask.size == 0:
                display_masks.append(mask)
                continue
            x1, y1, x2, y2 = int(box_disp[0]), int(box_disp[1]), int(box_disp[2]), int(box_disp[3])
            new_w = x2 - x1
            new_h = y2 - y1
            if new_w < 1 or new_h < 1:
                display_masks.append(mask)
                continue
            display_masks.append(cv2.resize(mask.astype(np.uint8),
                                 (new_w, new_h), interpolation=cv2.INTER_NEAREST))

        # 绘制检测框
        visualize.draw_boxes(display_img, display_boxes, cls_ids, scores,
                             coco_names, visualize.rdk_colors)

        # 绘制实例掩码
        visualize.draw_masks(display_img, display_boxes, display_masks, cls_ids,
                             visualize.rdk_colors, alpha=0.4)

        # 绘制轮廓
        if not no_contour:
            visualize.draw_contours(display_img, display_boxes, display_masks, cls_ids,
                                    visualize.rdk_colors, thickness=1)

        # ----- 计算并显示 FPS -----
        frame_count += 1
        if frame_count == 1:
            fps_avg = 1.0 / max(infer_ms / 1000, 1e-6)
        else:
            instant_fps = 1.0 / max((time.perf_counter() - loop_start), 1e-6)
            fps_avg = (1 - alpha) * fps_avg + alpha * instant_fps

        fps_text = f"FPS: {fps_avg:.1f}  |  Infer: {infer_ms:.1f}ms  |  Det: {len(boxes)}"
        cv2.putText(display_img, fps_text, (10, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.6, color=(0, 255, 0), thickness=2)

        # ----- 保存帧（若启用） -----
        if video_writer is not None:
            video_writer.write(display_img)

        # ----- 显示 -----
        cv2.imshow(window_name, display_img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' 或 ESC
            print("[退出] 用户中断。")
            break

        # ----- 帧率控制：保持目标帧率 -----
        elapsed = time.time() - loop_start
        sleep_time = frame_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    # 关闭显示窗口
    cv2.destroyWindow(window_name)


def main() -> None:
    """主函数：视频流实时实例分割入口。

    解析参数、初始化模型和视频源、进入推理主循环。
    """
    opt = parse_arguments()
    soc = inspect.get_soc_name().lower()
    model_suffix = "nashp" if soc == "s600" else "nashe"

    # ----- 模型下载（若文件缺失） -----
    download_url = (
        f"https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_{soc}/"
        f"ultralytics_YOLO/yolo11n_seg_{model_suffix}_640x640_nv12.hbm"
    )
    file_io.download_model_if_needed(opt.model_path, download_url)

    # ----- 初始化模型配置 -----
    config = YoloV11SegConfig(
        model_path=opt.model_path,
        score_thres=opt.score_thres,
        nms_thres=opt.nms_thres,
        do_morph=not opt.no_morph,
    )

    # 实例化模型
    yolo11_seg = YoloV11Seg(config)

    # 配置运行时调度参数
    yolo11_seg.set_scheduling_params(priority=opt.priority, bpu_cores=opt.bpu_cores)

    # 打印模型信息
    inspect.print_model_info(yolo11_seg.model)

    # ----- 初始化视频源 -----
    cap = init_video_capture(opt.video_path, opt.camera_id)

    # ----- 初始化视频写入器 -----
    video_writer = init_video_writer(opt.save_video, opt.display_fps)

    # ----- 加载类别名称 -----
    coco_names = file_io.load_class_names(opt.label_file)

    # ----- 运行推理主循环 -----
    try:
        run_inference_loop(
            yolo11_seg=yolo11_seg,
            cap=cap,
            coco_names=coco_names,
            video_writer=video_writer,
            no_contour=opt.no_contour,
            target_fps=opt.display_fps,
            loop=opt.loop,
        )
    except KeyboardInterrupt:
        print("\n[退出] 用户中断。")
    finally:
        # ----- 释放资源 -----
        cap.release()
        if video_writer is not None:
            video_writer.release()
        print("[完成] 资源已释放。")


if __name__ == "__main__":
    main()
