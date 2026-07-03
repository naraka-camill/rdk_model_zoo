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

"""YOLO11-Pose 视频流实时姿态估计推理脚本。

本脚本支持从摄像头或视频文件读取视频流，对每一帧进行 YOLO11-Pose 姿态估计，
并在窗口中实时显示检测框、关键点和 FPS 信息。

工作流程：
    1) 解析命令行参数（选择摄像头或视频文件）。
    2) 下载模型文件（如果缺失）。
    3) 创建 YoloV11PoseConfig 并初始化 YoloV11Pose 运行时。
    4) 循环读取视频帧 -> 预处理 -> BPU 推理 -> 后处理（检测框 + 关键点）。
    5) 在帧上绘制检测框和关键点，显示实时 FPS。
    6) 按 'q' 键退出，按 'p' 键暂停/继续。

示例：
    # 使用默认摄像头（索引 0）
    python video_pose.py

    # 使用指定摄像头
    python video_pose.py --video-source 0

    # 使用视频文件
    python video_pose.py --video-source /path/to/video.mp4

    # 指定模型和阈值
    python video_pose.py --video-source 0 --score-thres 0.3 --nms-thres 0.6 --kpt-conf-thres 0.5
"""

import os
import cv2
import sys
import time
import argparse

sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.file_io as file_io
import utils.py_utils.inspect as inspect
import utils.py_utils.visualize as visualize
from yolo11pose import YoloV11Pose, YoloV11PoseConfig


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数。

    支持摄像头索引、视频文件路径、模型路径、阈值参数等配置。

    Returns:
        解析后的命令行参数对象。
    """
    soc = inspect.get_soc_name().lower()
    model_suffix = "nashp" if soc == "s600" else "nashe"

    parser = argparse.ArgumentParser(description="YOLO11-Pose 视频流实时姿态估计")

    parser.add_argument('--video-source', type=str, default='0',
                        help='视频输入源：摄像头索引（如 0）或视频文件路径。默认使用摄像头 0。')
    parser.add_argument('--model-path', type=str,
                        default=f'/opt/hobot/model/{soc}/basic/yolo11n_pose_{model_suffix}_640x640_nv12.hbm',
                        help='BPU 量化的 *.hbm 模型文件路径。')
    parser.add_argument('--priority', type=int, default=0,
                        help='模型优先级 (0~255)，0 最低，255 最高。')
    parser.add_argument('--bpu-cores', nargs='+', type=int, default=[0],
                        help='BPU 核心索引列表，例如 --bpu-cores 0 1。')
    parser.add_argument('--label-file', type=str,
                        default='../../test_data/coco_classes.names',
                        help='COCO 类别标签文件路径。')
    parser.add_argument('--nms-thres', type=float, default=0.7,
                        help='非极大值抑制 (NMS) 的 IoU 阈值。')
    parser.add_argument('--score-thres', type=float, default=0.25,
                        help='检测框置信度阈值。')
    parser.add_argument('--kpt-conf-thres', type=float, default=0.5,
                        help='关键点置信度阈值（用于可视化过滤）。')
    parser.add_argument('--display-scale', type=float, default=0.8,
                        help='显示窗口缩放比例，范围 (0, 1]。')
    parser.add_argument('--skip-frames', type=int, default=0,
                        help='跳帧处理：每处理一帧后跳过的帧数，用于提高速度。')

    return parser.parse_args()


def open_video_source(source: str) -> cv2.VideoCapture:
    """打开视频输入源。

    支持摄像头索引（数字字符串）或视频文件路径。

    Args:
        source: 摄像头索引或视频文件路径。

    Returns:
        配置好的 OpenCV VideoCapture 对象。

    Raises:
        RuntimeError: 如果无法打开视频源。
    """
    # 判断是否为摄像头索引（纯数字字符串）
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
        source_name = f"摄像头 {source}"
    else:
        cap = cv2.VideoCapture(source)
        source_name = os.path.basename(source)

    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频源：{source_name}")

    # 设置摄像头分辨率 640x640（仅对摄像头有效，视频文件会忽略）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    # 设置摄像头帧率为 30 FPS（仅对摄像头有效）
    cap.set(cv2.CAP_PROP_FPS, 30)

    # 获取实际生效的视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[信息] 视频源: {source_name}")
    print(f"[信息] 分辨率: {width} x {height}")
    if fps > 0:
        print(f"[信息] 帧率: {fps:.2f} FPS")
    if total_frames > 0:
        print(f"[信息] 总帧数: {total_frames}")

    return cap


def run_video_pose(args: argparse.Namespace) -> None:
    """视频流实时姿态估计主循环。

    从视频源逐帧读取，执行 YOLO11-Pose 推理，可视化结果并显示实时 FPS。

    Args:
        args: 命令行参数对象。

    Returns:
        None
    """
    # ── 1. 下载模型（如果本地不存在） ──────────────────────────────────────
    download_url = (
        f"https://archive.d-robotics.cc/downloads/rdk_model_zoo/"
        f"rdk_{inspect.get_soc_name().lower()}/ultralytics_YOLO/"
        f"yolo11n_pose_{'nashp' if inspect.get_soc_name().lower() == 's600' else 'nashe'}"
        f"_640x640_nv12.hbm"
    )
    file_io.download_model_if_needed(args.model_path, download_url)

    # ── 2. 初始化模型配置 ──────────────────────────────────────────────────
    config = YoloV11PoseConfig(
        model_path=args.model_path,
        score_thres=args.score_thres,
        nms_thres=args.nms_thres,
    )

    # ── 3. 实例化模型 ──────────────────────────────────────────────────────
    yolov11_pose = YoloV11Pose(config)
    yolov11_pose.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)

    # 打印模型基本信息
    inspect.print_model_info(yolov11_pose.model)

    # ── 4. 加载类别名称 ────────────────────────────────────────────────────
    coco_names = file_io.load_class_names(args.label_file)

    # ── 5. 打开视频源 ──────────────────────────────────────────────────────
    cap = open_video_source(args.video_source)

    # ── 6. 初始化 FPS 计算相关变量 ─────────────────────────────────────────
    frame_count = 0
    skip_counter = 0
    fps = 0.0
    fps_alpha = 0.1  # 平滑系数，用于指数移动平均
    prev_time = time.time()

    # 窗口名称
    window_name = "YOLO11-Pose 实时姿态估计"

    # 预先创建显示窗口，确保 cv2.imshow 能正常工作
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if args.display_scale < 1.0:
        cv2.resizeWindow(window_name, 0, 0)  # 让窗口自适应

    print("\n[操作说明]")
    print("  'q' 键 - 退出")
    print("  'p' 键 - 暂停/继续")
    print("  's' 键 - 保存当前帧截图\n")

    paused = False

    try:
        while True:
            # ── 6a. 暂停处理 ──────────────────────────────────────────────
            if paused:
                key = cv2.waitKey(100) & 0xFF
                if key == ord('p'):
                    paused = False
                    print("[信息] 继续处理")
                elif key == ord('q'):
                    break
                continue

            # ── 6b. 读取一帧 ──────────────────────────────────────────────
            ret, frame = cap.read()
            if not ret:
                print("[信息] 视频读取完毕或读取失败，退出循环。")
                break

            # ── 6c. 跳帧逻辑：跳过指定数量的帧以提高处理速度 ──────────────
            if args.skip_frames > 0 and skip_counter < args.skip_frames:
                skip_counter += 1
                frame_count += 1
                continue
            skip_counter = 0

            frame_count += 1
            img_h, img_w = frame.shape[:2]

            # ── 7. 推理：预处理 -> BPU 前向推理 -> 后处理 ─────────────────
            input_array = yolov11_pose.pre_process(frame)
            outputs = yolov11_pose.forward(input_array)
            boxes, scores, cls_ids, kpts_xy, kpts_score = \
                yolov11_pose.post_process(outputs, img_w, img_h)

            # ── 8. 可视化：绘制检测框和关键点 ─────────────────────────────
            if len(boxes) > 0:
                visualize.draw_boxes(frame, boxes, cls_ids, scores,
                                     coco_names, visualize.rdk_colors)
                visualize.draw_keypoints(frame, kpts_xy, kpts_score,
                                         kpt_conf_thresh=args.kpt_conf_thres)

            # ── 9. 计算并显示实时 FPS ─────────────────────────────────────
            current_time = time.time()
            dt = current_time - prev_time
            prev_time = current_time

            # 指数移动平均平滑 FPS
            instant_fps = 1.0 / dt if dt > 0 else 0.0
            fps = fps * (1 - fps_alpha) + instant_fps * fps_alpha

            # 在左上角显示 FPS 和检测信息
            info_text = [
                f"FPS: {fps:.1f}",
                f"检测目标: {len(boxes)}",
                f"帧: {frame_count}",
            ]
            for i, text in enumerate(info_text):
                cv2.putText(frame, text, (10, 30 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # ── 10. 缩放显示并展示 ────────────────────────────────────────
            display_scale = max(0.1, min(1.0, args.display_scale))
            if display_scale < 1.0:
                disp_w = int(img_w * display_scale)
                disp_h = int(img_h * display_scale)
                display_frame = cv2.resize(frame, (disp_w, disp_h))
            else:
                display_frame = frame

            cv2.imshow(window_name, display_frame)

            # ── 11. 键盘控制 ──────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[信息] 用户退出。")
                break
            elif key == ord('p'):
                paused = True
                print("[信息] 已暂停，按 'p' 键继续。")
            elif key == ord('s'):
                # 保存当前帧截图
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                save_path = f"pose_snapshot_{timestamp}.jpg"
                cv2.imwrite(save_path, frame)
                print(f"[信息] 截图已保存: {save_path}")

    except KeyboardInterrupt:
        print("\n[信息] 用户中断。")
    finally:
        # ── 12. 清理资源 ──────────────────────────────────────────────────
        cap.release()
        cv2.destroyAllWindows()
        print(f"[信息] 共处理 {frame_count} 帧，视频源已关闭。")


def main() -> None:
    """主入口函数。

    解析命令行参数并启动视频流实时姿态估计。

    Returns:
        None
    """
    args = parse_arguments()
    run_video_pose(args)


if __name__ == "__main__":
    main()
