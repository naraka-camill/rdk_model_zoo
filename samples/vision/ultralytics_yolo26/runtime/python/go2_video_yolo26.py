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
YOLO26 统一视频流实时推理脚本（宇树机器人 Go2 摄像头版）

本脚本通过宇树 SDK VideoClient 获取 Go2 机器人前置摄像头画面，
支持 YOLO26 多种任务：detect、seg、pose、cls、obb。
对每一帧进行 BPU 推理，实时绘制结果并在窗口中显示。

工作流程:
    1) 解析命令行参数（任务类型、模型路径、阈值等）。
    2) 若模型文件缺失，自动根据 SoC 类型下载。
    3) 根据 --task 创建对应的 YOLO26 模型封装实例。
    4) 通过 Unitree VideoClient 逐帧获取机器人摄像头画面。
    5) 预处理 -> BPU 推理 -> 后处理。
    6) 绘制结果并叠加 FPS 信息。
    7) 实时显示结果，并可选保存输出视频。

用法示例:
    # 目标检测
    python go2_video_yolo26.py --task detect --net-interface eth1

    # 实例分割
    python go2_video_yolo26.py --task seg --net-interface eth1

    # 关键点检测
    python go2_video_yolo26.py --task pose --net-interface eth1

    # 分类
    python go2_video_yolo26.py --task cls --net-interface eth1

    # 旋转框检测
    python go2_video_yolo26.py --task obb --net-interface eth1

    # 保存输出视频
    python go2_video_yolo26.py --task detect --net-interface eth1 --save-video result.mp4
"""

import os
import cv2
import sys
import time
import argparse
import numpy as np

sys.path.append(os.path.abspath("../../../../../"))

# YOLO26 模型封装
from yolo26_det import YOLO26Detect, YOLO26DetectConfig
from yolo26_seg import YOLO26Seg, YOLO26SegConfig
from yolo26_pose import YOLO26Pose, YOLO26PoseConfig
from yolo26_cls import YOLO26Cls, YOLO26ClsConfig
from yolo26_obb import YOLO26OBB, YOLO26OBBConfig

import utils.py_utils.file_io as file_io
import utils.py_utils.inspect as inspect
import utils.py_utils.visualize as visualize

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient


# ======================================================================
# 宇树机器人视频捕获封装
# ======================================================================

class UnitreeVideoCapture:
    """宇树机器人 VideoClient 的包装类，提供与 OpenCV VideoCapture 兼容的接口。"""
    def __init__(self, net_interface: str = "eth1"):
        ChannelFactoryInitialize(0, net_interface)

        self._client = VideoClient()
        self._client.SetTimeout(3.0)
        self._client.Init()

        # Go2 前置摄像头规格
        self._width = 640
        self._height = 480
        self._fps = 30.0
        self._opened = True

    def read(self):
        """读取一帧机器人摄像头画面。

        Returns:
            (ret, frame): ret 为 True 表示成功，frame 为 BGR numpy 数组。
        """
        code, data = self._client.GetImageSample()
        if code != 0:
            return False, None
        image_data = np.frombuffer(bytes(data), dtype=np.uint8)
        frame = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        if frame is None:
            return False, None
        return True, frame

    def isOpened(self):
        return self._opened

    def release(self):
        self._opened = False

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return self._width
        elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return self._height
        elif prop == cv2.CAP_PROP_FPS:
            return self._fps
        return 0

    def set(self, prop, value):
        return False


# ======================================================================
# 显示参数
# ======================================================================

DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 640


# ======================================================================
# 参数解析
# ======================================================================

def parse_arguments() -> argparse.Namespace:
    """解析命令行参数。"""
    soc = inspect.get_soc_name().lower()
    board_type = ""
    try:
        with open("/sys/class/boardinfo/board_type", "r", encoding="utf-8") as f:
            board_type = f.read().strip().lower()
    except Exception:
        board_type = soc

    if soc == "s600":
        model_march = "nash-p"
        model_suffix = "nashp"
        soc_dir = "rdk_s600"
    elif soc == "s100p" or "p" in board_type:
        model_march = "nash-m"
        model_suffix = "nashm"
        soc_dir = "rdk_s100"
    else:
        model_march = "nash-e"
        model_suffix = "nashe"
        soc_dir = "rdk_s100"

    task_model_map = {
        'detect': f'yolo26n_detect_{model_suffix}_640x640_nv12.hbm',
        'seg': f'yolo26n_seg_{model_suffix}_640x640_nv12.hbm',
        'pose': f'yolo26n_pose_{model_suffix}_640x640_nv12.hbm',
        'cls': f'yolo26n_cls_{model_suffix}_224x224_nv12.hbm',
        'obb': f'yolo26n_obb_{model_suffix}_640x640_nv12.hbm',
    }

    parser = argparse.ArgumentParser(
        description="YOLO26 视频流实时推理（宇树 Go2 摄像头）")

    parser.add_argument('--task', type=str, required=True,
                        choices=['detect', 'seg', 'pose', 'cls', 'obb'],
                        help="任务类型: detect, seg, pose, cls, obb")
    parser.add_argument('--model-path', type=str, default=None,
                        help='BPU 量化模型 (*.hbm) 路径')
    parser.add_argument('--priority', type=int, default=0,
                        help='模型优先级 (0~255)')
    parser.add_argument('--bpu-cores', nargs='+', type=int, default=[0],
                        help='BPU 核心索引列表')
    parser.add_argument('--net-interface', type=str, default='eth1',
                        help='机器人通信网卡接口名称')
    parser.add_argument('--label-file', type=str, default=None,
                        help='类别标签文件路径')
    parser.add_argument('--save-video', type=str, default=None,
                        help='输出视频保存路径（可选），例如 result.mp4')
    parser.add_argument('--score-thres', type=float, default=0.25,
                        help='检测置信度阈值')
    parser.add_argument('--nms-thres', type=float, default=0.45,
                        help='NMS 的 IoU 阈值')
    parser.add_argument('--topk', type=int, default=5,
                        help='[Cls] Top K 结果数')
    parser.add_argument('--kpt-conf-thres', type=float, default=0.5,
                        help='[Pose] 关键点可见性阈值')
    parser.add_argument('--angle-sign', type=float, default=1.0,
                        help='[OBB] 角度解码符号乘数')
    parser.add_argument('--angle-offset', type=float, default=0.0,
                        help='[OBB] 角度解码偏移')
    parser.add_argument('--no-morph', action='store_true',
                        help='[Seg] 禁用掩码后处理的形态学开运算')
    parser.add_argument('--no-contour', action='store_true',
                        help='[Seg] 禁用结果图上绘制轮廓线')
    parser.add_argument('--display-fps', type=int, default=30,
                        help='目标显示帧率（fps）')

    opt = parser.parse_args()

    # 记录 SoC/后缀信息供后续使用
    opt.model_march = model_march
    opt.model_suffix = model_suffix
    opt.soc_dir = soc_dir
    opt.task_model_map = task_model_map

    return opt


# ======================================================================
# 模型初始化
# ======================================================================

def init_model(opt: argparse.Namespace):
    """根据任务类型初始化对应的 YOLO26 模型实例。

    Args:
        opt: 解析后的命令行参数。

    Returns:
        初始化的模型实例。
    """
    # 若未指定模型路径，自动选择默认模型
    if opt.model_path is None:
        opt.model_path = os.path.join("..", "..", "model", opt.model_march,
                                      opt.task_model_map[opt.task])

    # 模型文件缺失时自动下载
    if not os.path.exists(opt.model_path):
        model_file = opt.task_model_map[opt.task]
        download_url = ("https://archive.d-robotics.cc/downloads/rdk_model_zoo/"
                        f"{opt.soc_dir}/Ultralytics_YOLO_OE_3.7.0/"
                        f"{opt.model_march}/{model_file}")
        file_io.download_model_if_needed(opt.model_path, download_url)

    if opt.task == 'detect':
        config = YOLO26DetectConfig(
            model_path=opt.model_path,
            score_thres=opt.score_thres,
            nms_thres=opt.nms_thres,
        )
        model = YOLO26Detect(config)

    elif opt.task == 'seg':
        config = YOLO26SegConfig(
            model_path=opt.model_path,
            score_thres=opt.score_thres,
            nms_thres=opt.nms_thres,
        )
        model = YOLO26Seg(config)

    elif opt.task == 'pose':
        config = YOLO26PoseConfig(
            model_path=opt.model_path,
            score_thres=opt.score_thres,
            nms_thres=opt.nms_thres,
        )
        model = YOLO26Pose(config)

    elif opt.task == 'cls':
        config = YOLO26ClsConfig(
            model_path=opt.model_path,
            topk=opt.topk,
        )
        model = YOLO26Cls(config)

    elif opt.task == 'obb':
        config = YOLO26OBBConfig(
            model_path=opt.model_path,
            score_thres=opt.score_thres,
            nms_thres=opt.nms_thres,
            angle_sign=opt.angle_sign,
            angle_offset=opt.angle_offset,
        )
        model = YOLO26OBB(config)

    else:
        raise ValueError(f"不支持的任务类型: {opt.task}")

    # 配置运行时调度参数
    model.set_scheduling_params(priority=opt.priority, bpu_cores=opt.bpu_cores)

    # 打印模型信息
    inspect.print_model_info(model.model)

    return model


# ======================================================================
# 视频源初始化
# ======================================================================

def init_video_capture(net_interface: str) -> UnitreeVideoCapture:
    """初始化宇树机器人视频捕获。

    Args:
        net_interface: 机器人通信网卡接口名称。

    Returns:
        配置好的 UnitreeVideoCapture 对象。
    """
    cap = UnitreeVideoCapture(net_interface)
    print(f"[视频源] Go2 机器人前置摄像头 (网卡: {net_interface})")
    print(f"[分辨率] {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f} x "
          f"{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}")
    print(f"[帧率  ] {cap.get(cv2.CAP_PROP_FPS):.2f} fps")
    return cap


# ======================================================================
# 视频写入器初始化
# ======================================================================

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


# ======================================================================
# 加载类别标签
# ======================================================================

def load_labels(opt: argparse.Namespace) -> list:
    """根据任务类型加载对应的类别标签文件。

    Args:
        opt: 命令行参数。

    Returns:
        类别名称列表。
    """
    if opt.task == 'cls':
        # 分类任务使用 ImageNet 标签
        label_path = (opt.label_file or
                      '../../../../../datasets/imagenet/imagenet_classes.names')
        return file_io.load_class_names(label_path)

    # detect / seg / pose 使用 COCO 标签
    default_label = '../../test_data/coco_classes.names'
    default_label = None if opt.task == 'obb' else default_label

    if opt.task == 'obb':
        label_path = opt.label_file or '../../../../../datasets/dotav1/dota_classes.names'
    else:
        label_path = opt.label_file or default_label

    if label_path and os.path.exists(label_path):
        return file_io.load_class_names(label_path)
    return []


# ======================================================================
# 推理与显示主循环
# ======================================================================

def run_inference_loop(
    model,
    task: str,
    cap: UnitreeVideoCapture,
    class_names: list,
    video_writer: cv2.VideoWriter | None,
    opt: argparse.Namespace,
) -> None:
    """执行实时推理主循环（宇树机器人摄像头版）。

    逐帧读取 -> 预处理 -> 推理 -> 后处理 -> 可视化 -> 显示/保存。

    Args:
        model: 初始化的 YOLO26 模型实例。
        task: 任务类型。
        cap: UnitreeVideoCapture 实例。
        class_names: 类别名称列表。
        video_writer: 视频写入器（可选）。
        opt: 命令行参数。
    """
    window_name = f"YOLO26-{task.upper()} 实时推理"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, DISPLAY_WIDTH, DISPLAY_HEIGHT)

    # FPS 计算相关变量
    frame_count = 0
    fps_avg = 0.0
    alpha = 0.1  # 滑动平均平滑系数

    # 帧间隔时间（秒）
    frame_interval = 1.0 / opt.display_fps

    print(f"\n[开始] YOLO26-{task} 实时推理中，按 'q' 或 ESC 退出...\n")

    while True:
        loop_start = time.time()

        # ----- 读取一帧 -----
        ret, frame = cap.read()
        if not ret:
            print("[结束] 视频读取完毕。")
            break

        frame_h, frame_w = frame.shape[:2]

        # ----- 预处理 -> BPU 推理 -> 后处理 -----
        t_pre_start = time.perf_counter()

        # 部分模型（如 cls）使用 predict 封装方法，部分需要手动三步
        if task == 'detect':
            boxes, scores, cls_ids = model.predict(frame)
        elif task == 'seg':
            boxes, scores, cls_ids, masks = model.predict(frame)
        elif task == 'pose':
            boxes, scores, cls_ids, kpts = model.predict(frame)
        elif task == 'cls':
            results = model.predict(frame)
        elif task == 'obb':
            results = model.predict(frame)

        t_infer_end = time.perf_counter()
        infer_ms = (t_infer_end - t_pre_start) * 1000

        # ----- 可视化 -----
        # 将帧缩放到显示尺寸
        display_img = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

        if task == 'detect':
            visualize.draw_boxes(display_img, boxes, cls_ids, scores,
                                 class_names, visualize.rdk_colors)
            det_count = len(boxes)

        elif task == 'seg':
            # 缩放坐标到显示尺寸
            scale_x = DISPLAY_WIDTH / frame_w
            scale_y = DISPLAY_HEIGHT / frame_h
            display_boxes = boxes.copy()
            display_boxes[:, 0] = boxes[:, 0] * scale_x
            display_boxes[:, 1] = boxes[:, 1] * scale_y
            display_boxes[:, 2] = boxes[:, 2] * scale_x
            display_boxes[:, 3] = boxes[:, 3] * scale_y

            # 缩放掩码到显示尺寸
            display_masks = []
            for box_disp, mask in zip(display_boxes, masks):
                if mask.size == 0:
                    display_masks.append(mask)
                    continue
                x1, y1, x2, y2 = (int(box_disp[0]), int(box_disp[1]),
                                  int(box_disp[2]), int(box_disp[3]))
                new_w, new_h = x2 - x1, y2 - y1
                if new_w < 1 or new_h < 1:
                    display_masks.append(mask)
                    continue
                display_masks.append(
                    cv2.resize(mask.astype(np.uint8), (new_w, new_h),
                               interpolation=cv2.INTER_NEAREST))

            visualize.draw_boxes(display_img, display_boxes, cls_ids, scores,
                                 class_names, visualize.rdk_colors)
            visualize.draw_masks(display_img, display_boxes, display_masks,
                                 cls_ids, visualize.rdk_colors, alpha=0.4)
            if not opt.no_contour:
                visualize.draw_contours(display_img, display_boxes,
                                        display_masks, cls_ids,
                                        visualize.rdk_colors, thickness=1)
            det_count = len(boxes)

        elif task == 'pose':
            scale_x = DISPLAY_WIDTH / frame_w
            scale_y = DISPLAY_HEIGHT / frame_h
            display_boxes = boxes.copy()
            display_boxes[:, 0] = boxes[:, 0] * scale_x
            display_boxes[:, 1] = boxes[:, 1] * scale_y
            display_boxes[:, 2] = boxes[:, 2] * scale_x
            display_boxes[:, 3] = boxes[:, 3] * scale_y

            display_kpts = kpts.copy()
            display_kpts[:, :, 0] = kpts[:, :, 0] * scale_x
            display_kpts[:, :, 1] = kpts[:, :, 1] * scale_y

            visualize.draw_pose(display_img, display_boxes, display_kpts,
                                kpt_conf_thres=opt.kpt_conf_thres,
                                scores=scores, class_ids=cls_ids,
                                colors=visualize.rdk_colors)
            det_count = len(boxes)

        elif task == 'cls':
            # 分类任务：在图像上显示 Top-K 结果
            idx2label = {}
            for i, name in enumerate(class_names):
                idx2label[i] = name
            visualize.print_classification_results(results, idx2label)

            # 在图像左上角显示分类结果
            y_offset = 30
            for rank, (cls_id, prob) in enumerate(results):
                label = class_names[cls_id] if cls_id < len(class_names) else f"ID:{cls_id}"
                text = f"Top-{rank + 1}: {label} ({prob:.3f})"
                cv2.putText(display_img, text, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            visualize.rdk_colors[rank % len(visualize.rdk_colors)], 2)
                y_offset += 30
            det_count = len(results)

        elif task == 'obb':
            visualize.draw_obb(display_img, results, class_names,
                               visualize.rdk_colors)
            det_count = len(results)

        else:
            det_count = 0

        # ----- 计算并显示 FPS -----
        frame_count += 1
        if frame_count == 1:
            fps_avg = 1.0 / max(infer_ms / 1000, 1e-6)
        else:
            instant_fps = 1.0 / max((time.perf_counter() - loop_start), 1e-6)
            fps_avg = (1 - alpha) * fps_avg + alpha * instant_fps

        fps_text = f"FPS: {fps_avg:.1f}  |  Infer: {infer_ms:.1f}ms  |  Det: {det_count}"
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

        # ----- 帧率控制 -----
        elapsed = time.time() - loop_start
        sleep_time = frame_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    cv2.destroyWindow(window_name)


# ======================================================================
# 主函数
# ======================================================================

def main() -> None:
    """主函数：视频流实时推理入口。"""
    opt = parse_arguments()

    # ----- 初始化模型 -----
    model = init_model(opt)

    # ----- 加载类别标签 -----
    class_names = load_labels(opt)

    # ----- 初始化宇树机器人视频源 -----
    cap = init_video_capture(opt.net_interface)

    # ----- 初始化视频写入器 -----
    video_writer = init_video_writer(opt.save_video, opt.display_fps)

    # ----- 运行推理主循环 -----
    try:
        run_inference_loop(
            model=model,
            task=opt.task,
            cap=cap,
            class_names=class_names,
            video_writer=video_writer,
            opt=opt,
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
