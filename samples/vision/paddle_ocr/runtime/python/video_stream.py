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

"""PaddleOCR real-time video stream demo.

This script captures video from a camera device or video file, runs the
two-stage PP-OCRv6 pipeline (detection + recognition) on selected frames,
and displays the result as a live video stream with text overlays.

To maintain real-time performance, OCR is only executed every N frames
(controlled by ``--process-interval``); intermediate frames reuse the most
recent detection results for display.

Controls:
    - ``q`` / ``ESC`` — Quit.
    - ``p`` — Pause / resume.
    - ``+`` / ``-`` — Increase / decrease processing interval.

Example:
    # USB camera at /dev/video0
    python3 video_stream.py

    # MIPI camera (RDK dev board camera)
    python3 video_stream.py --video-source 0 --video-type mipi

    # Video file
    python3 video_stream.py --video-source /path/to/video.mp4 --video-type file

    # Custom camera resolution
    python3 video_stream.py --video-width 1280 --video-height 720
"""

import os
import sys
import cv2
import argparse
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

# Add project root to sys.path so we can import utility modules.
sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.inspect as inspect
import utils.py_utils.file_io as file_io
from paddle_ocr import PaddleOCRDet, PaddleOCRDetConfig, PaddleOCRRec, PaddleOCRRecConfig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODEL_SOC = None  # cached after first resolve


def _resolve_model_soc() -> str:
    """Map the runtime SoC name to the PP-OCRv6 model variant directory."""
    global _MODEL_SOC
    if _MODEL_SOC is not None:
        return _MODEL_SOC
    soc = inspect.get_soc_name().lower().strip("()").strip()
    _MODEL_SOC = "s600" if soc == "s600" else "s100"
    return _MODEL_SOC


# ---------------------------------------------------------------------------
# OCR Pipeline Runner
# ---------------------------------------------------------------------------

@dataclass
class OCRFrameResult:
    """Holds the OCR results for a single processed frame.

    Attributes:
        boxes: List of ``(4, 2)`` polygon vertex arrays for each text region.
        texts: List of recognized text strings corresponding to each box.
        frame_with_boxes: Copy of the frame with polygon boxes drawn on it.
    """
    boxes: List[np.ndarray]
    texts: List[str]
    frame_with_boxes: np.ndarray


class OCRVideoRunner:
    """Runs PaddleOCR detection + recognition on a video stream.

    Encapsulates model loading, frame preprocessing, OCR inference, and
    result overlay rendering.

    Args:
        det_model_path: Path to the detection ``.hbm`` model.
        rec_model_path: Path to the recognition ``.hbm`` model.
        label_file: Path to the character vocabulary file.
        font_path: Path to a TrueType font file for text rendering.
        threshold: Binarization threshold for the detection mask.
        ratio_prime: Contour expansion ratio.
        priority: BPU scheduling priority.
        bpu_cores: BPU core indices for inference.
    """

    def __init__(
        self,
        det_model_path: str,
        rec_model_path: str,
        label_file: str,
        font_path: str,
        threshold: float = 0.5,
        ratio_prime: float = 2.7,
        priority: int = 0,
        bpu_cores: Optional[List[int]] = None,
    ) -> None:
        # Download models if missing
        model_soc = _resolve_model_soc()
        det_url = (
            "https://archive.d-robotics.cc/downloads/rdk_model_zoo/"
            f"rdk_{model_soc}/paddle_ocr/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm"
        )
        rec_url = (
            "https://archive.d-robotics.cc/downloads/rdk_model_zoo/"
            f"rdk_{model_soc}/paddle_ocr/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm"
        )
        file_io.download_model_if_needed(det_model_path, det_url)
        file_io.download_model_if_needed(rec_model_path, rec_url)

        # Load character list
        with open(label_file, 'r', encoding='utf-8') as f:
            char_list = ['blank'] + [line.rstrip('\n') for line in f]
        char_list.append(' ')
        self.char_list = char_list

        self.font_path = font_path

        # Initialize detection model
        det_config = PaddleOCRDetConfig(
            model_path=det_model_path,
            ratio_prime=ratio_prime,
            threshold=threshold,
        )
        self.det = PaddleOCRDet(det_config)
        self.det.set_scheduling_params(
            priority=priority,
            bpu_cores=bpu_cores or [0],
        )
        inspect.print_model_info(self.det.model)

        # Initialize recognition model
        rec_config = PaddleOCRRecConfig(model_path=rec_model_path)
        self.rec = PaddleOCRRec(rec_config)
        self.rec.set_scheduling_params(
            priority=priority,
            bpu_cores=bpu_cores or [0],
        )
        inspect.print_model_info(self.rec.model)

        # Cached results for frames not processed
        self._last_result: Optional[OCRFrameResult] = None

    def process_frame(self, frame: np.ndarray) -> OCRFrameResult:
        """Run the full OCR pipeline on a single frame.

        Args:
            frame: Input BGR frame.

        Returns:
            An ``OCRFrameResult`` containing bounding boxes, recognized texts,
            and the annotated frame.
        """
        img_h, img_w = frame.shape[:2]

        # --- Detection ---
        input_tensor = self.det.pre_process(frame)
        det_outputs = self.det.forward(input_tensor)
        img_boxes, cropped_images, boxes_list = self.det.post_process(
            det_outputs, frame, img_w, img_h
        )

        # --- Recognition ---
        recognized_texts = []
        for crop in cropped_images:
            rec_input = self.rec.pre_process(crop)
            rec_outputs = self.rec.forward(rec_input)
            text = self.rec.post_process(rec_outputs, self.char_list)
            recognized_texts.append(text)

        result = OCRFrameResult(
            boxes=boxes_list,
            texts=recognized_texts,
            frame_with_boxes=img_boxes,
        )
        self._last_result = result
        return result

    def overlay_text(self, frame: np.ndarray, result: OCRFrameResult) -> np.ndarray:
        """Draw recognized Chinese text on the frame, positioned above each box.

        Uses PIL/Pillow with a TrueType font for correct CJK rendering
        (OpenCV's ``putText`` cannot render Chinese characters).

        Args:
            frame: BGR frame to draw on.
            result: OCR results to render.

        Returns:
            The annotated frame (modified in-place).
        """
        from PIL import Image, ImageDraw, ImageFont

        font_size = 24
        font = ImageFont.truetype(self.font_path, font_size)

        # Convert BGR frame to RGB PIL image for drawing
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(img_pil)

        for text, box in zip(result.texts, result.boxes):
            if not text:
                continue
            # Compute top-center of the bounding box
            xs = box[:, 0]
            ys = box[:, 1]
            cx = int(np.mean(xs))
            y_top = int(np.min(ys)) - 8

            # Measure text size using PIL
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]

            text_x = cx - tw // 2
            text_y = y_top if y_top > th else y_top + th + 4

            # Draw white background rectangle
            draw.rectangle(
                [text_x - 2, text_y - th - 2, text_x + tw + 2, text_y + 4],
                fill=(255, 255, 255),
            )
            # Draw red Chinese text
            draw.text((text_x, text_y), text, font=font, fill=(255, 0, 0))

        # Convert back to BGR NumPy array
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def annotate_result(self, frame: np.ndarray, result: OCRFrameResult) -> np.ndarray:
        """Combine box drawing and text overlay on the frame.

        Args:
            frame: Original BGR frame.
            result: OCR results.

        Returns:
            Fully annotated frame.
        """
        annotated = self.overlay_text(result.frame_with_boxes, result)
        return annotated


# ---------------------------------------------------------------------------
# Video source helpers
# ---------------------------------------------------------------------------

def _open_camera(source: int, width: int, height: int, api: int) -> cv2.VideoCapture:
    """Open a camera device with the given settings."""
    cap = cv2.VideoCapture(source, api)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera source {source}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def _open_video_file(path: str) -> cv2.VideoCapture:
    """Open a video file."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {path}")
    return cap


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the real-time PaddleOCR video stream demo.

    Parses command-line arguments, initializes the camera/video source, loads
    the OCR models, and enters the main loop that captures frames, optionally
    runs OCR, and displays the annotated result.
    """
    model_soc = _resolve_model_soc()

    parser = argparse.ArgumentParser(
        description="PaddleOCR real-time video stream demo (PP-OCRv6).")

    # Model & data paths
    parser.add_argument('--det-model-path', type=str,
                        default=f'/opt/hobot/model/{model_soc}/basic/'
                                f'PP-OCRv6_det_infer-deploy_640x640_nv12.hbm',
                        help='Path to BPU quantized detection model (*.hbm).')
    parser.add_argument('--rec-model-path', type=str,
                        default=f'/opt/hobot/model/{model_soc}/basic/'
                                f'PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm',
                        help='Path to BPU quantized recognition model (*.hbm).')
    parser.add_argument('--label-file', type=str,
                        default='../../test_data/ppocrv6_dict.txt',
                        help='Path to the character vocabulary file.')
    parser.add_argument('--font-path', type=str,
                        default='../../test_data/FangSong.ttf',
                        help='Path to a TrueType font file for text rendering.')

    # Video source
    parser.add_argument('--video-source', type=str, default='0',
                        help='Video source. Use a number (e.g., 0) for a camera '
                             'device, or a file path for a video file.')
    parser.add_argument('--video-type', type=str,
                        choices=['auto', 'camera', 'mipi', 'usb', 'file'],
                        default='auto',
                        help='Force video source type. "auto" infers from source. '
                             '"mipi" uses CAP_V4L2, "usb" uses CAP_ANY, '
                             '"file" opens a video file.')
    parser.add_argument('--video-width', type=int, default=640,
                        help='Desired camera capture width.')
    parser.add_argument('--video-height', type=int, default=640,
                        help='Desired camera capture height.')

    # Processing settings
    parser.add_argument('--process-interval', type=int, default=3,
                        help='Run OCR on every Nth frame (1 = every frame). '
                             'Higher values improve throughput at the cost of '
                             'responsiveness.')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Binarization threshold for the detection mask.')
    parser.add_argument('--ratio-prime', type=float, default=2.7,
                        help='Contour expansion ratio.')
    parser.add_argument('--priority', type=int, default=0,
                        help='Model scheduling priority (0~255).')
    parser.add_argument('--bpu-cores', nargs='+', type=int, default=[0],
                        help='BPU core indices, e.g. --bpu-cores 0 1.')
    parser.add_argument('--display-scale', type=float, default=1.0,
                        help='Scale factor for the display window '
                             '(e.g., 0.5 for half-size).')

    # Output
    parser.add_argument('--save-video', type=str, default=None,
                        help='Optional path to save the output video stream.')

    opt = parser.parse_args()

    # --- Resolve video source ---
    video_type = opt.video_type
    try:
        source_int = int(opt.video_source)
        is_numeric = True
    except ValueError:
        source_int = opt.video_source
        is_numeric = False

    if video_type == 'auto':
        video_type = 'file' if not is_numeric else 'camera'

    print(f"[Video] Source: {opt.video_source}, Type: {video_type}")

    if video_type == 'file':
        cap = _open_video_file(opt.video_source)
    elif video_type == 'mipi':
        # RDK dev board MIPI camera typically uses V4L2 API
        cap = _open_camera(source_int, opt.video_width, opt.video_height,
                           cv2.CAP_V4L2)
    else:
        # USB camera or default
        cap = _open_camera(source_int, opt.video_width, opt.video_height,
                           cv2.CAP_ANY)

    # Read first frame to determine actual resolution
    ret, test_frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Failed to read first frame from video source.")
    actual_h, actual_w = test_frame.shape[:2]
    print(f"[Video] Resolution: {actual_w} x {actual_h}")

    # --- Initialize OCR ---
    print("[OCR] Loading models ...")
    runner = OCRVideoRunner(
        det_model_path=opt.det_model_path,
        rec_model_path=opt.rec_model_path,
        label_file=opt.label_file,
        font_path=opt.font_path,
        threshold=opt.threshold,
        ratio_prime=opt.ratio_prime,
        priority=opt.priority,
        bpu_cores=opt.bpu_cores,
    )

    # --- Video writer (optional) ---
    video_writer = None
    if opt.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        out_w = int(actual_w * opt.display_scale) * 2  # side-by-side
        out_h = int(actual_h * opt.display_scale)
        video_writer = cv2.VideoWriter(opt.save_video, fourcc, fps,
                                       (out_w, out_h))

    # --- Main loop ---
    frame_count = 0
    process_interval = opt.process_interval
    paused = False

    print("[Info] Controls:  q=Quit  p=Pause  +/-=Interval  r=Reset OCR")
    print(f"[Info] Processing every {process_interval} frame(s).")

    # Process the first frame immediately
    last_result = runner.process_frame(test_frame)
    last_annotated = runner.annotate_result(test_frame, last_result)
    fps_text = ""

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("[Info] End of video stream.")
                break

            frame_count += 1

            # Run OCR every N frames
            if frame_count % process_interval == 0:
                result = runner.process_frame(frame)
                annotated = runner.annotate_result(frame, result)
                # Print recognized texts
                if result.texts:
                    print(f"[Frame {frame_count}] Texts: {result.texts}")
            else:
                # Reuse last result — redraw boxes on current frame
                if runner._last_result is not None:
                    annotated = runner.annotate_result(frame, runner._last_result)
                else:
                    annotated = frame

            # FPS counter
            fps_text = f"Interval: {process_interval} | Frame: {frame_count}"

        # Build side-by-side display: original | annotated
        display_scale = opt.display_scale
        if display_scale != 1.0:
            disp_w = int(actual_w * display_scale)
            disp_h = int(actual_h * display_scale)
            frame_small = cv2.resize(frame, (disp_w, disp_h))
            annotated_small = cv2.resize(annotated, (disp_w, disp_h))
        else:
            frame_small = frame
            annotated_small = annotated

        # Side-by-side
        combined = np.hstack((frame_small, annotated_small))

        # Overlay FPS / status info
        info_text = f"[PAUSED] " if paused else ""
        info_text += fps_text
        cv2.putText(combined, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("PaddleOCR Real-time (original | result)", combined)

        if video_writer and not paused:
            video_writer.write(combined)

        # --- Keyboard handling ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q or ESC
            print("[Info] User quit.")
            break
        elif key == ord('p'):
            paused = not paused
            print(f"[Info] {'Paused' if paused else 'Resumed'}.")
        elif key == ord('+') or key == ord('='):
            process_interval = min(process_interval + 1, 30)
            print(f"[Info] Process interval increased to {process_interval}.")
        elif key == ord('-') or key == ord('_'):
            process_interval = max(process_interval - 1, 1)
            print(f"[Info] Process interval decreased to {process_interval}.")
        elif key == ord('r'):
            # Force re-process current frame
            if not paused:
                result = runner.process_frame(frame)
                annotated = runner.annotate_result(frame, result)
                if result.texts:
                    print(f"[Frame {frame_count} - Manual] Texts: {result.texts}")

    # --- Cleanup ---
    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    print("[Done] Video stream closed.")


if __name__ == "__main__":
    main()
