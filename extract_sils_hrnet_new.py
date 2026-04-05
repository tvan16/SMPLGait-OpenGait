#!/usr/bin/env python3
"""
Extract silhouettes dùng YOLO + Mask2Former (giống pipeline GaitDataset).
Cấu trúc giống extract_sils_hrnet.py: npz-root, output-dir, only-seq, overwrite, output-size, v.v.

Usage:
  python extract_sils_hrnet_new.py --npz-root smpl_model_data/DataFinal/3D_SMPLs_temp \\
    --output-dir smpl_model_data/DataFinal/DataFinal-sils-m2f

Requires: ultralytics, transformers, torch, PIL. File yolov8s.pt trong thư mục chạy hoặc đường dẫn đúng.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

OPENGAIT_ROOT = Path(__file__).resolve().parent
TEMP_NPZ_ROOT = OPENGAIT_ROOT / "smpl_model_data" / "DataFinal" / "3D_SMPLs_temp"
OUT_SIL_ROOT = OPENGAIT_ROOT / "smpl_model_data" / "DataFinal" / "DataFinal-sils-m2f"

# Config YOLO + Mask2Former
YOLO_WEIGHTS = "yolov8s.pt"
SEG_CKPT = "facebook/mask2former-swin-large-ade-semantic"
DET_CONF = 0.10
PAD_RATIO = 0.25


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def largest_component(mask_255: np.ndarray) -> np.ndarray:
    binary = (mask_255 > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return mask_255
    areas = stats[1:, cv2.CC_STAT_AREA]
    k = 1 + int(np.argmax(areas))
    return (labels == k).astype(np.uint8) * 255


def crop_and_resize(
    mask: np.ndarray,
    output_size: int = 64,
    padding_ratio: float = 0.1,
    fit_height: bool = True,
) -> np.ndarray:
    """Crop theo bbox người, resize về output_size x output_size."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((output_size, output_size), dtype=np.uint8)

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    h = y_max - y_min + 1
    w = x_max - x_min + 1
    mask_h, mask_w = mask.shape[0], mask.shape[1]
    is_portrait = mask_h > mask_w
    pad_ratio_use = padding_ratio * 1.8 if is_portrait else padding_ratio
    pad_h = int(h * pad_ratio_use)
    pad_w = int(w * pad_ratio_use)
    y_min = max(0, y_min - pad_h)
    y_max = min(mask_h - 1, y_max + pad_h)
    x_min = max(0, x_min - pad_w)
    x_max = min(mask_w - 1, x_max + pad_w)
    if is_portrait:
        extend_below = int(h * 0.8)
        y_max = min(mask_h - 1, y_max + extend_below)
    crop = mask[y_min : y_max + 1, x_min : x_max + 1]
    crop_h, crop_w = crop.shape[:2]
    if crop_h == 0 or crop_w == 0:
        return np.zeros((output_size, output_size), dtype=np.uint8)

    if fit_height:
        scale = output_size / crop_h
        new_h = output_size
        new_w = int(round(crop_w * scale))
        if new_w > output_size:
            resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            x0 = (new_w - output_size) // 2
            resized = resized[:, x0 : x0 + output_size]
            canvas = np.zeros((output_size, output_size), dtype=np.uint8)
            canvas[:, :] = resized
            return canvas
        resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        canvas = np.zeros((output_size, output_size), dtype=np.uint8)
        x_off = (output_size - new_w) // 2
        canvas[:, x_off : x_off + new_w] = resized
        return canvas

    scale = output_size / max(crop_h, crop_w)
    new_h = int(round(crop_h * scale))
    new_w = int(round(crop_w * scale))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((output_size, output_size), dtype=np.uint8)
    y_off = (output_size - new_h) // 2
    x_off = (output_size - new_w) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    return canvas


# ---------------------------------------------------------------------------
# YOLO + Mask2Former pipeline
# ---------------------------------------------------------------------------
def semantic_pred_map(frame_bgr: np.ndarray, processor, seg_model, device) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = seg_model(**inputs)
    pred = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size[::-1]]
    )[0]
    return pred.cpu().numpy()


def detect_best_person_bbox(frame_bgr: np.ndarray, det_model, conf=0.10, pad_ratio=0.25):
    H, W = frame_bgr.shape[:2]
    res = det_model.predict(frame_bgr, conf=conf, verbose=False)[0]
    boxes = res.boxes
    if boxes is None or len(boxes) == 0:
        return None
    cls = boxes.cls.cpu().numpy().astype(int)
    xyxy = boxes.xyxy.cpu().numpy()
    person_idx = np.where(cls == 0)[0]
    if len(person_idx) == 0:
        return None
    areas = (xyxy[person_idx, 2] - xyxy[person_idx, 0]) * (
        xyxy[person_idx, 3] - xyxy[person_idx, 1]
    )
    best_i = person_idx[int(np.argmax(areas))]
    x1, y1, x2, y2 = xyxy[best_i]
    bw, bh = (x2 - x1), (y2 - y1)
    pad = pad_ratio * max(bw, bh)
    x1 = int(max(0, x1 - pad))
    y1 = int(max(0, y1 - pad))
    x2 = int(min(W - 1, x2 + pad))
    y2 = int(min(H - 1, y2 + pad))
    return (x1, y1, x2, y2)


def crop_by_bbox(arr: np.ndarray, bbox):
    x1, y1, x2, y2 = bbox
    return arr[y1 : y2 + 1, x1 : x2 + 1].copy()


def make_silhouette_full_frame(
    frame_bgr: np.ndarray,
    det_model,
    processor,
    seg_model,
    device,
    person_id: int,
    conf=0.10,
    pad_ratio=0.25,
) -> np.ndarray:
    """Sil full frame (H,W) 0/255. YOLO bbox → ROI seg → paste; fallback full-frame seg."""
    H, W = frame_bgr.shape[:2]
    bbox = detect_best_person_bbox(frame_bgr, det_model, conf=conf, pad_ratio=pad_ratio)

    if bbox is not None:
        roi_img = crop_by_bbox(frame_bgr, bbox)
        pred_roi = semantic_pred_map(roi_img, processor, seg_model, device)
        sil_roi = (pred_roi == person_id).astype(np.uint8) * 255
        sil_roi = largest_component(sil_roi)
        sil_full = np.zeros((H, W), dtype=np.uint8)
        x1, y1, x2, y2 = bbox
        Hroi, Wroi = roi_img.shape[:2]
        if sil_roi.shape != (Hroi, Wroi):
            sil_roi = cv2.resize(sil_roi, (Wroi, Hroi), interpolation=cv2.INTER_NEAREST)
        sil_full[y1 : y2 + 1, x1 : x2 + 1] = sil_roi
        sil_full = largest_component(sil_full)
        return sil_full

    pred = semantic_pred_map(frame_bgr, processor, seg_model, device)
    sil_full = (pred == person_id).astype(np.uint8) * 255
    sil_full = largest_component(sil_full)
    return sil_full


def generate_sils_for_seq(
    det_model,
    processor,
    seg_model,
    device,
    person_id: int,
    npz_dir: Path,
    out_sil_dir: Path,
    output_size: int = 64,
    padding_ratio: float = 0.15,
    fit_height: bool = True,
    skip_existing: bool = True,
    det_conf: float = 0.10,
    pad_ratio: float = 0.25,
) -> int:
    ensure_dir(out_sil_dir)
    npz_files = sorted(npz_dir.glob("*.npz"))
    if not npz_files:
        print(f"  ⚠ No .npz files in {npz_dir}")
        return 0

    saved = 0
    skipped = 0

    for npz_path in npz_files:
        out_path = out_sil_dir / f"{npz_path.stem}.png"
        if skip_existing and out_path.exists():
            skipped += 1
            continue

        try:
            data = np.load(npz_path, allow_pickle=True)
            res = data["results"][()]
        except Exception as e:
            print(f"  ⚠ Error loading {npz_path.name}: {e}")
            continue

        if res.get("smpl_thetas") is None or len(res.get("smpl_thetas", [])) == 0:
            continue

        png_path = npz_path.with_suffix(".png")
        if not png_path.exists():
            print(f"  ⚠ PNG not found for {npz_path.name}: {png_path.name}")
            continue

        frame = cv2.imread(str(png_path))
        if frame is None:
            print(f"  ⚠ Failed to read image {png_path}")
            continue

        person_mask = make_silhouette_full_frame(
            frame,
            det_model,
            processor,
            seg_model,
            device,
            person_id,
            conf=det_conf,
            pad_ratio=pad_ratio,
        )
        if np.count_nonzero(person_mask) == 0:
            continue

        sil_out = crop_and_resize(
            person_mask,
            output_size=output_size,
            padding_ratio=padding_ratio,
            fit_height=fit_height,
        )
        cv2.imwrite(str(out_path), sil_out)
        saved += 1

    if skipped:
        print(f"  (đã có sẵn {skipped} frame, giữ nguyên)")
    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Extract silhouettes: YOLO + Mask2Former (cấu trúc giống extract_sils_hrnet)."
    )
    parser.add_argument("--output-size", type=int, default=64)
    parser.add_argument("--padding-ratio", type=float, default=0.2)
    parser.add_argument("--no-fit-height", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--npz-root", type=str, default=None)
    parser.add_argument("--only-seq", type=str, default=None, metavar="SEQ")
    parser.add_argument(
        "--only-ids",
        type=str,
        default=None,
        help="Chỉ xử lý các id này, cách nhau bởi dấu phẩy (e.g. 00011,00024).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--det-conf", type=float, default=DET_CONF, help="YOLO confidence")
    parser.add_argument("--pad-ratio", type=float, default=PAD_RATIO, help="Padding quanh bbox YOLO")
    args = parser.parse_args()

    out_sil_root = Path(args.output_dir) if args.output_dir else OUT_SIL_ROOT
    if not out_sil_root.is_absolute():
        out_sil_root = OPENGAIT_ROOT / out_sil_root

    npz_root = Path(args.npz_root) if args.npz_root else TEMP_NPZ_ROOT
    if not npz_root.is_absolute():
        npz_root = Path.cwd() / npz_root

    if not npz_root.exists():
        print(f"Thư mục {npz_root} không tồn tại!")
        return

    print("Loading YOLO...")
    det_model = YOLO(YOLO_WEIGHTS)
    print("Loading Mask2Former...")
    processor = AutoImageProcessor.from_pretrained(SEG_CKPT)
    seg_model = Mask2FormerForUniversalSegmentation.from_pretrained(SEG_CKPT).to(
        args.device
    ).eval()
    person_id = None
    for k, v in seg_model.config.id2label.items():
        if v.lower() == "person":
            person_id = int(k)
            break
    if person_id is None:
        raise ValueError("Không tìm thấy label 'person' trong Mask2Former.")
    print("Done.")

    if args.only_seq:
        print(f"Chỉ xử lý seq: {args.only_seq}.")
    only_ids = None
    if args.only_ids:
        only_ids = {x.strip() for x in args.only_ids.split(",") if x.strip()}
        print(f"Chỉ xử lý ids: {sorted(only_ids)}.")

    total_saved = 0
    seq_count = 0
    limit = args.limit

    for id_dir in sorted(npz_root.iterdir()):
        if not id_dir.is_dir():
            continue
        person_id_name = id_dir.name
        if only_ids is not None and person_id_name not in only_ids:
            continue

        for cam_dir in sorted(id_dir.iterdir()):
            if not cam_dir.is_dir() or not cam_dir.name.startswith("cam"):
                continue
            cam_id = cam_dir.name

            for seq_dir in sorted(cam_dir.iterdir()):
                if not seq_dir.is_dir() or not seq_dir.name.startswith("seq"):
                    continue
                if args.only_seq is not None and seq_dir.name != args.only_seq:
                    continue

                seq_id = seq_dir.name
                npz_files = list(seq_dir.glob("*.npz"))
                if not npz_files:
                    continue

                out_sil_dir = out_sil_root / person_id_name / cam_id / seq_id

                print(f"\nGenerating silhouettes (YOLO+Mask2Former) for {person_id_name}/{cam_id}/{seq_id}:")
                saved = generate_sils_for_seq(
                    det_model=det_model,
                    processor=processor,
                    seg_model=seg_model,
                    device=args.device,
                    person_id=person_id,
                    npz_dir=seq_dir,
                    out_sil_dir=out_sil_dir,
                    output_size=args.output_size,
                    padding_ratio=args.padding_ratio,
                    fit_height=not args.no_fit_height,
                    skip_existing=not args.overwrite,
                    det_conf=args.det_conf,
                    pad_ratio=args.pad_ratio,
                )
                print(f"  → Saved {saved} silhouettes to {out_sil_dir}")

                total_saved += saved
                seq_count += 1
                if limit and seq_count >= limit:
                    break
            if limit and seq_count >= limit:
                break
        if limit and seq_count >= limit:
            break

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Sequences processed: {seq_count}")
    print(f"  Total silhouettes saved: {total_saved}")
    print("=" * 60)


if __name__ == "__main__":
    main()
