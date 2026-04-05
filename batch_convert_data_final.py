#!/usr/bin/env python3
"""
Pipeline cho datasets/Data-Final: video .mkv → ROMP (.npz) → .pkl.

Cấu trúc Data-Final:
  Data-Final/<batch_id>/<mã_sinh_viên_B2xDCxxxxx>/<kiểu_áo>/<video>.mkv
  - Mỗi người (mã SV) có 5 kiểu áo: AoDai, Bag, Causual, Coat, Saudi.
  - Mỗi kiểu áo có 5 video: Goc1, Goc2, Goc3, Goc4, Goc5.
  - Goc5 thường bị xoay (điện thoại dọc); có thể chỉnh thẳng bằng ffmpeg hoặc bỏ qua.

Ví dụ:
  datasets/Data-Final/362026/B25DCTN012/AoDai/B25DCTN012_...-Goc1.mkv ... Goc5.mkv
"""
import argparse
import re
import subprocess
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pickle

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Mặc định: data nằm trong Data-Final/362026 (sau khi unzip 362026.zip)
VIDEO_ROOT = Path("datasets/Data-Final/362026")
TEMP_NPZ_ROOT = Path("smpl_model_data/DataFinal/3D_SMPLs_temp")
OUT_ROOT = Path("smpl_model_data/DataFinal/DataFinal-smpls-pkl")
ROTATED_TEMP_DIR = Path("smpl_model_data/DataFinal/rotated_temp")

OPENGAIT_ROOT = Path(__file__).resolve().parent

CONDITION_ORDER = ["AoDai", "Bag", "Causual", "Coat", "Saudi"]
VIDEO_EXTENSIONS = (".mkv", ".mov", ".mp4")

# Sắp xếp video theo Goc1, Goc2, Goc3, Goc4, Goc5
GOC_PATTERN = re.compile(r"Goc(\d+)", re.IGNORECASE)


def _goc_sort_key(path: Path) -> tuple:
    m = GOC_PATTERN.search(path.stem)
    if m:
        return (int(m.group(1)), path.name)
    return (99, path.name)


def get_video_frame_count(video_path):
    if not HAS_CV2:
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count if count >= 0 else None


def rotate_video_ffmpeg(input_path: Path, transpose: int, output_path: Path) -> bool:
    """
    Xoay video bằng ffmpeg.
    transpose=1: 90° CW, transpose=2: 90° CCW, transpose=3: 180°.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if transpose == 3:
        vf = "transpose=2,transpose=2"
    else:
        vf = f"transpose={transpose}"
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-vf", vf,
        "-c:a", "copy",
        "-loglevel", "warning",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=600)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def rotate_video_opencv(input_path: Path, transpose: int, output_path: Path) -> bool:
    """
    Xoay video bằng OpenCV (fallback khi không có ffmpeg).
    transpose=1 → 90° CW, transpose=2 → 90° CCW, transpose=3 → 180°.
    """
    if not HAS_CV2:
        return False
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        return False
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if transpose == 3:
        rot = cv2.ROTATE_180
        out_w, out_h = w, h
    else:
        rot = cv2.ROTATE_90_CLOCKWISE if transpose == 1 else cv2.ROTATE_90_COUNTERCLOCKWISE
        out_w, out_h = h, w
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))
    if not out.isOpened():
        cap.release()
        return False
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rot = cv2.rotate(frame, rot)
            out.write(frame_rot)
    finally:
        cap.release()
        out.release()
    return output_path.exists()


def rotate_video(input_path: Path, transpose: int, output_path: Path) -> bool:
    """Thử ffmpeg trước, không có thì dùng OpenCV để xoay video Goc5 (giữ nguyên đuôi, ví dụ .mkv)."""
    if rotate_video_ffmpeg(input_path, transpose, output_path):
        return True
    if HAS_CV2:
        if rotate_video_opencv(input_path, transpose, output_path):
            return True
    print(f"  ✗ Không xoay được (cần ffmpeg hoặc OpenCV): {input_path.name}")
    return False


def get_video_for_processing(video_path: Path, is_goc5: bool, goc5_rotate: int, temp_dir: Path) -> Path:
    """
    Trả về đường dẫn video để đưa vào ROMP.
    Nếu là Goc5 và goc5_rotate in (1, 2, 3) thì tạo bản xoay tạm và trả về path đó.
    1=90° CW, 2=90° CCW, 3=180°.
    """
    if not is_goc5 or goc5_rotate not in (1, 2, 3):
        return video_path
    temp_dir.mkdir(parents=True, exist_ok=True)
    out_rotated = temp_dir / f"{video_path.stem}_rotated{video_path.suffix}"
    if rotate_video(video_path, goc5_rotate, out_rotated):
        return out_rotated
    return video_path  # fallback: dùng nguyên bản (không xoay)


def extract_smpl_from_video(video_path, output_npz_dir, verbose_stderr=False):
    output_npz_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "romp",
        "--mode=video",
        f"--input={video_path}",
        f"--save_path={output_npz_dir}",
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        npz_files = list(output_npz_dir.glob("*.npz"))
        if npz_files:
            total_frames = get_video_frame_count(video_path)
            msg = f"  ⚠ ROMP báo lỗi nhưng đã tạo {len(npz_files)} .npz, tiếp tục..."
            if total_frames is not None:
                msg += f" (video có {total_frames} frame tổng)"
            print(msg)
            if verbose_stderr and e.stderr:
                for line in e.stderr.strip().split("\n")[-8:]:
                    print(f"     ROMP: {line}")
            return True
        print(f"  ✗ Error: {e.stderr}")
        return False


def convert_npz_to_pkl(npz_dir, output_pkl_path):
    output_pkl_path.parent.mkdir(parents=True, exist_ok=True)
    thetas_list = []
    npz_files = sorted(npz_dir.glob("*.npz"))
    if not npz_files:
        return False
    for npz_path in npz_files:
        try:
            data = np.load(npz_path, allow_pickle=True)
            res = data["results"][()]
            if res.get("smpl_thetas") is None or len(res.get("smpl_thetas", [])) == 0:
                continue
            cam = np.array(res["cam"][0], dtype=np.float32)
            smpl_pose = np.array(res["smpl_thetas"][0], dtype=np.float32)
            smpl_shape = np.array(res["smpl_betas"][0], dtype=np.float32)
            theta = np.concatenate([cam, smpl_pose, smpl_shape], axis=0)
            thetas_list.append(theta)
        except Exception as e:
            print(f"  ⚠ {npz_path.name}: {e}")
            continue
    if not thetas_list:
        return False
    smpl_seq = np.stack(thetas_list, axis=0)
    with open(output_pkl_path, "wb") as f:
        pickle.dump(smpl_seq, f)
    print(f"  ✓ Saved {smpl_seq.shape} to {output_pkl_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Data-Final: video → .npz → .pkl (mã SV = 1 người, 5 kiểu áo, 5 góc; Goc5 có thể xoay thẳng)."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Thư mục gốc chứa mã SV (mặc định: datasets/Data-Final/362026).",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default=None,
        help="Thư mục lưu .pkl output (mặc định: smpl_model_data/DataFinal/DataFinal-smpls-pkl).",
    )
    parser.add_argument(
        "--temp-npz-root",
        type=str,
        default=None,
        help="Thư mục lưu .npz tạm (mặc định: smpl_model_data/DataFinal/3D_SMPLs_temp).",
    )
    parser.add_argument(
        "--max-per-condition",
        type=int,
        default=5,
        help="Số video tối đa mỗi kiểu áo. Mặc định 5 (gồm Goc5). Đặt 4 để bỏ Goc5.",
    )
    parser.add_argument(
        "--goc5-rotate",
        type=int,
        choices=[0, 1, 2, 3],
        default=2,
        help="Goc5 bị xoay: 0=bỏ qua, 1=90° CW, 2=90° CCW, 3=180°. Mặc định 2.",
    )
    parser.add_argument(
        "--keep-npz",
        action="store_true",
        help="Không xóa thư mục .npz sau khi convert → pkl (để chạy extract sil).",
    )
    parser.add_argument(
        "--verbose-romp",
        action="store_true",
        help="In thêm stderr khi ROMP báo lỗi.",
    )
    args = parser.parse_args()

    video_root = Path(args.data_root) if args.data_root else VIDEO_ROOT
    if not video_root.is_absolute():
        video_root = OPENGAIT_ROOT / video_root

    if not video_root.exists():
        print(f"Thư mục không tồn tại: {video_root}")
        return

    global OUT_ROOT, TEMP_NPZ_ROOT, ROTATED_TEMP_DIR
    if args.out_root:
        OUT_ROOT = Path(args.out_root)
        if not OUT_ROOT.is_absolute():
            OUT_ROOT = OPENGAIT_ROOT / OUT_ROOT
    else:
        if not OUT_ROOT.is_absolute():
            OUT_ROOT = OPENGAIT_ROOT / OUT_ROOT

    if args.temp_npz_root:
        TEMP_NPZ_ROOT = Path(args.temp_npz_root)
        if not TEMP_NPZ_ROOT.is_absolute():
            TEMP_NPZ_ROOT = OPENGAIT_ROOT / TEMP_NPZ_ROOT
    else:
        if not TEMP_NPZ_ROOT.is_absolute():
            TEMP_NPZ_ROOT = OPENGAIT_ROOT / TEMP_NPZ_ROOT

    if not ROTATED_TEMP_DIR.is_absolute():
        ROTATED_TEMP_DIR = OPENGAIT_ROOT / ROTATED_TEMP_DIR

    max_per = args.max_per_condition
    keep_npz = args.keep_npz
    goc5_rotate = args.goc5_rotate

    # Bỏ Goc5 nếu không xoay và vẫn lấy 5 video → thực tế lấy 5; nếu goc5_rotate=0 và max_per=5 thì cần lọc bỏ Goc5
    person_folders = sorted([d for d in video_root.iterdir() if d.is_dir()])
    person_id_map = {f.name: f"{i:05d}" for i, f in enumerate(person_folders, start=1)}

    print(f"Data-Final: {len(person_folders)} người (mã SV), tối đa {max_per} video/kiểu áo")
    print(f"VIDEO_ROOT={video_root}")
    print(f"OUT={OUT_ROOT}, TEMP_NPZ={TEMP_NPZ_ROOT}")
    print(f"Goc5: {'bỏ qua' if goc5_rotate == 0 else f'xoay transpose={goc5_rotate} (1=CW, 2=CCW)'}")
    print(f"Person ID: {list(person_id_map.keys())[:5]}...\n")

    total = 0
    done = 0
    rotated_temp_dir = ROTATED_TEMP_DIR

    for person_folder in person_folders:
        person_name = person_folder.name
        person_id = person_id_map[person_name]

        for cam_idx, cond in enumerate(CONDITION_ORDER):
            cond_dir = person_folder / cond
            if not cond_dir.is_dir():
                continue

            videos = [
                f for f in cond_dir.iterdir()
                if f.suffix in VIDEO_EXTENSIONS
            ]
            videos_sorted = sorted(videos, key=_goc_sort_key)
            if goc5_rotate == 0:
                videos_sorted = [v for v in videos_sorted if "Goc5" not in v.stem]
            videos_sorted = videos_sorted[:max_per]

            for seq_idx, video_path in enumerate(videos_sorted):
                total += 1
                cam_id = f"cam{cam_idx:02d}"
                seq_id = f"seq{seq_idx:02d}"
                is_goc5 = "Goc5" in video_path.stem

                temp_npz_dir = TEMP_NPZ_ROOT / person_id / cam_id / seq_id
                output_pkl_path = OUT_ROOT / person_id / cam_id / seq_id / f"{seq_id}.pkl"

                if output_pkl_path.exists():
                    print(f"[{total}] ⏭ {person_name}/{cond}/{video_path.name} (đã có)")
                    done += 1
                    continue

                video_to_use = get_video_for_processing(
                    video_path, is_goc5, goc5_rotate,
                    rotated_temp_dir,
                )
                used_rotated = video_to_use != video_path

                print(f"\n[{total}] {person_name}/{cond}/{video_path.name} → {person_id}/{cam_id}/{seq_id}" + (" [đã xoay Goc5]" if used_rotated else ""))

                ok = extract_smpl_from_video(video_to_use, temp_npz_dir, verbose_stderr=args.verbose_romp)
                if used_rotated and video_to_use.exists():
                    try:
                        video_to_use.unlink()
                    except OSError:
                        pass

                npz_files = list(temp_npz_dir.glob("*.npz")) if temp_npz_dir.exists() else []

                if not ok and not npz_files:
                    continue
                if convert_npz_to_pkl(temp_npz_dir, output_pkl_path):
                    done += 1
                    if temp_npz_dir.exists() and not keep_npz:
                        shutil.rmtree(temp_npz_dir)

    # Dọn temp xoay nếu còn
    if rotated_temp_dir.exists():
        try:
            shutil.rmtree(rotated_temp_dir)
        except OSError:
            pass

    print(f"\n{'='*60}")
    print(f"Total: {total}, Done: {done}, Output: {OUT_ROOT}")
    print("Tiếp theo: extract sil với extract_sils_hrnet_new.py --npz-root <TEMP_NPZ> --output-dir <SIL_ROOT>")
    print("=" * 60)


if __name__ == "__main__":
    main()
