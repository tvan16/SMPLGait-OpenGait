# SMPLGait — from videos to inference

This guide covers **data preparation**, **preprocessing**, and **running inference** with SMPLGait on custom data. Set up Python (PyTorch, CUDA) as in [docs/0.get_started.md](../../docs/0.get_started.md).

---

## 1. Directory layout and models to download

### 1.1 Video input (for `batch_convert_data_final.py`)

The script walks **each direct child folder under `--data-root`** as one person (subject ID). Inside each person folder there are **condition** subfolders (e.g. clothing types), each containing video files (`.mkv`, `.mp4`, …). Videos are mapped to `seq00`, `seq01`, … according to sorted names (e.g. Goc1…Goc5). See the docstring in `batch_convert_data_final.py` for details.

### 1.2 Data layout after preprocessing (for inference)

After steps 2 and 3 you should have a fixed tree **`/<id>/camXX/seqYY/`** (e.g. `00001/cam00/seq00`):

| Item | Content |
|------|---------|
| `…-smpls-pkl/` | Each `seq*` contains a `.pkl` with an SMPL sequence of shape `(T, 85)`. |
| `…-sils-m2f/` | Same `id/cam/seq` tree as smpls-pkl; one sil `.png` per frame (64×64). |
| `….json` | `TRAIN_SET` / `TEST_SET` — each entry is an **`id` folder name** (e.g. `"00001"`). |

Place the inference checkpoint for example at:

`work/checkpoints/SMPLGait/SMPLGait-180000.pt`  
(the `work/` directory sits next to `opengait/`).

### 1.3 Models and weights

| Step | Component | Notes |
|------|-------------|--------|
| Video → SMPL (step 2) | **ROMP** | Install ROMP; the `romp --mode=video` CLI must be on your PATH. |
| Sil (step 3) | **`yolov8s.pt`** | Put it in the **working directory** when you run `extract_sils_hrnet_new.py`, or set `YOLO_WEIGHTS` in the script to the correct path. |
| Sil (step 3) | **Mask2Former** | First run downloads `facebook/mask2former-swin-large-ade-semantic` from Hugging Face (network required). |
| Inference (step 4) | **SMPLGait** | **`SMPLGait-180000.pt`** (e.g. Gait3D-Parsing / OpenGait release): [Hugging Face — opengait/OpenGait](https://huggingface.co/opengait/OpenGait); set the path in `configs/smplgait/smplgait.yaml` (`restore_hint`). |

Python packages for step 3 include `ultralytics`, `transformers`, `torch`, `PIL`, `opencv-python`, etc.

---

## 2. Video → ROMP frames → `.pkl` (`batch_convert_data_final.py`)

Run from the **OpenGait repository root** (where `opengait/` and `batch_convert_data_final.py` live).

Example with videos under `datasets/Data-PTIT`, writing `.pkl` and ROMP temp under `smpl_model_data/Data-PTIT/`:

```bash
python batch_convert_data_final.py \
  --data-root datasets/Data-PTIT \
  --out-root smpl_model_data/Data-PTIT/Data-PTIT-smpls-pkl \
  --temp-npz-root smpl_model_data/Data-PTIT/3D_SMPLs_temp \
  --keep-npz
```

- **`--keep-npz`:** keep per-frame `.npz` + `.png` so step 3 can build silhouettes. Omit it if you only need `.pkl` and will not run step 3 (the script may delete temp after each video depending on its logic).

Common options: `--max-per-condition`, `--goc5-rotate`, `--verbose-romp` (see `python batch_convert_data_final.py -h`).

---

## 3. Silhouette extraction (`extract_sils_hrnet_new.py`)

Run from the **repository root** once **`3D_SMPLs_temp`** contains paired `.npz` / `.png` for each frame. Ensure **`yolov8s.pt`** is available (or fix the path in the script).

```bash
python extract_sils_hrnet_new.py \
  --npz-root smpl_model_data/Data-PTIT/3D_SMPLs_temp \
  --output-dir smpl_model_data/Data-PTIT/Data-PTIT-sils-m2f \
  --device cuda
```

- Re-runs **skip** frames that already have sils unless you pass **`--overwrite`**.
- Provide a **JSON** partition listing the `id`s that exist under `…-smpls-pkl` (e.g. `smpl_model_data/Data-PTIT/Data-PTIT.json`).

---

## 4. Inference

1. Edit **`configs/smplgait/smplgait.yaml`** — set the fields described in the header comment: `dataset_root`, `dataset_partition`, `sil_dir_name`, `restore_hint`, and optionally `dataset_name`.

2. Run from the **`opengait/`** directory:

```bash
cd opengait
python -m torch.distributed.launch --nproc_per_node=1 main.py \
  --cfgs ../configs/smplgait/smplgait.yaml \
  --phase test
```

You need at least **one GPU**; `nproc_per_node` should match the number of GPUs used.

---

## Appendix: other configs in this folder

- **`smplgait_gait3d_parsing.yaml`** — optional **Gait3D + parsing** benchmark template from upstream (only if you use that dataset).

See also [docs/2.prepare_dataset.md](../../docs/2.prepare_dataset.md).
