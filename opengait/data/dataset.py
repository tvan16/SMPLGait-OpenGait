import os
import pickle
import os.path as osp
from pathlib import Path

import cv2
import numpy as np
import torch.utils.data as tordata
import json

from utils import get_msg_mgr


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _to_frame_list(obj):
    """Chuyển nội dung pickle thành list các frame (numpy)."""
    if isinstance(obj, np.ndarray):
        if obj.ndim == 3:
            return [obj[i].copy() for i in range(obj.shape[0])]
        if obj.ndim == 2:
            return [obj[i].copy() for i in range(obj.shape[0])]
        if obj.ndim == 1:
            return [obj.copy()]
    if isinstance(obj, list):
        return [np.asarray(x) for x in obj]
    raise TypeError(
        "Unsupported pickle content type: {}; expected ndarray or list.".format(type(obj))
    )


def _is_smpl_frame(f):
    a = np.asarray(f)
    if a.ndim == 1 and a.shape[0] == 85:
        return True
    if a.ndim == 2 and a.shape[1] == 85:
        return True
    return False


def _is_smpl_seq(frames):
    if not frames:
        return False
    return _is_smpl_frame(frames[0])


def _parallel_sil_dir(pkl_path, sil_dir_name):
    """
    .../X-smpls-pkl/.../seq/xxx.pkl  ->  .../<sil_dir_name>/.../seq/
    """
    if not sil_dir_name:
        return None
    parent = Path(pkl_path).resolve().parent
    parts = list(parent.parts)
    new_parts = []
    replaced = False
    for p in parts:
        if str(p).endswith("-smpls-pkl"):
            new_parts.append(sil_dir_name)
            replaced = True
        else:
            new_parts.append(p)
    if not replaced:
        return None
    return Path(*new_parts)


def _load_gray_png(path, out_hw=(64, 64)):
    """Đọc sil PNG; nếu file rỗng/corrupt hoặc OpenCV không đọc được thì dùng ma trận 0 (giống frame thiếu)."""
    try:
        if path is not None and osp.isfile(str(path)) and osp.getsize(str(path)) == 0:
            try:
                get_msg_mgr().log_debug("Empty sil PNG (0 bytes), using zeros: {}".format(path))
            except Exception:
                pass
    except OSError:
        pass
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        try:
            get_msg_mgr().log_debug("Cannot read sil PNG, using zeros: {}".format(path))
        except Exception:
            pass
        return np.zeros(out_hw, dtype=np.float32)
    arr = arr.astype(np.float32)
    if out_hw is not None and (arr.shape[0] != out_hw[0] or arr.shape[1] != out_hw[1]):
        arr = cv2.resize(arr, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_NEAREST)
    return arr


def _load_sil_png_sequence(sil_dir, num_frames, out_hw=(64, 64)):
    if sil_dir is None or not sil_dir.is_dir():
        return [np.zeros(out_hw, dtype=np.float32) for _ in range(num_frames)]
    pngs = sorted(sil_dir.glob("*.png"))
    out = []
    for i in range(num_frames):
        if i < len(pngs):
            out.append(_load_gray_png(pngs[i], out_hw=out_hw))
        else:
            out.append(np.zeros(out_hw, dtype=np.float32))
    return out


def _is_pickle_entry(path):
    if not osp.isfile(path):
        return False
    b = osp.basename(path)
    if b.endswith(".pkl"):
        return True
    if b.startswith("sils-") or b.startswith("smpls-"):
        return True
    return False


def _sort_pkl_paths(paths):
    """Ưu tiên file sil trước smpl (merge Gait3D: sils-* trước smpls-*)."""

    def pri(p):
        b = osp.basename(p).lower()
        if b.startswith("sils"):
            return 0
        if b.startswith("smpls") or "smpl" in b:
            return 1
        return 2

    return sorted(paths, key=lambda p: (pri(p), osp.basename(p)))


class DataSet(tordata.Dataset):
    def __init__(self, data_cfg, training):
        """
        seqs_info: mỗi phần tử [label, type, view, paths] — paths là list file .pkl trong leaf.

        SMPLGait (2 modality):
        - Một file .pkl (T, 85): SMPL; sil lấy từ thư mục song song thay *-smpls-pkl* bằng
          data_cfg['sil_dir_name'] (cùng .../id/cam/seq/*.png).
        - Hai file (sils-*, smpls-* hoặc sil vs smpl): load cả hai, độ dài T phải khớp.
        """
        self.sil_dir_name = data_cfg.get("sil_dir_name", None)
        self.__dataset_parser(data_cfg, training)
        self.cache = data_cfg["cache"]
        self.label_list = [seq_info[0] for seq_info in self.seqs_info]
        self.types_list = [seq_info[1] for seq_info in self.seqs_info]
        self.views_list = [seq_info[2] for seq_info in self.seqs_info]

        self.label_set = sorted(list(set(self.label_list)))
        self.types_set = sorted(list(set(self.types_list)))
        self.views_set = sorted(list(set(self.views_list)))
        self.seqs_data = [None] * len(self)
        self.indices_dict = {label: [] for label in self.label_set}
        for i, seq_info in enumerate(self.seqs_info):
            self.indices_dict[seq_info[0]].append(i)
        if self.cache:
            self.__load_all_data()

    def __len__(self):
        return len(self.seqs_info)

    def __loader__(self, paths):
        paths = sorted(paths)
        pkl_paths = [p for p in paths if _is_pickle_entry(p)]
        if not pkl_paths:
            raise ValueError(
                "- Loader - no valid pickle entries in paths: {}".format(paths)
            )

        if len(pkl_paths) == 1:
            return self._load_from_single_pkl(pkl_paths[0])

        pkl_paths = _sort_pkl_paths(pkl_paths)
        loaded = []
        for pth in pkl_paths:
            obj = _load_pickle(pth)
            frames = _to_frame_list(obj)
            loaded.append((pth, frames))

        sil_frames, smpl_frames = None, None
        for pth, frames in loaded:
            if _is_smpl_seq(frames):
                if smpl_frames is not None:
                    raise ValueError("Multiple SMPL sequences in one leaf: {}".format(paths))
                smpl_frames = [np.asarray(f, dtype=np.float32).reshape(-1)[:85] for f in frames]
            else:
                if sil_frames is not None:
                    raise ValueError("Multiple silhouette sequences in one leaf: {}".format(paths))
                sil_frames = [np.asarray(f, dtype=np.float32) for f in frames]

        if sil_frames is None or smpl_frames is None:
            if len(loaded) == 2:
                f0, f1 = loaded[0][1], loaded[1][1]
                if _is_smpl_seq(f0) and not _is_smpl_seq(f1):
                    smpl_frames = [np.asarray(f, dtype=np.float32).reshape(-1)[:85] for f in f0]
                    sil_frames = [np.asarray(f, dtype=np.float32) for f in f1]
                elif _is_smpl_seq(f1) and not _is_smpl_seq(f0):
                    smpl_frames = [np.asarray(f, dtype=np.float32).reshape(-1)[:85] for f in f1]
                    sil_frames = [np.asarray(f, dtype=np.float32) for f in f0]
                else:
                    raise ValueError(
                        "Cannot infer sil/smpl from two pkls under {}".format(paths)
                    )
            else:
                raise ValueError(
                    "Expected sil + smpl for SMPLGait, got ambiguous pkls: {}".format(paths)
                )

        if len(sil_frames) != len(smpl_frames):
            raise ValueError(
                "Sil length {} != SMPL length {} for paths {}".format(
                    len(sil_frames), len(smpl_frames), paths
                )
            )
        return [sil_frames, smpl_frames]

    def _load_from_single_pkl(self, pth):
        obj = _load_pickle(pth)
        frames = _to_frame_list(obj)

        if _is_smpl_seq(frames):
            smpl_frames = []
            for f in frames:
                v = np.asarray(f, dtype=np.float32).reshape(-1)
                if v.size < 85:
                    raise ValueError("SMPL vector dim {} < 85 in {}".format(v.size, pth))
                smpl_frames.append(v[:85])
            sil_dir = _parallel_sil_dir(pth, self.sil_dir_name)
            sil_frames = _load_sil_png_sequence(sil_dir, len(smpl_frames))
            return [sil_frames, smpl_frames]

        return [frames]

    def __getitem__(self, idx):
        if not self.cache:
            data_list = self.__loader__(self.seqs_info[idx][-1])
        elif self.seqs_data[idx] is None:
            data_list = self.__loader__(self.seqs_info[idx][-1])
            self.seqs_data[idx] = data_list
        else:
            data_list = self.seqs_data[idx]
        seq_info = self.seqs_info[idx]
        return data_list, seq_info

    def __load_all_data(self):
        for idx in range(len(self)):
            self.__getitem__(idx)

    def __dataset_parser(self, data_config, training):
        dataset_root = data_config["dataset_root"]
        try:
            data_in_use = data_config["data_in_use"]
        except KeyError:
            data_in_use = None

        with open(data_config["dataset_partition"], "rb") as f:
            partition = json.load(f)
        train_set = partition["TRAIN_SET"]
        test_set = partition["TEST_SET"]
        label_list = os.listdir(dataset_root)
        train_set = [label for label in train_set if label in label_list]
        test_set = [label for label in test_set if label in label_list]
        miss_pids = [label for label in label_list if label not in (train_set + test_set)]
        msg_mgr = get_msg_mgr()

        def log_pid_list(pid_list):
            if len(pid_list) >= 3:
                msg_mgr.log_info(
                    "[%s, %s, ..., %s]" % (pid_list[0], pid_list[1], pid_list[-1])
                )
            else:
                msg_mgr.log_info(pid_list)

        if len(miss_pids) > 0:
            msg_mgr.log_debug("-------- Miss Pid List --------")
            msg_mgr.log_debug(miss_pids)
        if training:
            msg_mgr.log_info("-------- Train Pid List --------")
            log_pid_list(train_set)
        else:
            msg_mgr.log_info("-------- Test Pid List --------")
            log_pid_list(test_set)

        def get_seqs_info_list(label_set):
            seqs_info_list = []
            for lab in label_set:
                for typ in sorted(os.listdir(osp.join(dataset_root, lab))):
                    for vie in sorted(os.listdir(osp.join(dataset_root, lab, typ))):
                        seq_info = [lab, typ, vie]
                        seq_path = osp.join(dataset_root, *seq_info)
                        entries = sorted(os.listdir(seq_path))
                        seq_dirs = [
                            osp.join(seq_path, d)
                            for d in entries
                            if _is_pickle_entry(osp.join(seq_path, d))
                        ]
                        if seq_dirs:
                            if data_in_use is not None:
                                seq_dirs = [
                                    d
                                    for d, use_bl in zip(seq_dirs, data_in_use)
                                    if use_bl
                                ]
                            seqs_info_list.append([*seq_info, seq_dirs])
                        else:
                            msg_mgr.log_debug(
                                "Find no pickle entry in %s-%s-%s."
                                % (lab, typ, vie)
                            )
            return seqs_info_list

        self.seqs_info = (
            get_seqs_info_list(train_set)
            if training
            else get_seqs_info_list(test_set)
        )
