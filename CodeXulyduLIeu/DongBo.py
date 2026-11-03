#!/usr/bin/env python3
"""Căn chỉnh một cặp MP4/CSI CSV cho web demo, không chia lại dữ liệu.

Với đoạn Test đã xuất, dùng meta_test để khớp chính xác pkt_id/timestamp.
Với video mới không có metadata, ghép gần đúng theo thứ tự và timestamp.
"""
from __future__ import annotations
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE.parent / 'Data'
REFERENCE_ROOT = HERE / 'reference_meta'
SEGMENT = re.compile(r'^test_\d+_(\d+)_(\d+)$')
REQUIRED_CSI = {'timestamp_sec', 'pkt_id', 'csi_matrix'}


def read_csi(path: Path) -> list[dict]:
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not REQUIRED_CSI.issubset(reader.fieldnames or []):
            raise ValueError(f'CSV cần các cột {sorted(REQUIRED_CSI)}: {path}')
        rows = [row for row in reader if row.get('timestamp_sec') and row.get('csi_matrix')]
    rows.sort(key=lambda row: float(row['timestamp_sec']))
    if not rows:
        raise ValueError(f'CSI CSV không có gói hợp lệ: {path}')
    return rows


def video_info(path: Path) -> tuple[int, float]:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f'Không mở được video: {path}')
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if count < 1 or not np.isfinite(fps) or fps <= 0:
            raise ValueError(f'Video thiếu số frame hoặc FPS hợp lệ: {path}')
        return count, fps
    finally:
        cap.release()


def infer_test_location(video: Path):
    try:
        parts = video.parent.resolve().relative_to(DATA_ROOT.resolve()).parts
    except ValueError:
        return None
    if len(parts) < 4 or parts[0] not in ('DataSoluong', 'DataHanhDong'):
        return None
    match = SEGMENT.fullmatch(parts[-1])
    if not match:
        return None
    return parts[0], '/'.join(parts[1:-1]), int(match[1]), int(match[2])


def exact_test_pairs(csi: list[dict], meta_file: Path, record_id: str,
                     first: int, last: int, frame_count: int) -> list[dict]:
    with meta_file.open(encoding='utf-8-sig', newline='') as f:
        selected = [row for row in csv.DictReader(f)
                    if row.get('record_id') == record_id and first <= int(float(row['pair_index_30hz'])) <= last]
    selected.sort(key=lambda row: int(float(row['pair_index_30hz'])))
    if len(selected) != frame_count or frame_count != last-first+1:
        raise ValueError(f'Metadata Test không khớp {frame_count} frame của {record_id} [{first}-{last}]')
    by_packet = defaultdict(list)
    for index, row in enumerate(csi):
        by_packet[str(row['pkt_id'])].append(index)
    used, aligned = set(), []
    for row in selected:
        index = int(float(row['pair_index_30hz']))
        target = float(row['timestamp_sec'])
        candidates = [(abs(float(csi[j]['timestamp_sec'])-target), j)
                      for j in by_packet[str(row['pkt_id'])] if j not in used]
        if not candidates:
            raise ValueError(f'Thiếu CSI pkt_id={row["pkt_id"]} tại pair {index}')
        error, j = min(candidates)
        if error > 0.005:
            raise ValueError(f'CSI sai timestamp tại pair {index}: lệch {error:.6f}s')
        used.add(j)
        aligned.append({'frame_index': index-first, 'pair_index_30hz': index,
                        'record_id': record_id, 'alignment_mode': 'exact_test_metadata',
                        **csi[j]})
    return aligned


def estimated_pairs(csi: list[dict], frame_count: int, record_id: str) -> list[dict]:
    if len(csi) < frame_count:
        raise ValueError(f'CSI có {len(csi)} gói, ít hơn {frame_count} frame; không tự nhân bản CSI')
    times = np.array([float(r['timestamp_sec']) for r in csi], dtype=float)
    targets = np.linspace(times[0], times[-1], frame_count)
    aligned = []
    next_index = 0
    for frame_index, target in enumerate(targets):
        last_possible = len(csi)-(frame_count-frame_index)
        guess = int(np.searchsorted(times, target))
        candidates = [max(next_index, min(last_possible, j)) for j in (guess-1, guess)]
        j = min(candidates, key=lambda k: abs(times[k]-target))
        next_index = j+1
        aligned.append({'frame_index': frame_index, 'pair_index_30hz': frame_index,
                        'record_id': record_id, 'alignment_mode': 'estimated_from_two_files',
                        **csi[j]})
    return aligned


def align_pair(video: Path, csi_csv: Path, reference_meta: Path | None = None,
               record_id: str | None = None) -> list[dict]:
    video, csi_csv = Path(video), Path(csi_csv)
    if not video.is_file() or not csi_csv.is_file():
        raise FileNotFoundError(f'Cần cả video và CSI: {video}, {csi_csv}')
    frame_count, _ = video_info(video)
    csi = read_csi(csi_csv)
    location = infer_test_location(video)
    if location and reference_meta is None:
        dataset, inferred_record, first, last = location
        candidate = REFERENCE_ROOT / f'{dataset}_meta_test.csv'
        if candidate.is_file():
            reference_meta = candidate
    if reference_meta is not None:
        if not location:
            raise ValueError('Metadata Test chỉ áp dụng cho video trong cấu trúc Data/<dataset>/<record>/test_.../')
        _, inferred_record, first, last = location
        return exact_test_pairs(csi, Path(reference_meta), inferred_record, first, last, frame_count)
    return estimated_pairs(csi, frame_count, record_id or (location[1] if location else video.stem))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', type=Path, required=True)
    p.add_argument('--csi', type=Path, required=True)
    p.add_argument('--reference-meta', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    rows = align_pair(a.video, a.csi, a.reference_meta)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f'{len(rows)} cặp; mode={rows[0]["alignment_mode"]}; output={a.output}')

if __name__ == '__main__':
    main()
