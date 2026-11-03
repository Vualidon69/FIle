#!/usr/bin/env python3
"""Trích 73 CSI + 768 V-JEPA từ một cặp MP4/CSV của web demo.

Dùng lại chính xác công thức CSI và cách tiền xử lý/pooling V-JEPA của pipeline
huấn luyện. Không tạo nhãn, không chia lại Train/Validation/Test.
"""
from __future__ import annotations
import argparse
import io
import math
import re
from collections import deque
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from DongBo import align_pair, infer_test_location

HERE = Path(__file__).resolve().parent
WEB_ROOT = HERE.parent / 'CSI_VLFormer_Local_Web'
CSI_RATE_HZ = 30.0
TEMPORAL_WINDOW_SAMPLES = 64
CSI_NAMES = [f'csi_amp_{i:03d}' for i in range(62)] + [
    'csi_amp_mean', 'csi_amp_std', 'csi_amp_var', 'csi_temporal_diff',
    'csi_motion_energy', 'csi_subcarrier_var', 'csi_spectral_energy',
    'csi_doppler_peak_hz', 'csi_doppler_centroid_hz',
    'csi_doppler_bandwidth_hz', 'csi_doppler_peak_power',
]
RGB_NAMES = [f'video_vjepa_{i:04d}' for i in range(768)]

def parse_numbers(value) -> list[float]:
    if value is None:
        return []
    values = re.findall(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?", str(value))
    try:
        return [float(value) for value in values]
    except ValueError:
        return []

def csi_amplitudes(raw_value) -> list[float]:
    """Convert collector I/Q pairs after its four-value header to amplitudes."""
    values = parse_numbers(raw_value)
    if len(values) <= 4:
        return []
    values = values[4:]
    values = values[:len(values) // 2 * 2]
    if not values:
        return []
    return [math.hypot(real, imag) for real, imag in zip(values[0::2], values[1::2])]

def csi_temporal_features(vector, pair_index, previous, history):
    """Packet statistics plus temporal-spectrum/Doppler proxies at 30 CSI packets/s."""
    values = np.asarray(vector, dtype=np.float64)
    result = {
        "csi_amp_mean": float(np.mean(values)),
        "csi_amp_std": float(np.std(values)),
        "csi_amp_var": float(np.var(values)),
    }

    if previous is None:
        result["csi_temporal_diff"] = 0.0
        result["csi_motion_energy"] = 0.0
    else:
        previous_index, previous_values = previous
        delta_sec = max((pair_index - previous_index) / CSI_RATE_HZ, 1.0 / CSI_RATE_HZ)
        delta = values - previous_values
        result["csi_temporal_diff"] = float(np.mean(np.abs(delta)) / delta_sec)
        result["csi_motion_energy"] = float(np.mean(delta * delta) / delta_sec)

    history.append((int(pair_index), values.copy(), result["csi_amp_mean"]))
    recent = list(history)[-TEMPORAL_WINDOW_SAMPLES:]
    if len(recent) >= 2:
        matrix = np.stack([item[1] for item in recent])
        result["csi_subcarrier_var"] = float(np.mean(np.var(matrix, axis=0)))
    else:
        result["csi_subcarrier_var"] = 0.0

    # Re-sample irregularly selected packet indices onto a 30 Hz grid before FFT.
    if len(recent) >= 8 and recent[-1][0] - recent[0][0] >= 7:
        packet_indices = np.asarray([item[0] for item in recent], dtype=float)
        signal = np.asarray([item[2] for item in recent], dtype=float)
        grid = np.arange(max(packet_indices[0], packet_indices[-1] - (TEMPORAL_WINDOW_SAMPLES - 1)),
                         packet_indices[-1] + 1, dtype=float)
        sampled = np.interp(grid, packet_indices, signal)
        sampled = sampled - sampled.mean()
        spectrum = np.fft.rfft(sampled)
        power = np.abs(spectrum) ** 2
        frequencies = np.fft.rfftfreq(len(sampled), d=1.0 / CSI_RATE_HZ)
        if len(power) > 1:
            motion_power = power[1:]
            motion_freqs = frequencies[1:]
            total_power = float(motion_power.sum())
        else:
            motion_power = np.array([], dtype=float)
            motion_freqs = np.array([], dtype=float)
            total_power = 0.0
        if total_power > 1e-12:
            peak_index = int(np.argmax(motion_power))
            centroid = float(np.sum(motion_freqs * motion_power) / total_power)
            bandwidth = float(np.sqrt(np.sum(((motion_freqs - centroid) ** 2) * motion_power) / total_power))
            result["csi_spectral_energy"] = total_power
            result["csi_doppler_peak_hz"] = float(motion_freqs[peak_index])
            result["csi_doppler_centroid_hz"] = centroid
            result["csi_doppler_bandwidth_hz"] = bandwidth
            result["csi_doppler_peak_power"] = float(motion_power[peak_index])
        else:
            result["csi_spectral_energy"] = 0.0
            result["csi_doppler_peak_hz"] = 0.0
            result["csi_doppler_centroid_hz"] = 0.0
            result["csi_doppler_bandwidth_hz"] = 0.0
            result["csi_doppler_peak_power"] = 0.0
    else:
        result["csi_spectral_energy"] = 0.0
        result["csi_doppler_peak_hz"] = 0.0
        result["csi_doppler_centroid_hz"] = 0.0
        result["csi_doppler_bandwidth_hz"] = 0.0
        result["csi_doppler_peak_power"] = 0.0
    return result

def extract_csi(aligned: list[dict]) -> np.ndarray:
    """Tạo đúng 73 cột CSI từ các gói đã căn chỉnh; reset ở đầu đoạn Test."""
    history = deque(maxlen=TEMPORAL_WINDOW_SAMPLES)
    previous = None
    rows = []
    previous_index = None
    for item in aligned:
        pair_index = int(item['pair_index_30hz'])
        if previous_index is not None and pair_index != previous_index + 1:
            history.clear(); previous = None
        values = csi_amplitudes(item['csi_matrix'])
        if len(values) != 62 or not np.isfinite(values).all():
            raise ValueError(f'CSI pair {pair_index} có {len(values)} biên độ; cần 62 giá trị hữu hạn')
        summary = csi_temporal_features(values, pair_index, previous, history)
        row = dict(zip(CSI_NAMES[:62], values)) | summary
        rows.append([row[name] for name in CSI_NAMES])
        previous = (pair_index, np.asarray(values, dtype=np.float64))
        previous_index = pair_index
    result = np.asarray(rows, dtype=np.float32)
    if result.shape != (len(aligned), 73) or not np.isfinite(result).all():
        raise ValueError('Đặc trưng CSI không đúng chiều hoặc có NaN/Inf')
    return result


def preprocess_frame(frame_bgr: np.ndarray) -> torch.Tensor:
    """JPEG 90 rồi resize/crop 384, ImageNet mean/std như extractor gốc."""
    ok, encoded = cv2.imencode('.jpg', frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise ValueError('Không mã hóa được khung hình JPEG')
    with Image.open(io.BytesIO(encoded.tobytes())) as image:
        image = image.convert('RGB')
        width, height = image.size
        short = int(256.0 / 224.0 * 384)
        scale = short / min(width, height)
        image = image.resize((round(width*scale), round(height*scale)), Image.Resampling.BILINEAR)
        left = max(0, (image.width-384)//2)
        top = max(0, (image.height-384)//2)
        image = image.crop((left, top, left+384, top+384))
        array = np.asarray(image, dtype=np.float32)/255.0
    tensor = torch.from_numpy(array).permute(2,0,1).contiguous()
    mean = torch.tensor((.485,.456,.406),dtype=tensor.dtype)[:,None,None]
    std = torch.tensor((.229,.224,.225),dtype=tensor.dtype)[:,None,None]
    return (tensor-mean)/std


def load_vjepa(device: torch.device, model_dir: Path):
    source = model_dir/'source_meta'
    weights = model_dir/'vjepa2_1_vitb_dist_vitG_384.pt'
    if not (source/'hubconf.py').is_file() or not weights.is_file():
        raise FileNotFoundError(f'Cần mã nguồn Meta và checkpoint V-JEPA trong {model_dir}')
    loaded = torch.hub.load(str(source), 'vjepa2_1_vit_base_384', source='local',
                            pretrained=False, trust_repo=True, num_frames=8)
    encoder = loaded[0] if isinstance(loaded,(tuple,list)) else loaded
    state = torch.load(weights,map_location='cpu',weights_only=True)
    if 'ema_encoder' not in state:
        raise KeyError(f'Checkpoint V-JEPA thiếu ema_encoder: {weights}')
    clean = {}
    for key,value in state['ema_encoder'].items():
        for prefix in ('module.','backbone.'):
            while key.startswith(prefix):
                key=key[len(prefix):]
        clean[key]=value
    encoder.load_state_dict(clean,strict=True)
    encoder=encoder.to(device).eval()
    encoder.requires_grad_(False)
    return encoder


def pool_output(output) -> np.ndarray:
    if isinstance(output,(tuple,list)):
        tensors=[value for value in output if torch.is_tensor(value)]
        if not tensors:
            raise TypeError('V-JEPA không trả tensor')
        output=tensors[-1]
    if output.ndim==2:
        pooled=output
    elif output.ndim>=3:
        pooled=output.reshape(output.shape[0],-1,output.shape[-1]).mean(dim=1)
    else:
        raise ValueError(f'Shape V-JEPA không hỗ trợ: {tuple(output.shape)}')
    return pooled.float().cpu().numpy()


def extract_rgb(video: Path, aligned: list[dict], encoder, device: torch.device) -> np.ndarray:
    """Mã hóa clip 8 frame không chồng lấn; vector clip gán cho 8 dòng."""
    indices=[int(item['frame_index']) for item in aligned]
    if indices!=list(range(len(indices))):
        raise ValueError('Đoạn video cần một frame liên tiếp cho mỗi dòng CSI đã căn chỉnh')
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f'Không mở được video: {video}')
    vectors=np.empty((len(aligned),768),dtype=np.float32)
    try:
        for start in range(0,len(aligned),8):
            frames=[]
            for _ in range(min(8,len(aligned)-start)):
                ok,frame=cap.read()
                if not ok:
                    raise ValueError(f'Video kết thúc trước frame {start+len(frames)}')
                frames.append(preprocess_frame(frame))
            if not frames:
                break
            while len(frames)<8:
                frames.append(frames[-1])
            clip=torch.stack(frames,dim=1).unsqueeze(0).to(device)
            with torch.inference_mode():
                vector=pool_output(encoder(clip))
            if vector.shape!=(1,768) or not np.isfinite(vector).all():
                raise ValueError(f'Embedding V-JEPA sai shape/giá trị: {vector.shape}')
            vectors[start:min(start+8,len(aligned))]=vector[0]
    finally:
        cap.release()
    return vectors


def make_windows(n: int, steps: int = 16, stride: int = 16) -> np.ndarray:
    return np.asarray([np.arange(start,start+steps,dtype=np.int32)
                       for start in range(0,n-steps+1,stride)],dtype=np.int32).reshape(-1,steps)


def normalize_for_model(csi: np.ndarray, rgb: np.ndarray, dataset: str) -> tuple[np.ndarray, np.ndarray]:
    """Áp dụng đúng mean/std của tập Train, theo thứ tự cột lúc huấn luyện."""
    stats_path = WEB_ROOT / 'models' / dataset / 'normalization.npz'
    if not stats_path.is_file():
        raise FileNotFoundError(f'Thiếu thống kê chuẩn hóa từ Train: {stats_path}')
    with np.load(stats_path, allow_pickle=False) as stats:
        names = stats['feature_names'].tolist()
        mean = stats['mean']
        scale = stats['scale']
    if names != CSI_NAMES + RGB_NAMES:
        raise ValueError(f'Thứ tự đặc trưng trong {stats_path} khác đầu vào hiện tại')
    raw = np.concatenate((csi, rgb), axis=1)
    if raw.shape[1] != len(mean) or len(mean) != len(scale):
        raise ValueError(f'Số chiều thống kê chuẩn hóa không khớp: {stats_path}')
    normalized = (np.where(np.isfinite(raw), raw, mean) - mean) / scale
    normalized = normalized.astype(np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError('Đặc trưng sau chuẩn hóa có NaN/Inf')
    return normalized[:, :len(CSI_NAMES)], normalized[:, len(CSI_NAMES):]


def feature_pair(video: Path, csi_csv: Path, *, device: str = 'auto',
                 reference_meta: Path | None = None,
                 model_dir: Path | None = None,
                 dataset: str | None = None) -> dict:
    if dataset is None:
        location = infer_test_location(Path(video))
        dataset = location[0] if location else None
    if dataset not in ('DataSoluong', 'DataHanhDong'):
        raise ValueError('Cần --dataset DataSoluong hoặc DataHanhDong cho video tải lên')
    aligned=align_pair(video,csi_csv,reference_meta)
    csi=extract_csi(aligned)
    if device=='auto':
        device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    torch_device=torch.device(device)
    encoder=load_vjepa(torch_device,model_dir or WEB_ROOT/'models/vjepa21')
    rgb=extract_rgb(Path(video),aligned,encoder,torch_device)
    model_csi,model_rgb=normalize_for_model(csi,rgb,dataset)
    windows=make_windows(len(aligned))
    if not len(windows):
        raise ValueError(f'Đoạn chỉ có {len(aligned)} cặp; cần ít nhất 16')
    result={'csi':csi,'rgb':rgb,'model_csi':model_csi,'model_rgb':model_rgb,
            'dataset':dataset,'windows':windows,
            'pair_index_30hz':np.array([int(x['pair_index_30hz']) for x in aligned],dtype=np.int32),
            'timestamp_sec':np.array([float(x['timestamp_sec']) for x in aligned],dtype=np.float64),
            'alignment_mode':aligned[0]['alignment_mode']}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',type=Path,required=True)
    p.add_argument('--csi',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference-meta',type=Path)
    p.add_argument('--device',choices=('auto','cuda','mps','cpu'),default='auto')
    p.add_argument('--model-dir',type=Path)
    p.add_argument('--dataset',choices=('DataSoluong','DataHanhDong'),
                   help='Bắt buộc cho video tải lên ngoài các cặp Test đã biết')
    a=p.parse_args()
    result=feature_pair(a.video,a.csi,device=a.device,reference_meta=a.reference_meta,
                        model_dir=a.model_dir,dataset=a.dataset)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,**result)
    print(f'Đã tạo {len(result["csi"])} cặp, {len(result["windows"])} cửa sổ 16/16; '
          f'CSI={result["csi"].shape[1]}, RGB={result["rgb"].shape[1]}, '
          f'mode={result["alignment_mode"]}; output={a.output}')

if __name__=='__main__':
    main()
