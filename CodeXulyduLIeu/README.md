# Chuẩn bị đầu vào cho web demo

Thư mục này nhận một video MP4 và tệp CSI CSV cùng phiên ghi. Hai tệp Python chỉ xử lý dữ liệu, không huấn luyện và không chia lại Train, Validation, Test.

| Tệp | Chức năng |
| --- | --- |
| `DongBo.py` | Ghép từng khung hình với một gói CSI. Với đoạn Test trong `../Data/`, đối chiếu `record_id`, chỉ số cặp, `pkt_id` và timestamp với `reference_meta/*_meta_test.csv`. Với video mới không có metadata, ghép ước lượng theo mốc thời gian của CSV. |
| `Trich_feature.py` | Tạo 73 đặc trưng CSI và 768 embedding RGB V-JEPA 2.1, rồi chia thành cửa sổ 16 mẫu, bước 16. Xuất cả đặc trưng thô và đặc trưng đã chuẩn hóa cho model. |
| `reference_meta/` | Metadata của các đoạn Test đã chọn, dùng để ghép đúng video và CSI. Không chứa nhãn để đưa vào suy luận. |

Trọng số V-JEPA và mã nguồn Meta dùng để nạp model ngoại tuyến nằm ở `../CSI_VLFormer_Local_Web/models/vjepa21/`. Mỗi tác vụ có `normalization.npz` trong thư mục model riêng, chứa trung bình và độ lệch chuẩn tính từ đúng tập Train của 841 cột. Không dùng thống kê Test để chuẩn hóa.

Ví dụ với một cặp Test:

```bash
python Trich_feature.py --video "/duong/dan/test_01_000964_001323/video.mp4" --csi "/duong/dan/test_01_000964_001323/csi.csv" --output "/duong/dan/features.npz" --device mps
```

Với video tải lên ngoài `../Data/`, thêm `--dataset DataSoluong` hoặc `--dataset DataHanhDong`. Khi không có metadata chung, `alignment_mode` trong tệp kết quả là `estimated_from_two_files`. Cần hai nguồn cùng phiên và CSV có cột `timestamp_sec`, `pkt_id`, `csi_matrix`.

Trong tệp NPZ: `csi` và `rgb` là đặc trưng thô; `model_csi` và `model_rgb` là đặc trưng đã chuẩn hóa; `windows` chứa vị trí các cửa sổ 16/16; `pair_index_30hz`, `timestamp_sec`, `alignment_mode` cho biết cách ghép. Mã suy luận web cần dùng `model_csi` và `model_rgb` để đưa vào Fusion và Qwen.

Mã trong `CSI_VLFormer_Local_Web/inference/` hiện chưa được nối với hai tệp này. Cần hoàn thiện phần đó trước khi web dự đoán được từ video mới.
