# Các mô hình dùng cho web demo

Thư mục này giữ trọng số của hai tác vụ DataSoluong và DataHanhDong. Một phiên suy luận cần V-JEPA 2.1 để tạo đặc trưng Video RGB, đặc trưng CSI được trích xuất theo đúng quy trình huấn luyện, khối Fusion và Perceiver, rồi Qwen3-VL gốc cùng LoRA để tạo câu trả lời. Hiện thư mục Qwen3-VL gốc vẫn chưa có trọng số.

## Thư mục chung

| Thư mục hoặc tệp | Chức năng |
| --- | --- |
| `Qwen3-VL-2B-Instruct/` | Vị trí đặt **toàn bộ model Qwen3-VL-2B-Instruct gốc**. Hiện chỉ là chỗ trống; LoRA không thay thế model gốc. |
| `vjepa21/vjepa2_1_vitb_dist_vitG_384.pt` | Trọng số V-JEPA 2.1 ViT-B dùng để tạo embedding từ clip Video RGB. Đây chỉ là bộ mã hóa RGB, không hợp nhất RGB với CSI. |
| `vjepa21/source_meta/` | Mã nguồn Meta dùng để nạp V-JEPA ngoại tuyến; đi cùng checkpoint trên. |
| `shared_processor/` | Bản sao processor Qwen3-VL từ DataSoluong, dùng chung khi phù hợp. `SOURCE.txt` ghi nguồn bản sao. Hai tác vụ cũng có processor trong thư mục riêng. |

## DataSoluong và DataHanhDong

Hai thư mục tác vụ có vai trò giống nhau. Mã web phải chọn đúng thư mục theo tác vụ đang dự đoán.

| Tệp hoặc thư mục trong mỗi tác vụ | Chức năng |
| --- | --- |
| `fusion_best_checkpoint.pt` | Checkpoint Fusion được huấn luyện trước bước Qwen. Dùng làm nguồn khởi tạo hoặc đối chiếu. Khi suy luận với Qwen đã fine-tune, ưu tiên trạng thái Fusion trong `best_checkpoint.pt` của cùng tác vụ. |
| `best_checkpoint.pt` | Checkpoint Qwen tốt nhất theo Validation. Chứa trạng thái LoRA, Fusion, teacher và soft projector, cùng thông tin huấn luyện. **Không chứa trọng số Qwen3-VL gốc.** |
| `best_lora_adapter/adapter_model.safetensors` | Trọng số LoRA xuất riêng để nạp bằng PEFT. Tương ứng với `lora_state` trong `best_checkpoint.pt`; không cần nạp cả hai bản LoRA cùng lúc. |
| `best_lora_adapter/adapter_config.json` | Cấu hình LoRA, gồm rank, alpha, các lớp được gắn adapter và đường dẫn model gốc tại thời điểm xuất. Đường dẫn Kaggle trong cấu hình cần được trỏ sang model Qwen gốc ở máy chạy web. |
| `processor/` | Tokenizer, mẫu hội thoại và cấu hình xử lý ảnh/video cho Qwen. Chuẩn bị đầu vào trước khi gọi model; không chứa trọng số học được. |
| `training_config.json` | Ghi cấu hình của lượt huấn luyện, số đặc trưng, đường dẫn nguồn và các tham số. Các đường dẫn Kaggle là đường dẫn cũ, không dùng nguyên xi trên máy local. |
| `normalization.npz` | Trung bình, độ lệch chuẩn và thứ tự 841 cột tính từ tập Train của đúng tác vụ. Dùng để chuẩn hóa 73 CSI và 768 RGB khi suy luận. |

### Tệp riêng của DataSoluong

| Tệp | Chức năng |
| --- | --- |
| `best_multimodal_projector.pt` | Bản xuất riêng của Fusion và soft projector tại epoch tốt nhất. Nội dung cần thiết cũng có trong `best_checkpoint.pt`; giữ để đối chiếu hoặc nạp riêng khi cần. |
| `best_lora_adapter/README.md` | Tài liệu do PEFT tạo kèm adapter. |

### Tệp riêng của DataHanhDong

| Tệp | Chức năng |
| --- | --- |
| `test_summary.json` | Tóm tắt đánh giá trên tập Test của lượt DataHanhDong. Đây là báo cáo, không phải trọng số mô hình. |

Processor và adapter trong `DataHanhDong/` được **xuất lại sau khi huấn luyện** từ model Qwen gốc và `best_checkpoint.pt`, rồi chép vào đây. Adapter đã được đối chiếu với 112 tensor LoRA trong checkpoint; processor đã được kiểm tra nạp thành công. Chúng không nằm trong gói kết quả đầu tiên tải về.

## Các tệp trong processor

| Tệp | Chức năng |
| --- | --- |
| `tokenizer.json`, `vocab.json`, `merges.txt` | Từ vựng và quy tắc biến văn bản thành token. |
| `tokenizer_config.json`, `special_tokens_map.json`, `added_tokens.json` | Cấu hình tokenizer và các token đặc biệt hoặc bổ sung. |
| `chat_template.jinja` | Định dạng hội thoại và câu hỏi đưa vào Qwen. |
| `preprocessor_config.json` | Cấu hình xử lý ảnh. |
| `video_preprocessor_config.json` | Cấu hình xử lý đầu vào video của processor. |

## Thứ tự nạp khi hoàn thiện web

1. Từ MP4, dùng V-JEPA 2.1 tạo embedding RGB; từ CSV, tạo 73 đặc trưng CSI và căn chỉnh theo thời gian.
2. Nạp Qwen3-VL-2B-Instruct gốc và processor.
3. Chọn checkpoint đúng tác vụ. Nạp LoRA, Fusion và soft projector từ `best_checkpoint.pt`, hoặc dùng adapter xuất riêng cho phần LoRA.
4. Tạo đầu ra từ ảnh RGB và các token đa phương thức. Không dùng nhãn thật làm đầu vào suy luận.

Các file mô hình đã có, nhưng mã trong `inference/` của web hiện mới là khung. Cần nối và kiểm tra toàn bộ quy trình trước khi dùng để dự đoán.
