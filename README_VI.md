# Số Hóa Thư Giấy — Letter Processing Worker

[Tiếng Việt](README_VI.md) | [English](README.md)

Hệ thống xử lý ảnh và worker nền tảng (standalone, deterministic) chuyên dụng phục vụ việc số hóa, phục chế ảnh chụp thư tay, phát hiện biên dạng giấy, nắn thẳng phối cảnh 3D, khử bóng đổ không đồng đều, làm sạch trang giấy và lọc bỏ nét chữ lem từ mặt sau.

Được thiết kế **tách biệt hoàn toàn với OCR và nhận dạng văn bản**, chỉ tập trung chuyên sâu vào thị giác máy tính (computer vision) và phục chế hình ảnh chất lượng cao.

> 📘 **Tài liệu hướng dẫn chi tiết & code mẫu tích hợp**: Xem tại [USAGE.md](USAGE.md).

---

## Minh Họa Kết Quả (Trước & Sau Phục Chế)

| 📷 Ảnh chụp gốc (Before) | ✨ Bản phục chế Vintage (After) |
| :---: | :---: |
| <img src="assets/before.jpg" alt="Ảnh chụp ban đầu" width="360" /> | <img src="assets/after.jpg" alt="Bản scan phục chế" width="360" /> |
| *Ảnh chụp nghiêng, bóng tối loang lổ, chữ hằn trang sau* | *Nắn thẳng phối cảnh 3D, sạch nền, giữ sắc ngà & vân giấy* |

---

## 1. Kiến Trúc & Quy Trình 7 Giai Đoạn (Pipeline Overview)

```
[Ảnh Chụp Thô]
    │
    ▼
01. Đọc & Kiểm Tra An Toàn (Xoay EXIF, chặn bom giải nén, kiểm tra dung lượng)
    │
    ▼
02. Dò Biên Dạng Giấy (Multi-pass Canny + Otsu, xếp hạng độ lồi & diện tích)
    │
    ├─► Độ tin cậy < 0.80 ──► [awaiting_corners] (Giao diện yêu cầu người dùng xác nhận 4 góc)
    │                                │ (Người dùng kéo chỉnh 4 góc)
    │                                └─────────┐
    ▼                                          ▼
03. Nắn Thẳng Phối Cảnh 3D (Thuật toán Zhang 2003, khôi phục tỷ lệ vật lý thực tế)
    │
    ▼
04. Khử Bóng Đổ Nền Giấy (Morphological Dilation trích xuất trường sáng không hào quang)
    │
    ▼
05. Tách Nét Chữ & Phục Chế Màu (Lọc chữ lem trang sau, tái tạo màu theo Preset)
    │
    ▼
06. Kiểm Định Chất Lượng QA (Độ nhòe Laplacian, độ phủ mực, tính toàn vẹn)
    │
    ▼
07. Lưu Trữ Nguyên Tử (Atomic Persist, chống ghi đè lỗi, kèm metadata JSON)
```

---

## 2. Các Phong Cách Phục Chế (Restoration Presets)

| Preset | Mô tả | Định dạng xuất | Phù hợp nhất |
|---|---|---|---|
| **Pure B&W** (`pure_bw`) | Nhị phân hóa tương phản cao (Adaptive Gaussian hoặc Otsu), nền trắng tinh (255) và nét mực đen tuyền (0). | PNG 2D Nhị phân | Văn bản photocopy, bản in sạch, tài liệu lưu trữ |
| **Natural** (`natural`) | Giữ đa mức xám tự nhiên, làm sạch đều nền giấy nhưng giữ trọn sắc độ đậm nhạt của bút mực, bút chì. | PNG 2D Đơn sắc | Thư viết tay thông thường, nét bút chì thanh mảnh |
| **Vintage** (`vintage`) | Giấy màu vàng ngà hoài niệm (`#DACEBD`), nét mực sepia nâu đen truyền thống (`#2A2016`), giữ vi vân giấy thật. | PNG 3D Màu (RGB) | Thư tay kỷ niệm, thư cổ, tạo phong vị hoài cổ chân thực |

---

## 3. Bảng Tra Cứu Tham Số Cấu Hình

### 3.1 Cài Đặt Phục Chế (`RestoreSettings`)

| Tham số | Kiểu dữ liệu | Giá trị hợp lệ | Mặc định | Ý nghĩa & Ứng dụng |
|---|---|---|---|---|
| `preset` | string | `"pure_bw"`, `"natural"`, `"vintage"` | `"vintage"` | Phong cách thẩm mỹ đầu ra. |
| `aspect_ratio` | string | `"auto"`, `"a4"`, `"notebook"`, `"letter"`, `"free"` | `"auto"` | **Tỷ lệ khổ giấy**: `"auto"` dùng hình học 3D (Zhang 2003) khôi phục tỷ lệ thực, triệt tiêu hiện tượng dẹt/kéo dãn ngang khi chụp góc nghiêng. |
| `paper_warmth` | float | `0.0` đến `1.0` | `0.75` | **Sắc ngà giấy thư** *(dành cho preset vintage)*: `0.0` (kem sáng), `0.75` (vàng ngà nguyên bản), `1.0` (giấy cổ đậm đà). |
| `bleed_through_reduction` | float | `0.50` đến `0.98` | `0.82` | **Lọc chữ hằn trang sau**: Tự động triệt tiêu nét mực mờ in hằn từ mặt sau của giấy tập học sinh mỏng. |
| `margin_inset` | float | `0.0` đến `0.10` | `0.0` | **Xén mép lề giấy**: Cắt thụt lề vào trong sau khi nắn thẳng (ví dụ `0.02` = 2%), giúp loại bỏ lỗ gáy xoắn lò xo hoặc bóng mép bàn. |
| `denoise` | string | `"off"`, `"light"`, `"medium"` | `"light"` | Mức độ khử nhiễu hạt: `"light"` bảo vệ nét mực mảnh; `"medium"` khử hạt mạnh cho ảnh chụp thiếu sáng. |
| `threshold_method` | string | `"adaptive"`, `"otsu"` | `"adaptive"` | Thuật toán phân ngưỡng *(chỉ áp dụng cho `pure_bw`)*. |
| `threshold_strength` | float | `0.0` đến `1.0` | `0.5` | Độ đậm nét chữ: giá trị thấp nét đậm đà; giá trị cao nét thanh mảnh. |

---

### 3.2 Biến Môi Trường & Cấu Hình Worker (`WorkerConfig`)

| Biến môi trường | Kiểu | Mặc định | Mô tả |
|---|---|---|---|
| `STORAGE_ROOT` | string / path | `.` | Thư mục gốc sandbox, chống tấn công duyệt thư mục (`../`). |
| `DATABASE_PATH` | string / path | `letter_worker.db` | Đường dẫn file cơ sở dữ liệu hàng đợi SQLite. |
| `MAX_IMAGE_FILE_SIZE_BYTES` | integer | `52428800` (50 MB) | Dung lượng file tối đa cho phép tải lên. |
| `MAX_IMAGE_PIXELS` | integer | `50000000` (50 MP) | Giới hạn số lượng pixel tối đa phòng chống bom giải nén (Decompression Bomb). |
| `MAX_OUTPUT_DIMENSION` | integer | `8192` | Kích thước cạnh dài tối đa của ảnh đầu ra (tự động thu nhỏ tỷ lệ nếu vượt quá). |
| `CONFIDENCE_AUTO_ACCEPT` | float | `0.80` | Độ tin cậy ≥ 0.80 sẽ tự động nắn thẳng mà không dừng hỏi người dùng. |
| `CONFIDENCE_AWAIT_CONFIRMATION` | float | `0.50` | Độ tin cậy < 0.80 sẽ tạm dừng ở `awaiting_corners` để giao diện hiển thị 4 góc gợi ý. |
| `JOB_LEASE_TIMEOUT_SECONDS` | integer | `300` (5 phút) | Thời gian hết hạn lease để tự động giải phóng job nếu worker bị crash đột ngột. |
| `JOB_MAX_ATTEMPTS` | integer | `3` | Số lần thử lại tối đa khi gặp lỗi bất định trước khi đánh dấu thất bại. |

---

## 4. Các Mô Hình Tích Hợp (Integration Patterns)

### Mô hình A: Dùng Trực Tiếp Như Thư Viện Python (In-Memory / Fast Execution)

Phù hợp cho các hệ thống nhúng, API FastAPI, Celery worker, script tự động hóa hoặc Jupyter Notebook:

```python
from letter_worker import process_letter

# Phục chế ảnh nhanh trong bộ nhớ (hỗ trợ đường dẫn file, PIL Image hoặc NumPy array)
restored_rgb, metadata = process_letter(
    image="uploads/letter.jpg",
    preset="vintage",              # "pure_bw", "natural", "vintage"
    aspect_ratio="auto",           # "auto", "notebook", "a4", "letter", "free"
    paper_warmth=0.75,             # 0.75: vàng ngà hoài niệm
    bleed_through_reduction=0.82,  # Khử chữ lem trang sau
    margin_inset=0.01,             # Xén 1% mép viền loại bỏ gáy lò xo
    output_path="outputs/clean.png", # (Tùy chọn) Lưu file tự động
)

print("Kích thước đầu ra:", metadata["output_dimensions"])
print("Độ sắc nét:", metadata["quality"]["blur_score"])
print("Trạng thái QA:", metadata["quality"]["status"])
```

---

### Mô hình B: Hàng Đợi SQLite & Worker Daemon Đa Tiến Trình (Multi-process Safe)

Dành cho kiến trúc Microservices nơi Backend (Node.js/Express, Next.js, Go, PHP, Java) giao tiếp bất đồng bộ với Python Worker qua SQLite WAL:

1. **Backend đẩy job vào SQLite**:
   ```sql
   INSERT INTO jobs (job_id, status, stage, progress, input_path, output_dir, settings_json, created_at)
   VALUES ('job_001', 'queued', 'queued', 0, 'uploads/letter.jpg', 'outputs', '{"preset":"vintage","aspect_ratio":"auto"}', datetime('now'));
   ```
2. **Worker Daemon tự động nhận job, cập nhật tiến trình và xử lý**:
   Worker sử dụng cơ chế transaction `BEGIN IMMEDIATE` đảm bảo không bao giờ bị claim trùng giữa nhiều worker chạy song song.
3. **Frontend Polling tiến trình**:
   API đọc cột `stage`, `progress`, `message_code` để hiển thị trạng thái động cho người dùng.

---

### Mô hình C: Gọi Qua Lệnh CLI Subprocess

```bash
# Phục chế trực tiếp 1 file ảnh
letter-worker process photo.jpg --output-dir ./outputs --preset vintage --aspect-ratio auto
```

Kết quả in ra stdout dạng JSON chuẩn:
```json
{
  "job_id": "cli_1773901234",
  "status": "completed",
  "output_path": "outputs/cli_1773901234_vintage.png",
  "metadata_path": "outputs/cli_1773901234_metadata.json",
  "confidence": 0.89,
  "quality": {
    "status": "passed",
    "blur_score": 142.5,
    "ink_coverage": 0.084,
    "width": 2620,
    "height": 3668
  },
  "duration_seconds": 1.15
}
```

---

## 5. Hướng Dẫn Cài Đặt Nhanh

### Yêu cầu hệ thống
* Python 3.11 trở lên
* Trình quản lý gói [uv](https://docs.astral.sh/uv/) (hoặc pip tiêu chuẩn)

### Chạy kiểm thử tự động (52 unit & integration tests)
```bash
uv run pytest -v
```

### Các lệnh CLI chính
```bash
# Khởi động Worker Daemon
uv run python main.py worker --db ./letter_worker.db --storage .

# Khởi tạo bảng dữ liệu SQLite
uv run python main.py init-db --db ./letter_worker.db

# Đưa một job vào hàng đợi
uv run python main.py enqueue job_001 photo.jpg --output-dir ./outputs --preset vintage --db ./letter_worker.db

# Kiểm tra trạng thái job
uv run python main.py status job_001 --db ./letter_worker.db

# Phục chế trực tiếp không qua hàng đợi
uv run python main.py process photo.jpg --output-dir ./outputs --preset vintage
```

---

## 6. Đóng Gói & Triển Khai Với Docker

Dự án đi kèm Dockerfile chuẩn Production chạy dưới quyền non-root user (`USER worker`):

```bash
# Build Docker image
docker build -t letter-worker:latest .

# Chạy container gắn volume dữ liệu
docker run -d \
  --name letter-worker \
  -v /path/to/data:/app/data \
  -e STORAGE_ROOT=/app/data \
  -e DATABASE_PATH=/app/data/letter_worker.db \
  letter-worker:latest
```

---

## 7. Giấy Phép & Bản Quyền

Phát triển bởi Kunniii. Mã nguồn được phân phối dưới giấy phép MIT.
