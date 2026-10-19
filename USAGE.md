# Hướng Dẫn Sử Dụng & Tích Hợp — Letter Processing Worker

Tài liệu này hướng dẫn chi tiết cách sử dụng trực tiếp và cách tích hợp Python Worker này vào hệ thống của bạn (Backend API, Frontend UI, Docker).

---

## Mục Lục
1. [Chạy thử nghiệm nhanh (One-shot CLI)](#1-chay-thu-nghiem-nhanh-one-shot-cli)
2. [Chạy chế độ Worker Daemon & Hàng đợi SQLite](#2-chay-che-do-worker-daemon--hang-doi-sqlite)
3. [Tích hợp với Backend API (Node.js, Python FastAPI, Go)](#3-tich-hop-voi-backend-api)
4. [Quy trình phối hợp với Frontend UI (Xác nhận 4 góc giấy)](#4-quy-trinh-phoi-hop-voi-frontend-ui)
5. [Bảng tra cứu chi tiết các thông số (Parameters)](#5-bang-tra-cuu-chi-tiet-cac-thong-so)
6. [Bảng mã lỗi & Bảng tiến trình giao diện](#6-bang-ma-loi--bang-tien-trinh-giao-dien)
7. [Triển khai với Docker](#7-trien-khai-voi-docker)

---

## 1. Chạy thử nghiệm nhanh (One-shot CLI)

Chế độ này xử lý trực tiếp 1 ảnh mà **không cần chạy database SQLite** hay Worker daemon. Phù hợp để kiểm tra thuật toán hoặc tích hợp dạng CLI script.

### 1.1 Lệnh cơ bản
```bash
# Phục chế theo phong cách tự nhiên (Natural)
uv run python main.py process input.jpg --output-dir ./output --preset natural

# Phục chế theo phong cách thư cổ (Vintage: giấy ngà, mực sepia)
uv run python main.py process input.jpg --output-dir ./output --preset vintage

# Phục chế trắng đen tương phản cao (Pure B&W)
uv run python main.py process input.jpg --output-dir ./output --preset pure_bw
```

### 1.2 Chỉ định sẵn 4 góc giấy bằng tay
Nếu bạn đã có tọa độ 4 góc (thứ tự theo chiều kim đồng hồ: `[Trên-Trái, Trên-Phải, Dưới-Phải, Dưới-Trái]`):
```bash
uv run python main.py process input.jpg --output-dir ./output --preset pure_bw \
  --corners "[[120, 80], [780, 110], [750, 690], [100, 640]]"
```

### 1.3 Tùy chỉnh thông số khử nhiễu và độ đậm nét mực
```bash
# Tăng nét chữ đậm hơn với threshold-strength = 0.3
uv run python main.py process input.jpg --output-dir ./output \
  --preset pure_bw \
  --denoise light \
  --threshold-method adaptive \
  --threshold-strength 0.3
```

Kết quả in ra màn hình ở dạng JSON chuẩn:
```json
{
  "job_id": "cli_1773901234",
  "status": "completed",
  "output_path": "output/cli_1773901234_pure_bw.png",
  "metadata_path": "output/cli_1773901234_metadata.json",
  "confidence": 0.88,
  "quality": {
    "status": "passed",
    "blur_score": 142.5,
    "ink_coverage": 0.084,
    "width": 1240,
    "height": 1754
  },
  "duration_seconds": 0.84
}
```

---

## 2. Chạy chế độ Worker Daemon & Hàng đợi SQLite

Dùng cho môi trường dịch vụ (production/staging). Backend API đẩy job vào database, Worker chạy ngầm để lấy job và xử lý tự động.

### Bước 1: Khởi tạo database
```bash
uv run python main.py init-db --db ./letter_worker.db
```

### Bước 2: Đẩy một job vào hàng đợi (Enqueue)
```bash
uv run python main.py enqueue job_001 input.jpg --output-dir ./output --preset natural --db ./letter_worker.db
```

### Bước 3: Xem trạng thái hiện tại của job
```bash
uv run python main.py status job_001 --db ./letter_worker.db
```

### Bước 4: Khởi động Worker Daemon
```bash
uv run python main.py worker --db ./letter_worker.db --storage .
```
Worker sẽ chạy liên tục, tự động claim job, cập nhật tiến độ (progress/stage), gia hạn heartbeat và tự động thu hồi các job bị treo do sự cố.

---

## 3. Tích hợp với Backend API

### 3.1 Cấu trúc bảng `jobs` trong SQLite
Backend API (Node.js, Next.js, Go, Python) chỉ cần kết nối tới file `letter_worker.db`:

```sql
CREATE TABLE jobs (
    job_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,         -- 'queued', 'loading', 'detecting', 'awaiting_corners', 'rectifying', 'restoring', 'quality_check', 'completed', 'failed'
    stage TEXT,                   -- Công đoạn hiện tại
    progress INTEGER DEFAULT 0,   -- 0 đến 100
    message_code TEXT,            -- Mã tiến trình cho UI
    input_path TEXT NOT NULL,     -- Đường dẫn file ảnh gốc (tương đối theo STORAGE_ROOT)
    output_dir TEXT NOT NULL,     -- Thư mục xuất kết quả
    output_path TEXT,             -- Đường dẫn file ảnh đã phục chế
    metadata_path TEXT,           -- Đường dẫn file JSON metadata
    corners_json TEXT,            -- JSON mảng 4 góc: [{"x": 100, "y": 80}, ...]
    settings_json TEXT,           -- JSON cấu hình: {"preset": "vintage", ...}
    error_code TEXT,              -- Mã lỗi nếu thất bại
    error_message TEXT,           -- Chi tiết lỗi
    attempts INTEGER DEFAULT 0,
    max_attempts INTEGER DEFAULT 3,
    worker_id TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    heartbeat_at TEXT,
    finished_at TEXT
);
```

---

### 3.2 Ví dụ tích hợp với Node.js / Express (hoặc Next.js API Route)

Dùng thư viện `better-sqlite3`:

```javascript
import Database from 'better-sqlite3';
const db = new Database('letter_worker.db');

// 1. API: Tạo job mới khi user upload ảnh
app.post('/api/jobs', (req, res) => {
  const jobId = `job_${Date.now()}`;
  const inputRelativePath = `uploads/${req.file.filename}`;
  const settings = JSON.stringify({
    preset: req.body.preset || 'natural',
    denoise: 'light',
    threshold_strength: 0.5
  });

  const stmt = db.prepare(`
    INSERT INTO jobs (
      job_id, status, stage, progress, message_code,
      input_path, output_dir, settings_json, max_attempts, created_at
    ) VALUES (?, 'queued', 'queued', 0, 'QUEUED', ?, 'outputs', ?, 3, datetime('now'))
  `);

  stmt.run(jobId, inputRelativePath, settings);
  res.json({ success: true, jobId });
});

// 2. API: Lấy trạng thái tiến độ job (dùng cho Polling hoặc SSE)
app.get('/api/jobs/:jobId', (req, res) => {
  const row = db.prepare('SELECT * FROM jobs WHERE job_id = ?').get(req.params.jobId);
  if (!row) return res.status(404).json({ error: 'Job not found' });

  res.json({
    jobId: row.job_id,
    status: row.status,
    progress: row.progress,
    stage: row.stage,
    messageCode: row.message_code,
    outputPath: row.output_path,
    corners: row.corners_json ? JSON.parse(row.corners_json) : null
  });
});

// 3. API: Người dùng xác nhận 4 góc giấy (khi job ở trạng thái awaiting_corners)
app.post('/api/jobs/:jobId/confirm-corners', (req, res) => {
  const cornersJson = JSON.stringify(req.body.corners); // [{x, y}, {x, y}, {x, y}, {x, y}]
  
  db.prepare(`
    UPDATE jobs
    SET status = 'queued',
        stage = 'queued',
        progress = 0,
        message_code = 'CORNERS_CONFIRMED',
        corners_json = ?
    WHERE job_id = ?
  `).run(cornersJson, req.params.jobId);

  res.json({ success: true, message: 'Corners confirmed, job re-queued' });
});
```

---

### 3.3 Ví dụ tích hợp với Python FastAPI (trực tiếp không qua SQLite daemon)

```python
from fastapi import FastAPI, UploadFile, File
from pathlib import Path
from letter_worker.config import WorkerConfig
from letter_worker.processor import LetterProcessor
from letter_worker.schemas import LetterJob, RestoreSettings

app = FastAPI()
processor = LetterProcessor(WorkerConfig(storage_root="./storage"))

@app.post("/restore")
async def restore_letter(file: UploadFile = File(...), preset: str = "vintage"):
    upload_path = Path("storage/uploads") / file.filename
    upload_path.parent.mkdir(parents=True, exist_ok=True)
    with open(upload_path, "wb") as f:
        f.write(await file.read())

    job = LetterJob(
        job_id=f"job_{file.filename}",
        input_path=f"uploads/{file.filename}",
        output_dir="outputs",
        settings=RestoreSettings(preset=preset)
    )

    result = processor.process(job)
    return result.model_dump()
```

---

### 3.4 Dùng như thư viện Python độc lập (`process_letter`)

Nếu bạn đang viết script Python, xử lý hàng loạt (batch job), hay chạy trong Jupyter Notebook, hàm `process_letter` cho phép phục chế ảnh trong bộ nhớ (in-memory) chỉ với 1 dòng lệnh:

```python
from letter_worker import process_letter

# Nhận file ảnh, numpy array, hoặc PIL Image và trả về (restored_rgb, metadata)
restored_img, meta = process_letter(
    image="input.jpg",
    preset="vintage",              # "pure_bw", "natural", "vintage"
    aspect_ratio="auto",           # "auto", "a4", "notebook", "letter", "free"
    paper_warmth=0.75,             # 0.0 (kem sáng) -> 0.75 (vàng ngà) -> 1.0 (cổ đậm)
    bleed_through_reduction=0.82,  # Lọc chữ in hằn mờ từ mặt sau
    margin_inset=0.01,             # Cắt bớt viền mép giấy và lỗ xoắn lò xo
    output_path="output/clean.png" # (Tùy chọn) Lưu file tự động
)

print("Kích thước xuất:", meta["output_dimensions"])
print("Độ sắc nét:", meta["quality"]["blur_score"])
```

---

## 4. Quy trình phối hợp với Frontend UI

Khi người dùng chụp một bức thư bằng điện thoại, giấy thư có thể bị xiên hoặc nền chụp phức tạp:

```
[Người dùng tải ảnh lên]
           │
           ▼
[Worker tự động dò 4 góc giấy]
           │
  ┌────────┴─────────────────────────┐
  │                                  │
[Độ tin cậy >= 0.80]               [Độ tin cậy < 0.80]
  │                                  │
  │                                  ▼
  │                         Worker chuyển trạng thái:
  │                         status = "awaiting_corners"
  │                         và lưu tọa độ gợi ý vào corners_json
  │                                  │
  │                                  ▼
  │                         Frontend hiển thị ảnh gốc + 4 điểm kéo thả
  │                         Người dùng kéo chỉnh lại 4 góc mép giấy
  │                                  │
  │                                  ▼
  │                         Frontend gửi 4 góc lên API
  │                         API set status = "queued"
  │                                  │
  └─────────────────┬────────────────┘
                    │
                    ▼
[Worker làm phẳng phối cảnh (Perspective Rectification)]
                    │
                    ▼
[Khử bóng & Làm sạch nền theo Preset đã chọn]
                    │
                    ▼
[Kiểm tra chất lượng QA & Lưu ảnh kết quả]
```

### Cách tính tọa độ góc cho Frontend
- Tọa độ `x, y` là tọa độ pixel thật trên ảnh gốc (đã theo chiều xoay chuẩn EXIF).
- Nếu trên giao diện web bạn scale ảnh nhỏ lại để vừa màn hình (ví dụ `display_width = 400px`, `natural_width = 2400px`), hãy nhân lại tỷ lệ tương ứng trước khi gửi về API:

```javascript
// Quy đổi tọa độ click / kéo thả từ giao diện về kích thước ảnh gốc
const scaleX = img.naturalWidth / img.clientWidth;
const scaleY = img.naturalHeight / img.clientHeight;

const real_x = Math.round(canvas_x * scaleX);
const real_y = Math.round(canvas_y * scaleY);
```

---

## 5. Bảng tra cứu chi tiết các thông số

### 5.1 Cài đặt phong cách phục chế (`RestoreSettings`)

| Tên thông số | Kiểu | Giá trị hợp lệ | Mặc định | Ý nghĩa & Ứng dụng thực tế |
|---|---|---|---|---|
| `preset` | string | `"pure_bw"`, `"natural"`, `"vintage"` | `"natural"` | **pure_bw**: Nền trắng tinh, chữ đen đậm (phù hợp in ấn, photo lại).<br>**natural**: Nền trắng sáng, giữ sắc độ xám tự nhiên của mực viết tay, bút chì.<br>**vintage**: Nền giấy màu ngà ấm (`#DACEBD`), mực màu sepia nâu đen truyền thống (`#2A2016`). |
| `aspect_ratio` | string | `"auto"`, `"a4"`, `"notebook"`, `"letter"`, `"free"` | `"auto"` | **Tỷ lệ khổ giấy**:<br>- `auto`: Tự động khôi phục tỷ lệ 3D thực bằng thuật toán điểm tụ phối cảnh (Zhang 2003), loại bỏ hiện tượng dãn ngang khi chụp góc nghiêng.<br>- `notebook`: Vở học sinh/sổ tay (1:1.35).<br>- `a4`: Khổ A4 chuẩn quốc tế (1:1.414).<br>- `letter`: US Letter (1:1.294).<br>- `free`: 2D tự do theo độ dài cạnh thô. |
| `paper_warmth` | float | `0.0` đến `1.0` | `0.75` | **Sắc ngà giấy thư** *(dành cho preset vintage)*:<br>- `0.0`: Kem sáng tinh khôi.<br>- `0.75`: Vàng ngà hoài niệm nguyên bản.<br>- `1.0`: Giấy cổ phong vị đậm đà. |
| `bleed_through_reduction` | float | `0.50` đến `0.98` | `0.82` | **Lọc chữ in hằn trang sau**:<br>Tự động ép nét chữ mờ có tương phản ≥ 0.82 × độ sáng nền giấy về màu giấy sạch, xóa bỏ chữ lem từ trang sau của giấy tập học sinh mỏng. |
| `margin_inset` | float | `0.0` đến `0.10` | `0.0` | **Xén mép lề giấy**:<br>Tỷ lệ cắt thụt lề vào trong sau khi nắn thẳng (ví dụ `0.02` = 2%), giúp loại bỏ lỗ gáy lò xo, bóng mép bàn, ghim giấy. |
| `denoise` | string | `"off"`, `"light"`, `"medium"` | `"light"` | Mức độ khử nhiễu hạt.<br>- `light`: Giữ nguyên nét mực mảnh và dấu chấm câu.<br>- `medium`: Khử sạch hạt nhiễu đối với ảnh chụp thiếu sáng. |
| `threshold_method` | string | `"adaptive"`, `"otsu"` | `"adaptive"` | *(Dùng cho `pure_bw`)*.<br>- `adaptive`: Phân ngưỡng cục bộ thích ứng Gaussian.<br>- `otsu`: Phân ngưỡng toàn cục theo biểu đồ histogram. |
| `threshold_strength` | float | `0.0` đến `1.0` | `0.5` | Độ nhạy tách nét chữ.<br>- `0.1 - 0.4`: Nét chữ đậm đà, giữ được chữ nhạt.<br>- `0.6 - 0.9`: Nét chữ thanh mảnh, lọc bỏ vết ố bẩn mờ. |
| `adaptive_block_size` | int (số lẻ) | ví dụ: `21, 31, 41` | `None` (auto) | Kích thước cửa sổ tính ngưỡng. Mặc định tự tính theo tỷ lệ độ phân giải ảnh. |
| `adaptive_c` | float | ví dụ: `5.0 - 15.0` | `None` (auto) | Độ dời ngưỡng cục bộ. Mặc định tự tính từ `threshold_strength`. |

---

### 5.2 Cấu hình môi trường Worker (`WorkerConfig`)

| Biến môi trường | Mặc định | Mô tả |
|---|---|---|
| `STORAGE_ROOT` | `.` | Thư mục gốc chứa ảnh. Worker từ chối mọi file nằm ngoài thư mục này. |
| `DATABASE_PATH` | `letter_worker.db` | Đường dẫn file SQLite lưu hàng đợi. |
| `MAX_IMAGE_FILE_SIZE_BYTES` | `52428800` (50MB) | Dung lượng tối đa của 1 file ảnh. |
| `MAX_IMAGE_PIXELS` | `50000000` (50MP) | Số pixel tối đa của ảnh (chống tràn RAM). |
| `MAX_OUTPUT_DIMENSION` | `8192` | Cạnh tối đa của ảnh sau khi làm phẳng. |
| `CONFIDENCE_AUTO_ACCEPT` | `0.80` | Ngưỡng điểm tin cậy để tự động phục chế không cần hỏi người dùng. |
| `JOB_LEASE_TIMEOUT_SECONDS` | `300` | Sau 5 phút nếu worker treo, job sẽ được cấp lại cho worker khác. |

---

## 6. Bảng mã lỗi & Bảng tiến trình giao diện

### 6.1 Mã tiến trình (`message_code`) dành cho UI
Frontend có thể ánh xạ các mã này thành lời dẫn tương tác sinh động:

| Mã `message_code` | Tiến độ | Lời dẫn giao diện gợi ý |
|---|---|---|
| `IMAGE_LOADING` | 10% | *"Đang tiếp nhận và kiểm tra ảnh bức thư..."* |
| `CORNER_DETERMINATION` | 25% | *"Đang nhận diện trang giấy và tìm kiếm 4 góc thư..."* |
| `USING_USER_CORNERS` | 35% | *"Căn chỉnh theo các góc trang thư bạn đã chỉ định..."* |
| `AWAITING_CORNER_CONFIRMATION` | 40% | *"Độ tin cậy chưa tuyệt đối. Mời bạn kéo chỉnh lại 4 góc trên ảnh."* |
| `PERSPECTIVE_RECTIFICATION` | 50% | *"Đang nắn thẳng phối cảnh và kéo phẳng mặt giấy..."* |
| `RESTORATION_APPLIED` | 75% | *"Đang khử bóng đổ, làm sạch trang và phục chế nét mực..."* |
| `QUALITY_ASSURANCE` | 90% | *"Đang kiểm định chất lượng: độ nét và độ tương phản nét chữ..."* |
| `PROCESSING_COMPLETED` | 100% | *"Phục chế hoàn tất! Bản thư đã sẵn sàng."* |

### 6.2 Bảng mã lỗi chuẩn (`error_code`)

| Mã lỗi | Nguyên nhân | Hướng xử lý |
|---|---|---|
| `INVALID_IMAGE` | File ảnh rỗng, hỏng hoặc định dạng không hỗ trợ | Báo người dùng chọn lại ảnh hợp lệ |
| `IMAGE_TOO_LARGE` | Dung lượng > 50 MB hoặc độ phân giải > 50 MP | Nén hoặc giảm kích thước ảnh trước khi gửi |
| `INVALID_CORNERS` | 4 góc bị tự cắt (hình cánh bướm), lõm hoặc trùng điểm | Yêu cầu kéo lại 4 góc thành tứ giác lồi |
| `PAPER_NOT_DETECTED`| Không tìm thấy trang giấy trên ảnh | Chuyển sang màn hình chọn góc thủ công |
| `RECTIFICATION_FAILED`| Lỗi toán học khi tính ma trận phối cảnh | Thử lại hoặc điều chỉnh 4 góc |
| `OUTPUT_VALIDATION_FAILED`| Kết quả bị toàn trắng, toàn đen hoặc quá mờ | Thử lại với preset khác hoặc ảnh rõ nét hơn |
| `STORAGE_ERROR` | Đường dẫn cố tình thoát ra ngoài thư mục an toàn | Kiểm tra lại đường dẫn file |

---

## 7. Triển khai với Docker

Chạy Worker dưới dạng một Docker container độc lập, gắn volume chung thư mục dữ liệu với Backend:

### Build Image
```bash
docker build -t letter-worker:latest .
```

### Chạy Container
```bash
docker run -d \
  --name letter-worker \
  --restart unless-stopped \
  -v C:/data/letter_app:/app/data \
  -e STORAGE_ROOT=/app/data \
  -e DATABASE_PATH=/app/data/letter_worker.db \
  letter-worker:latest
```
Trong cấu hình trên:
- Thư mục dữ liệu trên máy chủ: `C:/data/letter_app`
- Backend API chỉ cần ghi ảnh vào `C:/data/letter_app/uploads` và insert vào `C:/data/letter_app/letter_worker.db`.
- Worker trong container sẽ tự động lấy job, xuất ảnh vào `C:/data/letter_app/outputs`.
