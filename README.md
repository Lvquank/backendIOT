# Backend IOT Parking

Backend Node.js + SQLite cho he thong quan ly bai xe va script webcam nhan dien bien so bang YOLO + OCR.

## Yeu Cau

- Node.js 18+.
- Python 3.10+.
- Webcam hoat dong tren may chay `plate_recognizer.py`.
- Model YOLO tai `model/best.pt`.

## Cai Dat Backend

```powershell
cd D:\AppNhanDienUser\backend
npm install
```

Chay server:

```powershell
npm start
```

Server mac dinh chay tai:

```text
http://localhost:3000
```

Database SQLite se duoc tao tu dong tai:

```text
data/parking.db
```

Khi database moi duoc tao, `database.js` se seed san user va bien so mau, gom:

```text
user id 1: 99E122268
user id 2: 51F97022
```

## Cai Dat Python Cho Webcam

```powershell
cd D:\AppNhanDienUser\backend
py -m pip install -r requirements.txt
py -m pip install "fast-plate-ocr[onnx]"
```

Neu chua co model, dat file YOLO vao:

```text
D:\AppNhanDienUser\backend\model\best.pt
```

## Chay Nhan Dien Bien So

Mo terminal thu nhat de chay backend:

```powershell
cd D:\AppNhanDienUser\backend
npm start
```

Mo terminal thu hai de chay webcam:

```powershell
cd D:\AppNhanDienUser\backend
py plate_recognizer.py --status IN
```

Hoac chay che do xe ra:

```powershell
py plate_recognizer.py --status OUT
```

Trong cua so webcam:

```text
I = chuyen sang che do IN
O = chuyen sang che do OUT
Q = thoat
```

## Logic IN / OUT

- `IN`: bien so phai ton tai trong bang `vehicles`, neu dung se ghi log vao `access_logs`.
- `OUT`: bien so phai ton tai trong bang `vehicles` va log hop le gan nhat cua bien do phai la `IN`, neu dung se ghi log `OUT`.
- Bien so khong tim thay hoac khong du dieu kien se bi tu choi va khong ghi vao `/api/logs`.

## API Chinh

Quet bien so:

```http
POST /api/plate/scan
Content-Type: application/json
```

Body:

```json
{
  "plate_number": "99E122268",
  "status": "IN"
}
```

Xem log webcam:

```http
GET /api/logs
```

Thong ke log webcam:

```http
GET /api/logs/stats
```

Quan ly bien so:

```http
GET    /api/vehicles
POST   /api/vehicles
DELETE /api/vehicles/:id
```

## Thu Muc Khong Commit

Repo da ignore cac thu muc/file runtime:

```text
node_modules/
data/*.db
data/*.db-shm
data/*.db-wal
debug_crops/
__pycache__/
*.pyc
```

## Loi Thuong Gap

Neu webcam khong mo duoc:

```text
[CAM] Cannot open webcam index=0
```

Kiem tra webcam co dang bi ung dung khac dung khong, hoac doi `WEBCAM_INDEX` trong `plate_recognizer.py`.

Neu OCR khong load duoc, cai lai:

```powershell
py -m pip install "fast-plate-ocr[onnx]"
```

Neu backend khong nhan request tu webcam, dam bao `npm start` dang chay va `BACKEND_URL` trong `plate_recognizer.py` la:

```text
http://localhost:3000/api/plate/scan
```
