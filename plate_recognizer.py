"""
plate_recognizer.py  (v4 - fast-plate-ocr + crop debug)
=========================================================
OCR: fast-plate-ocr (ONNX MobileViT) - xin hon Tesseract nhieu lan
     Khong can MSVC, hoat dong Python 3.14
     Cai: pip install "fast-plate-ocr[onnx]"

Tinh nang debug:
  - Luu anh crop bien so vao thu muc debug_crops de xem
  - OCR chay background thread -- webcam khong lag
  - Majority voting qua nhieu frame
"""

import cv2
import re
import os
import time
import argparse
import threading
import collections
import requests
import numpy as np
from ultralytics import YOLO


def console_text(value) -> str:
    return str(value).encode('ascii', errors='ignore').decode('ascii')


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Cáº¤U HÃŒNH
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
MODEL_PATH    = r'D:\AppNhanDienUser\backend\model\best.pt'
BACKEND_URL   = 'http://localhost:3000/api/plate/scan'
CONF_THRESH   = 0.40          # ngÆ°á»¡ng YOLO (tháº¥p hÆ¡n Ä‘á»ƒ detect dá»… hÆ¡n)
COOLDOWN_SEC  = 4.0
MIN_PLATE_LEN = 4
WEBCAM_INDEX  = 0
DEFAULT_SCAN_STATUS = 'IN'
TWO_LINE_ASPECT_MAX = 2.45

# Debug: lÆ°u áº£nh crop ra file
SAVE_CROPS    = True          # True = lÆ°u áº£nh crop ra thÆ° má»¥c bÃªn dÆ°á»›i
CROP_DIR      = r'D:\AppNhanDienUser\backend\debug_crops'
# Voting window
VOTE_WINDOW   = 10

# MÃ u
C_MATCH   = (50,  220,  80)
C_NOMATCH = (0,   165, 255)
C_DETECT  = (200, 180,   0)
C_WHITE   = (255, 255, 255)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  KHá»žI Táº O fast-plate-ocr
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print('[OCR] Loading fast-plate-ocr model...')
try:
    from fast_plate_ocr import LicensePlateRecognizer
    # european-plates-mobile-vit-v2-model: nháº¹, nhanh, tá»‘t cho chá»¯ sá»‘
    _ocr = LicensePlateRecognizer('european-plates-mobile-vit-v2-model')
    HAS_OCR = True
    print('[OCR] OK fast-plate-ocr ready')
except Exception as e:
    HAS_OCR = False
    _ocr    = None
    print(f'[OCR] ERROR cannot load OCR: {e}')
    print('[OCR]   Run: pip install "fast-plate-ocr[onnx]"')

# Táº¡o thÆ° má»¥c debug crops
if SAVE_CROPS:
    os.makedirs(CROP_DIR, exist_ok=True)
    print(f'[DEBUG] Crop images saved to: {CROP_DIR}')


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  TIá»€N Xá»¬ LÃ áº¢NH CROP
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def preprocess_for_ocr(img_bgr: np.ndarray) -> np.ndarray:
    """
    Chuáº©n bá»‹ áº£nh crop cho fast-plate-ocr.
    Model yÃªu cáº§u áº£nh GRAYSCALE (1 channel), khÃ´ng pháº£i BGR.
    """
    h, w = img_bgr.shape[:2]
    # Scale lÃªn tá»‘i thiá»ƒu 50px chiá»u cao
    if h < 50:
        scale = 50.0 / h
        img_bgr = cv2.resize(img_bgr, (int(w * scale), 50),
                             interpolation=cv2.INTER_CUBIC)
    # Chuyá»ƒn sang grayscale (model chá»‰ nháº­n 1 channel)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # CLAHE tÄƒng tÆ°Æ¡ng pháº£n
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    gray  = clahe.apply(gray)
    # Denoise nháº¹
    gray  = cv2.fastNlMeansDenoising(gray, h=10)
    return gray


def clean_plate(text: str) -> str:
    """LÃ m sáº¡ch káº¿t quáº£ OCR â€” chá»‰ giá»¯ kÃ½ tá»± biá»ƒn sá»‘ VN."""
    text = text.upper().strip()
    # Bá» kÃ½ tá»± rÃ¡c
    text = re.sub(r'[^A-Z0-9]', '', text)
    return text


def crop_text_region(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    pts = cv2.findNonZero(binary)
    if pts is None:
        return img_bgr

    x, y, bw, bh = cv2.boundingRect(pts)
    pad_x = max(6, int(w * 0.04))
    pad_y = max(3, int(h * 0.08))
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + bw + pad_x)
    y2 = min(h, y + bh + pad_y)
    return img_bgr[y1:y2, x1:x2]


def split_two_line_plate(img_bgr: np.ndarray) -> list[np.ndarray]:
    """Tach crop bien so 2 hang thanh [hang tren, hang duoi] neu co the."""
    h, w = img_bgr.shape[:2]
    if h < 80 or w / max(h, 1) > TWO_LINE_ASPECT_MAX:
        return []

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    row_score = np.count_nonzero(binary, axis=1)
    y_from = int(h * 0.38)
    y_to = int(h * 0.66)
    if y_to <= y_from:
        return []

    split_y = y_from + int(np.argmin(row_score[y_from:y_to]))
    top_h = split_y
    bottom_h = h - split_y
    if top_h < h * 0.25 or bottom_h < h * 0.25:
        return []

    pad = max(4, int(h * 0.04))
    top = crop_text_region(img_bgr[0:min(h, split_y + pad), :])
    bottom = crop_text_region(img_bgr[max(0, split_y - pad):h, :])
    return [top, bottom]


def run_ocr(img_bgr: np.ndarray) -> str:
    gray = preprocess_for_ocr(img_bgr)
    result = _ocr.run(gray)

    if not result:
        return ''

    item = result[0]

    if hasattr(item, 'plate'):
        raw = item.plate or ''
    elif isinstance(item, str):
        raw = item
    elif isinstance(item, (list, tuple)) and len(item) > 0:
        raw = str(item[0])
    else:
        d = getattr(item, '__dict__', {})
        raw = str(next(iter(d.values()), '')) if d else str(item)

    return clean_plate(raw)


def ocr_crop(img_bgr: np.ndarray) -> str:
    """Cháº¡y fast-plate-ocr trÃªn áº£nh crop biá»ƒn sá»‘ (grayscale input)."""
    if not HAS_OCR or img_bgr is None or img_bgr.size == 0:
        return ''
    try:
        full_text = run_ocr(img_bgr)

        lines = split_two_line_plate(img_bgr)
        if len(lines) == 2:
            top_text = run_ocr(lines[0])
            bottom_text = run_ocr(lines[1])
            split_text = clean_plate(f'{top_text}{bottom_text}')
            if top_text and bottom_text and len(split_text) >= MIN_PLATE_LEN:
                return split_text

        return full_text
        gray   = preprocess_for_ocr(img_bgr)   # tráº£ vá» (H, W) grayscale
        result = _ocr.run(gray)                # tráº£ vá» list[PlatePrediction]

        if not result:
            return ''

        item = result[0]

        # fast-plate-ocr tráº£ vá» PlatePrediction object vá»›i attribute .plate
        if hasattr(item, 'plate'):
            raw = item.plate or ''
        elif isinstance(item, str):
            raw = item
        elif isinstance(item, (list, tuple)) and len(item) > 0:
            raw = str(item[0])
        else:
            # Fallback: láº¥y giÃ¡ trá»‹ Ä‘áº§u tiÃªn trong __dict__ náº¿u cÃ³
            d = getattr(item, '__dict__', {})
            raw = str(next(iter(d.values()), '')) if d else str(item)

        return clean_plate(raw)
    except Exception as e:
        print(f'  [OCR-ERR] {e}')
        return ''


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  LÆ¯U CROP DEBUG
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_crop_counter = 0
def save_crop(img_bgr: np.ndarray, plate_text: str = ''):
    """LÆ°u áº£nh crop ra file Ä‘á»ƒ debug."""
    global _crop_counter
    if not SAVE_CROPS or img_bgr is None or img_bgr.size == 0:
        return
    _crop_counter += 1
    suffix = f'_{plate_text}' if plate_text else ''
    name   = os.path.join(CROP_DIR, f'crop_{_crop_counter:04d}{suffix}.jpg')
    cv2.imwrite(name, img_bgr)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  BACKGROUND OCR WORKER (khÃ´ng lag)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class OcrWorker:
    def __init__(self):
        self._lock  = threading.Lock()
        self._img   = None
        self._busy  = False
        self._votes = collections.Counter()
        self._best  = ''
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, img_bgr: np.ndarray):
        with self._lock:
            if not self._busy:
                self._img  = img_bgr.copy()
                self._busy = True

    def get_best(self) -> str:
        with self._lock:
            return self._best

    def reset(self):
        with self._lock:
            self._votes.clear()
            self._best = ''

    def _loop(self):
        while True:
            img = None
            with self._lock:
                if self._busy and self._img is not None:
                    img = self._img
                    self._img = None

            if img is not None:
                text = ocr_crop(img)
                with self._lock:
                    if len(text) >= MIN_PLATE_LEN:
                        self._votes[text] += 1
                        # Trim Ä‘á»ƒ trÃ¡nh counter phÃ¬nh to
                        if sum(self._votes.values()) > VOTE_WINDOW * 5:
                            self._votes = collections.Counter(
                                {k: max(1, v//2) for k, v in self._votes.items()}
                            )
                        winner = self._votes.most_common(1)[0][0]
                        self._best = winner
                        top3 = dict(self._votes.most_common(3))
                        print(f'  [VOTE] {top3} -> {winner!r}')

                    # LÆ°u crop Ä‘á»ƒ debug (dÃ¹ OCR ra gÃ¬)
                    save_crop(img, text)

                    self._busy = False
            else:
                time.sleep(0.01)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Gá»¬I BACKEND
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def normalize_scan_status(value: str) -> str:
    return 'OUT' if str(value).upper() == 'OUT' else 'IN'


def parse_args():
    parser = argparse.ArgumentParser(description='Webcam bien so xe IN/OUT')
    parser.add_argument(
        '--status',
        choices=['IN', 'OUT', 'in', 'out'],
        default=DEFAULT_SCAN_STATUS,
        help='Che do quet ban dau: IN hoac OUT',
    )
    return parser.parse_args()


def send_to_backend(plate: str, scan_status: str) -> dict | None:
    try:
        r = requests.post(
            BACKEND_URL,
            json={'plate_number': plate, 'status': normalize_scan_status(scan_status)},
            timeout=3,
        )
        return r.json()
    except Exception as e:
        print(f'  [NET] {e}')
        return None


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Váº¼ UI
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def draw_box(frame, x1, y1, x2, y2, label, color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    L = 16
    for cx, cy in [(x1,y1),(x2,y1),(x1,y2),(x2,y2)]:
        dx, dy = (1 if cx==x1 else -1), (1 if cy==y1 else -1)
        cv2.line(frame,(cx,cy),(cx+dx*L,cy),C_WHITE,3)
        cv2.line(frame,(cx,cy),(cx,cy+dy*L),C_WHITE,3)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (x1, y1-th-10), (x1+tw+8, y1), color, -1)
    cv2.putText(frame, label, (x1+4, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)


def draw_hud(frame, last_plate, last_result, last_time, now, live_ocr, scan_status):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (w, 68), (10,10,20), -1)
    cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, f'BIEN SO XE SCANNER  MODE:{scan_status}  [I/O doi mode]', (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_WHITE, 2)

    if live_ocr:
        cv2.putText(frame, f'OCR live: {live_ocr}', (w-280, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.56, (200,200,100), 1)

    if last_plate:
        matched = last_result and last_result.get('matched')
        color   = C_MATCH if matched else C_NOMATCH
        status  = 'CHO PHEP' if matched else 'TU CHOI'
        cd      = max(0.0, COOLDOWN_SEC-(now-last_time))
        cd_str  = f'  CD:{cd:.1f}s' if cd > 0 else ''
        cv2.putText(frame, f'Last: {last_plate}  [{status}]{cd_str}',
                    (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)
    else:
        cv2.putText(frame, 'Chua phat hien bien so...', (12,55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (130,130,130), 1)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  MAIN
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class PlateRecognizer:
    def __init__(self, scan_status=DEFAULT_SCAN_STATUS):
        print('[YOLO] Loading model...')
        self.model      = YOLO(MODEL_PATH)
        self.ocr_worker = OcrWorker()
        print('[YOLO] OK ready\n')
        self.last_sent_time  = 0.0
        self.last_plate_text = ''
        self.last_result     = None
        self._frame_idx      = 0
        self.scan_status     = normalize_scan_status(scan_status)

    def set_scan_status(self, scan_status: str):
        new_status = normalize_scan_status(scan_status)
        if self.scan_status != new_status:
            self.scan_status = new_status
            self.last_sent_time = 0.0
            self.last_plate_text = ''
            self.last_result = None
            self.ocr_worker.reset()
            print(f'\n[MODE] Chuyen sang {self.scan_status}')

    def run(self):
        cap = cv2.VideoCapture(WEBCAM_INDEX)
        if not cap.isOpened():
            print(f'[CAM] Cannot open webcam index={WEBCAM_INDEX}')
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        print('='*56)
        print('  WEBCAM BIEN SO XE')
        print(f'  Backend : {BACKEND_URL}')
        print(f'  Mode    : {self.scan_status}  ([I]=IN, [O]=OUT)')
        print(f'  Crops   : {CROP_DIR}  (SAVE_CROPS={SAVE_CROPS})')
        print('  [I] IN   [O] OUT   [Q] thoat')
        print('='*56 + '\n')

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            now = time.time()
            self._frame_idx += 1

            results = self.model.predict(frame, conf=CONF_THRESH, verbose=False)

            has_det = False
            for res in results:
                for box in res.boxes:
                    has_det = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    x1 = max(0,x1); y1 = max(0,y1)
                    x2 = min(frame.shape[1]-1, x2)
                    y2 = min(frame.shape[0]-1, y2)
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    # Gá»­i cho OCR má»—i 3 frame
                    if self._frame_idx % 3 == 0:
                        self.ocr_worker.submit(crop)

                    best = self.ocr_worker.get_best()

                    # Gá»­i backend
                    if (
                        len(best) >= MIN_PLATE_LEN
                        and (now - self.last_sent_time) >= COOLDOWN_SEC
                        and best != self.last_plate_text
                    ):
                        print(f'\n[SCAN] {self.scan_status} >>> {best}  (conf={conf:.0%})')
                        result = send_to_backend(best, self.scan_status)
                        self.last_sent_time  = now
                        self.last_plate_text = best
                        self.last_result     = result
                        self.ocr_worker.reset()
                        if result:
                            if result.get('matched'):
                                u = result['user']
                                print(f'[DB]   OK {console_text(u["full_name"])} ({u["username"]})')
                            else:
                                print(f'[DB]   DENIED {console_text(result.get("message", best))}')

                    # Váº½ box
                    if len(best) >= MIN_PLATE_LEN:
                        matched = (
                            self.last_result is not None
                            and self.last_result.get('matched')
                            and self.last_plate_text == best
                        )
                        color = C_MATCH if matched else C_NOMATCH
                        label = f'{best}  {conf:.0%}'
                    else:
                        color = C_DETECT
                        label = f'Scanning... {conf:.0%}'

                    draw_box(frame, x1, y1, x2, y2, label, color)

            # Reset voting khi khÃ´ng cÃ³ detection
            if not has_det and self._frame_idx % 30 == 0:
                self.ocr_worker.reset()

            draw_hud(frame, self.last_plate_text, self.last_result,
                     self.last_sent_time, now, self.ocr_worker.get_best(), self.scan_status)

            cv2.imshow('Bien So Xe Scanner | [Q] thoat', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('i'):
                self.set_scan_status('IN')
            elif key == ord('o'):
                self.set_scan_status('OUT')
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print('[CAM] Webcam closed.')
        if SAVE_CROPS:
            print(f'[DEBUG] {_crop_counter} crop images saved to: {CROP_DIR}')


if __name__ == '__main__':
    args = parse_args()
    PlateRecognizer(args.status).run()
