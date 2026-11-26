#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monitoraggio Video AI 24/7 (Cloud Worker v4 — Anti-Oggetti Fissi + FFmpeg Linux)
================================================================================
- Filtro Movimento (Background Subtractor): ignora corrimano, colonnine e oggetti fissi
- Blocco Anti-Ripetizione: non invia mai messaggi ripetuti sullo stesso punto
- Soglia confidenza 58% (scarta riflessi/oggetti al 42%, rileva persone vere 77-85%)
- Lettore ibrido OpenCV + FFmpeg nativo Linux + Auto-Foto Avvio + Comando /foto
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import threading
from collections import deque
from datetime import datetime, timezone, timedelta

import cv2
import numpy as np
import requests
from ultralytics import YOLO

TZ_ITALIA = timezone(timedelta(hours=1))

CONFIDENCE_THRESHOLD = 0.58
YOLO_IMGSZ = 1280
COOLDOWN_TELEGRAM_SEC = 60
FPS_ANALISI = 2
DURATA_MASSIMA_SEC = int(float(os.environ.get("MAX_RUN_HOURS", "5.4")) * 3600)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def log(msg: str):
    ora = datetime.now(TZ_ITALIA).strftime("%d/%m/%Y %H:%M:%S")
    print(f"[{ora}] {msg}", flush=True)


def carica_config_cifrata():
    raw = os.environ.get("MOLO_CONFIG", "").strip()
    if not raw:
        raise RuntimeError(
            "Variabile cifrata MOLO_CONFIG mancante! "
            "Inseriscila in GitHub -> Settings -> Secrets and variables -> Actions."
        )
    return json.loads(raw)


def risolvi_stream_rtsp_me(embed_url: str, max_tentativi_risveglio: int = 20):
    m = re.search(r"rtsp\.me/embed/([A-Za-z0-9_-]+)", embed_url)
    if not m:
        return embed_url, time.time() + 900

    cam_id = m.group(1)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    try:
        session.get(f"https://rtsp.me/embed/{cam_id}/", timeout=10)
    except Exception:
        pass

    token_api = f"https://rtsp.me/api/embed/{cam_id}/token/"
    resp1 = session.get(
        token_api,
        headers={"Referer": f"https://rtsp.me/embed/{cam_id}/"},
        timeout=12,
    )
    resp1.raise_for_status()
    data1 = resp1.json()
    if data1.get("status") != "ok" or not data1.get("token"):
        raise RuntimeError(f"Token non valido: {data1.get('status')}")

    token = data1["token"]
    edge_server = data1.get("url", "https://fn.rtsp.me").rstrip("/")

    stream_api = f"{edge_server}/token/{token}"
    resp2 = session.get(
        stream_api,
        headers={"Referer": "https://rtsp.me/"},
        timeout=12,
    )
    resp2.raise_for_status()
    data2 = resp2.json()

    hls_url = data2.get("hls") or data2.get("ios")
    expires = min(float(data2.get("expires", time.time() + 1200)), time.time() + 900)
    if not hls_url:
        raise RuntimeError("Nessun URL HLS restituito")

    last_code = 0
    for _ in range(max_tentativi_risveglio):
        try:
            r_hls = session.get(hls_url, headers={"Referer": "https://rtsp.me/"}, timeout=8)
            last_code = r_hls.status_code
            if r_hls.status_code == 200 and "#EXTM3U" in r_hls.text:
                return hls_url, expires
        except Exception:
            pass
        time.sleep(1.5)

    raise RuntimeError(f"Playlist HLS non pronta dopo 30s (HTTP {last_code})")


def cattura_frame_ffmpeg(hls_url: str) -> np.ndarray:
    if not shutil.which("ffmpeg"):
        return None
    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-user_agent", USER_AGENT,
        "-headers", "Referer: https://rtsp.me/\r\nOrigin: https://rtsp.me\r\n",
        "-i", hls_url,
        "-frames:v", "1",
        "-f", "image2pipe",
        "-vcodec", "mjpeg",
        "-q:v", "3",
        "-",
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15, check=False)
    if res.returncode == 0 and len(res.stdout) > 1000:
        arr = np.frombuffer(res.stdout, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return None


class CameraWatchdogStream:
    def __init__(self, idx: int, name: str, embed_url: str):
        self.idx = idx
        self.label = f"Cam-{idx}"
        self.name = name
        self.embed_url = embed_url
        self.lock = threading.Lock()
        self.latest_frame = None
        self.last_frame_ts = 0.0
        self.frame_seq = 0
        self.status_msg = "Avvio connessione..."
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _set_status(self, msg: str):
        with self.lock:
            self.status_msg = msg

    def _push_frame(self, frm: np.ndarray):
        with self.lock:
            self.latest_frame = frm
            self.last_frame_ts = time.time()
            self.frame_seq += 1
            self.status_msg = f"Online (frame #{self.frame_seq})"

    def _loop(self):
        cap = None
        hls_url = ""
        expires = 0.0
        use_ffmpeg_fallback = False

        while self.running:
            now = time.time()
            stalled = (now - self.last_frame_ts) > 25.0 and self.last_frame_ts > 0

            if not hls_url or now >= expires or stalled:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                    cap = None
                try:
                    self._set_status("Risveglio stream RTSP.ME in corso...")
                    hls_url, expires = risolvi_stream_rtsp_me(self.embed_url)
                    self._set_status("Stream pronto, apertura flusso video...")

                    cap = cv2.VideoCapture(hls_url)
                    frm = None
                    if cap.isOpened():
                        for _ in range(8):
                            ret, f = cap.read()
                            if ret and f is not None:
                                frm = f
                                break
                            time.sleep(0.3)

                    if frm is not None:
                        use_ffmpeg_fallback = False
                        self._push_frame(frm)
                        log(f"Stream [{self.label}] attivo (OpenCV).")
                        continue

                    if cap is not None:
                        try:
                            cap.release()
                        except Exception:
                            pass
                        cap = None

                    frm_ff = cattura_frame_ffmpeg(hls_url)
                    if frm_ff is not None:
                        use_ffmpeg_fallback = True
                        self._push_frame(frm_ff)
                        log(f"Stream [{self.label}] attivo (FFmpeg nativo).")
                        continue
                    else:
                        self._set_status("Attesa primo keyframe video...")
                        hls_url = ""
                except Exception as e:
                    self._set_status(f"Riconnessione ({type(e).__name__}: {str(e)[:60]})")
                    hls_url = ""
                time.sleep(3.0)
                continue

            if not use_ffmpeg_fallback and cap is not None and cap.isOpened():
                ret, frm = cap.read()
                if ret and frm is not None:
                    self._push_frame(frm)
                else:
                    frm_ff = cattura_frame_ffmpeg(hls_url)
                    if frm_ff is not None:
                        use_ffmpeg_fallback = True
                        self._push_frame(frm_ff)
                    else:
                        time.sleep(0.5)
            else:
                frm_ff = cattura_frame_ffmpeg(hls_url)
                if frm_ff is not None:
                    self._push_frame(frm_ff)
                else:
                    hls_url = ""
                    time.sleep(1.0)

    def get_latest(self, last_seen_seq: int):
        with self.lock:
            if self.latest_frame is None or self.frame_seq == last_seen_seq:
                return None, last_seen_seq
            if (time.time() - self.last_frame_ts) > 30.0:
                return None, last_seen_seq
            return self.latest_frame.copy(), self.frame_seq

    def get_status(self) -> str:
        with self.lock:
            return self.status_msg


def invia_foto_telegram(bot_token: str, chat_id: str, frame_bgr: np.ndarray, messaggio: str):
    if not bot_token or not chat_id:
        return
    tg_img = frame_bgr
    if tg_img.shape[1] > 1280:
        h_new = int(tg_img.shape[0] * (1280 / tg_img.shape[1]))
        tg_img = cv2.resize(tg_img, (1280, h_new))
    ok, buf = cv2.imencode(".jpg", tg_img, [int(cv2.IMWRITE_JPEG_QUALITY), 84])
    if not ok:
        return
    jpg_bytes = buf.tobytes()

    def _worker():
        url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
        try:
            requests.post(
                url,
                data={"chat_id": chat_id, "caption": messaggio},
                files={"photo": ("allarme.jpg", jpg_bytes, "image/jpeg")},
                timeout=30,
            )
        except Exception:
            pass

    threading.Thread(target=_worker, daemon=True).start()


def invia_video_telegram(bot_token: str, chat_id: str, frames_bgr: list, caption: str, fps: int = 3):
    if not bot_token or not chat_id or not frames_bgr:
        return

    def _worker():
        tmp_raw = os.path.join(tempfile.gettempdir(), f"raw_{int(time.time()*1000)}.mp4")
        tmp_h264 = os.path.join(tempfile.gettempdir(), f"clip_{int(time.time()*1000)}.mp4")
        send_file = tmp_raw
        try:
            writer = cv2.VideoWriter(tmp_raw, cv2.VideoWriter_fourcc(*"mp4v"), max(1, fps), (960, 540))
            for f in frames_bgr:
                writer.write(cv2.resize(f, (960, 540)))
            writer.release()

            if shutil.which("ffmpeg"):
                subprocess.run(
                    [
                        "ffmpeg", "-y", "-loglevel", "error",
                        "-i", tmp_raw,
                        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        tmp_h264,
                    ],
                    timeout=25,
                    check=False,
                )
                if os.path.exists(tmp_h264) and os.path.getsize(tmp_h264) > 1000:
                    send_file = tmp_h264

            with open(send_file, "rb") as vf:
                requests.post(
                    f"https://api.telegram.org/bot{bot_token}/sendVideo",
                    data={"chat_id": chat_id, "caption": caption, "supports_streaming": "true"},
                    files={"video": ("allarme_molo.mp4", vf, "video/mp4")},
                    timeout=45,
                )
        except Exception:
            pass
        finally:
            for p in (tmp_raw, tmp_h264):
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass

    threading.Thread(target=_worker, daemon=True).start()


def invia_messaggio_telegram(bot_token: str, chat_id: str, testo: str):
    if not bot_token or not chat_id:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": testo},
            timeout=10,
        )
    except Exception:
        pass


def punto_dentro_roi(px: int, py: int, roi_norm, w: int, h: int) -> bool:
    if not roi_norm or len(roi_norm) < 3:
        return True
    pts = np.array([[int(pt[0] * w), int(pt[1] * h)] for pt in roi_norm], dtype=np.int32)
    return cv2.pointPolygonTest(pts, (float(px), float(py)), False) >= 0


def oggetto_in_movimento(frame_gray: np.ndarray, bg_float: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> bool:
    """
    Confronta il riquadro rilevato con lo sfondo storico del molo:
    se è un oggetto fisso (corrimano, colonnina, bitta, barca ormeggiata),
    non c'è variazione rispetto allo sfondo e viene ignorato al 100%.
    """
    if bg_float is None:
        return False
    h, w = frame_gray.shape[:2]
    bx1, by1 = max(0, x1), max(0, y1)
    bx2, by2 = min(w, x2), min(h, y2)
    if (bx2 - bx1) < 8 or (by2 - by1) < 8:
        return False
    roi_curr = frame_gray[by1:by2, bx1:bx2]
    roi_bg = cv2.convertScaleAbs(bg_float[by1:by2, bx1:bx2])
    diff = cv2.absdiff(roi_curr, roi_bg)
    changed_ratio = float(np.count_nonzero(diff > 22)) / float(diff.size)
    return changed_ratio >= 0.10


def bersaglio_gia_notificato(cx: int, cy: int, last_points: list, soglia_px: int = 55) -> bool:
    """
    Impedisce l'invio di messaggi ripetuti per un bersaglio fermo nello stesso punto.
    """
    for px, py in last_points:
        if ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5 < soglia_px:
            return True
    return False


def main():
    avvio_ts = time.time()
    cfg = carica_config_cifrata()
    bot_token = cfg["telegram_bot_token"]
    chat_id = str(cfg["telegram_chat_id"])
    cameras = cfg["cameras"]
    rois = cfg.get("rois", {})

    log(f"Avvio Monitoraggio Cloud 24/7 v4 ({len(cameras)} telecamere configurate)...")
    model = YOLO("yolov8s.pt")

    if os.environ.get("SEND_STARTUP_MSG", "1") == "1":
        invia_messaggio_telegram(
            bot_token,
            chat_id,
            "☁️ Server Cloud 24/7 v4 AVVIATO (Filtro Anti-Oggetti Fissi attivo)!\n"
            f"📷 Connessione in corso a: {', '.join(cameras.keys())}...",
        )

    cams_state = [
        {
            "idx": idx + 1,
            "name": nome,
            "stream": CameraWatchdogStream(idx + 1, nome, url),
            "last_seq": -1,
            "last_annotated": None,
            "bg_float": None,
            "last_alarm_points": [],
            "sent_first_photo": False,
            "last_telegram": 0.0,
            "buffer": deque(maxlen=12),
            "post_record_remaining": 0,
            "pending_clip_caption": "",
        }
        for idx, (nome, url) in enumerate(cameras.items())
    ]

    last_update_id = 0
    try:
        r_up = requests.get(f"https://api.telegram.org/bot{bot_token}/getUpdates", timeout=8).json()
        for u in r_up.get("result", []):
            last_update_id = max(last_update_id, u["update_id"])
    except Exception:
        pass

    last_cmd_check = 0.0
    cicli_totali = 0
    intervallo = 1.0 / max(1, FPS_ANALISI)

    while (time.time() - avvio_ts) < DURATA_MASSIMA_SEC:
        t0 = time.time()

        if (t0 - last_cmd_check) >= 3.5:
            last_cmd_check = t0
            try:
                r_up = requests.get(
                    f"https://api.telegram.org/bot{bot_token}/getUpdates",
                    params={"offset": last_update_id + 1, "timeout": 0},
                    timeout=5,
                ).json()
                for u in r_up.get("result", []):
                    last_update_id = max(last_update_id, u["update_id"])
                    msg_obj = u.get("message") or {}
                    txt = (msg_obj.get("text") or "").strip().lower()
                    sender_chat = str((msg_obj.get("chat") or {}).get("id", ""))
                    if sender_chat == chat_id and txt in ("/foto", "foto", "/stato", "stato"):
                        now_str = datetime.now(TZ_ITALIA).strftime("%d/%m/%Y %H:%M:%S")
                        inviate = 0
                        stati = []
                        for st in cams_state:
                            stati.append(f"• {st['name']}: {st['stream'].get_status()}")
                            if st["last_annotated"] is not None:
                                invia_foto_telegram(
                                    bot_token,
                                    chat_id,
                                    st["last_annotated"],
                                    f"📸 Foto Live ({now_str})\n📷 Telecamera: {st['name']}\n✅ Controlli AI eseguiti: {cicli_totali}",
                                )
                                inviate += 1
                        if inviate == 0:
                            invia_messaggio_telegram(
                                bot_token,
                                chat_id,
                                "⏳ Stato connessione telecamere:\n" + "\n".join(stati),
                            )
            except Exception:
                pass

        for state in cams_state:
            nome = state["name"]
            frame, seq = state["stream"].get_latest(state["last_seq"])
            if frame is None:
                continue
            state["last_seq"] = seq
            cicli_totali += 1

            h, w = frame.shape[:2]
            roi_norm = rois.get(nome, [])
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_gray = cv2.GaussianBlur(frame_gray, (5, 5), 0)
            if state["bg_float"] is None or state["bg_float"].shape != frame_gray.shape:
                state["bg_float"] = frame_gray.astype(np.float32)

            if roi_norm and len(roi_norm) >= 3:
                xs = [int(pt[0] * w) for pt in roi_norm]
                ys = [int(pt[1] * h) for pt in roi_norm]
                x0 = max(0, min(xs) - 80)
                y0 = max(0, min(ys) - 80)
                x1_roi = min(w, max(xs) + 80)
                y1_roi = min(h, max(ys) + 80)
                crop_img = frame[y0:y1_roi, x0:x1_roi]
            else:
                x0, y0 = 0, 0
                crop_img = frame

            results = model.predict(
                source=crop_img,
                classes=[0],
                conf=CONFIDENCE_THRESHOLD,
                imgsz=YOLO_IMGSZ,
                verbose=False,
            )

            annotated = frame.copy()
            if roi_norm and len(roi_norm) >= 3:
                pts = np.array([[int(pt[0] * w), int(pt[1] * h)] for pt in roi_norm], dtype=np.int32)
                cv2.polylines(annotated, [pts], True, (255, 200, 0), 3)

            persone_sul_molo = 0
            max_conf = 0.0
            current_points = []

            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
                    x1, y1, x2, y2 = bx1 + x0, by1 + y0, bx2 + x0, by2 + y0
                    conf = float(box.conf[0])
                    piedi_x = int((x1 + x2) / 2)
                    piedi_y = int(y2)
                    centro_y = int((y1 + y2) / 2)

                    sul_molo = punto_dentro_roi(piedi_x, piedi_y, roi_norm, w, h) or punto_dentro_roi(
                        piedi_x, centro_y, roi_norm, w, h
                    )
                    if not sul_molo:
                        continue

                    if not oggetto_in_movimento(frame_gray, state["bg_float"], x1, y1, x2, y2):
                        continue

                    if bersaglio_gia_notificato(piedi_x, centro_y, state["last_alarm_points"]):
                        continue

                    persone_sul_molo += 1
                    current_points.append((piedi_x, centro_y))
                    max_conf = max(max_conf, conf)
                    color = (0, 0, 255)
                    label = f"PERSONA RILEVATA ({conf:.0%})"

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
                    cv2.circle(annotated, (piedi_x, piedi_y), 6, color, -1)
                    cv2.putText(
                        annotated,
                        label,
                        (x1, max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                    )

            if persone_sul_molo == 0:
                state["last_alarm_points"] = []

            alpha_bg = 0.02 if persone_sul_molo > 0 else 0.12
            cv2.accumulateWeighted(frame_gray, state["bg_float"], alpha_bg)

            now_str = datetime.now(TZ_ITALIA).strftime("%d/%m/%Y %H:%M:%S")
            status_color = (0, 0, 255) if persone_sul_molo > 0 else (0, 200, 0)
            cv2.putText(
                annotated,
                f"CLOUD 24/7 | Cam: {nome} | {now_str} | Persone: {persone_sul_molo}",
                (15, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                status_color,
                2,
            )

            state["last_annotated"] = annotated
            state["buffer"].append(annotated)

            if not state["sent_first_photo"] and os.environ.get("SEND_STARTUP_MSG", "1") == "1":
                state["sent_first_photo"] = True
                invia_foto_telegram(
                    bot_token,
                    chat_id,
                    annotated,
                    f"✅ Telecamera [{nome}] CONNESSA al Cloud 24/7!\n🕒 {now_str}",
                )

            if state["post_record_remaining"] > 0:
                state["post_record_remaining"] -= 1
                if state["post_record_remaining"] == 0:
                    invia_video_telegram(
                        bot_token,
                        chat_id,
                        list(state["buffer"]),
                        state["pending_clip_caption"],
                        fps=max(2, FPS_ANALISI),
                    )

            now_ts = time.time()
            if persone_sul_molo > 0 and (now_ts - state["last_telegram"] >= COOLDOWN_TELEGRAM_SEC):
                state["last_telegram"] = now_ts
                state["last_alarm_points"] = current_points
                msg = (
                    f"🚨 PERSONA RILEVATA SUL MOLO! (Cloud 24/7)\n"
                    f"📷 Telecamera: {nome}\n"
                    f"🕒 Ora: {now_str}\n"
                    f"👤 Persone rilevate: {persone_sul_molo} (conf. {max_conf:.0%})"
                )
                invia_foto_telegram(bot_token, chat_id, annotated, msg)
                state["pending_clip_caption"] = f"🎥 Video rilevamento — Cam: {nome} ({now_str})"
                state["post_record_remaining"] = 4

        elapsed = time.time() - t0
        if elapsed < intervallo:
            time.sleep(intervallo - elapsed)

    log("Fine turno programmato (5.4 ore): avvio automatico turno successivo...")


if __name__ == "__main__":
    main()
