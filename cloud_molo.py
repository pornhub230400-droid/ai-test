#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monitoraggio Video AI 24/7 (Cloud Worker)
=========================================
Nessun link telecamera, nome o token Telegram Ã¨ presente in questo file.
Tutti i dati sensibili vengono letti esclusivamente dalla variabile cifrata
di ambiente MOLO_CONFIG (GitHub Encrypted Secrets).
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

CONFIDENCE_THRESHOLD = 0.45
COOLDOWN_TELEGRAM_SEC = 45
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
            "Inseriscila in GitHub -> Settings -> Secrets and variables -> Actions -> New repository secret."
        )
    return json.loads(raw)


def risolvi_stream_rtsp_me(embed_url: str, max_tentativi_risveglio: int = 10):
    m = re.search(r"rtsp\.me/embed/([A-Za-z0-9_-]+)", embed_url)
    if not m:
        return embed_url, time.time() + 3600

    cam_id = m.group(1)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    token_api = f"https://rtsp.me/api/embed/{cam_id}/token/"
    resp1 = session.get(
        token_api,
        headers={"Referer": f"https://rtsp.me/embed/{cam_id}/"},
        timeout=10,
    )
    resp1.raise_for_status()
    data1 = resp1.json()
    if data1.get("status") != "ok" or not data1.get("token"):
        raise RuntimeError("Token stream non valido")

    token = data1["token"]
    edge_server = data1.get("url", "https://fn.rtsp.me").rstrip("/")

    stream_api = f"{edge_server}/token/{token}"
    resp2 = session.get(
        stream_api,
        headers={"Referer": "https://rtsp.me/"},
        timeout=10,
    )
    resp2.raise_for_status()
    data2 = resp2.json()

    hls_url = data2.get("hls") or data2.get("ios")
    expires = float(data2.get("expires", time.time() + 1800))
    if not hls_url:
        raise RuntimeError("Nessun URL HLS disponibile")

    for _ in range(max_tentativi_risveglio):
        try:
            r_hls = session.get(hls_url, headers={"Referer": "https://rtsp.me/"}, timeout=8)
            if r_hls.status_code == 200 and "#EXTM3U" in r_hls.text:
                return hls_url, expires
        except Exception:
            pass
        time.sleep(1.2)

    return hls_url, expires


def invia_foto_telegram(bot_token: str, chat_id: str, frame_bgr: np.ndarray, messaggio: str):
    if not bot_token or not chat_id:
        return
    tg_img = frame_bgr
    if tg_img.shape[1] > 1280:
        h_new = int(tg_img.shape[0] * (1280 / tg_img.shape[1]))
        tg_img = cv2.resize(tg_img, (1280, h_new))
    ok, buf = cv2.imencode(".jpg", tg_img, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        return
    jpg_bytes = buf.tobytes()

    def _worker():
        url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
        try:
            resp = requests.post(
                url,
                data={"chat_id": chat_id, "caption": messaggio},
                files={"photo": ("allarme.jpg", jpg_bytes, "image/jpeg")},
                timeout=30,
            )
            if resp.status_code != 200:
                log("Avviso servizio notifiche")
        except Exception:
            log("Errore di rete durante invio Telegram")

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
            log("Errore invio clip video")
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


def main():
    avvio_ts = time.time()
    cfg = carica_config_cifrata()
    bot_token = cfg["telegram_bot_token"]
    chat_id = cfg["telegram_chat_id"]
    cameras = cfg["cameras"]
    rois = cfg.get("rois", {})

    log(f"Avvio Monitoraggio Cloud 24/7 ({len(cameras)} telecamere configurate)...")
    model = YOLO("yolov8n.pt")

    if os.environ.get("SEND_STARTUP_MSG", "1") == "1":
        invia_messaggio_telegram(
            bot_token,
            chat_id,
            "â˜ï¸ Server Cloud 24/7 ATTIVO (Configurazione Cifrata Privata)!\n"
            f"ðŸ“· Telecamere sorvegliate: {', '.join(cameras.keys())}\n"
            "ðŸŸ¢ Il monitoraggio gira in cloud: puoi spegnere tranquillamente il PC.",
        )

    cams_state = [
        {
            "idx": idx + 1,
            "name": nome,
            "embed_url": url,
            "cap": None,
            "expires": 0,
            "fail_count": 0,
            "last_telegram": 0.0,
            "buffer": deque(maxlen=14),
            "post_record_remaining": 0,
            "pending_clip_caption": "",
        }
        for idx, (nome, url) in enumerate(cameras.items())
    ]

    intervallo = 1.0 / max(1, FPS_ANALISI)

    while (time.time() - avvio_ts) < DURATA_MASSIMA_SEC:
        t0 = time.time()

        for state in cams_state:
            nome = state["name"]
            cam_label = f"Cam-{state['idx']}"
            embed_url = state["embed_url"]

            need_refresh = (
                state["cap"] is None
                or not state["cap"].isOpened()
                or time.time() > (state["expires"] - 60)
                or state["fail_count"] >= 12
            )
            if need_refresh:
                if state["cap"] is not None:
                    try:
                        state["cap"].release()
                    except Exception:
                        pass
                try:
                    hls_url, expires = risolvi_stream_rtsp_me(embed_url)
                    state["cap"] = cv2.VideoCapture(hls_url)
                    state["expires"] = expires
                    state["fail_count"] = 0
                    log(f"Stream [{cam_label}] connesso.")
                except Exception:
                    log(f"Riconnessione [{cam_label}] in corso...")
                    time.sleep(3)
                    continue

            for _ in range(2):
                state["cap"].grab()
            ret, frame = state["cap"].read()
            if not ret or frame is None:
                state["fail_count"] += 1
                continue

            state["fail_count"] = 0
            h, w = frame.shape[:2]
            roi_norm = rois.get(nome, [])

            results = model.predict(
                source=frame,
                classes=[0],
                conf=CONFIDENCE_THRESHOLD,
                verbose=False,
            )

            annotated = frame.copy()
            if roi_norm and len(roi_norm) >= 3:
                pts = np.array([[int(pt[0] * w), int(pt[1] * h)] for pt in roi_norm], dtype=np.int32)
                cv2.polylines(annotated, [pts], True, (255, 200, 0), 3)

            persone_sul_molo = 0
            max_conf = 0.0

            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    piedi_x = int((x1 + x2) / 2)
                    piedi_y = int(y2)

                    if punto_dentro_roi(piedi_x, piedi_y, roi_norm, w, h):
                        persone_sul_molo += 1
                        max_conf = max(max_conf, conf)
                        color = (0, 0, 255)
                        label = f"PERSONA RILEVATA ({conf:.0%})"
                    else:
                        color = (160, 160, 160)
                        label = f"Fuori zona ({conf:.0%})"

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.circle(annotated, (piedi_x, piedi_y), 6, color, -1)
                    cv2.putText(
                        annotated,
                        label,
                        (x1, max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        color,
                        2,
                    )

            now_str = datetime.now(TZ_ITALIA).strftime("%d/%m/%Y %H:%M:%S")
            status_color = (0, 0, 255) if persone_sul_molo > 0 else (0, 200, 0)
            cv2.putText(
                annotated,
                f"CLOUD 24/7 | Cam: {nome} | {now_str} | Persone: {persone_sul_molo}",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                status_color,
                2,
            )

            state["buffer"].append(annotated)

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
                msg = (
                    f"ðŸš¨ PERSONA RILEVATA SUL MOLO! (Cloud 24/7)\n"
                    f"ðŸ“· Telecamera: {nome}\n"
                    f"ðŸ•’ Ora: {now_str}\n"
                    f"ðŸ‘¤ Persone rilevate: {persone_sul_molo} (conf. {max_conf:.0%})"
                )
                invia_foto_telegram(bot_token, chat_id, annotated, msg)
                state["pending_clip_caption"] = f"ðŸŽ¥ Video rilevamento (6 sec) â€” Cam: {nome} ({now_str})"
                state["post_record_remaining"] = 8

        elapsed = time.time() - t0
        if elapsed < intervallo:
            time.sleep(intervallo - elapsed)

    log("Fine turno programmato (5.4 ore): avvio automatico turno successivo...")
    for state in cams_state:
        if state["cap"] is not None:
            try:
                state["cap"].release()
            except Exception:
                pass


if __name__ == "__main__":
    main()
