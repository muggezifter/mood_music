#!/usr/bin/env python3
"""
Webcam Mood Detector → PureData
================================
Captures webcam frames, analyses the visible face for emotional expression,
and sends the mood name as a TCP message to PureData [netreceive] whenever
the detected mood changes.

Detected moods (matching rhythm_changes.py):
  happiness · sadness · anger · fear · surprise · disgust · contempt · neutral

Two analysis backends are available via --backend:

  deepface (default)
    Fast, runs locally, frame rate is easily real-time.
    Requires:  pip install opencv-python deepface tf-keras

  ollama
    Uses a local vision LLM (default: gemma3:12b, already pulled).
    Slower (~2-10 s per frame) but requires no extra Python packages.
    Override model with --ollama-model (e.g. llava:34b).

PureData setup:
  [netreceive 3001 1]   ← TCP, outputs symbols like "happiness"

Usage examples:
  python mood_detector.py
  python mood_detector.py --backend ollama
  python mood_detector.py --pd-port 3001 --interval 0.4 --debounce 3
  python mood_detector.py --backend ollama --ollama-model llava:34b
  python mood_detector.py --no-display                    # headless / ssh session
  python mood_detector.py --color-model bw                # greyscale preview
  python mood_detector.py --color-model duotone           # two-tone preview using mood colour
  python mood_detector.py --detector-backend opencv       # faster face detector
  python mood_detector.py --detector-backend retinaface   # most accurate (default)
"""

import argparse
import base64
import json
import queue
import socket
import sys
import threading
import time
import urllib.error
import urllib.request

# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_PD_HOST      = '127.0.0.1'
DEFAULT_PD_PORT      = 3000 
DEFAULT_CAMERA       = 0
DEFAULT_INTERVAL     = 0.5        # seconds between analyses
DEFAULT_CONFIDENCE   = 25.0       # minimum DeepFace confidence % to accept
DEFAULT_DEBOUNCE     = 2          # consecutive identical readings before send
DEFAULT_BACKEND           = 'deepface'
DEFAULT_DETECTOR_BACKEND  = 'retinaface'
DEFAULT_OLLAMA_MODEL      = 'gemma3:12b'
DEFAULT_OLLAMA_URL   = 'http://localhost:11434'
DEFAULT_CROP_SIDES      = 0.25       # fraction to discard from each side (0 = disabled)
DEFAULT_MIN_FACE_HEIGHT = 33         # face must be this % of frame height to count

# ─────────────────────────────────────────────────────────────────────────────
# Mood vocabulary and mappings
# ─────────────────────────────────────────────────────────────────────────────

MOODS = frozenset({
    'happiness', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'contempt', 'neutral',
})

# DeepFace label → program mood name
DEEPFACE_MAP: dict[str, str] = {
    'happy':    'happiness',
    'sad':      'sadness',
    'angry':    'anger',
    'fear':     'fear',
    'surprise': 'surprise',
    'disgust':  'disgust',
    'neutral':  'neutral',
}

# BGR colors for the on-screen overlay
MOOD_COLORS: dict[str, tuple] = {
    'happiness': (  0, 215, 255),
    'sadness':   (180,  80,  40),
    'anger':     ( 30,  30, 220),
    'fear':      (160,  30, 190),
    'surprise':  ( 20, 200, 200),
    'disgust':   ( 20, 130,  60),
    'contempt':  (130, 130, 130),
    'neutral':   (200, 200, 200),
}

# ─────────────────────────────────────────────────────────────────────────────
# Shared state  (analysis worker writes, main thread reads)
# ─────────────────────────────────────────────────────────────────────────────

_frame_q     = queue.Queue(maxsize=1)
_result_lock = threading.Lock()
_result: dict = {'mood': None, 'confidence': 0.0, 'face_present': False}
_stop_evt    = threading.Event()

# ─────────────────────────────────────────────────────────────────────────────
# PureData TCP sender
# ─────────────────────────────────────────────────────────────────────────────

class PDSender:
    """
    Persistent TCP connection to PureData [netreceive].
    Reconnects automatically on failure.
    Message format: 'mood ;\\n'  — parsed by [netreceive] as a PD symbol.
    """

    def __init__(self, host: str, port: int) -> None:
        self.host  = host
        self.port  = port
        self._sock = None
        self._lock = threading.Lock()

    @property
    def connected(self) -> bool:
        return self._sock is not None

    def send(self, mood: str) -> bool:
        msg = f"{mood} ;\n".encode('ascii')
        with self._lock:
            for _ in range(2):       # one retry after reconnect
                try:
                    if self._sock is None:
                        self._sock = socket.create_connection(
                            (self.host, self.port), timeout=2.0
                        )
                    self._sock.sendall(msg)
                    return True
                except OSError:
                    self._drop()
        return False

    def _drop(self) -> None:
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def close(self) -> None:
        with self._lock:
            self._drop()

# ─────────────────────────────────────────────────────────────────────────────
# Analysis backends
# ─────────────────────────────────────────────────────────────────────────────

def _analyze_deepface(frame, min_conf: float, detector_backend: str,
                      min_face_height_pct: float = 0.0) -> tuple[str | None, float]:
    """Returns (mood, confidence%) using DeepFace, or (None, 0.0).

    Uses enforce_detection=True so DeepFace raises ValueError when no face is
    present.  Catching that is the only reliable way to get a 'no face' signal;
    with enforce_detection=False DeepFace always returns a result regardless.
    """
    try:
        from deepface import DeepFace  # lazy: loads TF on first call
        results  = DeepFace.analyze(
            img_path=frame,
            actions=['emotion'],
            enforce_detection=True,
            detector_backend=detector_backend,
            silent=True,
        )
        # Reject the detection if the face is too small relative to the frame.
        if min_face_height_pct > 0:
            frame_h = frame.shape[0]
            face_h  = results[0].get('region', {}).get('h', frame_h)
            if face_h / frame_h * 100 < min_face_height_pct:
                return None, 0.0
        emotions = results[0]['emotion']          # {label: confidence_pct}
        dominant = max(emotions, key=emotions.get)
        conf     = float(emotions[dominant])
        if conf < min_conf:
            return None, conf
        return DEEPFACE_MAP.get(dominant), conf
    except ValueError:
        # DeepFace could not detect a face in the frame.
        return None, 0.0
    except Exception:
        return None, 0.0


def _analyze_ollama(frame, model: str, base_url: str) -> tuple[str | None, float]:
    """Returns (mood, 100.0) via a local Ollama vision LLM, or (None, 0)."""
    try:
        import cv2  # may not be installed in ollama-only mode
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
    except ImportError:
        # cv2 not installed — caller should have verified this before reaching here
        return None, 0.0
    if not ok:
        return None, 0.0

    b64 = base64.b64encode(buf.tobytes()).decode()
    prompt = (
        "Look carefully at the person's facial expression in this image. "
        "Choose the single best-matching emotion from this list: "
        "happiness, sadness, anger, fear, surprise, disgust, contempt, neutral. "
        "Reply with exactly that one word and nothing else."
    )
    payload = json.dumps({
        'model':   model,
        'prompt':  prompt,
        'images':  [b64],
        'stream':  False,
        'options': {'temperature': 0},
    }).encode()

    try:
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/api/generate",
            data=payload,
            headers={'Content-Type': 'application/json'},
        )
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            data = json.loads(resp.read())
        word = data.get('response', '').strip().lower()
        word = word.split()[0] if word else ''    # take first token only
        return (word if word in MOODS else None), 100.0
    except Exception as exc:
        print(f"[ollama] error: {exc}", flush=True)
        return None, 0.0

# ─────────────────────────────────────────────────────────────────────────────
# Analysis worker thread
# ─────────────────────────────────────────────────────────────────────────────

def _put_frame(frame) -> None:
    """Discard any stale queued frame and enqueue the latest."""
    try:
        _frame_q.get_nowait()
    except queue.Empty:
        pass
    try:
        _frame_q.put_nowait(frame.copy())
    except queue.Full:
        pass


def analysis_worker(args: argparse.Namespace) -> None:
    # Pre-warm DeepFace so the real-time loop doesn't stall on first use.
    if args.backend == 'deepface':
        try:
            import numpy as np
            from deepface import DeepFace
            blank = np.zeros((64, 64, 3), dtype='uint8')
            # enforce_detection=False here only: blank frame has no face by
            # design; we just want TF/model weights loaded before the live loop.
            DeepFace.analyze(img_path=blank, actions=['emotion'],
                             enforce_detection=False,
                             detector_backend=args.detector_backend,
                             silent=True)
        except Exception:
            pass

    print("[detector] Ready.", flush=True)

    while not _stop_evt.is_set():
        try:
            frame = _frame_q.get(timeout=1.0)
        except queue.Empty:
            continue

        if args.backend == 'ollama':
            mood, conf = _analyze_ollama(frame, args.ollama_model, args.ollama_url)
        else:
            mood, conf = _analyze_deepface(frame, args.confidence,
                                           args.detector_backend,
                                           args.min_face_height)

        if mood:
            with _result_lock:
                _result['mood']         = mood
                _result['confidence']   = conf
                _result['face_present'] = True
        else:
            with _result_lock:
                _result['face_present'] = False

        # Rate-limit: sleep for interval, but wake immediately on stop.
        _stop_evt.wait(args.interval)

# ─────────────────────────────────────────────────────────────────────────────
# On-screen overlay  (only used when --no-display is not set)
# ─────────────────────────────────────────────────────────────────────────────

def _apply_color_model(frame, color_model: str, mood_color: tuple | None = None):
    """Return a display copy of *frame* transformed by *color_model*.

    full    – no change (default)
    bw      – greyscale
    duotone – shadow→highlight tint using the current mood colour
    """
    import cv2
    import numpy as np
    if color_model == 'bw':
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if color_model == 'duotone':
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        alpha = gray.astype(np.float32) / 255.0
        shadow    = np.array([15, 10, 30], dtype=np.float32)          # dark indigo
        hi_bgr    = mood_color if mood_color else (220, 200, 50)       # amber fallback
        highlight = np.array(hi_bgr, dtype=np.float32)
        out = shadow + (highlight - shadow) * alpha[:, :, np.newaxis]
        return np.clip(out, 0, 255).astype(np.uint8)
    return frame.copy()   # 'full'


def _draw_overlay(frame, mood: str | None, confidence: float,
                  pd_ok: bool, face_ok: bool = True) -> None:
    import cv2
    h, w = frame.shape[:2]
    if not face_ok:
        color = (80, 80, 80)
        label = 'NO FACE'
    else:
        color = MOOD_COLORS.get(mood, (200, 200, 200)) if mood else (160, 160, 160)
        label = mood.upper() if mood else 'DETECTING\u2026'

    # Dark footer bar
    cv2.rectangle(frame, (0, h - 56), (w, h), (22, 22, 22), -1)

    # Mood label
    cv2.putText(frame, label, (14, h - 16),
                cv2.FONT_HERSHEY_DUPLEX, 1.05, color, 2, cv2.LINE_AA)

    # Confidence bar
    if mood and confidence:
        bar_x = w - 228
        bar_w = int(200 * min(confidence, 100) / 100)
        cv2.rectangle(frame, (bar_x, h - 46), (bar_x + bar_w, h - 32), color, -1)
        cv2.putText(frame, f"{confidence:.0f}%", (bar_x, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)

    # PD connection indicator dot
    dot = (0, 210, 0) if pd_ok else (55, 55, 210)
    cv2.circle(frame, (w - 14, h - 40), 6, dot, -1)
    cv2.putText(frame, "FUDI", (w - 46, h - 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, dot, 1, cv2.LINE_AA)


def _draw_centered_overlay(frame, mood: str | None, confidence: float,
                           pd_ok: bool, face_ok: bool = True) -> None:
    """Variant used with --crop-display: mood name centred on the frame."""
    import cv2
    import numpy as np
    h, w = frame.shape[:2]
    label = 'NO FACE' if not face_ok else (mood.upper() if mood else 'DETECTING\u2026')

    font       = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1.4
    thickness  = 2
    color      = (255, 255, 255)   # always white
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    tx = (w - tw) // 2
    ty = int(h * 0.75) + th // 2   # 75% from the top

    cv2.putText(frame, label, (tx, ty), font, font_scale, color, thickness, cv2.LINE_AA)

    # Confidence bar (centred, near the bottom)
    if face_ok and confidence:
        bar_total = min(w - 40, 200)
        bar_fill  = int(bar_total * min(confidence, 100) / 100)
        bar_x     = (w - bar_total) // 2
        cv2.rectangle(frame, (bar_x, h - 22), (bar_x + bar_fill, h - 12),
                      (255, 255, 255), -1)
        cv2.putText(frame, f"{confidence:.0f}%",
                    (bar_x + bar_total + 6, h - 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # PD connection indicator dot (top-right corner)
    dot = (0, 210, 0) if pd_ok else (55, 55, 210)
    cv2.circle(frame, (w - 14, 20), 6, dot, -1)
    cv2.putText(frame, "FUDI", (w - 46, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, dot, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Webcam facial mood detector → PureData [netreceive]",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--pd-host',
                   default=DEFAULT_PD_HOST,
                   help="PureData netreceive host")
    p.add_argument('--pd-port',
                   type=int, default=DEFAULT_PD_PORT,
                   help="PureData netreceive TCP port")
    p.add_argument('--camera',
                   type=int, default=DEFAULT_CAMERA,
                   help="Webcam device index")
    p.add_argument('--interval',
                   type=float, default=DEFAULT_INTERVAL,
                   help="Seconds between emotion analyses")
    p.add_argument('--confidence',
                   type=float, default=DEFAULT_CONFIDENCE,
                   help="Min DeepFace confidence %% to accept a reading "
                        "(deepface backend only)")
    p.add_argument('--debounce',
                   type=int, default=DEFAULT_DEBOUNCE,
                   help="Consecutive identical readings required before "
                        "signalling a mood change")
    p.add_argument('--backend',
                   choices=['deepface', 'ollama'], default=DEFAULT_BACKEND,
                   help="Analysis backend")
    p.add_argument('--detector-backend',
                   dest='detector_backend',
                   choices=['retinaface', 'opencv', 'ssd', 'dlib',
                            'mtcnn', 'fastmtcnn', 'yunet', 'yolov8'],
                   default=DEFAULT_DETECTOR_BACKEND,
                   help="DeepFace face detector (deepface backend only); "
                        "retinaface = most accurate, opencv = fastest")
    p.add_argument('--ollama-model',
                   default=DEFAULT_OLLAMA_MODEL,
                   help="Ollama vision model name (ollama backend only)")
    p.add_argument('--ollama-url',
                   default=DEFAULT_OLLAMA_URL,
                   help="Ollama API base URL")
    p.add_argument('--no-display',
                   action='store_true',
                   help="Headless mode: do not open a preview window")
    p.add_argument('--color-model',
                   choices=['full', 'bw', 'duotone'], default='full',
                   dest='color_model',
                   help="Preview window colour model: "
                        "full (default, normal colour), "
                        "bw (greyscale), "
                        "duotone (two-tone tint using the current mood colour)")
    p.add_argument('--crop',
                   type=float, default=DEFAULT_CROP_SIDES,
                   metavar='FRAC',
                   help="Fraction of frame width to discard from each side "
                        "before analysis (0 = full frame, 0.25 = middle 50%%)"
                        "; cropped region is shown with guide lines in the preview")
    p.add_argument('--crop-display',
                   action='store_true',
                   dest='crop_display',
                   help="Show only the active (cropped) region in the preview window "
                        "instead of the full frame; has no effect when --crop is 0")
    p.add_argument('--scale',
                   type=int, default=100,
                   metavar='PCT',
                   help="Resize the preview window to PCT%% of its natural size "
                        "(e.g. 75 = 75%%, 150 = 150%%; default: 100)")
    p.add_argument('--min-face-height',
                   type=float, default=DEFAULT_MIN_FACE_HEIGHT,
                   dest='min_face_height',
                   metavar='PCT',
                   help="Minimum face height as a percentage of the frame height "
                        "required to accept a detection (0 = disabled, "
                        f"default: {DEFAULT_MIN_FACE_HEIGHT}%%; deepface backend only)")
    args = p.parse_args()

    if args.interval <= 0:
        p.error("--interval must be positive")
    if args.debounce < 1:
        p.error("--debounce must be >= 1")
    if not (0.0 <= args.crop < 0.5):
        p.error("--crop must be in the range [0, 0.5)")
    if args.scale <= 0:
        p.error("--scale must be a positive integer")
    if args.min_face_height < 0:
        p.error("--min-face-height must be >= 0")
    return args

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Import cv2 here so a missing install gives a clear message.
    try:
        import cv2
    except ImportError:
        sys.exit(
            "opencv-python is not installed.\n"
            "Run:  pip install opencv-python\n"
            "Then for the deepface backend also:  pip install deepface tf-keras"
        )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"Error: cannot open camera index {args.camera}")
    # Ask for an impossibly large frame; the driver clamps to its maximum.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  10000)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 10000)
    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera   : {cam_w}×{cam_h} (max resolution)")

    if not args.no_display:
        # WINDOW_GUI_NORMAL hides the Qt toolbar row of pictogram buttons.
        cv2.namedWindow('Mood Detector  [q / Esc = quit]',
                        cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)

    sender = PDSender(args.pd_host, args.pd_port)

    print(f"Backend  : {args.backend}"
          + (f"  ({args.ollama_model})" if args.backend == 'ollama' else ""))
    print(f"Sending  : TCP  {args.pd_host}:{args.pd_port}  [FUDI]")
    print(f"Interval : {args.interval}s   Debounce: {args.debounce}   "
          f"Camera: {args.camera}")
    if args.crop > 0:
        print(f"Crop     : {args.crop:.0%} from each side  "
              f"(using middle {1 - 2*args.crop:.0%} of the frame)")
    if args.backend == 'deepface' and args.min_face_height > 0:
        print(f"Min face : {args.min_face_height:.0f}% of frame height")
    if args.backend == 'deepface':
        print("Loading emotion model (first run may be slow)…", flush=True)
    print()

    worker = threading.Thread(
        target=analysis_worker, args=(args,),
        daemon=True, name='analysis-worker',
    )
    worker.start()

    confirmed_mood   = None
    candidate_mood   = None
    candidate_count  = 0
    face_confirmed   = None   # None = not yet known, True = face, False = no face
    face_candidate   = None
    face_cand_count  = 0

    # Tell rhythm_changes.py to pause until we confirm a face
    sender.send('face_off')

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            # Crop sides for analysis while keeping the full frame for display.
            if args.crop > 0:
                _h, _w = frame.shape[:2]
                _x1 = int(_w * args.crop)
                _x2 = _w - _x1
                _put_frame(frame[:, _x1:_x2])
            else:
                _put_frame(frame)

            with _result_lock:
                raw_mood = _result['mood']
                raw_conf = _result['confidence']
                raw_face = _result['face_present']

            # Face-presence debounce (independent of mood debounce)
            if raw_face == face_candidate:
                face_cand_count += 1
            else:
                face_candidate  = raw_face
                face_cand_count = 1

            if face_cand_count >= args.debounce and face_candidate != face_confirmed:
                token = 'face_on' if face_candidate else 'face_off'
                sender.send(token)
                state = 'detected' if face_candidate else 'lost'
                print(f"  [face {state}]", flush=True)
                face_confirmed = face_candidate

            # Mood debounce: only act after N consecutive identical readings.
            if raw_mood == candidate_mood:
                candidate_count += 1
            else:
                candidate_mood  = raw_mood
                candidate_count = 1

            if candidate_mood and candidate_count >= args.debounce:
                if candidate_mood != confirmed_mood:
                    ok_send = sender.send(candidate_mood)
                    tag = "→ FUDI" if ok_send else "→ FUDI (not connected)"
                    print(f"  {candidate_mood:<12}  conf={raw_conf:5.1f}%  {tag}",
                          flush=True)
                    confirmed_mood = candidate_mood

            if not args.no_display:
                mood_clr = MOOD_COLORS.get(confirmed_mood) if confirmed_mood else None
                display_frame = _apply_color_model(frame, args.color_model, mood_clr)
                display_frame = cv2.flip(display_frame, 1)
                # Optionally restrict the display to the active crop region.
                if args.crop > 0 and args.crop_display:
                    _dh, _dw = display_frame.shape[:2]
                    _dx1 = int(_dw * args.crop)
                    _dx2 = _dw - _dx1
                    display_frame = display_frame[:, _dx1:_dx2]
                else:
                    # Draw guide lines only when showing the full frame with a crop.
                    if args.crop > 0:
                        _dh, _dw = display_frame.shape[:2]
                        _dx1 = int(_dw * args.crop)
                        _dx2 = _dw - _dx1
                        cv2.line(display_frame, (_dx1, 0), (_dx1, _dh), (0, 255, 255), 1)
                        cv2.line(display_frame, (_dx2, 0), (_dx2, _dh), (0, 255, 255), 1)
                # Scale the video frame first so overlay elements are drawn at a
                # fixed pixel size regardless of the --scale value.
                if args.scale != 100:
                    _sh, _sw = display_frame.shape[:2]
                    _nw = max(1, int(_sw * args.scale / 100))
                    _nh = max(1, int(_sh * args.scale / 100))
                    display_frame = cv2.resize(display_frame, (_nw, _nh),
                                              interpolation=cv2.INTER_LINEAR)
                if args.crop > 0 and args.crop_display:
                    _draw_centered_overlay(display_frame, confirmed_mood, raw_conf,
                                           sender.connected, face_confirmed is True)
                else:
                    _draw_overlay(display_frame, confirmed_mood, raw_conf,
                                  sender.connected, face_confirmed is True)
                cv2.imshow('Mood Detector  [q / Esc = quit]', display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
            else:
                time.sleep(0.03)       # cap CPU in headless mode

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        _stop_evt.set()
        worker.join(timeout=3.0)
        cap.release()
        cv2.destroyAllWindows()
        sender.send('face_off')
        sender.close()
        print("Stopped.")


if __name__ == '__main__':
    main()
