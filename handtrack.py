"""

AIR THEREMIN / DJ LOOPER                       
  Left hand  Y  →  Volume                                 
  Right hand X  →  Pitch  (80 Hz – 1760 Hz, log scale)   
  Left pinch    →  Toggle drum beat (120 BPM)             
  Press Q to quit                                         


Requirements:
    pip3 install opencv-python mediapipe sounddevice numpy
"""

import cv2
import mediapipe as mp
import math
import threading
import numpy as np
import sounddevice as sd

# ══════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════
SAMPLE_RATE  = 44100
BLOCK_SIZE   = 512
BPM          = 120
STEPS        = 16           # 16th-note grid
STEP_SAMPLES = int(SAMPLE_RATE * 60 / (BPM * 4))
LOOP_SAMPLES = STEP_SAMPLES * STEPS


# ══════════════════════════════════════════════════════════
#  PRE-BAKED DRUM SAMPLES
# ══════════════════════════════════════════════════════════
def _make_kick(dur=0.30):
    t     = np.linspace(0, dur, int(SAMPLE_RATE * dur))
    pitch = np.exp(-t * 25) * 160 + 45           # falling pitch envelope
    phase = np.cumsum(pitch / SAMPLE_RATE) * 2 * np.pi
    return (np.sin(phase) * np.exp(-t * 12) * 0.92).astype(np.float32)

def _make_snare(dur=0.18):
    rng   = np.random.default_rng(42)
    t     = np.linspace(0, dur, int(SAMPLE_RATE * dur))
    noise = rng.standard_normal(len(t)).astype(np.float32)
    tone  = np.sin(2 * np.pi * 185 * t).astype(np.float32)
    env   = np.exp(-t * 28).astype(np.float32)
    return (noise * 0.72 + tone * 0.28) * env * 0.58

def _make_hihat(dur=0.055):
    rng   = np.random.default_rng(7)
    t     = np.linspace(0, dur, int(SAMPLE_RATE * dur))
    noise = rng.standard_normal(len(t)).astype(np.float32)
    return (noise * np.exp(-t * 85) * 0.22).astype(np.float32)

KICK_WAVE  = _make_kick()
SNARE_WAVE = _make_snare()
HIHAT_WAVE = _make_hihat()

# Pattern rows: 16 steps each
PATTERN = {
    'kick':  np.array([1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0], dtype=np.int8),
    'snare': np.array([0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0], dtype=np.int8),
    'hat':   np.array([1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0], dtype=np.int8),
}


# ══════════════════════════════════════════════════════════
#  SHARED STATE  (written by video thread, read by audio CB)
# ══════════════════════════════════════════════════════════
class _State:
    def __init__(self):
        # User-controlled params
        self.target_freq  = 440.0   # Hz – right hand X
        self.target_vol   = 0.0     # 0-1 – left hand Y
        self.beat_on      = False   # left pinch toggle
        self.right_active = False   # right hand present → synth on
        # Internal audio-thread state (not locked, only audio thread writes)
        self._smooth_freq = 440.0
        self._smooth_vol  = 0.0
        self._osc_phase   = 0.0
        self._beat_pos    = 0
        # Beat position shared back to video thread for grid display
        self.beat_pos_read = 0
        self.lock = threading.Lock()

state = _State()


# ══════════════════════════════════════════════════════════
#  AUDIO CALLBACK  (runs in real-time audio thread)
# ══════════════════════════════════════════════════════════
def _audio_cb(outdata: np.ndarray, frames: int, time_info, status):
    with state.lock:
        t_freq     = state.target_freq
        t_vol      = state.target_vol
        beat_on    = state.beat_on
        right_on   = state.right_active

    # ── Smooth parameters to avoid zipper noise ───────────
    # EMA toward target, per block (≈86 updates/sec at 512 frames)
    a_freq = 0.18
    a_vol  = 0.12
    f0 = state._smooth_freq
    f1 = f0 + a_freq * (t_freq - f0)
    v0 = state._smooth_vol
    v1 = v0 + a_vol  * (t_vol  - v0)
    state._smooth_freq = f1
    state._smooth_vol  = v1

    # ── Theremin oscillator (smooth freq ramp within block) ─
    freq_ramp   = np.linspace(f0, f1, frames)
    phase_delta = freq_ramp / SAMPLE_RATE
    phases      = (state._osc_phase + np.cumsum(phase_delta)) % 1.0
    state._osc_phase = float(phases[-1])

    # Warm additive synth: fundamental + detuned copy + harmonics
    p2 = np.pi * 2
    osc = (np.sin(p2 * phases)             * 0.52 +
           np.sin(p2 * (phases + 0.0035))  * 0.26 +   # slight detune = chorus
           np.sin(p2 * phases * 2)         * 0.14 +
           np.sin(p2 * phases * 3)         * 0.06 +
           np.sin(p2 * phases * 4)         * 0.02)

    vol_ramp = np.linspace(v0, v1, frames)
    synth    = osc * vol_ramp * 0.38 if right_on else np.zeros(frames, np.float32)

    # ── Drum loop ─────────────────────────────────────────
    beat_out = np.zeros(frames, np.float32)
    if beat_on:
        pos   = (state._beat_pos + np.arange(frames)) % LOOP_SAMPLES
        steps = (pos // STEP_SAMPLES).astype(np.int32)
        spos  = (pos %  STEP_SAMPLES).astype(np.int32)

        for wave, pat in ((KICK_WAVE,  PATTERN['kick']),
                          (SNARE_WAVE, PATTERN['snare']),
                          (HIHAT_WAVE, PATTERN['hat'])):
            wlen = len(wave)
            mask = pat[steps].astype(bool) & (spos < wlen)
            beat_out[mask] += wave[spos[mask]]

        state._beat_pos = int((state._beat_pos + frames) % LOOP_SAMPLES)
    else:
        state._beat_pos = 0

    with state.lock:
        state.beat_pos_read = state._beat_pos

    # ── Mix & clip ────────────────────────────────────────
    mixed = np.clip(synth + beat_out, -1.0, 1.0).astype(np.float32)
    outdata[:, 0] = mixed
    outdata[:, 1] = mixed


# ══════════════════════════════════════════════════════════
#  HAND TRACKING HELPERS
# ══════════════════════════════════════════════════════════
mp_hands    = mp.solutions.hands
mp_drawing  = mp.solutions.drawing_utils
hand_model  = mp_hands.Hands(min_detection_confidence=0.7,
                              min_tracking_confidence=0.7,
                              max_num_hands=2)

HAND_DOT  = mp_drawing.DrawingSpec(color=(90,90,90),  thickness=1, circle_radius=2)
HAND_LINE = mp_drawing.DrawingSpec(color=(120,120,120), thickness=1)

def pinch_info(lm, w, h):
    """Returns (distance_px, thumb_pt, index_pt)."""
    tx = int(lm.landmark[4].x * w);  ty = int(lm.landmark[4].y * h)
    ix = int(lm.landmark[8].x * w);  iy = int(lm.landmark[8].y * h)
    return math.hypot(ix-tx, iy-ty), (tx, ty), (ix, iy)

def palm_xy(lm, w, h):
    """Return the normalised (0-1) position of the mid-palm landmark."""
    return lm.landmark[9].x, lm.landmark[9].y


# ══════════════════════════════════════════════════════════
#  VISUAL HUD
# ══════════════════════════════════════════════════════════
FONT   = cv2.FONT_HERSHEY_SIMPLEX
C_PINK = (200, 80, 230)   # left hand colour (BGR)
C_CYAN = (230, 210, 30)   # right hand colour
C_GRN  = (60,  210, 80)
C_WHT  = (230, 230, 230)
C_DIM  = (90,  90,  90)

NOTES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
def freq_to_note(f):
    if f <= 0: return "---"
    try:
        n = round(12 * math.log2(f / 440.0)) + 69
        return f"{NOTES[n % 12]}{n//12 - 1}"
    except Exception:
        return "---"

def draw_hud(frame, h, w, freq, vol, beat_on, beat_pos,
             left_pinch, left_present, right_present):
    ov = frame.copy()

    # Subtle side-panel tint
    cv2.rectangle(ov, (0,   0), (170, h), (20, 5,  10), -1)
    cv2.rectangle(ov, (w-170, 0), (w, h), (5,  5,  22), -1)
    cv2.addWeighted(ov, 0.40, frame, 0.60, 0, frame)

    # ── Title bar ─────────────────────────────────────────
    cv2.rectangle(frame, (w//2-110, 5), (w//2+110, 42), (10,10,10), -1)
    cv2.putText(frame, "AIR THEREMIN", (w//2-100, 33), FONT, 0.75, C_WHT, 2)

    # ── LEFT PANEL ────────────────────────────────────────
    lx = 12
    cv2.putText(frame, "LEFT HAND",   (lx, 28),  FONT, 0.44, C_PINK, 1)
    cv2.putText(frame, "y-axis",      (lx, 44),  FONT, 0.36, C_DIM,  1)
    cv2.putText(frame, "VOLUME",      (lx, 65),  FONT, 0.50, C_WHT,  1)

    # Volume bar
    bar_top, bar_bot = 80, 280
    bar_fill = int(vol * (bar_bot - bar_top))
    cv2.rectangle(frame, (lx,    bar_top), (lx+40, bar_bot),          (40,20,40),  -1)
    cv2.rectangle(frame, (lx, bar_bot-bar_fill), (lx+40, bar_bot),    C_PINK,      -1)
    cv2.rectangle(frame, (lx,    bar_top), (lx+40, bar_bot),          C_DIM,        1)
    cv2.putText(frame, f"{int(vol*100)}%", (lx+2, bar_bot+18), FONT, 0.45, C_WHT, 1)

    # Beat toggle status
    beat_col = C_GRN if beat_on else C_DIM
    cv2.putText(frame, "[ PINCH ]",    (lx,   bar_bot+42), FONT, 0.40, C_DIM,   1)
    cv2.putText(frame, "BEAT: " + ("ON " if beat_on else "OFF"), 
                (lx, bar_bot+60), FONT, 0.50, beat_col, 1)

    if left_pinch:
        cv2.putText(frame, "PINCH!", (lx, bar_bot+88), FONT, 0.6, C_GRN, 2)

    # Hand presence dot
    dot_col = C_PINK if left_present else C_DIM
    cv2.circle(frame, (lx+55, h-15), 7, dot_col, -1)

    # ── RIGHT PANEL ───────────────────────────────────────
    rx = w - 158
    cv2.putText(frame, "RIGHT HAND",  (rx, 28), FONT, 0.44, C_CYAN, 1)
    cv2.putText(frame, "x-axis",      (rx, 44), FONT, 0.36, C_DIM,  1)
    cv2.putText(frame, "PITCH",       (rx, 65), FONT, 0.50, C_WHT,  1)

    # Frequency bar
    f_norm    = np.clip((math.log2(max(freq,1)) - math.log2(80)) /
                        (math.log2(1760) - math.log2(80)), 0, 1)
    f_fill    = int(f_norm * (bar_bot - bar_top))
    fx        = w - 52
    cv2.rectangle(frame, (fx, bar_top), (fx+40, bar_bot),           (20,20,40), -1)
    cv2.rectangle(frame, (fx, bar_bot-f_fill), (fx+40, bar_bot),    C_CYAN,     -1)
    cv2.rectangle(frame, (fx, bar_top), (fx+40, bar_bot),           C_DIM,       1)

    note = freq_to_note(freq)
    cv2.putText(frame, f"{int(freq)} Hz", (rx,   bar_bot+18), FONT, 0.42, C_WHT,  1)
    cv2.putText(frame, note,              (rx,   bar_bot+38), FONT, 0.60, C_CYAN, 1)
    cv2.putText(frame, "A2  ──  A6",      (rx,   bar_bot+58), FONT, 0.34, C_DIM,  1)

    dot_col = C_CYAN if right_present else C_DIM
    cv2.circle(frame, (w-100, h-15), 7, dot_col, -1)

    # ── BEAT STEP GRID (bottom centre) ────────────────────
    current_step = beat_pos // STEP_SAMPLES if beat_on else -1
    gx, gy = w//2 - 120, h - 48
    for i in range(16):
        x1, y1 = gx + i * 15, gy
        x2, y2 = x1 + 12,    gy + 20
        has_k  = PATTERN['kick'][i]
        has_s  = PATTERN['snare'][i]
        has_h  = PATTERN['hat'][i]
        col    = (50,50,50)
        if has_k:  col = (60, 30, 70)
        if has_s:  col = (30, 50, 70)
        if has_h and not has_k and not has_s: col = (40,40,40)
        if i == current_step: col = (0, 230, 180)
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), C_DIM, 1)

    # Beat labels
    cv2.putText(frame, "1",     (gx+1,    gy-4), FONT, 0.3, C_DIM, 1)
    cv2.putText(frame, "2",     (gx+61,   gy-4), FONT, 0.3, C_DIM, 1)
    cv2.putText(frame, "3",     (gx+121-15, gy-4), FONT, 0.3, C_DIM, 1)
    cv2.putText(frame, "4",     (gx+181-30, gy-4), FONT, 0.3, C_DIM, 1)

    # Press Q hint
    cv2.putText(frame, "Q = quit", (w-68, h-4), FONT, 0.30, C_DIM, 1)


# ══════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════
def run():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    left_pinch_prev = False

    with sd.OutputStream(samplerate=SAMPLE_RATE, channels=2, dtype='float32',
                         callback=_audio_cb, blocksize=BLOCK_SIZE):
        print("🎵  Air Theremin running — press Q to quit")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame  = cv2.flip(frame, 1)
            h, w   = frame.shape[:2]
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hand_model.process(rgb)

            left_lm     = None
            right_lm    = None
            left_pinch  = False
            right_pinch = False

            if result.multi_hand_landmarks:
                for lm, handedness in zip(result.multi_hand_landmarks,
                                          result.multi_handedness):
                    label = handedness.classification[0].label   # 'Left' | 'Right'
                    mp_drawing.draw_landmarks(frame, lm,
                        mp_hands.HAND_CONNECTIONS, HAND_DOT, HAND_LINE)

                    dist, tp, ip = pinch_info(lm, w, h)
                    pinching = dist < 42
                    col = C_GRN if pinching else (160, 160, 160)
                    cv2.circle(frame, tp, 9, col, -1)
                    cv2.circle(frame, ip, 9, col, -1)
                    cv2.line(frame, tp, ip, col, 2)

                    if label == 'Left':
                        left_lm    = lm
                        left_pinch = pinching
                    else:
                        right_lm    = lm
                        right_pinch = pinching

            # ── Update shared audio state ──────────────────
            with state.lock:
                # Left hand → volume (raise hand = louder)
                if left_lm:
                    _, py = palm_xy(left_lm, w, h)
                    vol = float(np.clip(1.0 - py, 0, 1) ** 0.65)  # non-linear feel
                    state.target_vol = vol
                else:
                    state.target_vol = 0.0

                # Right hand → pitch (slide left/right)
                if right_lm:
                    px, _ = palm_xy(right_lm, w, h)
                    # Log scale A2 (110 Hz) → A6 (1760 Hz)
                    freq = 110.0 * (1760.0 / 110.0) ** float(np.clip(px, 0, 1))
                    state.target_freq  = freq
                    state.right_active = True
                else:
                    state.right_active = False

                # Left pinch → toggle beat (leading edge)
                if left_pinch and not left_pinch_prev:
                    state.beat_on = not state.beat_on

                freq     = state.target_freq
                vol      = state.target_vol
                beat_on  = state.beat_on
                beat_pos = state.beat_pos_read

            left_pinch_prev = left_pinch

            # ── Draw HUD ──────────────────────────────────
            draw_hud(frame, h, w, freq, vol, beat_on, beat_pos,
                     left_pinch, left_lm is not None, right_lm is not None)

            cv2.imshow("Air Theremin", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("👋  Theremin closed.")

if __name__ == "__main__":
    run()