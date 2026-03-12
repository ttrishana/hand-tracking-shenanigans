"""
╔═══════════════════════════════════════════════════╗
║   NARUTO — SHADOW CLONE JUTSU                     ║
║   影分身の術                                        ║
║                                                   ║
║   Perform the 4 hand signs in order:              ║
║   鼠 Rat → 亥 Boar → 戌 Dog → 午 Horse            ║
║   Hold each sign for ~0.8s                        ║
║                                                   ║
║   pip install opencv-python mediapipe             ║
║               sounddevice numpy                   ║
╚═══════════════════════════════════════════════════╝
"""

import cv2
import numpy as np
import sounddevice as sd
import math, time, threading
from mediapipe.python.solutions import hands as mp_hands_sol
from mediapipe.python.solutions import drawing_utils as mp_draw
from mediapipe.python.solutions.hands import HAND_CONNECTIONS

# ─────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────
SAMPLE_RATE   = 44100
SIGN_HOLD_SEC = 0.75     # seconds to hold a sign to register
SIGN_TIMEOUT  = 6.0      # seconds before sequence resets

SIGNS       = ['Rat', 'Dog', 'Horse']
SIGN_KANJI  = ['鼠',  '戌',  '午']
SIGN_DESC   = ['Both fists together',
               'Flat hand over fist',
               'Index + thumb triangle']

# States
IDLE, SEQUENCE, TRIGGERING, SMOKE, CLONES, FADE = range(6)

FONT        = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD   = cv2.FONT_HERSHEY_DUPLEX
ORANGE      = (0, 130, 255)     # BGR
GOLD        = (0, 200, 255)
WHITE       = (255, 255, 255)
DIM         = (80,  80,  80)
RED         = (40,  40,  200)


# ─────────────────────────────────────────────────────────
#  SOUND SYNTHESIS
# ─────────────────────────────────────────────────────────
def _gen_sign_sfx():
    """Short chakra 'click' when a sign registers."""
    t   = np.linspace(0, 0.35, int(SAMPLE_RATE * 0.35))
    f   = np.linspace(420, 900, len(t))
    osc = np.sin(2*np.pi * np.cumsum(f / SAMPLE_RATE))
    env = np.exp(-t * 12)
    nse = np.random.default_rng(1).standard_normal(len(t)).astype(np.float32) * 0.15
    return ((osc * 0.55 + nse) * env * 0.55).astype(np.float32)

def _gen_jutsu_sfx():
    """Big explosion sound when all 4 signs complete."""
    dur = 2.2
    t   = np.linspace(0, dur, int(SAMPLE_RATE * dur))
    rng = np.random.default_rng(42)
    # Deep boom
    boom = np.sin(2*np.pi*55*t) * np.exp(-t * 4) * 0.9
    # Whoooosh
    f_sweep = np.linspace(800, 60, len(t))
    sweep   = np.sin(2*np.pi * np.cumsum(f_sweep / SAMPLE_RATE)) * np.exp(-t*3)*0.5
    # White noise crack
    crack   = rng.standard_normal(len(t)).astype(np.float32) * np.exp(-t * 8) * 0.45
    # High shimmer
    shimmer = np.sin(2*np.pi*2400*t) * np.exp(-t*25) * 0.2
    combined = boom + sweep + crack + shimmer
    return np.clip(combined * 0.75, -1, 1).astype(np.float32)

SIGN_SFX  = _gen_sign_sfx()
JUTSU_SFX = _gen_jutsu_sfx()

def play(wave):
    threading.Thread(target=sd.play, args=(wave, SAMPLE_RATE),
                     kwargs={'blocking': False}, daemon=True).start()


# ─────────────────────────────────────────────────────────
#  HAND GEOMETRY HELPERS
# ─────────────────────────────────────────────────────────
def _tip_pip(lm, tip, pip):
    """True if finger tip is above (smaller y) its PIP — i.e. extended."""
    return lm.landmark[tip].y < lm.landmark[pip].y

def finger_states(lm):
    """[index, middle, ring, pinky] True=extended."""
    return [_tip_pip(lm, 8, 6), _tip_pip(lm, 12, 10),
            _tip_pip(lm, 16, 14), _tip_pip(lm, 20, 18)]

def thumb_extended(lm):
    """Thumb tip significantly above its IP joint."""
    return lm.landmark[4].y < lm.landmark[3].y - 0.02

def is_fist(lm):
    return not any(finger_states(lm))

def is_flat(lm):
    return all(finger_states(lm))

def is_gun(lm):
    """Index (+ thumb) extended, rest curled → L-shape."""
    f = finger_states(lm)
    return f[0] and not f[1] and not f[2] and not f[3]

def palm_center(lm):
    """Normalised (x,y) of palm landmark 9."""
    return lm.landmark[9].x, lm.landmark[9].y

def hands_near(lm1, lm2, thresh=0.30):
    x1,y1 = palm_center(lm1); x2,y2 = palm_center(lm2)
    return math.hypot(x2-x1, y2-y1) < thresh

# ─── SIGN CLASSIFIERS ─────────────────────────────────────
def detect_sign(sign_name, lm_left, lm_right):
    """Returns True if both hands form the given sign."""
    if lm_left is None or lm_right is None:
        return False
    near = hands_near(lm_left, lm_right)
    if sign_name == 'Rat':
        # Both fists, hands together
        return is_fist(lm_left) and is_fist(lm_right) and near
    elif sign_name == 'Boar':
        # Both open/flat, hands close (fingers interlocked upward)
        return is_flat(lm_left) and is_flat(lm_right) and near
    elif sign_name == 'Dog':
        # One flat, one fist — hands overlapping
        combo = (is_flat(lm_left) and is_fist(lm_right)) or \
                (is_fist(lm_left) and is_flat(lm_right))
        return combo and near
    elif sign_name == 'Horse':
        # Both gun shapes (index + thumb) forming a triangle
        return is_gun(lm_left) and is_gun(lm_right) and near
    return False


# ─────────────────────────────────────────────────────────
#  SMOKE EFFECT
# ─────────────────────────────────────────────────────────
class SmokeSystem:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.puffs = []   # each: [x, y, r, max_r, vx, vy, alpha]

    def burst(self):
        rng = np.random.default_rng(int(time.time()*1000) % 99999)
        cx, cy = self.w//2, self.h//2
        for _ in range(28):
            angle = rng.uniform(0, 2*math.pi)
            speed = rng.uniform(3, 9)
            self.puffs.append({
                'x': cx + rng.uniform(-60, 60),
                'y': cy + rng.uniform(-60, 60),
                'r': rng.uniform(20, 50),
                'max_r': rng.uniform(self.w*0.35, self.w*0.75),
                'vx': math.cos(angle)*speed,
                'vy': math.sin(angle)*speed,
                'alpha': 1.0,
                'grow': rng.uniform(8, 18),
            })

    def update(self, shrink=False):
        alive = []
        for p in self.puffs:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vx'] *= 0.93
            p['vy'] *= 0.93
            if shrink:
                p['r'] -= p['grow'] * 0.6
                p['alpha'] -= 0.025
                if p['r'] > 0 and p['alpha'] > 0:
                    alive.append(p)
            else:
                p['r'] = min(p['r'] + p['grow'], p['max_r'])
                alive.append(p)
        self.puffs = alive

    def draw(self, frame):
        if not self.puffs:
            return frame
        mask = np.zeros((self.h, self.w), np.float32)
        for p in self.puffs:
            cv2.circle(mask, (int(p['x']), int(p['y'])),
                       int(p['r']), float(p['alpha']), -1)
        mask = cv2.GaussianBlur(np.clip(mask, 0, 1), (61, 61), 0)
        smoke_col = np.full_like(frame, 215)   # off-white smoke
        alpha3 = np.dstack([mask]*3)
        out = (frame * (1 - alpha3) + smoke_col * alpha3)
        return np.clip(out, 0, 255).astype(np.uint8)

    def is_clear(self):
        return len(self.puffs) == 0


# ─────────────────────────────────────────────────────────
#  CLONE GRID  (5 side-by-side panels)
# ─────────────────────────────────────────────────────────
CLONE_TINTS = [
    np.array([1.0,  0.9,  0.7],  np.float32),   # warm/orange
    np.array([0.75, 0.85, 1.0],  np.float32),   # blue shadow
    np.array([1.0,  1.0,  1.0],  np.float32),   # original
    np.array([0.8,  1.0,  0.8],  np.float32),   # green
    np.array([1.0,  0.8,  0.95], np.float32),   # pink
]

def make_clone_grid(src_frame, h, w, reveal_frac):
    """Build the 5-clone grid. reveal_frac 0→1 controls how much shows."""
    cw = w // 5
    grid = np.zeros((h, w, 3), np.uint8)
    for i, tint in enumerate(CLONE_TINTS):
        panel = cv2.resize(src_frame, (cw, h))
        panel = (panel.astype(np.float32) * tint).clip(0, 255).astype(np.uint8)

        # Chakra glow border (orange)
        glow = panel.copy()
        cv2.rectangle(glow, (2, 2), (cw-2, h-2), (0, 140, 255), 4)
        panel = cv2.addWeighted(panel, 0.85, glow, 0.15, 0)

        # Clone number
        cv2.putText(panel, f"#{i+1}", (6, h-10), FONT, 0.55, (0,180,255), 1)

        # Wipe-in from bottom (reveal animation)
        reveal_rows = int(h * reveal_frac)
        start_row   = h - reveal_rows
        grid[start_row:h, i*cw:(i+1)*cw] = panel[start_row:h]

    # Big kanji title
    if reveal_frac > 0.6:
        alpha_txt = min((reveal_frac - 0.6) / 0.4, 1.0)
        overlay   = grid.copy()
        cv2.putText(overlay, "影分身の術",
                    (w//2 - 230, 70), FONT_BOLD, 1.8, (0, 190, 255), 3)
        cv2.putText(overlay, "SHADOW CLONE JUTSU",
                    (w//2 - 230, 115), FONT, 0.9, (0, 220, 255), 2)
        grid = cv2.addWeighted(grid, 1 - 0.9*alpha_txt,
                               overlay, 0.9*alpha_txt, 0)
    return grid


# ─────────────────────────────────────────────────────────
#  HUD HELPERS
# ─────────────────────────────────────────────────────────
def draw_sign_progress(frame, completed, current, hold_frac, h, w):
    """Draw the 4-sign progress bar across the top."""
    n       = len(SIGNS)
    box_w   = 130
    total_w = n * box_w + (n-1) * 12
    x0      = (w - total_w) // 2
    y0      = 12

    for i, (kanji, name) in enumerate(zip(SIGN_KANJI, SIGNS)):
        x1 = x0 + i * (box_w + 12)
        x2 = x1 + box_w
        done = i < completed
        active = i == completed and current is not None

        bg_col  = (10, 60, 5)  if done   else \
                  (5,  30, 55) if active else (10, 10, 10)
        brd_col = (0, 210, 90) if done   else \
                  ORANGE       if active else DIM

        cv2.rectangle(frame, (x1, y0), (x2, y0+54), bg_col, -1)
        cv2.rectangle(frame, (x1, y0), (x2, y0+54), brd_col, 2)

        # Kanji large
        cv2.putText(frame, kanji,  (x1+8,  y0+38), FONT_BOLD, 1.1,
                    (0,230,90) if done else GOLD if active else DIM, 2)
        cv2.putText(frame, name,   (x1+54, y0+20), FONT, 0.38,
                    WHITE if (done or active) else DIM, 1)
        cv2.putText(frame, SIGN_DESC[i], (x1+54, y0+36), FONT, 0.28,
                    DIM, 1)

        # Hold bar
        if active and hold_frac > 0:
            bar_len = int((box_w - 10) * hold_frac)
            cv2.rectangle(frame, (x1+5, y0+46), (x1+5+bar_len, y0+52),
                          ORANGE, -1)

        # Tick
        if done:
            cv2.putText(frame, "✓", (x1+4, y0+16), FONT, 0.5,
                        (0,230,90), 2)

    # Arrow connectors
    for i in range(n-1):
        ax = x0 + (i+1)*(box_w+12) - 10
        ay = y0 + 27
        col = (0,200,80) if i < completed else DIM
        cv2.arrowedLine(frame, (ax-2, ay), (ax+8, ay), col, 2, tipLength=0.6)


def draw_hint(frame, text, h, w, col=WHITE, size=0.7):
    (tw, _), _ = cv2.getTextSize(text, FONT, size, 2)
    cv2.putText(frame, text, (w//2 - tw//2, h - 14),
                FONT, size, col, 2)


# ─────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────
def run():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")

    detector = mp_hands_sol.Hands(
        min_detection_confidence=0.72,
        min_tracking_confidence=0.72,
        max_num_hands=2,
    )
    HAND_SPEC = mp_draw.DrawingSpec(color=(60,60,60), thickness=1, circle_radius=2)
    CONN_SPEC = mp_draw.DrawingSpec(color=(100,100,100), thickness=1)

    state          = IDLE
    completed      = 0          # how many signs done
    current_sign   = SIGNS[0]
    sign_start     = None
    last_sign_time = None
    clone_frame    = None
    effect_start   = None
    smoke          = None

    print("🥷  Shadow Clone Jutsu Tracker  — Q to quit")
    print("    Perform: Rat → Dog → Horse")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res   = detector.process(rgb)

        now = time.time()

        # ── Identify left / right hands ─────────────────────
        lm_left = lm_right = None
        if res.multi_hand_landmarks:
            for lm, handedness in zip(res.multi_hand_landmarks,
                                      res.multi_handedness):
                label = handedness.classification[0].label
                mp_draw.draw_landmarks(frame, lm, HAND_CONNECTIONS,
                                       HAND_SPEC, CONN_SPEC)
                if label == 'Left':
                    lm_left  = lm
                else:
                    lm_right = lm

        # ═══════════════════════════════════════════════════
        #  STATE MACHINE
        # ═══════════════════════════════════════════════════
        if state in (IDLE, SEQUENCE):

            # ── Timeout resets sequence ─────────────────────
            #if (last_sign_time and
            #        now - last_sign_time > SIGN_TIMEOUT and
            #        completed > 0):
            #    completed      = 0
            #    current_sign   = SIGNS[0]
            #    sign_start     = None
            #    last_sign_time = None
            #    state          = IDLE

            # ── Check current sign ──────────────────────────
            holding = detect_sign(current_sign, lm_left, lm_right)

            if holding:
                if sign_start is None:
                    sign_start = now
                hold_frac = min((now - sign_start) / SIGN_HOLD_SEC, 1.0)
                if hold_frac >= 1.0:
                    # Sign confirmed!
                    play(SIGN_SFX)
                    last_sign_time = now
                    completed     += 1
                    sign_start     = None

                    if completed == len(SIGNS):
                        # ALL DONE → TRIGGER JUTSU
                        clone_frame  = frame.copy()
                        effect_start = now
                        smoke        = SmokeSystem(w, h)
                        smoke.burst()
                        play(JUTSU_SFX)
                        state = TRIGGERING
                    else:
                        current_sign = SIGNS[completed]
                        state = SEQUENCE
            else:
                hold_frac  = 0.0
                sign_start = None

            # ── DRAW ────────────────────────────────────────
            # Subtle vignette overlay
            vig = np.zeros_like(frame)
            cv2.rectangle(vig, (0,0), (w,h), (0,0,0), -1)
            frame = cv2.addWeighted(frame, 0.92, vig, 0.08, 0)

            draw_sign_progress(frame, completed, current_sign,
                               hold_frac, h, w)

            # Current sign name large
            if lm_left and lm_right:
                msg = f"Hold: {current_sign} ({SIGN_KANJI[min(completed, len(SIGN_KANJI)-1)]})"
                col = ORANGE if holding else WHITE
                draw_hint(frame, msg, h, w, col)
            else:
                draw_hint(frame, "Show BOTH hands", h, w, DIM)

            # Orange glow border when holding
            if holding and hold_frac > 0:
                thickness = int(hold_frac * 8) + 1
                cv2.rectangle(frame, (0, 0), (w-1, h-1), ORANGE, thickness)

        # ────────────────────────────────────────────────────
        elif state == TRIGGERING:
            elapsed = now - effect_start
            # Flash white → start smoke
            flash = max(0, 1.0 - elapsed * 5)
            if flash > 0:
                wht = np.full_like(frame, 255)
                frame = cv2.addWeighted(frame, 1-flash, wht, flash, 0)
            smoke.update()
            frame = smoke.draw(frame)
            # After smoke fills screen (≈1.5s), transition
            if elapsed > 1.4:
                state        = SMOKE
                effect_start = now

        # ────────────────────────────────────────────────────
        elif state == SMOKE:
            elapsed = now - effect_start
            smoke.update()
            frame = smoke.draw(frame)
            if elapsed > 0.6:
                # Start clearing
                smoke.update(shrink=True)
            if elapsed > 1.8 or smoke.is_clear():
                state        = CLONES
                effect_start = now

        # ────────────────────────────────────────────────────
        elif state == CLONES:
            elapsed     = now - effect_start
            reveal_frac = min(elapsed / 1.2, 1.0)

            frame = make_clone_grid(clone_frame, h, w, reveal_frac)

            # Smoke clears over clones
            if elapsed < 1.0:
                smoke.update(shrink=True)
                frame = smoke.draw(frame)

            # Countdown to fade back
            remaining = max(0, 5.0 - elapsed)
            if elapsed > 2.0:
                cv2.putText(frame,
                            f"Clones disperse in {remaining:.0f}s ...",
                            (w//2-180, h-14), FONT, 0.6, DIM, 1)

            if elapsed > 5.0:
                state        = FADE
                effect_start = now

        # ────────────────────────────────────────────────────
        elif state == FADE:
            elapsed   = now - effect_start
            fade_frac = min(elapsed / 0.8, 1.0)
            clone_view = make_clone_grid(clone_frame, h, w, 1.0)
            frame = cv2.addWeighted(clone_view, 1 - fade_frac,
                                    frame, fade_frac, 0)
            if fade_frac >= 1.0:
                # Full reset
                state          = IDLE
                completed      = 0
                current_sign   = SIGNS[0]
                sign_start     = None
                last_sign_time = None
                clone_frame    = None
                smoke          = None
                print("🥷  Ready for next jutsu!")

        # ── Show ────────────────────────────────────────────
        cv2.imshow("Shadow Clone Jutsu  [Q = quit]", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("Goodbye, shinobi. 🍃")


if __name__ == "__main__":
    run()