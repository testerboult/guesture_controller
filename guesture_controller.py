import cv2
import mediapipe as mp
import pyautogui
from collections import deque
import time
import math

pyautogui.FAILSAFE = False

# Screen resolution
screen_w, screen_h = pyautogui.size()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_face = mp.solutions.face_mesh

# Distance helper
def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

# Gesture parameters
PINCH_ON = 0.015    # pinch close threshold
PINCH_OFF = 0.030   # pinch release threshold (hysteresis)
DOUBLE_TAP_TIME = 0.35
HOLD_TIME = 0.4

# Cursor smoothing (exponential moving average)
# Alpha in (0..1] where lower = smoother (slower), higher = more responsive
# Typical values: 0.12 (very smooth), 0.25 (balanced), 0.45 (responsive)
SMOOTHING = 0.25

# Smoothed cursor state (initialized at runtime)
smoothed_x = None
smoothed_y = None

# Left-hand swipe detection parameters
SWIPE_MIN_DISPLACEMENT = 0.20  # normalized units (x coordinate)
SWIPE_MAX_DURATION = 2      # seconds (time window over which swipe should occur)
SWIPE_COOLDOWN = 1          # seconds (cooldown between swipe events)

# Left hand state
left_history = deque(maxlen=16)  # stores tuples (timestamp, x)
last_swipe_time = 0.0

# Gesture states
is_pinching = False        # currently pinching
is_holding = False         # drag mode (True when mouseDown is active)
pinch_start_time = 0.0     # time pinch started
last_tap_time = 0.0        # last tap time (for double-tap)

cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=2,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.7) as hands:

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)



        if results.multi_hand_landmarks and results.multi_handedness:
            for handLms, handData in zip(results.multi_hand_landmarks, results.multi_handedness):

                hand_type = handData.classification[0].label
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

                # Only Right hand controls mouse
                if hand_type != "Right":
                    continue

                index_tip = handLms.landmark[8]
                thumb_tip = handLms.landmark[4]

                # Move mouse cursor (with exponential moving average smoothing)
                raw_x = index_tip.x * screen_w
                raw_y = index_tip.y * screen_h

                # initialize smoothed coords on first use
                if smoothed_x is None or smoothed_y is None:
                    smoothed_x = raw_x
                    smoothed_y = raw_y

                # apply EMA smoothing
                smoothed_x = SMOOTHING * raw_x + (1.0 - SMOOTHING) * smoothed_x
                smoothed_y = SMOOTHING * raw_y + (1.0 - SMOOTHING) * smoothed_y

                mouse_x = int(smoothed_x)
                mouse_y = int(smoothed_y)

                pyautogui.moveTo(mouse_x, mouse_y, duration=0.01)

                d = dist(index_tip, thumb_tip)

                # Handle pinch/click/hold using helper to keep logic tidy
                def process_pinch(distance, frame):
                    """Handle pinch start/release, tap/double-tap and hold (drag).

                    Uses globals:
                      - is_pinching, pinch_start_time, is_holding, last_tap_time
                    """

                    global is_pinching, pinch_start_time, is_holding, last_tap_time

                    # PINCH START
                    if not is_pinching and distance < PINCH_ON:
                        is_pinching = True
                        is_holding = False
                        pinch_start_time = time.time()

                    # PINCH RELEASE
                    elif is_pinching and distance > PINCH_OFF:
                        # mark that pinch ended; decide click vs release below
                        is_pinching = False

                        # If hold wasn't activated, this was a TAP / CLICK
                        if not is_holding:
                            now = time.time()

                            # Double tap check
                            if now - last_tap_time < DOUBLE_TAP_TIME:
                                pyautogui.doubleClick()
                                cv2.putText(frame, "DOUBLE TAP", (10, 15),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                            else:
                                pyautogui.click()
                                cv2.putText(frame, "CLICK", (10, 15),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                            last_tap_time = now

                        # If we were holding (mouseDown was active), we'll let
                        # the HOLD RELEASE logic below call mouseUp() and clear state.

                    # HOLD DETECTION (pinch held longer than HOLD_TIME)
                    if is_pinching and not is_holding:
                        if time.time() - pinch_start_time > HOLD_TIME:
                            pyautogui.mouseDown()
                            is_holding = True

                    # HOLD RELEASE (if we had a hold and pinch ended)
                    if not is_pinching and is_holding:
                        pyautogui.mouseUp()
                        is_holding = False

                process_pinch(d, frame)

                # ---------- LEFT-HAND SWIPE (app switch) ----------
                # For left hand only, accumulate wrist (landmark 0) x positions
                if hand_type == 'Left':
                    wrist_x = handLms.landmark[0].x
                    left_history.append((time.time(), wrist_x))

                    # check for a swipe if history is long enough
                    def check_left_swipe():
                        global last_swipe_time
                        if len(left_history) < 3:
                            return

                        now = time.time()
                        # avoid triggering too often
                        if now - last_swipe_time < SWIPE_COOLDOWN:
                            return

                        # find the earliest point inside the time window
                        earliest_idx = 0
                        for i, (t, x) in enumerate(left_history):
                            if now - t <= SWIPE_MAX_DURATION:
                                earliest_idx = i
                                break

                        t0, x0 = left_history[earliest_idx]
                        t1, x1 = left_history[-1]
                        dur = t1 - t0
                        disp = x1 - x0

                        # require the swipe to be reasonably quick and large
                        if dur <= 0 or dur > SWIPE_MAX_DURATION:
                            return

                        if abs(disp) >= SWIPE_MIN_DISPLACEMENT:
                            # Rightward movement -> swipe right -> next app
                            if disp > 0:
                                pyautogui.hotkey('alt', 'tab')
                                cv2.putText(frame, 'SWIPE RIGHT → NEXT APP', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                            else:
                                # Leftward movement -> swipe left -> previous app
                                pyautogui.hotkey('alt', 'shift', 'tab')
                                cv2.putText(frame, 'SWIPE LEFT ← PREV APP', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

                            last_swipe_time = now

                    check_left_swipe()

        cv2.putText(frame, "Right-Hand Mouse Control", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Mouse Control with Double Tap + Hold", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()


#It's not the complete version,if you have any ideas feel free to cantact me
#My Instagram handle is cyboult
#Email:- cyboult@gmail.com
#contact me at discord   :-  cyboult_

