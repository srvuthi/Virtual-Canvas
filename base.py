import cv2
import numpy as np
import mediapipe as mp
import math

# -----------------------------
# Helper: angle between 3 points
# -----------------------------
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle


def is_fist(hand_landmarks, w, h):
    """
    Detect if hand is making a fist by checking if all fingertips
    are below their respective PIP joints (curled down)
    """
    # Finger tip and PIP joint landmark indices
    fingers = [
        (8, 6),   # Index
        (12, 10), # Middle
        (16, 14), # Ring
        (20, 18)  # Pinky
    ]
    
    curled_count = 0
    for tip_idx, pip_idx in fingers:
        tip_y = hand_landmarks.landmark[tip_idx].y * h
        pip_y = hand_landmarks.landmark[pip_idx].y * h
        
        # Finger is curled if tip is below (greater y value) PIP joint
        if tip_y > pip_y:
            curled_count += 1
    
    # Fist if at least 3 fingers are curled
    return curled_count >= 3


# Webcam
cap = cv2.VideoCapture(0)

# Canvas
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

prev_x, prev_y = None, None
erase_prev_x, erase_prev_y = None, None

print("Virtual Canvas started")
print("Index finger UP = DRAW (green)")
print("FIST = ERASE (red)")
print("'c' = clear canvas | 'q' = quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    mode = "None"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            h, w, _ = frame.shape

            # Check if fist is made
            if is_fist(hand_landmarks, w, h):
                mode = "ERASE"
                
                # Use wrist position as eraser center
                wrist = (int(hand_landmarks.landmark[0].x * w),
                        int(hand_landmarks.landmark[0].y * h))
                
                # Erase in a circular area
                cv2.circle(canvas, wrist, 30, (0, 0, 0), -1)
                
                # Show eraser cursor
                cv2.circle(frame, wrist, 30, (0, 0, 255), 2)
                
                prev_x, prev_y = None, None
            else:
                # ---- Index finger landmarks ----
                mcp = (int(hand_landmarks.landmark[5].x * w),
                       int(hand_landmarks.landmark[5].y * h))
                pip = (int(hand_landmarks.landmark[6].x * w),
                       int(hand_landmarks.landmark[6].y * h))
                dip = (int(hand_landmarks.landmark[7].x * w),
                       int(hand_landmarks.landmark[7].y * h))
                tip = (int(hand_landmarks.landmark[8].x * w),
                       int(hand_landmarks.landmark[8].y * h))

                # Angle at PIP joint
                angle = calculate_angle(mcp, pip, tip)

                # Draw only if finger is straight
                if angle > 160:
                    mode = "DRAW"
                    if prev_x is not None:
                        cv2.line(canvas, (prev_x, prev_y), tip, (0, 255, 0), 5)
                    prev_x, prev_y = tip
                else:
                    prev_x, prev_y = None, None

                # Show angle (debug)
                cv2.putText(frame, f"Index angle: {int(angle)}",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)

            # Draw hand skeleton
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    else:
        prev_x, prev_y = None, None

    # Display current mode
    color = (0, 255, 0) if mode == "DRAW" else (0, 0, 255) if mode == "ERASE" else (255, 255, 255)
    cv2.putText(frame, f"Mode: {mode}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, color, 2)

    combined = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)
    cv2.imshow("Virtual Canvas", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)

cap.release()
cv2.destroyAllWindows()
print("Virtual Canvas closed.")
