import cv2
import mediapipe as mp
import numpy as np

# Initialize camera
cap = cv2.VideoCapture(0)

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

canvas = None
prev_x, prev_y = 0, 0

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    if canvas is None:
        canvas = np.zeros((h, w, 3), np.uint8)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            # Index finger tip
            x = int(hand.landmark[8].x * w)
            y = int(hand.landmark[8].y * h)

            # Check if finger is up
            if hand.landmark[8].y < hand.landmark[6].y:

                cv2.circle(frame, (x, y), 8, (255, 0, 255), -1)

                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x, y

                cv2.line(canvas, (prev_x, prev_y), (x, y), (255, 0, 255), 5)
                prev_x, prev_y = x, y

            else:
                prev_x, prev_y = 0, 0

    # Merge drawing with frame
    frame = cv2.add(frame, canvas)

    cv2.imshow("Air Canvas", frame)

    key = cv2.waitKey(1)

    if key == ord('c'):
        canvas = np.zeros((h, w, 3), np.uint8)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()