import cv2
import mediapipe as mp
import numpy as np
import os

print("DEBUG - MediaPipe is loading from:", mp.__file__)

# איתור אוטומטי של הסרטון שלך בשולחן העבודה
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
video_path = 'C:\AInav_Projects\IMG_4570.MOV'   #os.path.join(desktop_path, "IMG_4570.MOV")

print("טוען את הסרטון ומאתחל את מערכת הניווט...")

# אתחול הראייה הממוחשבת
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # אופטימיזציה לסרטון 4K אנכי (Downsampling לגודל מסך נוח)
    h, w, _ = frame.shape
    target_height = 720
    scale = target_height / h
    target_width = int(w * scale)
    small_frame = cv2.resize(frame, (target_width, target_height))

    # עיבוד שלד היד לחסימות (Occlusion)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    # כותרת חיווי
    cv2.putText(small_frame, "AInav Core - Markerless Tracking", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if results.multi_hand_landmarks:
        cv2.putText(small_frame, "STATE: Occlusion (Hand Mask Active)", (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                small_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        cv2.putText(small_frame, "STATE: Clear Line of Sight", (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # הצגת התוצאה במסך
    cv2.imshow('AInav Core Engine', small_frame)

    # לחיצה על Q תסגור את הסרטון
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("ההרצה הסתיימה."