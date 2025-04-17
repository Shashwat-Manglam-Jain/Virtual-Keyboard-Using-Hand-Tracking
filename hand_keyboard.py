import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# Define virtual keys
keys = [['Q', 'W', 'E', 'R', 'T', 'Y'],
        ['A', 'S', 'D', 'F', 'G', 'H'],
        ['Z', 'X', 'C', 'V', 'B', 'N']]

key_size = 60  # key width and height

# Draw keyboard on screen
def draw_keyboard(img):
    for i in range(len(keys)):
        for j, key in enumerate(keys[i]):
            x = j * key_size + 100
            y = i * key_size + 100
            cv2.rectangle(img, (x, y), (x + key_size, y + key_size), (255, 0, 0), 2)
            cv2.putText(img, key, (x + 20, y + 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    return img

# Check if finger tip is pressing a key
def detect_pressed_key(x_tip, y_tip):
    for i in range(len(keys)):
        for j, key in enumerate(keys[i]):
            x = j * key_size + 100
            y = i * key_size + 100
            if x < x_tip < x + key_size and y < y_tip < y + key_size:
                return key
    return None

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Set width
cap.set(4, 720)   # Set height

prev_pressed = None
press_time = 0

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read from webcam.")
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    # ✅ Draw keyboard and update the image
    img = draw_keyboard(img)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            # Get tip of index finger (landmark #8)
            index_tip = handLms.landmark[8]
            h, w, _ = img.shape
            x_tip = int(index_tip.x * w)
            y_tip = int(index_tip.y * h)

            # Draw circle on tip
            cv2.circle(img, (x_tip, y_tip), 10, (0, 255, 0), -1)

            # Check if finger is on a key
            key_pressed = detect_pressed_key(x_tip, y_tip)
            current_time = time.time()

            if key_pressed:
                if key_pressed != prev_pressed or (current_time - press_time) > 1:
                    pyautogui.press(key_pressed.lower())
                    print("Pressed:", key_pressed)
                    prev_pressed = key_pressed
                    press_time = current_time

    cv2.imshow("Virtual Hand Keyboard", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
