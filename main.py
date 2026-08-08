import cv2
import mediapipe as mp
import math
import tkinter as tk
from PIL import Image, ImageTk


# =========================
# SETTINGS
# =========================

IMAGE_PATH = "images.jfif"

MIN_SCALE = 0.3     # min scale of picture
MAX_SCALE = 10.0    # max scale of picture
ZOOM_SPEED = 0.05   # approach speed


# =========================
# MEDIAPIPE
# =========================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)


# =========================
# CAMERA
# =========================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Failed: camera is not found")
    exit()


# =========================
# IMAGE
# =========================

original_image = Image.open(IMAGE_PATH)

scale = 1.0


# =========================
# WINDOW WITH PHOTO
# =========================

root = tk.Tk()
root.title("Gesture Image Viewer")
root.geometry("900x650")

image_label = tk.Label(root)
image_label.pack(fill="both", expand=True)


# =========================
# WINDOW WITH CAMERA
# =========================

camera_window = tk.Toplevel(root)
camera_window.title("Camera")
camera_window.geometry("640x480")

camera_label = tk.Label(camera_window)
camera_label.pack(fill="both", expand=True)



# =========================
# DISTANCE BETWEEN FINGERS
# =========================

def distance(point1, point2):

    return math.sqrt(
        (point1.x - point2.x) ** 2 +
        (point1.y - point2.y) ** 2
    )


# =========================
# PHOTO DISPLAY
# =========================

def show_image():

    global scale

    width = int(original_image.width * scale)
    height = int(original_image.height * scale)

    width = max(1, width)
    height = max(1, height)

    image = original_image.resize(
        (width, height),
        Image.Resampling.LANCZOS
    )

    photo = ImageTk.PhotoImage(image)

    image_label.configure(image=photo)
    image_label.image = photo


# =========================
# SHOW CAMERA
# =========================

def show_camera(frame):

    # OpenCV BGR -> RGB
    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    image = Image.fromarray(frame_rgb)

    # Camera window size
    image = image.resize(
        (640, 480),
        Image.Resampling.LANCZOS
    )

    photo = ImageTk.PhotoImage(image)

    camera_label.configure(image=photo)
    camera_label.image = photo


# =========================
# CAMERA PROCESSING
# =========================

def process_camera():

    global scale

    ret, frame = camera.read()

    if not ret:
        root.after(10, process_camera)
        return

    # DSLR camera
    frame = cv2.flip(frame, 1)

    # BGR -> RGB
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    result = hands.process(rgb)

    # =========================
    # IF WE FAND HAND
    # =========================

    if result.multi_hand_landmarks:

        hand = result.multi_hand_landmarks[0]

        # big finder
        thumb = hand.landmark[
            mp_hands.HandLandmark.THUMB_TIP
        ]

        # Index finger
        index = hand.landmark[
            mp_hands.HandLandmark.INDEX_FINGER_TIP
        ]

        # Distanse
        dist = distance(
            thumb,
            index
        )

        print(
            f"distance = {dist:.3f}, "
            f"scale = {scale:.2f}"
        )

        # =========================
        # ZOOM
        # =========================

        if dist < 0.1:

            scale -= ZOOM_SPEED

        elif dist > 0.3:

            scale += ZOOM_SPEED

        scale = max(
            MIN_SCALE,
            min(MAX_SCALE, scale)
        )

        # =========================
        # DRAWING A HAND ON THE CAMERA
        # =========================

        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        # Finger coordinates
        h, w, _ = frame.shape

        thumb_x = int(thumb.x * w)
        thumb_y = int(thumb.y * h)

        index_x = int(index.x * w)
        index_y = int(index.y * h)

        # Line between the fingers
        cv2.line(
            frame,
            (thumb_x, thumb_y),
            (index_x, index_y),
            (0, 255, 0),
            3
        )

        # Dots
        cv2.circle(
            frame,
            (thumb_x, thumb_y),
            10,
            (0, 0, 255),
            -1
        )

        cv2.circle(
            frame,
            (index_x, index_y),
            10,
            (0, 0, 255),
            -1
        )

        # Info
        cv2.putText(
            frame,
            f"Distance: {dist:.3f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Scale: {scale:.2f}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    else:

        cv2.putText(
            frame,
            "Hand not detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    # Show camera
    show_camera(frame)

    # Update photo
    show_image()

    # Next frame
    root.after(
        10,
        process_camera
    )


# =========================
# CLOSE
# =========================

def close_program():

    camera.release()
    hands.close()

    root.destroy()


root.protocol(
    "WM_DELETE_WINDOW",
    close_program
)

camera_window.protocol(
    "WM_DELETE_WINDOW",
    close_program
)


# =========================
# START
# =========================

show_image()

root.after(
    10,
    process_camera
)

root.mainloop()
