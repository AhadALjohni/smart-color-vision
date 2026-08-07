import cv2
import time


# =========================================================
# إعدادات البرنامج
# =========================================================

CAMERA_INDEX = 0

MIN_AREA = 700

SHOW_MASK = False

BRIGHTNESS_VALUE = 25


# =========================================================
# فتح الكاميرا
# =========================================================

camera = cv2.VideoCapture(CAMERA_INDEX)

if not camera.isOpened():
    print("Could not open the camera.")
    raise SystemExit



camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


# =========================================================
# تعريف الألوان
# =========================================================

colors = {

    "Orange": {
        "lower": (11, 100, 80),
        "upper": (24, 255, 255),
        "box_color": (0, 165, 255)
    },

    "Yellow": {
        "lower": (25, 90, 90),
        "upper": (34, 255, 255),
        "box_color": (0, 255, 255)
    },

    "Green": {
        "lower": (35, 60, 60),
        "upper": (85, 255, 255),
        "box_color": (0, 255, 0)
    },

    "Blue": {
        "lower": (90, 70, 60),
        "upper": (120, 255, 255),
        "box_color": (255, 0, 0)
    },

    "Indigo": {
        "lower": (121, 60, 50),
        "upper": (135, 255, 255),
        "box_color": (130, 0, 75)
    },

    "Violet": {
        "lower": (136, 50, 50),
        "upper": (169, 255, 255),
        "box_color": (255, 0, 255)
    }
}


lower_red1 = (0, 90, 70)
upper_red1 = (10, 255, 255)

lower_red2 = (170, 90, 70)
upper_red2 = (180, 255, 255)

red_box_color = (0, 0, 255)


# =========================================================
# Mask
# =========================================================

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (7, 7)
)


# =========================================================
# FPS
# =========================================================

previous_time = time.time()

fps = 0.0

smoothed_fps = 0.0


# =========================================================
# دالة تحسين الإضاءة
# =========================================================

def improve_brightness(frame):

    hsv_image = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )

    h, s, v = cv2.split(hsv_image)

    v = cv2.add(
        v,
        BRIGHTNESS_VALUE
    )

    hsv_image = cv2.merge(
        (h, s, v)
    )

    improved = cv2.cvtColor(
        hsv_image,
        cv2.COLOR_HSV2BGR
    )

    return improved


# =========================================================
# =========================================================

def draw_transparent_panel(
    frame,
    x1,
    y1,
    x2,
    y2,
    alpha=0.55
):

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        (0, 0, 0),
        -1
    )

    cv2.addWeighted(
        overlay,
        alpha,
        frame,
        1 - alpha,
        0,
        frame
    )


# =========================================================
# =========================================================

def draw_label(
    frame,
    text,
    x,
    y,
    color
):

    font = cv2.FONT_HERSHEY_SIMPLEX

    font_scale = 0.6

    thickness = 2

    text_size, baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness
    )

    text_width = text_size[0]
    text_height = text_size[1]

    label_y = y - 10

    if label_y - text_height - 10 < 0:
        label_y = y + text_height + 20

    top_left = (
        x,
        label_y - text_height - 10
    )

    bottom_right = (
        x + text_width + 14,
        label_y + 5
    )

    cv2.rectangle(
        frame,
        top_left,
        bottom_right,
        color,
        -1
    )

    cv2.putText(
        frame,
        text,
        (x + 7, label_y - 2),
        font,
        font_scale,
        (255, 255, 255),
        thickness
    )


# =========================================================
# =========================================================

def process_color(
    frame,
    mask,
    color_name,
    box_color
):

    object_count = 0

    # -----------------------------
    # -----------------------------

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Blur بسيط
    mask = cv2.GaussianBlur(
        mask,
        (5, 5),
        0
    )

    
    _, mask = cv2.threshold(
        mask,
        127,
        255,
        cv2.THRESH_BINARY
    )


    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area < MIN_AREA:
            continue

        object_count += 1

    

        x, y, w, h = cv2.boundingRect(
            contour
        )

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            box_color,
            2
        )


        cv2.drawContours(
            frame,
            [contour],
            -1,
            box_color,
            2
        )



        moments = cv2.moments(
            contour
        )

        if moments["m00"] != 0:

            center_x = int(
                moments["m10"] /
                moments["m00"]
            )

            center_y = int(
                moments["m01"] /
                moments["m00"]
            )

        else:

            center_x = x + w // 2
            center_y = y + h // 2


        cv2.circle(
            frame,
            (center_x, center_y),
            5,
            box_color,
            -1
        )

        cv2.circle(
            frame,
            (center_x, center_y),
            9,
            (255, 255, 255),
            2
        )



        label_text = (
            f"{color_name} | "
            f"Area: {int(area)} | "
            f"({center_x},{center_y})"
        )

        draw_label(
            frame,
            label_text,
            x,
            y,
            box_color
        )

    return object_count, mask

while True:

    success, frame = camera.read()

    if not success:
        break


    frame = cv2.flip(
        frame,
        1
    )


   
    frame = improve_brightness(
        frame
    )


    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )



    total_objects = 0

    color_counts = {}

    all_masks = []



    red_mask1 = cv2.inRange(
        hsv,
        lower_red1,
        upper_red1
    )

    red_mask2 = cv2.inRange(
        hsv,
        lower_red2,
        upper_red2
    )

    red_mask = cv2.bitwise_or(
        red_mask1,
        red_mask2
    )


    red_count, cleaned_red_mask = process_color(
        frame,
        red_mask,
        "Red",
        red_box_color
    )

    color_counts["Red"] = red_count

    total_objects += red_count

    all_masks.append(
        cleaned_red_mask
    )


    for color_name, color_data in colors.items():

        mask = cv2.inRange(
            hsv,
            color_data["lower"],
            color_data["upper"]
        )

        count, cleaned_mask = process_color(
            frame,
            mask,
            color_name,
            color_data["box_color"]
        )

        color_counts[color_name] = count

        total_objects += count

        all_masks.append(
            cleaned_mask
        )



    current_time = time.time()

    delta_time = (
        current_time -
        previous_time
    )

    previous_time = current_time

    if delta_time > 0:

        fps = 1 / delta_time

        # تنعيم FPS
        smoothed_fps = (
            0.9 * smoothed_fps
            +
            0.1 * fps
        )


    draw_transparent_panel(
        frame,
        15,
        15,
        310,
        150,
        0.60
    )


    cv2.putText(
        frame,
        " COLOR ",
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"FPS: {smoothed_fps:.1f}",
        (30, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )


    cv2.putText(
        frame,
        f"Objects: {total_objects}",
        (30, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )


    detected = []

    for color_name, count in color_counts.items():

        if count > 0:

            detected.append(
                f"{color_name}:{count}"
            )


    if detected:

        detected_text = "  ".join(
            detected
        )

    else:

        detected_text = "Waiting for colors..."


    cv2.putText(
        frame,
        detected_text,
        (30, 132),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (200, 200, 200),
        1
    )


    height, width = frame.shape[:2]

    cv2.putText(
        frame,
        "Q: Quit    M: Mask",
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1
    )

    cv2.imshow(
        " Color ",
        frame
    )


    if SHOW_MASK:

        combined_mask = all_masks[0]

        for current_mask in all_masks[1:]:

            combined_mask = cv2.bitwise_or(
                combined_mask,
                current_mask
            )

        cv2.imshow(
            "Detection Mask",
            combined_mask
        )

    else:

        cv2.destroyWindow(
            "Detection Mask"
        ) if cv2.getWindowProperty(
            "Detection Mask",
            cv2.WND_PROP_VISIBLE
        ) >= 1 else None


    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):
        break

    if key == ord("m"):

        SHOW_MASK = not SHOW_MASK


camera.release()

cv2.destroyAllWindows()