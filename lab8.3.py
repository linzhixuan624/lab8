import cv2
import numpy as np

def fly_on_center(frame, fly_pic, x, y, w, h):

    center_x = x + w // 2
    center_y = y + h // 2

    fly_h = fly_pic.shape[0]
    fly_w = fly_pic.shape[1]

    fly_overlay_x = center_x - fly_w // 2
    fly_overlay_y = center_y - fly_h // 2
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]

    fly_x_start = max(0, fly_overlay_x)
    fly_y_start = max(0, fly_overlay_y)

    fly_x_end = min(frame_w, fly_overlay_x + fly_w)
    fly_y_end = min(frame_h, fly_overlay_y + fly_h)

    fly_crop_x_start = max(0, -fly_overlay_x)
    fly_crop_y_start = max(0, -fly_overlay_y)
    fly_crop_x_end = fly_crop_x_start + (fly_x_end - fly_x_start)
    fly_crop_y_end = fly_crop_y_start + (fly_y_end - fly_y_start)

    if fly_x_start < fly_x_end and fly_y_start < fly_y_end:
        frame_roi = frame[fly_y_start:fly_y_end, fly_x_start:fly_x_end]
        fly_cropped = fly_pic[fly_crop_y_start:fly_crop_y_end, fly_crop_x_start:fly_crop_x_end]

        if fly_cropped.shape[2] == 4:
            alpha = fly_cropped[:, :, 3] / 255.0
            alpha_inv = 1.0 - alpha

            for c in range(0, 3):
                frame_roi[:, :, c] = (alpha * fly_cropped[:, :, c] +
                                      alpha_inv * frame_roi[:, :, c])
        else:
            frame_roi[:] = fly_cropped

if __name__ == "__main__":
    fly_image = cv2.imread('fly64.png', cv2.IMREAD_UNCHANGED)
    
    cap = cv2.VideoCapture(0)
    target_size = (640, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("error")
            break

        frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
        _, binary_frame = cv2.threshold(blur_frame, 80, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            main_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(main_contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            fly_on_center(frame, fly_image, x, y, w, h)

        cv2.imshow('image', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()