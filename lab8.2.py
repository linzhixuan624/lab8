import cv2


left_count, right_count = 0, 0
cap = cv2.VideoCapture(0)
target_size = (640, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, target_size, cv2.INTER_LINEAR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

    frame_w = frame.shape[1]
    cv2.line(frame, (frame_w//2, 0), (frame_w//2, frame.shape[0]), (0, 255, 255), 2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        center_x = x + w // 2
        if center_x < frame_w // 2:
            left_count += 1
        else:
            right_count += 1

        cv2.putText(frame, f"Left: {left_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Right: {right_count}", (330, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()