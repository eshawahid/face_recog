import cv2
import requests
from datetime import datetime

url = 'http://127.0.0.1:5000/recognize'
video_file_path = 'C:\\Users\\PMLS\\Desktop\\face_recognition\\face_recognition\\video\\1.mp4'

video_capture = cv2.VideoCapture(video_file_path)

recognized_name = None
timestamp = None

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break

    try:
        _, buffer = cv2.imencode('.jpg', frame)
        response = requests.post(url, files={'image': ('frame.jpg', buffer.tobytes(), 'image/jpeg')})

        if response.status_code == 200:
            result = response.json()
            recognized_name = result.get('name', 'Unknown')
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            break  # Exit loop as soon as a face is recognized

    except Exception as e:
        print(f"Error processing frame: {e}")
        continue

status_text = f"Face Recognized: {recognized_name}" if recognized_name else "Face not matched"

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break

    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if recognized_name:
    print(f"Face Matched: {recognized_name}, Time: {timestamp}")
else:
    print("Face not matched")

video_capture.release()
cv2.destroyAllWindows()
