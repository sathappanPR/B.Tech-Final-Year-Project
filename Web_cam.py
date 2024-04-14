# from ultralytics import YOLO
# import cv2
# while True:
#     model = YOLO(r"D:\Finalyearproject\80_epochs\runs\detect\train\weights\best.pt")
#     results = model.predict(source = "0", show = True)
#     print(results)

from ultralytics import YOLO
import cv2

# Load YOLOv5 model with the specified weights
model = YOLO(r"D:\Finalyearproject\80_epochs\runs\detect\train\weights\best.pt")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame from webcam
    ret, frame = cap.read()

    # Perform object detection on the frame and display the results
    results = model.predict(frame, show=True)
    print(results)

    # Check for key press
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
