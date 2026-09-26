import cv2
import os

name = input("Enter your name: ")

dataset_path = os.path.join("dataset", name)

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

camera = cv2.VideoCapture(0)

face_detector = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

count = 0

print("Camera started.")
print("Look at the camera...")
print("Press 'q' to stop.")

while True:

    ret, frame = camera.read()

    if not ret:
        print("Unable to access webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        count += 1

        face = gray[y:y+h, x:x+w]

        file_path = os.path.join(
            dataset_path,
            f"{count}.jpg"
        )

        cv2.imwrite(file_path, face)

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Images: {count}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("Face Dataset Collection", frame)

    if count >= 20:
        print("20 face images captured successfully.")
        break

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print("Dataset saved successfully.")