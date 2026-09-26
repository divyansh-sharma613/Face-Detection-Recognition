import cv2
import os
import numpy as np


# ==========================================
# FACE DETECTION & RECOGNITION SYSTEM
# CODSOFT TASK 5
# ==========================================


# Face detector
face_detector = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)


# LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()


# ==========================================
# DATASET
# ==========================================

dataset_path = "dataset"

faces = []
labels = []

label_names = {}

current_label = 0


# ==========================================
# CHECK DATASET
# ==========================================

if not os.path.exists(dataset_path):

    print("Dataset folder not found.")
    print("Please create the dataset first.")

    exit()


# ==========================================
# LOAD FACE IMAGES
# ==========================================

for person_name in os.listdir(dataset_path):

    person_path = os.path.join(
        dataset_path,
        person_name
    )

    if not os.path.isdir(person_path):
        continue


    label_names[current_label] = person_name


    for image_name in os.listdir(person_path):

        image_path = os.path.join(
            person_path,
            image_name
        )


        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )


        if image is None:
            continue


        detected_faces = face_detector.detectMultiScale(
            image,
            scaleFactor=1.2,
            minNeighbors=5
        )


        for (x, y, w, h) in detected_faces:

            faces.append(
                image[y:y+h, x:x+w]
            )

            labels.append(
                current_label
            )


    current_label += 1


# ==========================================
# TRAIN RECOGNIZER
# ==========================================

if len(faces) == 0:

    print("No face images found in dataset.")

    exit()


print("Training face recognition model...")

recognizer.train(
    faces,
    np.array(labels)
)

print("Training completed successfully.")


# ==========================================
# START WEBCAM
# ==========================================

camera = cv2.VideoCapture(0)


print()
print("Face Detection and Recognition Started")
print("Press 'q' to quit.")


while True:

    ret, frame = camera.read()


    if not ret:

        print("Unable to access webcam.")

        break


    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    detected_faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5
    )


    for (x, y, w, h) in detected_faces:

        face = gray[y:y+h, x:x+w]


        label, confidence = recognizer.predict(face)


        # Convert confidence into similarity percentage
        similarity = max(
            0,
            min(
                100,
                100 - confidence
            )
        )


        if similarity >= 50:

            name = label_names.get(
                label,
                "Unknown"
            )

        else:

            name = "Unknown"


        # Draw rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )


        # Display name
        cv2.putText(
            frame,
            name,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


    # Display window
    cv2.imshow(
        "Face Detection and Recognition - CODSOFT Task 5",
        frame
    )


    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ==========================================
# CLOSE CAMERA
# ==========================================

camera.release()

cv2.destroyAllWindows()

print("Program closed.")