import cv2
import os

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def extract_faces_from_frames(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):
        if file.endswith(".jpg"):
            img_path = os.path.join(input_folder, file)
            img = cv2.imread(img_path)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for i, (x, y, w, h) in enumerate(faces):
                face = img[y:y+h, x:x+w]

                face_path = os.path.join(output_folder, f"{file}_face_{i}.jpg")
                cv2.imwrite(face_path, face)

    print(f"Faces extracted → {output_folder}")