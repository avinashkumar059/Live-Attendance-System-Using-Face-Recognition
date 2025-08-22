import cv2
import os
import csv
import json
import time
import numpy as np
from datetime import datetime
from keras_facenet import FaceNet
from sklearn.neighbors import KNeighborsClassifier

# Create attendance directory if it doesn't exist
attendance_dir = "attendance"
os.makedirs(attendance_dir, exist_ok=True)

def load_label_map():
    with open("label_map.json", "r") as f:
        data = json.load(f)
    # Reverse mapping {int_id -> label_str}
    return {int(v): k for k, v in data.items()}

def get_today_filename():
    today = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(attendance_dir, f"attendance_{today}.csv")

def already_marked_today(enroll, filename):
    if not os.path.exists(filename):
        return False
    with open(filename, 'r') as f:
        for line in f.readlines():
            if enroll in line:
                return True
    return False

def recognize_faces():
    # ✅ Check model files
    if not (os.path.exists("faces_embeddings.npy") and 
            os.path.exists("faces_labels.npy") and 
            os.path.exists("label_map.json")):
        print("❌ Model files not found. Run train_model.py first.")
        return

    # ✅ Load embeddings & labels
    faces_embeddings = np.load("faces_embeddings.npy", allow_pickle=True)
    faces_labels = np.load("faces_labels.npy", allow_pickle=True)
    label_map = load_label_map()

    print(f"🔍 Loaded {len(faces_embeddings)} embeddings for {len(set(faces_labels))} classes")

    # ✅ Train KNN classifier
    knn = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
    knn.fit(list(faces_embeddings), list(faces_labels))

    # ✅ Load FaceNet model
    embedder = FaceNet()

    # ✅ Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)
    attendance = {}
    filename = get_today_filename()

    if not os.path.exists(filename):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Date', 'Time', 'Enrollment', 'Name'])

    print("\n📸 Starting FaceNet recognition. Will run for 15 seconds...\n")

    start_time = time.time()
    end_time = start_time + 15

    while time.time() < end_time:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to access webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]  
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

            try:
                # ✅ Get embedding
                embeddings = embedder.embeddings([face_rgb])
                if embeddings is None or len(embeddings) == 0:
                    raise ValueError("No embedding generated")
                emb = embeddings[0]

                # ✅ Predict using KNN
                pred_id = knn.predict([emb])[0]
                dist, _ = knn.kneighbors([emb], n_neighbors=1)
                distance = dist[0][0]

                # ✅ Threshold for unknown
                THRESHOLD = 0.9
                if distance < THRESHOLD:
                    label = label_map.get(int(pred_id), "Unknown")
                else:
                    label = "Unknown"

            except Exception as e:
                print(f"⚠️ Error processing face: {e}")
                cv2.putText(frame, "Error", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                continue

            # ✅ Mark attendance
            if label != "Unknown":
                try:
                    enroll, name = label.split('_', 1)
                except ValueError:
                    enroll, name = "???", "Unknown"

                if enroll not in attendance and not already_marked_today(enroll, filename):
                    now = datetime.now()
                    date = now.strftime("%Y-%m-%d")
                    time_str = now.strftime("%H:%M:%S")
                    attendance[enroll] = (date, time_str, enroll, name)

                cv2.putText(frame, f"{name} ({enroll})", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Unknown", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

        # ✅ Show remaining time
        remaining = int(end_time - time.time())
        cv2.putText(frame, f"Time left: {remaining}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("FaceNet Recognition Attendance", frame)
        if cv2.waitKey(1) == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

    # ✅ Save attendance
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        for record in attendance.values():
            writer.writerow(record)

    print(f"\n✅ Attendance recorded in '{filename}'")

if __name__ == "__main__":
    recognize_faces()
