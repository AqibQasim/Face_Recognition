import cv2
import face_recognition
import os
import pickle
import numpy as np

ENCODING_DIR = 'encodings'

# Load all existing encodings
def load_known_faces():
    known_encodings = []
    known_names = []

    for file in os.listdir(ENCODING_DIR):
        if file.endswith('.pkl'):
            with open(os.path.join(ENCODING_DIR, file), 'rb') as f:
                data = pickle.load(f)
                known_encodings.append(data['encoding'])
                known_names.append(data['name'])
    return known_encodings, known_names


# Save new face encoding
def save_face_encoding(name, cnic, encoding):
    filename = f"{name.replace(' ', '_')}_{cnic}.pkl"
    data = {'name': name, 'cnic': cnic, 'encoding': encoding}
    with open(os.path.join(ENCODING_DIR, filename), 'wb') as f:
        pickle.dump(data, f)
    print(f"[INFO] Saved encoding for {name} ({cnic})")


def main():
    known_encodings, known_names = load_known_faces()

    cap = cv2.VideoCapture(0)
    print("[INFO] Starting webcam...")

    while True:
        ret, frame = cap.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb)
        face_encodings = face_recognition.face_encodings(rgb, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                matched_idx = matches.index(True)
                name = known_names[matched_idx]
                cv2.putText(frame, f"Welcome Back: {name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            else:
                cv2.putText(frame, "New Visitor - Press 'r' to Register", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.imshow("Visitor Management - Registration", frame)

        key = cv2.waitKey(1) & 0xFF

        # Press 'r' to register new user
        if key == ord('r') and face_encodings:
            name = input("Enter your full name: ")
            cnic = input("Enter your CNIC: ")
            save_face_encoding(name, cnic, face_encodings[0])
            known_encodings.append(face_encodings[0])
            known_names.append(name)

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
