import cv2
import face_recognition
import pickle
import os
from checkin_checkout import update_visitor_log
from new_user_registration import register_new_user

ENCODINGS_DIR = "encodings"

# Load all encodings
known_encodings = []
known_names = []
known_cnic = []

for filename in os.listdir(ENCODINGS_DIR):
    if filename.endswith(".pkl"):
        with open(os.path.join(ENCODINGS_DIR, filename), "rb") as f:
            data = pickle.load(f)
            known_encodings.append(data["encoding"])
            known_names.append(data["name"])
            known_cnic.append(data["cnic"])

video_capture = cv2.VideoCapture(0)

print("🔍 Scanning for faces...")

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_names[match_index]
            cnic = known_cnic[match_index]
            
            print(f"\n✅ Face recognized: {name} - {cnic}")
            update_visitor_log(name, cnic)
            print("🟢 Checked in/out successfully!")
            print("🙏 Thanks for visiting Tapal.")
        else:
            print("\n❌ Face not recognized.")
            print("👤 Registering new user...")
            register_new_user(face_encoding ,  frame)  # You must modify this to accept frame input
            print("📝 New user registered successfully.")

    cv2.imshow('Face Scanner - Tap Q to Quit', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
