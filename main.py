import cv2
import face_recognition
import pickle
import os
import time
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

last_seen = None
seen_start_time = None
RECOGNITION_DELAY = 1  # seconds

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    current_time = time.time()

    for i, face_encoding in enumerate(face_encodings):
        name = "New User"
        label_to_draw = "New User"

        matches = face_recognition.compare_faces(known_encodings, face_encoding)

        if True in matches:
            match_index = matches.index(True)
            name = known_names[match_index]
            cnic = known_cnic[match_index]
            label_to_draw = name

            if last_seen == cnic:
                if current_time - seen_start_time >= RECOGNITION_DELAY:
                    print(f"\n✅ Face recognized: {name} - {cnic}")
                    update_visitor_log(name, cnic)
                    print("🟢 Checked in/out successfully!")
                    print("🙏 Thanks for visiting Tapal.")
                    last_seen = None
                    seen_start_time = None
            else:
                last_seen = cnic
                seen_start_time = current_time

        else:
            if last_seen == "Unknown":
                if current_time - seen_start_time >= RECOGNITION_DELAY:
                    print("\n❌ Face not recognized.")
                    print("👤 Registering new user...")
                    register_new_user(face_encoding, frame)
                    print("📝 New user registered successfully.")
                    last_seen = None
                    seen_start_time = None
            else:
                last_seen = "Unknown"
                seen_start_time = current_time

        # Draw a box and label
        top, right, bottom, left = face_locations[i]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 20), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, label_to_draw, (left + 6, bottom - 6), font, 0.5, (0, 0, 0), 1)

    if not face_encodings:
        last_seen = None
        seen_start_time = None

    cv2.imshow('Face Scanner - Tap Q to Quit', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
