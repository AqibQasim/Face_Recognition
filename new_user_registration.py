import os
import pickle
import cv2

ENCODING_DIR = 'encodings'

def save_face_encoding(name, cnic, encoding):
    """Saves face encoding with name and CNIC into a .pkl file"""
    if not os.path.exists(ENCODING_DIR):
        os.makedirs(ENCODING_DIR)
    filename = f"{name.replace(' ', '_')}_{cnic}.pkl"
    data = {'name': name, 'cnic': cnic, 'encoding': encoding}
    with open(os.path.join(ENCODING_DIR, filename), 'wb') as f:
        pickle.dump(data, f)
    print(f"[INFO] Saved encoding for {name} ({cnic})")


def register_new_user(encoding, frame):
    """
    Triggered from main.py when a face is unrecognized.
    Accepts face encoding and frame, asks user for details, and saves.
    """
    name = input("📝 Enter your full name: ")
    cnic = input("🪪 Enter your CNIC: ")
    save_face_encoding(name, cnic, encoding)

    # Show welcome message briefly
    cv2.putText(frame, f"Welcome, {name}!", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "You are now registered!", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Registration Successful", frame)
    cv2.waitKey(2000)  
    cv2.destroyWindow("Registration Successful")
