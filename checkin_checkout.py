import cv2
import face_recognition
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

import datetime
checked_in_users = {}


ENCODING_DIR = 'encodings'
EXCEL_LOG = 'visitors.xlsx'


# Load known encodings
def load_known_faces():
    encodings, names, cnics = [], [], []
    for file in os.listdir(ENCODING_DIR):
        if file.endswith('.pkl'):
            with open(os.path.join(ENCODING_DIR, file), 'rb') as f:
                data = pickle.load(f)
                encodings.append(data['encoding'])
                names.append(data['name'])
                cnics.append(data['cnic'])
    return encodings, names, cnics


def handle_checkin_checkout(name, cnic):
    now = datetime.datetime.now()
    if cnic in checked_in_users:
        last_checkin = checked_in_users[cnic]
        delta = now - last_checkin

        if delta.total_seconds() < 300:  # less than 5 minutes
            print(f"{name} already checked in recently.")
            return
        else:
            # Mark as checkout
            checked_in_users.pop(cnic)
            print(f"{name} checked out at {now}")
            update_excel(name, cnic, checkin=False, timestamp=now)
    else:
        # First-time check-in
        checked_in_users[cnic] = now
        print(f"{name} checked in at {now}")
        update_excel(name, cnic, checkin=True, timestamp=now)



# Load or create Excel log
def load_log():
    if os.path.exists(EXCEL_LOG):
        return pd.read_excel(EXCEL_LOG)
    return pd.DataFrame(columns=["Name", "CNIC", "Check-in", "Check-out"])


# Save log to Excel
def save_log(df):
    df.to_excel(EXCEL_LOG, index=False)


# Check if already checked in but not out
def is_checked_in(df, cnic):
    last_entry = df[df["CNIC"] == cnic]
    if not last_entry.empty and pd.isna(last_entry.iloc[-1]["Check-out"]):
        return True
    return False


def update_excel(name, cnic, checkin=True, timestamp=None):
    import pandas as pd

    file = 'visitors.xlsx'
    timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")

    try:
        df = pd.read_excel(file)
    except FileNotFoundError:
        df = pd.DataFrame(columns=["Name", "CNIC", "Check-in", "Check-out"])

    row_index = df[df["CNIC"] == cnic].index

    if len(row_index) == 0:
        new_row = {"Name": name, "CNIC": cnic, "Check-in": timestamp_str if checkin else "", "Check-out": timestamp_str if not checkin else ""}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        idx = row_index[0]
        if checkin:
            if pd.isna(df.at[idx, "Check-in"]) or df.at[idx, "Check-in"] == "":
                df.at[idx, "Check-in"] = timestamp_str
            else:
                df.at[idx, "Check-in"] += f" | {timestamp_str}"
        else:
            if pd.isna(df.at[idx, "Check-out"]) or df.at[idx, "Check-out"] == "":
                df.at[idx, "Check-out"] = timestamp_str
            else:
                df.at[idx, "Check-out"] += f" | {timestamp_str}"

    df.to_excel(file, index=False)



def main():
    known_encodings, known_names, known_cnics = load_known_faces()
    log_df = load_log()

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
                idx = matches.index(True)
                name = known_names[idx]
                cnic = known_cnics[idx]

                handle_checkin_checkout(name, cnic)
                status = f"Detected: {name}"
            else:
                status = "Visitor not recognized"

            # Draw rectangle and label
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, status, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Visitor Management - Checkin/Checkout", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    save_log(log_df)
    print("[INFO] Log saved to Excel.")

if __name__ == "__main__":
    main()