import face_recognition
import cv2
import numpy as np
import os
import pickle
from datetime import datetime, timedelta
import pandas as pd
import tkinter as tk
from tkinter import messagebox
import threading
import time

EXCEL_FILE = "visitors.xlsx"
ENCODING_FOLDER = "encodings"

def load_known_faces(folder):
    encodings = []
    names = []
    cnics = []

    for filename in os.listdir(folder):
        if filename.endswith(".pkl"):
            filepath = os.path.join(folder, filename)
            with open(filepath, "rb") as f:
                data = pickle.load(f)
                encodings.append(data["encoding"])
                names.append(data["name"])
                cnics.append(data["cnic"])
    return encodings, names, cnics

def recognize_face(frame, known_encodings, known_names, known_cnic):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.45)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            return known_names[best_match_index], known_cnic[best_match_index]
    return None, None

def load_or_create_excel(file):
    if os.path.exists(file):
        return pd.read_excel(file)
    else:
        df = pd.DataFrame(columns=["Name", "CNIC", "Check-ins", "Check-outs"])
        df.to_excel(file, index=False)
        return df

def update_visitor_log(name, cnic):
    df = load_or_create_excel(EXCEL_FILE)
    cnic = str(cnic)
    now = datetime.now()

    if cnic in df["CNIC"].astype(str).values:
        idx = df[df["CNIC"].astype(str) == cnic].index[0]

        checkins = str(df.at[idx, "Check-ins"]) if pd.notna(df.at[idx, "Check-ins"]) else ""
        checkouts = str(df.at[idx, "Check-outs"]) if pd.notna(df.at[idx, "Check-outs"]) else ""

        checkin_list = checkins.split("\n") if checkins else []
        checkout_list = checkouts.split("\n") if checkouts else []

        def parse_times(lst):
            return [datetime.strptime(t.strip(), "%Y-%m-%d %H:%M:%S") for t in lst if t.strip()]

        parsed_checkins = parse_times(checkin_list)
        parsed_checkouts = parse_times(checkout_list)

        last_checkin = parsed_checkins[-1] if parsed_checkins else None
        last_checkout = parsed_checkouts[-1] if parsed_checkouts else None

        if last_checkin and (not last_checkout or last_checkin > last_checkout):
            if now - last_checkin >= timedelta(minutes=1):
                checkout_list.append(now.strftime("%Y-%m-%d %H:%M:%S"))
                print(f"{name} checked out.")
            else:
                print(f"{name} just checked in recently. Wait before checking out.")
                return
        else:
            if last_checkout and now - last_checkout < timedelta(minutes=1):
                print(f"{name} just checked out. Wait before checking in.")
                return
            checkin_list.append(now.strftime("%Y-%m-%d %H:%M:%S"))
            print(f"{name} checked in.")

        df.at[idx, "Check-ins"] = "\n".join(checkin_list)
        df.at[idx, "Check-outs"] = "\n".join(checkout_list)

    else:
        # New visitor, just log first check-in for now
        df.loc[len(df)] = [name, cnic, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ""]

    df.to_excel(EXCEL_FILE, index=False)

def is_new_user(cnic):
    df = load_or_create_excel(EXCEL_FILE)
    return str(cnic) not in df["CNIC"].astype(str).values

def show_user_form(temp_name, temp_cnic):
    def submit():
        name = name_entry.get()
        cnic = cnic_entry.get()
        if name and cnic:
            encoding_path = os.path.join(ENCODING_FOLDER, f"{name}_{cnic}.pkl")
            with open(encoding_path, "wb") as f:
                pickle.dump({"name": name, "cnic": cnic, "encoding": temp_encoding}, f)
            update_visitor_log(name, cnic)
            messagebox.showinfo("Saved", "New user saved successfully!")
            form.destroy()
        else:
            messagebox.showerror("Missing Info", "Please fill out all fields.")

    form = tk.Tk()
    form.title("New Visitor Registration")

    tk.Label(form, text="Full Name:").pack(pady=5)
    name_entry = tk.Entry(form, width=40)
    name_entry.pack(pady=5)
    name_entry.insert(0, temp_name)

    tk.Label(form, text="CNIC:").pack(pady=5)
    cnic_entry = tk.Entry(form, width=40)
    cnic_entry.pack(pady=5)
    cnic_entry.insert(0, temp_cnic)

    tk.Button(form, text="Submit", command=submit).pack(pady=15)

    form.mainloop()

# Global variable to keep encoding
temp_encoding = None

def main():
    global temp_encoding
    known_encodings, known_names, known_cnic = load_known_faces(ENCODING_FOLDER)
    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit...")

    new_user_triggered = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        name, cnic = recognize_face(frame, known_encodings, known_names, known_cnic)
        if name:
            print(f"Recognized: {name} ({cnic})")

            if is_new_user(cnic):
                if not new_user_triggered:
                    # Save the encoding for the new face
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(rgb_frame)
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    if face_encodings:
                        temp_encoding = face_encodings[0]
                        new_user_triggered = True
                        print("New user detected. Closing camera and showing form in 2 seconds...")
                        cap.release()
                        cv2.destroyAllWindows()
                        time.sleep(2)
                        show_user_form(name, cnic)
                        break
            else:
                update_visitor_log(name, cnic)
                cv2.putText(frame, f"{name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Face Recognition - Check-in/Check-out", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
