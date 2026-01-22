import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime

# ---------------------- PATH SETTINGS ----------------------
training_data_dir = r"E:\ML-pro\training_data"

# ---------------------- STUDENT LIST ----------------------
students = [
    "ABDUL WAHID", "ARUN BALAJI R", "MAHESH ARAVIND M.S", "DHANUSH S",
    "BHARATH P", "ANBU SELVAM N", "DHARUN CHANDRU B", "VICE PRINCIPAL"
]

# ---------------------- DAILY CSV WITH VERSION ----------------------
def get_daily_csv():
    today_str = datetime.now().strftime("%Y-%m-%d")
    base_path = rf"E:\ML-pro\attendance_{today_str}.csv"
    if not os.path.exists(base_path):
        return base_path
    i = 1
    while True:
        new_path = rf"E:\ML-pro\attendance_{today_str}({i}).csv"
        if not os.path.exists(new_path):
            return new_path
        i += 1

# ---------------------- TRAINING ----------------------
def train_recognizer(training_data_dir):
    faces, labels, names = [], [], []
    label_id = 0
    for student_name in students:
        folder_path = os.path.join(training_data_dir, student_name)
        if not os.path.isdir(folder_path):
            print(f"⚠️ Folder not found for student: {student_name}")
            continue
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"⚠️ Could not read image: {img_path}")
                continue
            faces.append(img)
            labels.append(label_id)
        names.append(student_name)
        label_id += 1

    if not faces:
        print(f"No valid training images found in {training_data_dir}")
        return None, []

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    print(f"✅ Training completed on {len(faces)} face(s).")
    return recognizer, names

# ---------------------- ATTENDANCE MANAGEMENT ----------------------
def initialize_csv():
    attendance_file = get_daily_csv()
    df = pd.DataFrame({
        'Roll_no': list(range(1, len(students)+1)),
        'Name': students,
        'Attendance': ['Absent'] * len(students),
        'Time': [''] * len(students),
        'Date': [''] * len(students)
    })
    df.to_csv(attendance_file, index=False)
    print(f"✅ Created new attendance file: {attendance_file}")
    return attendance_file

def mark_present(name, attendance_file):
    df = pd.read_csv(attendance_file)
    name = name.upper()
    if name in df['Name'].values:
        idx = df.index[df['Name'] == name][0]
        if df.at[idx, 'Attendance'] == 'Absent':
            df.at[idx, 'Attendance'] = 'Present'
            df.at[idx, 'Time'] = datetime.now().strftime("%H:%M:%S")
            df.at[idx, 'Date'] = datetime.now().strftime("%Y-%m-%d")
            df.to_csv(attendance_file, index=False)
            print(f"🟩 Marked Present: {name}")

# ---------------------- FACE RECOGNITION ----------------------
def start_recognition(recognizer, names, attendance_file):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not access webcam.")
        return

    print("🎥 Press 'q' to stop recognition.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            if recognizer is not None and names:
                label, confidence = recognizer.predict(roi)
                if confidence < 70:
                    name = names[label]
                    mark_present(name, attendance_file)
                    color = (0, 255, 0)
                    text = f"{name} ({int(confidence)}%)"
                else:
                    color = (0, 0, 255)
                    text = "Unknown"
            else:
                color = (0, 0, 255)
                text = "Unknown"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Attendance System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------------------- FINALIZE ATTENDANCE ----------------------
def finalize_attendance(attendance_file):
    df = pd.read_csv(attendance_file)
    total_present = df['Attendance'].value_counts().get('Present', 0)
    total_absent = df['Attendance'].value_counts().get('Absent', 0)

    # Add Present_Count and Absent_Count columns after last student
    df['Present_Count'] = ''
    df['Absent_Count'] = ''
    df.loc[len(df), ['Name', 'Present_Count', 'Absent_Count']] = ['Total', total_present, total_absent]

    df.to_csv(attendance_file, index=False)
    print(f"📊 Attendance Summary: Present={total_present}, Absent={total_absent}")
    print(f"📄 CSV updated with summary counts: {attendance_file}")

# ---------------------- MAIN ----------------------
def attendance_system():
    recognizer, names = train_recognizer(training_data_dir)
    attendance_file = initialize_csv()
    start_recognition(recognizer, names, attendance_file)
    finalize_attendance(attendance_file)

if __name__ == "__main__":
    attendance_system()
