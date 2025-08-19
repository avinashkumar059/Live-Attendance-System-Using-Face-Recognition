# Live-Attendance-System-Using-Face-Recognition
---------------------------------------------

# 📌 Project Overview:
This is a Python-based attendance management system that uses face recognition to automatically mark student or employee presence. 
The system captures faces in real-time, compares them with a stored dataset, and records attendance in a CSV file with a timestamp.

# 🖥️ Technologies Used:
- Python 3.10+
- OpenCV (with contrib)
- NumPy
- Pillow (PIL)
- Tkinter (for GUI)
- CSV module
- pyttsx3 (optional: voice feedback)

# 📁 Folder Structure:
- capture_images.py       → Register new student images
- train_model.py          → Train the recognizer model
- recognize.py            → Run face recognition and mark attendance
- main_gui.py             → GUI interface for all functions
- student_images/         → Folder that stores face images
- attendance_YYYY-MM-DD.csv → Attendance records per day
- trainer.yml             → Trained face recognizer model
- requirements.txt        → Python dependencies
- README.txt              → This file

# 📷 How to Use:
1. Run the GUI:
   > python main_gui.py

2. Click "Register Student"
   - Enter student name and enrollment number.
   - Webcam will capture 100 face images in ~10 seconds.

3. Click "Train Model"
   - Trains the model from all registered student images.

4. Click "Start Attendance"
   - Webcam opens and automatically recognizes known faces.
   - Attendance is saved in: `attendance_YYYY-MM-DD.csv`

# ✅ Features:
- Multi-face recognition (group support)
- Prevents duplicate attendance in same day
- Face data stored by enrollment_number_name format
- CSV export for easy review
- Modular code files for each task
- Real-time webcam display with face bounding box

# 📌 Tip:
Install all dependencies using:
> pip install -r requirements.txt

# 📁 Dependencies:
- opencv-contrib-python
- numpy
- pillow
- pyttsx3 (optional)

# 📞 Developer:
- Manish Rajbanshi
- Avinash Kumar Raut
- Avajit Kumar Kewrat
- Raju Kumar Sah
