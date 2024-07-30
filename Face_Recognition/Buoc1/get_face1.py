import tkinter as tk
from tkinter import messagebox
import cv2 as cv
import os
import numpy as np

class FaceDataCollector:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Data Collector")

        self.scale = 1.0
        self.face_detection_model = '../model/face_detection_yunet_2023mar.onnx'
        self.face_recognition_model = '../model/face_recognition_sface_2021dec.onnx'
        self.score_threshold = 0.9
        self.nms_threshold = 0.3
        self.top_k = 5000
        self.save_results = True

        self.detector = cv.FaceDetectorYN.create(
            self.face_detection_model, "", (320, 320), self.score_threshold, self.nms_threshold, self.top_k
        )
        self.recognizer = cv.FaceRecognizerSF.create(self.face_recognition_model, "")
        self.tm = cv.TickMeter()

        self.video_capture = cv.VideoCapture(0)
        self.frame_width = int(self.video_capture.get(cv.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.video_capture.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.detector.setInputSize([self.frame_width, self.frame_height])

        self.canvas = tk.Canvas(root, width=self.frame_width, height=self.frame_height)
        self.canvas.pack()

        self.name_label = tk.Label(root, text="Enter Name:")
        self.name_label.pack()
        self.name_entry = tk.Entry(root)
        self.name_entry.pack()

        self.capture_button = tk.Button(root, text="Capture Face", command=self.capture_face)
        self.capture_button.pack()

        self.quit_button = tk.Button(root, text="Quit", command=self.quit)
        self.quit_button.pack()

        self.update_frame()

    def update_frame(self):
        ret, frame = self.video_capture.read()
        if not ret:
            messagebox.showerror("Error", "Failed to capture video frame.")
            return

        frame_resized = cv.resize(frame, (0, 0), fx=self.scale, fy=self.scale)
        self.tm.start()
        faces = self.detector.detect(frame_resized)
        self.tm.stop()

        self.visualize(frame_resized, faces, self.tm.getFPS())

        frame_rgb = cv.cvtColor(frame_resized, cv.COLOR_BGR2RGB)
        self.photo = tk.PhotoImage(data=cv.imencode('.png', frame_rgb)[1].tobytes())
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.faces = faces
        self.frame = frame_resized

        self.root.after(10, self.update_frame)

    def capture_face(self):
        name = self.name_entry.get()
        if not name:
            messagebox.showerror("Error", "Please enter a name.")
            return

        if self.faces[1] is None:
            messagebox.showerror("Error", "No faces detected.")
            return

        face_align = self.recognizer.alignCrop(self.frame, self.faces[1][0])
        save_dir = os.path.join('collected_faces', name)
        os.makedirs(save_dir, exist_ok=True)
        file_name = os.path.join(save_dir, f'{name}_{len(os.listdir(save_dir)):04d}.bmp')
        cv.imwrite(file_name, face_align)
        messagebox.showinfo("Success", f"Captured and saved face image: {file_name}")

    def visualize(self, input, faces, fps, thickness=2):
        if faces[1] is not None:
            for idx, face in enumerate(faces[1]):
                coords = face[:-1].astype(np.int32)
                cv.rectangle(input, (coords[0], coords[1]), (coords[0] + coords[2], coords[1] + coords[3]), (0, 255, 0), thickness)
                for i in range(4, 13, 2):
                    cv.circle(input, (coords[i], coords[i + 1]), 2, (255, 0, 0), thickness)
        cv.putText(input, f'FPS: {fps:.2f}', (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def quit(self):
        self.video_capture.release()
        self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceDataCollector(root)
    root.mainloop()
