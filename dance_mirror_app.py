import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def calculate_similarity(user_landmarks, reference_landmarks,frame):
    landmarks_info = []
    if user_landmarks and reference_landmarks:
        for ul, rl in zip(user_landmarks.landmark, reference_landmarks.landmark):
            user_point = np.array([ul.x * frame.shape[1], ul.y * frame.shape[0]])
            ref_point = np.array([rl.x * frame.shape[1], rl.y * frame.shape[0]])
            distance = np.linalg.norm(user_point - ref_point)
            landmarks_info.append((ul, distance))
    return landmarks_info

class DanceMirrorApp:
    def __init__(self, window, window_title, video_source=0, dance_video_path="demovid.avi"):
        self.window = window
        self.window.title(window_title)
        self.vid = cv2.VideoCapture(video_source)
        self.dance_video = cv2.VideoCapture(dance_video_path)
        self.canvas = tk.Canvas(window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()
        self.btn_start = tk.Button(window, text="Start", width=15, command=self.start_dance)
        self.btn_start.pack(anchor=tk.CENTER, expand=True)
        self.delay = 15
        self.update()
        self.window.mainloop()

    def start_dance(self):
        self.dance_video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def update(self):
        ret, frame = self.vid.read()
        ret_dance, dance_frame = self.dance_video.read()
        if ret and ret_dance:
            frame = cv2.flip(frame, 1)
            dance_frame = cv2.resize(dance_frame, (frame.shape[1], frame.shape[0]))

            user_results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            dance_results = pose.process(cv2.cvtColor(dance_frame, cv2.COLOR_BGR2RGB))

            if user_results.pose_landmarks and dance_results.pose_landmarks:
                landmarks_info = calculate_similarity(user_results.pose_landmarks, dance_results.pose_landmarks, frame)

                for landmark, distance in landmarks_info:
                    draw_color = (0, 255, 0) if distance < 10 else (0, 255, 255) if distance < 20 else (0, 0, 255)
                    cv2.circle(frame, (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])), 5, draw_color, -1)

            alpha = 0.5
            blended = cv2.addWeighted(frame, 1 - alpha, dance_frame, alpha, 0)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update)

DanceMirrorApp(tk.Tk(), "Interactive Dance Mirror", video_source=0, dance_video_path="demovid.avi")