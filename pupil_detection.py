# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 00:39:37 2025

@author: Soumyadeep
"""

import cv2
import numpy as np

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)   # enhance global contrast

    faces = face_cascade.detectMultiScale(gray, 1.2, 6)
    iris_frame = np.zeros((150,150,3), dtype=np.uint8)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10, minSize=(40,40))
        for (ex, ey, ew, eh) in eyes:
            eye_gray = roi_gray[ey:ey+eh, ex:ex+ew]
            eye_color = roi_color[ey:ey+eh, ex:ex+ew]

            # --- 1️⃣ Pre-processing ---
            eye_eq = cv2.equalizeHist(eye_gray)
            eye_blur = cv2.GaussianBlur(eye_eq, (7,7), 0)

            # Adaptive threshold to highlight dark iris
            _, thresh = cv2.threshold(eye_blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

            # Morphological cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

            # --- 2️⃣ Circle detection (Iris) ---
            circles = cv2.HoughCircles(
                eye_blur,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=20,
                param1=100,       # edge detection threshold
                param2=18,        # smaller → more sensitive
                minRadius=8,
                maxRadius=40
            )

            if circles is not None:
                circles = np.uint16(np.around(circles))
                # choose the circle whose center lies in dark region
                best_circle = None
                min_intensity = 255
                for (ix, iy, ir) in circles[0, :]:
                    # sample intensity at center
                    if 0 <= iy < eye_blur.shape[0] and 0 <= ix < eye_blur.shape[1]:
                        center_intensity = eye_blur[iy, ix]
                        if center_intensity < min_intensity:
                            min_intensity = center_intensity
                            best_circle = (ix, iy, ir)

                if best_circle:
                    ix, iy, ir = best_circle
                    cv2.circle(eye_color, (ix, iy), ir, (0,0,255), 2)
                    cv2.circle(eye_color, (ix, iy), 2, (255,0,0), 3)

                    # --- 3️⃣ Crop iris safely ---
                    x1, y1 = max(ix-ir,0), max(iy-ir,0)
                    x2, y2 = min(ix+ir, eye_color.shape[1]), min(iy+ir, eye_color.shape[0])
                    iris_crop = eye_color[y1:y2, x1:x2]

                    if iris_crop.size != 0:
                        iris_frame = cv2.resize(iris_crop, (150,150))

            # draw eye ROI
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)

    cv2.imshow("Real-Time Iris Detection (Enhanced)", frame)
    cv2.imshow("Detected Iris", iris_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
