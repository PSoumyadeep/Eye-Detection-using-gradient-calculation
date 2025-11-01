# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 03:00:07 2025

@author: Soumyadeep
"""

import cv2
import numpy as np
import os

def preprocess_eye(eye_gray):
    """Enhance contrast and reduce noise for better gradient clarity."""
    eye_eq = cv2.equalizeHist(eye_gray)                      # normalize brightness
    eye_blur = cv2.GaussianBlur(eye_eq, (5, 5), 1)           # smooth small edges
    return eye_blur


def find_eye_center_accurate(eye_gray):
    """Improved gradient-based iris center detection (Timm & Barth inspired)."""
    eye = preprocess_eye(eye_gray)

    # Calculate gradients
    grad_x = cv2.Sobel(eye, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(eye, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(grad_x**2 + grad_y**2)

    # Normalize gradients
    grad_x /= (mag + 1e-5)
    grad_y /= (mag + 1e-5)

    rows, cols = eye.shape
    accum = np.zeros_like(eye, dtype=np.float64)

    # Weighted center voting
    for y in range(rows):
        for x in range(cols):
            if mag[y, x] > 60:  # only strong edges
                dx, dy = grad_x[y, x], grad_y[y, x]
                weight = mag[y, x] / 255.0
                for r in range(1, 30):  # search radius
                    cx = int(x - dx * r)
                    cy = int(y - dy * r)
                    if 0 <= cx < cols and 0 <= cy < rows:
                        accum[cy, cx] += weight

    # Smooth accumulator to stabilize detection
    accum = cv2.GaussianBlur(accum, (7, 7), 1.5)

    _, maxVal, _, maxLoc = cv2.minMaxLoc(accum)
    return maxLoc, maxVal


def crop_circular_iris(eye_img, center, radius=20):
    x, y = center
    h, w = eye_img.shape
    x1, y1 = max(0, x - radius), max(0, y - radius)
    x2, y2 = min(w, x + radius), min(h, y + radius)
    cropped = eye_img[y1:y2, x1:x2]

    mask = np.zeros_like(cropped, dtype=np.uint8)
    cv2.circle(mask, (radius, radius), radius, 255, -1)
    iris = cv2.bitwise_and(cropped, cropped, mask=mask)
    return iris


def main():
    cap = cv2.VideoCapture(0)
    os.makedirs("templates", exist_ok=True)
    counter = 0
    saved_once = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Region for eyes
        top = int(h * 0.35)
        bottom = int(h * 0.45)
        left = int(w * 0.25)
        right = int(w * 0.75)
        face_region = gray[top:bottom, left:right]
        fh, fw = face_region.shape

        left_eye_region = face_region[:, :fw // 2]
        right_eye_region = face_region[:, fw // 2:]

        # Improved detection
        (left_cx, left_cy), left_conf = find_eye_center_accurate(left_eye_region)
        (right_cx, right_cy), right_conf = find_eye_center_accurate(right_eye_region)

        left_center = (left + left_cx, top + left_cy)
        right_center = (left + fw // 2 + right_cx, top + right_cy)

        # Draw reticle
        cv2.circle(frame, left_center, 6, (0, 0, 255), 1)
        cv2.circle(frame, right_center, 6, (0, 0, 255), 1)
        cv2.circle(frame, left_center, 2, (0, 255, 0), -1)
        cv2.circle(frame, right_center, 2, (0, 255, 0), -1)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Save when stable and strong detection
        if not saved_once and left_conf > 12 and right_conf > 12:
            iris_left = crop_circular_iris(left_eye_region, (left_cx, left_cy), radius=22)
            iris_right = crop_circular_iris(right_eye_region, (right_cx, right_cy), radius=22)

            cv2.imwrite(f"templates/iris_left_{counter}.png", iris_left)
            cv2.imwrite(f"templates/iris_right_{counter}.png", iris_right)
            print(f"[INFO] Saved iris_left_{counter}.png and iris_right_{counter}.png")
            saved_once = True
            counter += 1

        cv2.imshow("High-Accuracy Iris Detection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            saved_once = False
            print("[INFO] Ready for next iris capture...")

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
