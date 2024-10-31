import cv2
import mediapipe as mp
import numpy as np
import os
import Quartz

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

lower_angle_limit = 38
upper_angle_limit = 44
count = 0
max_count = 20

# Initialize Webcam
cap = cv2.VideoCapture(0)

def calculate_angle(a, b, c):
    """Calculate the angle between three points to determine posture."""
    a = np.array(a)  # First point
    b = np.array(b)  # Middle point
    c = np.array(c)  # End point
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def is_screen_locked():
    """Check if the screen is locked (display is off)."""
    # Get display status using psutil
    d = Quartz.CGSessionCopyCurrentDictionary()
    if 'CGSSessionScreenIsLocked' in d.keys():
        return True
    return False

while cap.isOpened():
    if is_screen_locked():
        print("Screen is locked. Exiting...")
        break
    
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Convert back to BGR for OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        # Draw pose landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        landmarks = results.pose_landmarks.landmark

        # Get coordinates for the shoulders and hips for basic posture check
        shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
        shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
        # hip_left = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,
        #             landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
        # hip_right = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x,
        #              landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]
        nose = [landmarks[mp_pose.PoseLandmark.NOSE].x, landmarks[mp_pose.PoseLandmark.NOSE].y]


        # Calculate the angle between shoulders and hips
        angle = calculate_angle(shoulder_left, nose, shoulder_right)
        # print(count)
        # print(shoulder_left,shoulder_right, angle)
        if angle > upper_angle_limit or angle < lower_angle_limit  and count < max_count:
            count+=1
        # Alert if the angle suggests slouching (experiment with angle thresholds)
        if count == max_count:  # Adjust this threshold based on testing
            # cv2.putText(image, "Sit Straight!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            os.system('afplay ./sound.wav')
            count=0   # Reset alert cooldown


    # Display the output
    cv2.imshow("Posture Detection", image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
