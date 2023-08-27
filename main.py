import cv2
import mediapipe as mp
import numpy as np

mp_drawings = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture("http://192.168.0.124:8080/video")

def calculate_angle(a, b, c):
    a = np.array(a)  # first landmark coordinates
    b = np.array(b)  # second landmark coordinates
    c = np.array(c)  # third landmark coordinates

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Curl counter variables
state = None
count = 0

# setup matplotlib instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 480))

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        result = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = result.pose_landmarks.landmark  # return coordinates of landmark

            # Get coodinates left side
            l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            # Get coodinates left side
            r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            # Call calculate angle function
            l_angle = calculate_angle(l_hip, l_knee, l_ankle)
            r_angle = calculate_angle(r_hip, r_knee, r_ankle)

            # Visualize angle
            cv2.putText(image, str(l_angle),
                        tuple(np.multiply(l_knee, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(image, str(r_angle),
                        tuple(np.multiply(r_knee, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            # curl counter
            if l_angle < 50 and r_angle < 50:
                state = "sit"
            if l_angle > 160 and r_angle > 160 and state == "sit":
                state = "stand"
                count += 1
                print(count)

            # print(landmarks)
        except:
            pass

        # setup status box at corner of window
        cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)

        # Rep data inside status box
        cv2.putText(image, 'REPS', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(count), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 1, cv2.LINE_AA)

        # Render detection
        mp_drawings.draw_landmarks(image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                   mp_drawings.DrawingSpec(color = (0, 0, 255), thickness= 2, circle_radius = 2),
                                   mp_drawings.DrawingSpec(color = (255, 0, 0), thickness = 2, circle_radius = 2))

        cv2.imshow("MediaPipe feed", image)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()