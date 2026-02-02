import cv2
import numpy as np
import mediapipe as mp
import time

mp_drawing=mp.solutions.drawing_utils
mp_hands=mp.solutions.hands

def get_angle(a,b,c):
    ba=a-b
    bc=c-b
    cos=np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc))
    return np.degrees(np.arccos(cos))

def get_distance(p1,p2):
    return np.linalg.norm(p1-p2)

def in_range(val,low,high):
    return low<=val<=high

letter_ranges = {
    "A": {
        "index": (15,45),
        "middle": (10,40),
        "ring": (3,30),
        "pinky": (0,25),
        "thumb": (0.09,0.18)
    },
    "B": {
        "index": (170,180),
        "middle": (170,180),
        "ring": (170,180),
        "pinky": (170,180),
        "thumb": (0.05,0.07)
    },
    "C": {
        "index": (85,135),
        "middle": (60,120),
        "ring": (50,120),
        "pinky": (90,145),
        "thumb": (0.07,0.16)
    },
    "D": {
        "index": (168,180),
        "middle": (30,55),
        "ring": (20,100),
        "pinky": (5,50),
        "thumb": (0.06,0.16)
    },
    "E": {
        "index": (28,50),
        "middle": (15,32),
        "ring": (5,28),
        "pinky": (10,36),
        "thumb": (0.06,0.095)
    },
    "F": {
        "index": (58,160),
        "middle": (155,180),
        "ring": (155,180),
        "pinky": (155,180),
        "thumb": (0.07,0.125)
    },
    "G": {
        "index": (160,180),
        "middle": (0,60),
        "ring": (25,70),
        "pinky": (30,110),
        "thumb": (0.028,0.15)
    }
}


cap=cv2.VideoCapture(0)

angles=None
prediction="Waiting..."
INTERVAL=1
last_update_time=0

with mp_hands.Hands(min_detection_confidence=0.8,min_tracking_confidence=0.5,max_num_hands=1) as hands:
    while cap.isOpened():
        ret,frame=cap.read()
        image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image.flags.writeable=False
        results=hands.process(image)
        image.flags.writeable=True
        image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            hand=results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(
                image,
                hand,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(25,25,255),circle_radius=2,thickness=2),
                mp_drawing.DrawingSpec(color=(0,255,0))
            )

            current_time=time.time()
            if current_time-last_update_time>=INTERVAL:
                lm=hand.landmark

                index_angle=get_angle(
                    np.array([lm[5].x,lm[5].y,lm[5].z]),
                    np.array([lm[6].x,lm[6].y,lm[6].z]),
                    np.array([lm[8].x,lm[8].y,lm[8].z])
                )
                middle_angle=get_angle(
                    np.array([lm[9].x,lm[9].y,lm[9].z]),
                    np.array([lm[10].x,lm[10].y,lm[10].z]),
                    np.array([lm[12].x,lm[12].y,lm[12].z])
                )
                ring_angle=get_angle(
                    np.array([lm[13].x,lm[13].y,lm[13].z]),
                    np.array([lm[14].x,lm[14].y,lm[14].z]),
                    np.array([lm[16].x,lm[16].y,lm[16].z])
                )
                pinky_angle=get_angle(
                    np.array([lm[17].x,lm[17].y,lm[17].z]),
                    np.array([lm[18].x,lm[18].y,lm[18].z]),
                    np.array([lm[20].x,lm[20].y,lm[20].z])
                )

                thumb_tip=np.array([lm[4].x,lm[4].y,lm[4].z])
                index_mcp=np.array([lm[5].x,lm[5].y,lm[5].z])
                thumb_dist=get_distance(thumb_tip,index_mcp)

                angles={
                    "index":int(index_angle),
                    "middle":int(middle_angle),
                    "ring":int(ring_angle),
                    "pinky":int(pinky_angle),
                    "thumb":round(thumb_dist,3)
                }

                prediction = "Unknown"

                for letter, ranges in letter_ranges.items():
                   if all(in_range(angles[k], *ranges[k]) for k in ranges):    #in_range(angles[k], ranges[k][0], ranges[k][1])
                     prediction = letter
                     break


        if angles:
            cv2.putText(image,f"Index: {angles['index']}",(30,40),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
            cv2.putText(image,f"Middle: {angles['middle']}",(30,70),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
            cv2.putText(image,f"Ring: {angles['ring']}",(30,100),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
            cv2.putText(image,f"Pinky: {angles['pinky']}",(30,130),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
            cv2.putText(image,f"Thumb: {angles['thumb']}",(30,160),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
            cv2.putText(image,f"Prediction: {prediction}",(30,210),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)

        cv2.imshow("ASL Stable Sampling",image)
        if cv2.waitKey(10)&0xFF==ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
