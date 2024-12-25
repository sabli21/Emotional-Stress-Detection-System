import pandas as pd
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import datetime
from scipy.spatial import distance as dist
from scipy.spatial import distance_matrix
from scipy.special import expit
from imutils.video import FileVideoStream
import joblib
from sklearn import preprocessing
import pyqtgraph as pg
import sklearn

x = np.loadtxt("total_input_shuffle.txt", dtype=float, unpack=False)

scaler = preprocessing.StandardScaler().fit(x)

clf = joblib.load('my_model.pkl')

f = open("ear_fortrain2.txt", 'a')
f1 = open("ear_after_threshold2.txt", 'a')
f2 = open("ear_after_svm2.txt", 'r+')


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
    a = dist.euclidean(eye[1], eye[5])
    b = dist.euclidean(eye[2], eye[4])
    # compute the euclidean distance between the horizontal eye landmarks (x,y)-coordinates
    c = dist.euclidean(eye[0], eye[3])
    # compute the eye aspect ratio
    ear = (a + b) / (2.0 * c)
    return ear


ear = 0
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="", help="path to input video file")
args = vars(ap.parse_args())

ear_thresh = 0.22  # eye aspect ratio threshold to indicate blink
ear_consec_frames = 3  # no. of consecutive frames ear must be below threshold

counter = 0
counter1 = 0
total1 = 0
total2 = 0
blink_counter = list()
bpm = 0
bpm_list = []
avg_bpm = 0
eyebrow_points = []
outermouth_points = []
mouthcorner_points = []
stress_points = []

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# getting indexes of facial landmarks for right eye and left eye
(leyeStart, leyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(reyeStart, reyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

print("[INFO] starting video stream thread...")

# vs = FileVideoStream(args["video"]).start()
# fileStream = True

vs = VideoStream(src=0).start()
fileStream = False

time.sleep(1.0)

# ===========================#
window = np.zeros([1, 13])
frame_index = 0
buffer = list()

# ===========================#
now = datetime.datetime.now()
print(now)
while True:

    if fileStream and not vs.more():
        break

    frame = vs.read()
    frame = imutils.resize(frame, width=900)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # gettting points of eyebrows from the facial landmark
    (lBegin, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
    (rBegin, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]

    # getting lip points from facial landmarks
    (lower, upper) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    # detects faces in the grayscale frame
    rects = detector(gray, 0)

    # time1 = dt.now()

    # loop over the face detections
    for rect in rects:

        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy array
        landmarks = predictor(gray, rect)
        shape = face_utils.shape_to_np(landmarks)

        # left and right eye landmarks
        leftEye = shape[leyeStart:leyeEnd]
        rightEye = shape[reyeStart:reyeEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # average the eye aspect ratio for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # left and right eyebrow landmarks
        leyebrow = shape[lBegin:lEnd]
        reyebrow = shape[rBegin:rEnd]
        # mouth landmarks
        openmouth = shape[lower:upper]

        # nose bridge landmarks
        n1 = np.array([landmarks.part(28).x, landmarks.part(28).y])
        n2 = np.array([landmarks.part(29).x, landmarks.part(29).y])
        n3 = np.array([landmarks.part(30).x, landmarks.part(30).y])

        # outer mouth landmarks
        mouth_outer = np.array([[landmarks.part(49).x, landmarks.part(49).y],
                                [landmarks.part(50).x, landmarks.part(50).y],
                                [landmarks.part(51).x, landmarks.part(51).y],
                                [landmarks.part(52).x, landmarks.part(52).y],
                                [landmarks.part(53).x, landmarks.part(53).y],
                                [landmarks.part(55).x, landmarks.part(55).y],
                                [landmarks.part(56).x, landmarks.part(56).y],
                                [landmarks.part(57).x, landmarks.part(57).y],
                                [landmarks.part(58).x, landmarks.part(58).y],
                                [landmarks.part(59).x, landmarks.part(59).y],
                                [landmarks.part(60).x, landmarks.part(60).y],
                                [landmarks.part(64).x, landmarks.part(64).y]])
        # mouth corner landmarks
        mouth_corner = np.array([[landmarks.part(48).x, landmarks.part(48).y],
                                 [landmarks.part(54).x, landmarks.part(54).y]])
        print("mouth", mouth_corner)

        # left eyebrow landmarks
        left_eyebrow = np.array([[landmarks.part(17).x, landmarks.part(17).y],
                                 [landmarks.part(18).x, landmarks.part(18).y],
                                 [landmarks.part(19).x, landmarks.part(19).y],
                                 [landmarks.part(20).x, landmarks.part(20).y],
                                 [landmarks.part(21).x, landmarks.part(21).y]])

        # right eyebrow landmarks
        right_eyebrow = np.array([[landmarks.part(22).x, landmarks.part(22).y],
                                  [landmarks.part(23).x, landmarks.part(23).y],
                                  [landmarks.part(24).x, landmarks.part(24).y],
                                  [landmarks.part(25).x, landmarks.part(25).y],
                                  [landmarks.part(26).x, landmarks.part(26).y]])

        Ref = [n1, n2, n3]  # Reference vector with landmarks of nose bridge
        print("ref", Ref)

        # calculating normalization coefficient by getting the euclidean distance between upper and lower
        # reference landmarks i.e the size of the nose bridge
        normal_coefficient = dist.euclidean(n1, n3)
        # print("normal", normal_coefficient)

        # figuring out convex shape
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        reyebrowhull = cv2.convexHull(reyebrow)
        leyebrowhull = cv2.convexHull(leyebrow)
        openmouthhull = cv2.convexHull(openmouth)

        # calculating the outer mouth feature
        F1 = sum(distance_matrix(mouth_outer, Ref))
        outer_mouth_feature = sum(F1) / normal_coefficient
        outermouth_points.append(int(outer_mouth_feature))
        print("outer", outer_mouth_feature)

        # calculating the mouth corners feature
        F2 = sum(distance_matrix(mouth_corner, Ref))
        mouth_corner_feature = sum(F2) / normal_coefficient
        mouthcorner_points.append(int(mouth_corner_feature))
        print("corner", mouth_corner_feature)

        # calculating the eyebrows feature
        F3 = sum(distance_matrix(left_eyebrow, Ref) + distance_matrix(right_eyebrow, Ref))
        eyebrows_feature = sum(F3) / normal_coefficient
        eyebrow_points.append(int(eyebrows_feature))
        print("eyebrow", eyebrows_feature)

        cv2.drawContours(frame, [leftEyeHull], -1, (255, 0, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (255, 0, 0), 1)
        cv2.drawContours(frame, [reyebrowhull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [leyebrowhull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [openmouthhull], -1, (0, 0, 255), 1)
        for (x, y) in Ref:
            cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)

        # ============================#
        buffer.append(ear)

        if frame_index >= 12:
            # put the current EAR with +6 and -6 EARs in the window for SVM prediction
            window[0, :] = buffer[frame_index - 12: frame_index + 1]

            if buffer[frame_index - 6] < ear_thresh:
                counter1 += 1
                thresh_output = 1

            else:
                thresh_output = 0
                # if the eyes were closed for consecutive number of frames increment total number of blinks
                if counter1 >= ear_consec_frames:
                    total1 += 1
                # reset the ear frame counter
                counter1 = 0
            f1.write(str(thresh_output))
            f1.write('\n')

            window = scaler.transform(window)
            # print(window)

            output_predict = clf.predict(window)
            output_predict_write = output_predict[0]
            f2.write(str(output_predict_write))
            f2.write('\n')
            if output_predict == 1:
                counter += 1
            else:
                if counter >= 4:
                    total2 += 1
                    blink_counter.append(1)
                    print(blink_counter)
                counter = 0
                # when one minute has passed sum the count of blinks to get blinks per minute
                if (now + datetime.timedelta(minutes=1)) <= datetime.datetime.now():
                    bpm = sum(blink_counter)
                    print(bpm)
                    bpm_list.append(bpm)
                    avg_bpm = np.average(bpm_list)  # average blinks per minute or blink rate
                    blink_counter.clear()  # clear the blink counter so it can start filling again
                    now = datetime.datetime.now()  # reset the one minute timer

        # normalize the values of all features between 0 and 1
        outermouth_normalized = abs(outer_mouth_feature - np.min(outermouth_points)) / abs(np.max(outermouth_points) -
                                                                                           np.min(outermouth_points))
        print("outer normal", outermouth_normalized)
        mouthcorner_normalized = abs(mouth_corner_feature - np.min(mouthcorner_points)) / abs(
            np.max(mouthcorner_points) - np.min(mouthcorner_points))
        print("corners normal", mouthcorner_normalized)
        eyebrows_normalized = abs(eyebrows_feature - np.min(eyebrow_points)) / abs(np.max(eyebrow_points) -
                                                                                   np.min(eyebrow_points))
        print("eyebrow normal", eyebrows_normalized)
        normalized_value = (mouthcorner_normalized + outermouth_normalized)
        print("normal_value", normalized_value)
        # print("stress mouth", stress_value_mouth)
        # stress_value_corners = (np.exp(-mouthcorner_normalized))
        # print("stress eyebrow", stress_value_eyebrows)
        stress_value_eyebrows = (np.exp(-eyebrows_normalized))
        stress_value_mouth = (expit(normalized_value))
        # stress_value = expit(stress_value_mouth + stress_value_eyebrows)
        # print("stress", stress_value)

        # conditions to detect stress level
        if stress_value_mouth > 0.79 or stress_value_eyebrows > 0.85 or 6.0 >= avg_bpm > 0:
            stress_label = "High Stress"
        else:
            if stress_value_mouth > 0.75 or stress_value_eyebrows > 0.75:
                stress_label = "Low Stress"
            else:
                stress_label = "No Stress"

        cv2.putText(frame, "stress value(mouth): {}".format(str(int(stress_value_mouth * 100))), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 102, 255), 2)
        cv2.putText(frame, "stress value(eyebrows): {}".format(str(int(stress_value_eyebrows * 100))), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 102, 255), 2)
        #cv2.putText(frame, "Stress level: {}".format(stress_label), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    #(102, 102, 255), 2)
        cv2.putText(frame, "Blinks(SVM): {}".format(total2), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 102, 255),
                    2)
        cv2.putText(frame, "Blinks(Threshold = {0}): {1}".format(ear_thresh, total1), (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 102, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 102, 255), 2)
        cv2.putText(frame, "Blinks per minute: {} ".format(bpm), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (102, 102, 255), 2)
        cv2.putText(frame, "Blink Rate: {} ".format(avg_bpm), (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (102, 102, 255), 2)

        frame_index += 1

    f.write(str(ear))
    f.write('\n')

    cv2.imshow("Stress Monitor", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
f.close()
f1.close()
f2.close()
