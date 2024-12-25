import sys
import tkinter as tk
from tkinter import messagebox
from tkinter import *
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
from scipy import signal
from imutils.video import FileVideoStream
import joblib
from sklearn import preprocessing
from keras.models import load_model

x = np.loadtxt("total_input_shuffle.txt", dtype=float, unpack=False)

scaler = preprocessing.StandardScaler().fit(x)

clf = joblib.load('my_model.pkl')
model = load_model('first_5322_model.hdf5')

f = open("ear_fortrain2.txt", 'a')
f1 = open("ear_after_threshold2.txt", 'a')
f2 = open("ear_after_svm2.txt", 'r+')

ear = 0
ap = argparse.ArgumentParser()
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
high = 0

frame_ROI = np.zeros((10, 10, 3), np.uint8)
frame_out = np.zeros((10, 10, 3), np.uint8)
samples = []
buffer_size = 100
times = []
data_buffer = []
fps = 0
fft = []
freqs = []
t0 = time.time()
Bpm = 0
bpms = []
avg_bpms = 0


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
    a = dist.euclidean(eye[1], eye[5])
    b = dist.euclidean(eye[2], eye[4])
    # compute the euclidean distance between the horizontal eye landmarks (x,y)-coordinates
    c = dist.euclidean(eye[0], eye[3])
    # compute the eye aspect ratio
    ear = (a + b) / (2.0 * c)
    return ear


def face_detect(frame3):
    face_frame = np.zeros((10, 10, 3), np.uint8)
    mask = np.zeros((10, 10, 3), np.uint8)
    ROI1 = np.zeros((10, 10, 3), np.uint8)
    ROI2 = np.zeros((10, 10, 3), np.uint8)
    status = False

    if frame3 is None:
        return

    gray2 = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    rects2 = detector(gray2, 0)

    # assumpion: only 1 face is detected
    if len(rects2) > 0:
        status = True

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rects[0])
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        if y < 0:
            print("a")
            return frame3, face_frame, ROI1, ROI2, status, mask
        # if i==0:
        face_frame = frame3[y:y + h, x:x + w]
        # show the face number
        # cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10),
        #    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image

        # for (x, y) in shape:
        # cv2.circle(frame, (x, y), 1, (0, 0, 255), -1) #draw facial landmarks
        if face_frame.shape[:2][1] != 0:
            face_frame = imutils.resize(face_frame, width=256)

        face_frame = fa.align(frame3, gray2, rects2[0])  # align face

        grayf = cv2.cvtColor(face_frame, cv2.COLOR_BGR2GRAY)
        rectsf = detector(grayf, 0)

        if len(rectsf) > 0:
            shape2 = predictor(grayf, rectsf[0])
            shape2 = face_utils.shape_to_np(shape2)

            for (a, b) in shape2:
                cv2.circle(face_frame, (a, b), 1, (0, 0, 255), -1)  # draw facial landmarks

            cv2.rectangle(face_frame, (shape2[54][0], shape2[29][1]),  # draw rectangle on right and left cheeks
                          (shape2[12][0], shape2[33][1]), (0, 255, 0), 0)
            cv2.rectangle(face_frame, (shape2[4][0], shape2[29][1]),
                          (shape2[48][0], shape2[33][1]), (0, 255, 0), 0)

            ROI1 = face_frame[shape2[29][1]:shape2[33][1], shape2[54][0]:shape2[12][0]]  # right cheek

            ROI2 = face_frame[shape2[29][1]:shape2[33][1], shape2[4][0]:shape2[48][0]]  # left cheek

            # get the shape of face for color amplification
            rshape = np.zeros_like(shape2)
            rshape = face_remap(shape2)
            mask = np.zeros((face_frame.shape[0], face_frame.shape[1]))

            cv2.fillConvexPoly(mask, rshape[0:27], 1)
            # mask = np.zeros((face_frame.shape[0], face_frame.shape[1],3),np.uint8)
            # cv2.fillConvexPoly(mask, shape, 1)

        # cv2.imshow("face align", face_frame)

    else:
        cv2.putText(frame3, "No face detected",
                    (200, 200), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
        status = False
    return frame3, face_frame, ROI1, ROI2, status, mask


# some points in the facial landmarks need to be re-ordered
def face_remap(shape2):
    remapped_image = shape2.copy()
    # left eye brow
    remapped_image[17] = shape2[26]
    remapped_image[18] = shape2[25]
    remapped_image[19] = shape2[24]
    remapped_image[20] = shape2[23]
    remapped_image[21] = shape2[22]
    # right eye brow
    remapped_image[22] = shape2[21]
    remapped_image[23] = shape2[20]
    remapped_image[24] = shape2[19]
    remapped_image[25] = shape2[18]
    remapped_image[26] = shape2[17]
    # neatening
    remapped_image[27] = shape2[0]

    remapped_image = cv2.convexHull(shape2)
    return remapped_image


def extractColor(frame3):
    g = np.mean(frame3[:, :, 1])
    return g


def butter_bandpass(lowcut, highcut, fs, order=5):
    lowcut = lowcut + sys.float_info.epsilon
    highcut = highcut + sys.float_info.epsilon
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    Wn = [low, high]
    b, a = signal.butter(order, Wn, btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered_data = signal.lfilter(b, a, data)
    return filtered_data


def convert_dtype(x):
    x_float = x.astype('float32')
    return x_float


def normalize(x):
    x_n = (x - 0) / 255
    return x_n


def reshape(x):
    x_r = x.reshape((x.shape[0], x.shape[1], x.shape[2], 1))
    return x_r


def emotion_finder(faces, frame):
    emotions = {0: 'angry', 1: 'fear', 2: 'happy', 3: 'sad',
                4: 'surprised', 5: 'neutral'}
    x, y, w, h = face_utils.rect_to_bb(faces)
    frame = frame[y:y + h, x:x + w]
    roi_gray = cv2.resize(frame, (48, 48), interpolation=cv2.INTER_AREA)
    roi_gray = convert_dtype(np.array([roi_gray]))
    roi_gray = normalize(roi_gray)
    roi_gray = reshape(roi_gray)
    pr = model.predict(roi_gray)[0]
    print(pr)
    max_emo = np.argmax(pr)
    label = emotions[max_emo]
    if label in ['fear', 'sad', 'angry']:
        label = 'Stressed'
    else:
        label = 'Not Stressed'
    return label


def center(win):
    """
    centers a tkinter window
    :param win: the root or Toplevel window to center
    """
    win.update_idletasks()
    width = win.winfo_width()
    frm_width = win.winfo_rootx() - win.winfo_x()
    win_width = width + 2 * frm_width
    height = win.winfo_height()
    titlebar_height = win.winfo_rooty() - win.winfo_y()
    win_height = height + titlebar_height + frm_width
    x = win.winfo_screenwidth() // 2 - win_width // 2
    y = win.winfo_screenheight() // 2 - win_height // 2
    win.geometry('{}x{}+{}+{}'.format(width, height, x, y))
    win.deiconify()


print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
fa = face_utils.FaceAligner(predictor, desiredFaceWidth=256)

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
    frame = vs.read()
    frame = imutils.resize(frame, width=900)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame2 = vs.read()

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
        emotion = emotion_finder(rect, gray)
        cv2.putText(frame, emotion, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 255, 102), 2)
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
        cv2.putText(frame, "Blinks(SVM): {}".format(total2), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 102, 255),
                    2)
        cv2.putText(frame, "Blinks per minute: {} ".format(bpm), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (102, 102, 255), 2)
        cv2.putText(frame, "Blink Rate: {} ".format(avg_bpm), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (102, 102, 255), 2)

        frame_index += 1

        frame2, face_frame, ROI1, ROI2, status, mask = face_detect(frame2)

        frame_out = frame2
        frame_ROI = face_frame

        g1 = extractColor(ROI1)
        g2 = extractColor(ROI2)

        L = len(data_buffer)

        # calculate average green value of 2 ROIs
        g = (g1 + g2) / 2

        # remove sudden change, if the avg value change is over 10, use the mean of the data_buffer
        if abs(g - np.mean(data_buffer)) > 10 and L > 99:
            g = data_buffer[-1]

        times.append(time.time() - t0)
        data_buffer.append(g)

        # only process in a fixed-size buffer
        if L > buffer_size:
            data_buffer = data_buffer[-buffer_size:]
            times = times[-buffer_size:]
            bpms = bpms[-buffer_size // 2:]
            L = buffer_size

        processed = np.array(data_buffer)

        # start calculating after the first 10 frames
        if L == buffer_size:
            fps = float(L) / (times[-1] - times[
                0])  # calculate HR using a true fps of processor of the computer, not the fps the camera provide
            even_times = np.linspace(times[0], times[-1], L)

            processed = signal.detrend(processed)  # detrend the signal to avoid interference of light change
            interpolated = np.interp(even_times, times, processed)  # interpolation by 1
            interpolated = np.hamming(
                L) * interpolated  # make the signal become more periodic (avoid spectral leakage)
            # norm = (interpolated - np.mean(interpolated))/np.std(interpolated)#normalization
            norm = interpolated / np.linalg.norm(interpolated)
            raw = np.fft.rfft(norm * 30)  # do real fft with the normalization multiplied by 10

            freqs = float(fps) / L * np.arange(L / 2 + 1)
            freqs = 60. * freqs

            fft = np.abs(raw) ** 2  # get amplitude spectrum

            idx = np.where((freqs > 50) & (freqs < 180))  # the range of frequency that HR is supposed to be within
            pruned = fft[idx]
            pfreq = freqs[idx]

            freqs = pfreq
            fft = pruned

            idx2 = np.argmax(pruned)  # max in the range can be HR

            Bpm = freqs[idx2]
            bpms.append(Bpm)

            processed = butter_bandpass_filter(processed, 0, 1, fps, order=3)
            # ifft = np.fft.irfft(raw)
        samples = processed  # multiply the signal with 5 for easier to see in the plot

        if mask.shape[0] != 10:
            out = np.zeros_like(face_frame)
            mask = mask.astype(np.bool)
            out[mask] = face_frame[mask]
            if processed[-1] > np.mean(processed):
                out[mask, 2] = 180 + processed[-1] * 10
            face_frame[mask] = out[mask]

        # cv2.imshow("face", face_frame)
        # out = cv2.add(face_frame,out)
        # else:
        # cv2.imshow("face", face_frame)
        # frame_out = cv2.cvtColor(frame_out, cv2.COLOR_RGB2BGR)
        cv2.putText(frame, "FPS: " + str(float("{:.2f}".format(fps))),
                    (10, 180), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)

        # frame_ROI = cv2.cvtColor(frame_ROI, cv2.COLOR_RGB2BGR)
        frame_ROI = np.transpose(frame_ROI, (0, 1, 2)).copy()
        cv2.putText(frame, "Freq: " + str(float("{:.2f}".format(Bpm))),
                    (10, 210), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)

        if bpms.__len__() > 50:
            # if max(bpms - np.mean(bpms)) < 5:  # show HR if it is stable -the change is not over 5 bpm- for 3s
            cv2.putText(frame, "Heart rate: " + str(float("{:.2f}".format(np.mean(bpms)))) + " bpm",
                        (10, 240), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)
            avg_bpms = np.average(bpms)

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
        if (stress_value_mouth > 0.79 and 5.0 >= avg_bpm > 0 and avg_bpms > 67.0) or \
                (stress_value_eyebrows > 0.85 and 5.0 >= avg_bpm > 0 and avg_bpms > 67.0):
            stress_label = "High Stress"
            while True:
                high += 1

                if high >= 2:
                    root = tk.Tk()
                    root.withdraw()
                    MsgBox = tk.messagebox.showinfo('You seem overworked, please take a short break to rejuvenate')
                break

        else:
            if (stress_value_mouth > 0.79 and 5.0 >= avg_bpm > 0 and avg_bpms > 60.0) or \
                    (stress_value_eyebrows > 0.85 and 5.0 >= avg_bpm > 0 and avg_bpms > 60.0):
                stress_label = "Medium Stress"

            elif stress_value_mouth > 0.75 or stress_value_eyebrows > 0.78:
                stress_label = "Low Stress"

            else:
                stress_label = "No Stress"

        # cv2.putText(frame, "stress value(mouth):{}".format(str(int(stress_value_mouth * 100))), (10, 30),
        # cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 102, 255), 2)
        # cv2.putText(frame, "stress value(eyebrows):{}".format(str(int(stress_value_eyebrows * 100))), (300, 30),
        # cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 102, 255), 2)
        cv2.putText(frame, "Stress level:{}".format(stress_label), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (102, 102, 255), 2)
        # cv2.putText(frame, "Blinks(Threshold = {0}): {1}".format(ear_thresh, total1), (10, 60),
        # cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 102, 255), 2)
        # cv2.putText(frame, "EAR: {:.2f}".format(ear), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 102, 255),2)

    f.write(str(ear))
    f.write('\n')

    cv2.imshow("Heart-rate Monitor", frame_ROI)
    cv2.imshow("Stress Monitor", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
f.close()
f1.close()
f2.close()
