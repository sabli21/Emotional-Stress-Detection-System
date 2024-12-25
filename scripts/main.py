import sys
import warnings
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

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
fa = face_utils.FaceAligner(predictor, desiredFaceWidth=256)

global eyebrow_points, outermouth_points, mouthcorner_points

eyebrow_points = []
outermouth_points = []
mouthcorner_points = []


class VideoCapture(object):
    def __init__(self):
        # facial festures variables
        self.ear = 0
        self.ear_thresh = 0.22  # eye aspect ratio threshold to indicate blink
        self.ear_consec_frames = 3  # no. of consecutive frames ear must be below threshold
        self.counter = 0
        self.counter1 = 0
        self.total1 = 0
        self.total2 = 0
        self.blink_counter = list()
        self.bpm = 0
        self.bpm_list = []
        self.avg_bpm = 0

        # heart rate variables
        self.frame_ROI = np.zeros((10, 10, 3), np.uint8)
        self.frame_out = np.zeros((10, 10, 3), np.uint8)
        self.samples = []
        self.buffer_size = 100
        self.times = []
        self.data_buffer = []
        self.fps = 0
        self.fft = []
        self.freqs = []
        self.t0 = time.time()
        self.Bpm = 0
        self.bpms = []
        self.avg_bpms = 0
        # ===========================#
        self.window = np.zeros([1, 13])
        self.frame_index = 0
        self.buffer = list()

        # ===========================#

        print("[INFO] starting video stream thread...")
        self.vs = VideoStream(src=0).start()

        time.sleep(1.0)

        self.now = datetime.datetime.now()
        print('now:', self.now)

    def __del__(self):
        self.vs.stop()

    def get_frame(self):

        # while True:
        frame = self.vs.read()
        frame = cv2.flip(frame, 1)
        frame = imutils.resize(frame, width=700, height=700)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame2 = self.vs.read()

        # getting indexes of facial landmarks for right eye and left eye
        (leyeStart, leyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (reyeStart, reyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

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
            self.ear = (leftEAR + rightEAR) / 2.0

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
            print('mouth_points', outermouth_points)

            # calculating the mouth corners feature
            F2 = sum(distance_matrix(mouth_corner, Ref))
            mouth_corner_feature = sum(F2) / normal_coefficient
            mouthcorner_points.append(int(mouth_corner_feature))
            print("corner", mouth_corner_feature)

            # calculating the eyebrows feature
            F3 = sum(distance_matrix(left_eyebrow, Ref) + distance_matrix(right_eyebrow, Ref))
            eyebrows_feature = sum(F3) / normal_coefficient
            eyebrow_points.append(int(eyebrows_feature))
            print('eyebrow points', eyebrow_points)
            print("eyebrow", eyebrows_feature)

            cv2.drawContours(frame, [leftEyeHull], -1, (255, 0, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (255, 0, 0), 1)
            cv2.drawContours(frame, [reyebrowhull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [leyebrowhull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [openmouthhull], -1, (0, 0, 255), 1)
            for (x1, y) in Ref:
                cv2.circle(frame, (x1, y), 3, (255, 255, 255), -1)

            # ============================#
            self.buffer.append(self.ear)

            if self.frame_index >= 12:
                # put the current EAR with +6 and -6 EARs in the window for SVM prediction
                self.window[0, :] = self.buffer[self.frame_index - 12: self.frame_index + 1]

                if self.buffer[self.frame_index - 6] < self.ear_thresh:
                    self.counter1 += 1
                    thresh_output = 1
                else:
                    thresh_output = 0
                    # if the eyes were closed for consecutive number of frames increment total number of blinks
                    if self.counter1 >= self.ear_consec_frames:
                        self.total1 += 1
                    # reset the ear frame counter
                    self.counter1 = 0
                f1.write(str(thresh_output))
                f1.write('\n')

                self.window = scaler.transform(self.window)
                print(self.window)

                output_predict = clf.predict(self.window)
                output_predict_write = output_predict[0]
                f2.write(str(output_predict_write))
                f2.write('\n')
                if output_predict == 1:
                    self.counter += 1
                else:
                    if self.counter >= 4:
                        self.total2 += 1
                        self.blink_counter.append(1)
                        print(self.blink_counter)
                    self.counter = 0
                    # when one minute has passed sum the count of blinks to get blinks per minute
                    if (self.now + datetime.timedelta(minutes=1)) <= datetime.datetime.now():
                        self.bpm = sum(self.blink_counter)
                        print(self.bpm)
                        self.bpm_list.append(self.bpm)
                        self.avg_bpm = np.average(self.bpm_list)  # average blinks per minute or blink rate
                        self.blink_counter.clear()  # clear the blink counter so it can start filling again
                        self.now = datetime.datetime.now()  # reset the one minute timer

            cv2.putText(frame, "Blinks(SVM): {}".format(self.total2), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (102, 102, 255),
                        2)
            cv2.putText(frame, "Blinks per minute: {} ".format(self.bpm), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (102, 102, 255), 2)
            cv2.putText(frame, "Blink Rate: {:.1f} ".format(self.avg_bpm), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (102, 102, 255), 2)

            self.frame_index += 1
            print(self.frame_index)

            face_frame, ROI1, ROI2, status, mask = face_detect(frame2)
            print(status)
            frame_ROI = face_frame

            g1 = extractColor(ROI1)
            g2 = extractColor(ROI2)

            L = len(self.data_buffer)

            # calculate average green value of 2 ROIs
            g = (g1 + g2) / 2

            # remove sudden change, if the avg value change is over 10, use the mean of the data_buffer
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                if abs(g - np.mean(self.data_buffer)) > 10 and L > 99:
                    g = self.data_buffer[-1]

            self.times.append(time.time() - self.t0)
            self.data_buffer.append(g)

            # only process in a fixed-size buffer
            if L > self.buffer_size:
                self.data_buffer = self.data_buffer[-self.buffer_size:]
                self.times = self.times[-self.buffer_size:]
                self.bpms = self.bpms[-self.buffer_size // 2:]
                L = self.buffer_size

            processed = np.array(self.data_buffer)

            # start calculating after the first 10 frames
            if L == self.buffer_size:
                self.fps = float(L) / (self.times[-1] - self.times[
                    0])  # calculate HR using a true fps of processor of the computer, not the fps the camera provide
                even_times = np.linspace(self.times[0], self.times[-1], L)

                processed = signal.detrend(processed)  # detrend the signal to avoid interference of light change
                interpolated = np.interp(even_times, self.times, processed)  # interpolation by 1
                interpolated = np.hamming(
                    L) * interpolated  # make the signal become more periodic (advoid spectral leakage)
                # norm = (interpolated - np.mean(interpolated))/np.std(interpolated)#normalization
                norm = interpolated / np.linalg.norm(interpolated)
                raw = np.fft.rfft(norm * 30)  # do real fft with the normalization multiplied by 10

                self.freqs = float(self.fps) / L * np.arange(L / 2 + 1)
                self.freqs = 60. * self.freqs

                self.fft = np.abs(raw) ** 2  # get amplitude spectrum

                idx = np.where(
                    (self.freqs > 50) & (self.freqs < 180))  # the range of frequency that HR is supposed to be within
                pruned = self.fft[idx]
                pfreq = self.freqs[idx]

                self.freqs = pfreq
                self.fft = pruned

                idx2 = np.argmax(pruned)  # max in the range can be HR

                self.Bpm = self.freqs[idx2]
                self.bpms.append(self.Bpm)

                processed = butter_bandpass_filter(processed, 1, 1, self.fps, order=3)
                # ifft = np.fft.irfft(raw)
            # samples = processed  # multiply the signal with 5 for easier to see in the plot

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
            cv2.putText(frame, "FPS: " + str(float("{:.2f}".format(self.fps))),
                        (10, 180), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)

            # frame_ROI = cv2.cvtColor(frame_ROI, cv2.COLOR_RGB2BGR)
            # frame_ROI = np.transpose(frame_ROI, (0, 1, 2)).copy()
            cv2.putText(frame, "Freq: " + str(float("{:.2f}".format(self.Bpm))),
                        (10, 210), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)

            if self.bpms.__len__() > 50:
                # if max(self.bpms - np.mean(self.bpms)) < 5:  # show HR if it is stable -the change is not over 5 bpm- for 3s
                cv2.putText(frame, "Heart rate: " + str(float("{:.2f}".format(np.mean(self.bpms)))) + " bpm",
                            (10, 240), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)
                self.avg_bpms = np.average(self.bpms)

            # normalize the values of all features between 0 and 1
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                print('mouth points', outermouth_points)
                outermouth_normalized = abs(outer_mouth_feature - np.min(outermouth_points)) / abs(
                    np.max(outermouth_points) -
                    np.min(outermouth_points))
                print("outer normal", outermouth_normalized)
                mouthcorner_normalized = abs(mouth_corner_feature - np.min(mouthcorner_points)) / abs(
                    np.max(mouthcorner_points) - np.min(mouthcorner_points))
                print("corners normal", mouthcorner_normalized)
                eyebrows_normalized = abs(eyebrows_feature - np.min(eyebrow_points)) / abs(
                    np.max(eyebrow_points) - np.min(eyebrow_points))
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
            if (stress_value_mouth > 0.79 and 5.0 >= self.avg_bpm > 0 and self.avg_bpms > 67.0) or \
                    (stress_value_eyebrows > 0.85 and 5.0 >= self.avg_bpm > 0 and self.avg_bpms > 67.0):
                stress_label = "High Stress"
            else:
                if (stress_value_mouth > 0.79 and 5.0 >= self.avg_bpm > 0 and self.avg_bpms > 60.0) or \
                        (stress_value_eyebrows > 0.85 and 5.0 >= self.avg_bpm > 0 and self.avg_bpms > 60.0):
                    stress_label = "Medium Stress"

                elif stress_value_mouth > 0.75 or stress_value_eyebrows > 0.78:
                    stress_label = "Low Stress"

                else:
                    stress_label = "No Stress"

            # cv2.putText(frame, "stress value(mouth):{}".format(str(int(stress_value_mouth * 100))), (10, 30),
            # cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 102, 255), 2)
            # cv2.putText(frame, "stress value(eyebrows):{}".format(str(int(stress_value_eyebrows * 100))), (300, 30),
            # cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 102, 255), 2)
            cv2.putText(frame, "Stress level: {}".format(stress_label), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (102, 102, 255), 2)
            # cv2.putText(frame, "Blinks(Threshold = {0}): {1}".format(ear_thresh, total1), (10, 60),
            # cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 102, 255), 2)
            # cv2.putText(frame, "EAR: {:.2f}".format(ear), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 102, 255),2)

            f.write(str(self.ear))
            f.write('\n')

            # cv2.imshow("Heart-rate Monitor", frame_ROI)
            # cv2.imshow("Stress Monitor", frame)
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()


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
        (x, y, w, h) = face_utils.rect_to_bb(rects2[0])
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        if y < 0:
            print("a")
            return face_frame, ROI1, ROI2, status, mask
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
    return face_frame, ROI1, ROI2, status, mask


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
    roi = cv2.resize(frame, (48, 48), interpolation=cv2.INTER_AREA)
    roi = convert_dtype(np.array([roi]))
    roi = normalize(roi)
    roi = reshape(roi)
    pr = model.predict(roi)[0]
    print(pr)
    max_emo = np.argmax(pr)
    label = emotions[max_emo]
    if label in ['fear', 'sad', 'angry']:
        label = 'Stressed'
    else:
        label = 'Not Stressed'
    return label
