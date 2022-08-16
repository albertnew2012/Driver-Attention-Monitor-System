import math
import os
import time
from datetime import datetime
from typing import List

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pygame
from pygame import mixer

import utils

LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_EYEBROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
INNER_LIP = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
OUTER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
NOSE = [64, 6, 294, 94]
FACE_BOUNDARY = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176,
                 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
POSE = [33, 263, 1, 61, 291, 199]
radius = 1
red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)
yellow = (0, 255, 255)
black = (0, 0, 0)
white = (255, 255, 255)

IRIS_THRESHOLD = 0.55
EAR_THRESHOLD = 0.15
SOUND_VOLUME = 0.1
LOOP_DURATION = 30  # seconds
ALLOWEDCLOSURE = 0.8  # allowed maximum time for eye closure in seconds
FONTS = cv2.FONT_HERSHEY_COMPLEX
SHOW_PULSE = False


def moving_average(a, n=5):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


# Euclaidean distance
def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
    return distance


# For connecting range of dots
def drawPoints(indices: List[int], color=white):
    # draw point
    for i in range(len(indices)):
        cv2.circle(frame, mesh_points[indices[i], :2], radius, color, -1)


def eyeAspectRatio(indices):
    p1 = mesh_points[indices[0], :2]
    p2 = mesh_points[indices[13], :2]
    p3 = mesh_points[indices[10], :2]
    p4 = mesh_points[indices[8], :2]
    p5 = mesh_points[indices[6], :2]
    p6 = mesh_points[indices[3], :2]

    width = euclaideanDistance(p1, p4)
    height0 = euclaideanDistance(p2, p6)
    height1 = euclaideanDistance(p3, p4)
    EAR = (height0 + height1) / 2 / width
    return EAR


def mouthAspectRatio(indices=INNER_LIP):
    p1 = mesh_points[indices[0], :2]
    p2 = mesh_points[indices[4], :2]
    p3 = mesh_points[indices[6], :2]
    p4 = mesh_points[indices[11], :2]
    p5 = mesh_points[indices[14], :2]
    p6 = mesh_points[indices[16], :2]

    width = euclaideanDistance(p1, p4)
    height0 = euclaideanDistance(p2, p6)
    height1 = euclaideanDistance(p3, p4)
    MAR = (height0 + height1) / 2 / width
    return MAR


# Sound
def create_sound(aduio_path, volume=1.0):
    sound = pygame.mixer.Sound(aduio_path)
    sound.set_volume(volume)
    return sound


if __name__ == "__main__":
    # initialize mixer
    mixer.init()

    # loading in the voices/sounds
    voice_left = create_sound('audio/looking-left.wav', SOUND_VOLUME)
    voice_right = create_sound('audio/looking-right.wav', SOUND_VOLUME)
    voice_up = create_sound('audio/looking-up.wav', SOUND_VOLUME)
    voice_down = create_sound('audio/looking-down.wav', SOUND_VOLUME)
    voice_drowsy = create_sound('audio/alert.wav', SOUND_VOLUME)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)
    n = 0
    EAR = []
    CEF_COUNTER = 0
    CLOSED_EYES_FRAME = 3
    TOTAL_BLINKS = 0
    DEF_COUNTER = 0

    cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.set(cv2.CAP_PROP_FPS, 30)

    # video Recording setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc =  cv2.VideoWriter_fourcc(*'MJPG')
    now = datetime.now()
    if not os.path.isdir("output"):
        os.mkdir("output")
    output_file = os.path.join("./output", now.strftime("%y-%m-%d-%H_%M_%S") + '.mp4')
    video_writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'MJPG'), 30, (width, height))

    distortion_coeffs = np.zeros((4, 1))
    focal_length = width
    center = (width / 2, height / 2)
    cam_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype=np.float64)
    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    direction = None
    drowsy = None

    color = {"Looking Left": [utils.BLACK, utils.YELLOW], "Looking Right": [utils.BLACK, utils.YELLOW], \
             "Looking Up": [utils.BLACK, utils.YELLOW], "Looking Down": [utils.BLACK, utils.RED],
             "Looking Forward": [utils.WHITE, utils.GREEN]}

    prev_time = time.time()
    start = time.time()
    plt.ion()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        face_3d = []
        face_2d = []
        if results.multi_face_landmarks:
            # print(results.multi_face_landmarks[0].landmark)
            mesh_points = np.array(
                [np.multiply([p.x, p.y, p.z], [width, height, 1.0]) for p in results.multi_face_landmarks[0].landmark])

            nose_2d = mesh_points[1, :2].astype(np.float64)
            nose_3d = mesh_points[1].astype(np.float64)
            nose_3d[2] *= 4000
            face_2d = mesh_points[POSE, :2].astype(np.float64)
            face_3d = mesh_points[POSE].astype(np.float64)
            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            # See where the user's head tilting
            if y < -10:
                direction = "Looking Right"  # playsound('audio/left.wav')
                if not pygame.mixer.get_busy():
                    voice_right.play()
            elif y > 10:
                direction = "Looking Left"  # playsound('audio/right.wav')
                if not pygame.mixer.get_busy():
                    voice_left.play()
            elif x < -10:
                direction = "Looking Down"
                if not pygame.mixer.get_busy():
                    voice_down.play()
            elif x > 10:
                direction = "Looking Up"
                if not pygame.mixer.get_busy():
                    voice_up.play()
            else:
                direction = "Looking Forward"  # playsound('audio/center.wav')
            utils.colorBackgroundText(frame, f'Facing direction: {direction}', FONTS, 0.5, (10, 60), 1,
                                      color[direction][0], color[direction][1], 8, 8)

            # Display the nose direction
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_3d_projection[0][0][0] + y * 10), int(nose_3d_projection[0][0][1] - x * 10))
            cv2.line(frame, p1, p2, (255, 0, 0), 3)

            mesh_points = mesh_points.astype(int)
            cv2.polylines(frame, [mesh_points[LEFT_EYEBROW, :2]], True, (0, 255, 0), 1, cv2.LINE_AA)
            drawPoints(LEFT_EYEBROW)
            cv2.polylines(frame, [mesh_points[RIGHT_EYEBROW, :2]], True, (0, 255, 0), 1, cv2.LINE_AA)
            drawPoints(RIGHT_EYEBROW)
            cv2.polylines(frame, [mesh_points[LEFT_EYE, :2]], True, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.polylines(frame, [mesh_points[RIGHT_EYE, :2]], True, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.polylines(frame, [mesh_points[INNER_LIP, :2]], True, (0, 255, 0), 1, cv2.LINE_AA)
            drawPoints(INNER_LIP)
            cv2.polylines(frame, [mesh_points[OUTER_LIP, :2]], True, (0, 255, 0), 1, cv2.LINE_AA)
            drawPoints(OUTER_LIP)
            cv2.polylines(frame, [mesh_points[NOSE, :2]], True, (0, 255, 0), 1, cv2.LINE_AA)
            drawPoints(NOSE)
            cv2.polylines(frame, [mesh_points[FACE_BOUNDARY, :2]], True, (0, 255, 0), 1, cv2.LINE_AA)
            drawPoints(FACE_BOUNDARY)
            (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS, :2])
            (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS, :2])
            center_left = np.array([l_cx, l_cy], np.int32)
            center_right = np.array([r_cx, r_cy], np.int32)
            cv2.circle(frame, center_left, int(l_radius), (0, 0, 255), 1, cv2.LINE_AA)
            cv2.circle(frame, center_right, int(r_radius), (0, 0, 255), 1, cv2.LINE_AA)

            # calculate intersection
            left_eye_mask = cv2.drawContours(np.zeros(frame.shape[:2], np.uint8), [mesh_points[LEFT_EYE, :2]], -1, 1,
                                             cv2.FILLED)
            right_eye_mask = cv2.drawContours(np.zeros(frame.shape[:2], np.uint8), [mesh_points[RIGHT_EYE, :2]], -1, 1,
                                              cv2.FILLED)
            left_iris_mask = cv2.circle(np.zeros(frame.shape[:2], np.uint8), center_left, int(l_radius), 1, -1)
            right_iris_mask = cv2.circle(np.zeros(frame.shape[:2], np.uint8), center_right, int(r_radius), 1, -1)
            left_iris_open = cv2.bitwise_and(left_iris_mask, left_eye_mask)
            right_iris_open = cv2.bitwise_and(right_iris_mask, right_eye_mask)
            left_open_ratio = left_iris_open.sum() / left_iris_mask.sum()
            right_open_ratio = right_iris_open.sum() / right_iris_mask.sum()
            iris_visibility = (left_open_ratio + right_open_ratio) / 2
            # cv2.putText(frame, f"visible pupil ratio {left_open_ratio:0.2f}, {right_open_ratio:0.2f}", (25, 110),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.75, color=(0, 0, 255))
            utils.colorBackgroundText(frame, f"Iris visibility {left_open_ratio:0.2f}, {right_open_ratio:0.2f}", FONTS,
                                      0.5, (10, 95), 1, color[direction][0], color[direction][1], 8, 8)

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            utils.colorBackgroundText(frame, f'FPS: {fps:.1f}', FONTS, 0.5, (10, 25), 1, utils.BLACK, utils.WHITE, 8, 8)

            if iris_visibility < IRIS_THRESHOLD:
                DEF_COUNTER += 1
                # print(DEF_COUNTER)

                if DEF_COUNTER > int(ALLOWEDCLOSURE / (1 / fps)):
                    utils.colorBackgroundText(frame, f'Drowsy', cv2.FONT_HERSHEY_COMPLEX, 1, (10, 175), 2, utils.WHITE,
                                              utils.RED, 8, 8)
                    drowsy = True
                    if not pygame.mixer.get_busy():
                        voice_drowsy.play()

            else:
                DEF_COUNTER = 0
                drowsy = False

            if not drowsy and (left_EAR := eyeAspectRatio(LEFT_EYE)) < EAR_THRESHOLD or (
                    right_EAR := eyeAspectRatio(RIGHT_EYE)) < EAR_THRESHOLD:
                # cv2.putText(frame, "Blinking", (25,50),cv2.FONT_HERSHEY_SIMPLEX, 0.75, color=(0, 255, 0))
                n += 1
                # print(TOTAL_BLINKS)
                utils.colorBackgroundText(frame, f'Blink', cv2.FONT_HERSHEY_COMPLEX, 1, (10, 135), 2, utils.YELLOW, 8,
                                          8)
                CEF_COUNTER += 1
            else:
                if CEF_COUNTER > CLOSED_EYES_FRAME:
                    TOTAL_BLINKS += 1
                    CEF_COUNTER = 0

            EAR.append((left_EAR + right_EAR) / 2)

            if SHOW_PULSE:
                ear = moving_average(EAR, 5)
                plt.plot(ear[-100:])
                plt.xlim([0, 100])
                plt.ylim([0.12, 0.26])
                plt.title("Eye Blink Pulse")
                plt.xlabel("frameID")
                plt.ylabel("Eye Aspect Ratio")
                plt.draw()
                plt.pause(0.00000000000000001)
                plt.clf()

            if (MAR := mouthAspectRatio(INNER_LIP)) > 0.5:
                utils.colorBackgroundText(frame, f'Yawn', cv2.FONT_HERSHEY_COMPLEX, 1, (120, 135), 2, utils.YELLOW, 8,
                                          8)

        if time.time() - start > LOOP_DURATION:
            start = time.time()
            now = datetime.now()
            output_file = os.path.join("./output", now.strftime("%y-%m-%d-%H_%M_%S") + '.mp4')
            video_writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

        # Write the frame to the current video writer
        video_writer.write(frame)
        cv2.imshow('Driver Attention Monitor System', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
