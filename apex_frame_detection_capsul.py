# Source: Capsulnet, smic_preprocessing.py
import os
import sys
import cv2
import numpy as np
import pandas as pd
import face_recognition
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import csv

def detect_lmks(frame):
    lmks = face_recognition.face_landmarks(frame)
    if len(lmks) == 0:
        return None
    return lmks[0]


def get_cell(img, cell_location):
    point1, point2 = cell_location
    cell = img[point1[1]:point2[1], point1[0]:point2[0]]
    return cell


def get_cell_locations(lmks):
    def get_rect(center, width):
        point1 = np.array(center) - int(width / 2)
        point2 = np.array(center) + int(width / 2)
        return tuple(point1), tuple(point2)

    cells = {}
    if lmks is None:
        cell_width = 0
        return cells, cell_width

    cell_width = int((lmks['top_lip'][6][0] - lmks['top_lip'][0][0]) / 2)

    key = 'top_lip'
    points = np.array(lmks[key])
    left_lip_rect = get_rect(points[0], cell_width)
    right_lip_rect = get_rect(points[6], cell_width)
    cells['left_lip'] = left_lip_rect
    cells['right_lip'] = right_lip_rect

    key = 'chin'
    point = lmks[key][int(len(lmks[key]) / 2)]
    rect_point1 = (point[0] - int(cell_width / 2), point[1] - cell_width)
    rect_point2 = (point[0] + int(cell_width / 2), point[1])
    chin_rect = (rect_point1, rect_point2)
    cells['chin_rect'] = chin_rect

    key = 'nose_tip'
    point = lmks[key][0]
    left_nose_rect_point1 = (point[0] - cell_width, left_lip_rect[0][1] - cell_width)
    left_nose_rect_point2 = (point[0], left_lip_rect[0][1])
    left_nose_rect = (left_nose_rect_point1, left_nose_rect_point2)
    cells['left_nose'] = left_nose_rect

    point = lmks[key][4]
    right_nose_rect_point1 = (point[0], right_lip_rect[0][1] - cell_width)
    right_nose_rect_point2 = (point[0] + cell_width, right_lip_rect[0][1])
    right_nose_rect = (right_nose_rect_point1, right_nose_rect_point2)
    cells['right_nose'] = right_nose_rect

    key = 'left_eye'
    point = lmks[key][0]
    left_eye_rect_point1 = (point[0] - cell_width, int(point[1] - cell_width / 2))
    left_eye_rect_point2 = (point[0], int(point[1] + cell_width / 2))
    left_eye_rect = (left_eye_rect_point1, left_eye_rect_point2)
    cells['left_eye'] = left_eye_rect

    key = 'right_eye'
    point = lmks[key][3]
    right_eye_rect_point1 = (point[0], int(point[1] - cell_width / 2))
    right_eye_rect_point2 = (point[0] + cell_width, int(point[1] + cell_width / 2))
    right_eye_rect = (right_eye_rect_point1, right_eye_rect_point2)
    cells['right_eye'] = right_eye_rect

    left_point = lmks['left_eyebrow'][2]
    right_point = lmks['right_eyebrow'][2]
    center_point = (int((left_point[0] + right_point[0]) / 2),
                    int((left_point[1] + right_point[1]) / 2))

    center_eyebrow_rect = get_rect(center_point, cell_width)
    cells['center_eyebrow'] = center_eyebrow_rect

    left_rect_point1 = (int(center_point[0] - cell_width * 3 / 2),
                        int(center_point[1] - cell_width / 2))
    left_rect_point2 = (int(center_point[0] - cell_width * 1 / 2),
                        int(center_point[1] + cell_width / 2))
    left_eyebrow_rect = (left_rect_point1, left_rect_point2)
    cells['left_eyebrow'] = left_eyebrow_rect

    right_rect_point1 = (int(center_point[0] + cell_width * 1 / 2),
                         int(center_point[1] - cell_width / 2))
    right_rect_point2 = (int(center_point[0] + cell_width * 3 / 2),
                         int(center_point[1] + cell_width / 2))
    right_eyebrow_rect = (right_rect_point1, right_rect_point2)
    cells['right_eyebrow'] = right_eyebrow_rect

    return cells, cell_width


def compute_cell_difference(cell_t, cell_onset, cell_offset, cell_epsilon):
    numerator = (np.abs(cell_t - cell_onset) + 1.0)
    denominator = (np.abs(cell_t - cell_epsilon) + 1.0)
    difference = numerator / denominator

    numerator = (np.abs(cell_t - cell_offset) + 1.0)
    difference1 = numerator / denominator

    difference = difference + difference1

    return difference.mean()


def compute_cell_features(frame_t, on_frame, off_frame, frame_epsilon):
    lmks = detect_lmks(frame_t)
    cell_locations, cell_width = get_cell_locations(lmks)
    if cell_width == 0:
        return []
    cell_differences = {}
    frame_t = frame_t.astype(np.float32)
    on_frame = on_frame.astype(np.float32)
    off_frame = off_frame.astype(np.float32)
    frame_epsilon = frame_epsilon.astype(np.float32)

    for key in cell_locations:
        cell_location = cell_locations[key]
        cell_t = get_cell(frame_t, cell_location)
        cell_onset = get_cell(on_frame, cell_location)
        cell_offset = get_cell(off_frame, cell_location)
        cell_epsilon = get_cell(frame_epsilon, cell_location)

        cell_difference = compute_cell_difference(cell_t, cell_onset, cell_offset, cell_epsilon)
        cell_differences[key] = cell_difference
    return cell_differences


def find_apex_frame_of_frames(frames):
    epsilon = 1

    on_frame = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    off_frame = cv2.cvtColor(frames[-1], cv2.COLOR_BGR2GRAY)

    features = []

    for i in range(epsilon, len(frames)):
        frame_t = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        frame_epsilon = cv2.cvtColor(frames[i-epsilon], cv2.COLOR_BGR2GRAY)
        current_features = compute_cell_features(frame_t, on_frame, off_frame, frame_epsilon)
        feature = 0
        for key in current_features:
            feature += current_features[key]
        feature = feature / len(current_features)
        features.append(feature)

    padding = [0.0] * epsilon
    features = np.array(padding + features)
    apex_frame_idx = features.argmax()
    apex_frame = frames[apex_frame_idx]

    return apex_frame, features, apex_frame_idx


def draw_avg_plot(features, pred_apex_idx, data, clip_name):
    x = list(range(len(features)))
    plt.plot(x, features)
    plt.show()
    plt.axvline(x=pred_apex_idx, label='pred apex idx at={}'.format(pred_apex_idx), c='red')
    plt.legend()
    # plt.savefig('plots/{}/{}.png'.format(data, clip_name))
    plt.clf()
    plt.cla()
    plt.close()



def read_video_file(path):
    video_capture = cv2.VideoCapture(path)
    success = True
    frames = []
    while True:
        success, frame = video_capture.read()
        if success is False:
            break
        # out.write(frame)
        frames.append(frame)
    return frames


#frames = read_video_file("my_file_smile.avi")
# frames = read_video_file("/media/nastaran/HDD/DEAP/face_video/s01/s01_trial04.avi")
# frames = read_smic_files("/media/nastaran/HDD/SMIC/SMIC_all_raw/HS/s2/micro/surprise/s2_sur_02/")

#apex_frame, features, apex_frame_idx = find_apex_frame_of_frames(frames)
# print(features)
# print(apex_frame_idx)

#draw_avg_plot(features, apex_frame_idx, 'smic', "file_name")


def process_all():
    path = "/home/zsaf419/Documents/DEAP/face/face_video"
    files = os.listdir(path)
    files.sort()
    with open("apex-capsule.csv", 'w') as csv_file:
        writer = csv.writer(csv_file)
        row = ["file_name", "apex_frame", "frame_count"]
        writer.writerow(row)
        for participant_folder in files:
            trials_path = os.path.join(path, participant_folder)
            trials = os.listdir(trials_path)
            trials.sort()
            for file in trials:
                print(file)
                file_name = os.path.join(trials_path, file)
                frames = read_video_file(file_name)
                apex_frame, features, apex_frame_idx = find_apex_frame_of_frames(frames)
                pickle.dump((features, apex_frame_idx), open(
                    "pickles/{0}.pickle".format(file[:-4]), "wb"))
                row = [file, apex_frame_idx, len(frames)]
                writer.writerow(row)
                csv_file.flush()


process_all()
