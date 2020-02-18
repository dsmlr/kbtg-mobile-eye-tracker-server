import os
import pickle
import time

import cv2
import dlib
import ffmpeg
import numpy as np
import pandas as pd
import scipy.io as sio
from imutils import face_utils
from torch.utils import data


def _load_metadata(filename):
    return sio.loadmat(filename, squeeze_me=True, struct_as_record=False)['image_mean']


def _get_lreg(metadata_path):
    lreg = dict()
    filename_list = ['lr_eye_right_h', 'lr_eye_right_w', 'lr_eye_left_h', 'lr_eye_left_w', 'lr_face_h', 'lr_face_w']

    for filename in filename_list:
        lreg[filename] = pickle.load(open(os.path.join(metadata_path, filename), 'rb'))

    return lreg


METADATA_PATH = './metadata'
FACE_MEAN = _load_metadata(os.path.join(METADATA_PATH, 'mean_face_224.mat'))
LEFT_EYE_MEAN = _load_metadata(os.path.join(METADATA_PATH, 'mean_left_224.mat'))
RIGHT_EYE_MEAN = _load_metadata(os.path.join(METADATA_PATH, 'mean_right_224.mat'))
DETECTOR = dlib.get_frontal_face_detector()
PREDICTOR = dlib.shape_predictor(os.path.join(METADATA_PATH, 'shape_predictor_68_face_landmarks.dat'))
LREG = _get_lreg(METADATA_PATH)
PARAMS = {'batch_size': 20, 'shuffle': False, 'num_workers': 2}
# PARAMS = {'batch_size': 20, 'shuffle': False, 'num_workers': 0}
VALIDATION_POINTS = ['56,720', '360,56', '664,720', '360,1384']
TRAINING_POINTS = ['56,56', '664,56', '360,720', '56,1384', '664,1384']
FRAME_RANGE_LIST = [(68, 112), (128, 172), (188, 232), (248, 292), (308, 352), (368, 412), (428, 472), (488, 532),
                    (544, 592)]
# FRAME_RANGE_LIST = [(64, 123), (130, 156), (130, 156), (163, 189), (196, 222), (229, 255), (262, 288), (295, 321),
#                     (328, 354)]
DATA_FRAME = None
FRAMES = list()


class Extractor:
    @staticmethod
    def get_dataset_for_calibration(calibrate_video_path, x_positions, y_positions):
        global DATA_FRAME, FRAMES

        DATA_FRAME = None
        FRAMES = list()
        frames = list()

        video = cv2.VideoCapture(calibrate_video_path)

        print('Number of frames of calibrated video: {0}'.format(int(video.get(cv2.CAP_PROP_FRAME_COUNT))))

        rotate_code = Extractor.__check_rotation(calibrate_video_path)

        while video.isOpened():
            frame_exists, current_frame = video.read()
            if frame_exists:
                frames.append(Extractor.__correct_rotation(current_frame, rotate_code))
            else:
                break

        data_frame = pd.DataFrame(columns=['time', 'x', 'y', 'radius', 'idx_frame', 'u_idx'])

        for i, (x_pos, y_pos) in enumerate(zip(x_positions, y_positions)):
            frame_range = FRAME_RANGE_LIST[i]

            temp_df = pd.DataFrame([
                {'time': 0, 'x': x_pos, 'y': y_pos, 'radius': 56, 'idx_frame': j, 'u_idx': 0}
                for j in range(frame_range[0], frame_range[1])])

            data_frame = data_frame.append(temp_df, ignore_index=True)

        data_frame['xPos'] = (data_frame['x'].astype(float) / 111.282844) - 1.82
        data_frame['yPos'] = -(data_frame['y'].astype(float) / 111.196911) - 0.28
        data_frame['concat_xy'] = data_frame['x'].astype(str).str.cat(data_frame['y'].astype(str), sep=',')

        data_frame = data_frame.reset_index()

        labels = dict()
        for i in range(len(data_frame)):
            row = data_frame.iloc[i, :]
            labels[row['index']] = [row['xPos'], row['yPos']]

        DATA_FRAME = data_frame
        FRAMES = frames

        training_data_frame = data_frame.loc[(data_frame['concat_xy'].isin(TRAINING_POINTS)), :]
        validation_data_frame = data_frame.loc[(data_frame['concat_xy'].isin(VALIDATION_POINTS)), :]

        print(DATA_FRAME)

        validation_set = Dataset(validation_data_frame['index'].values, labels)
        validation_generator = data.DataLoader(validation_set, **PARAMS)

        training_set = Dataset(training_data_frame['index'].values, labels)
        training_generator = data.DataLoader(training_set, **PARAMS)

        return training_generator, validation_generator

    @staticmethod
    def extract_frames_from_video(video_path):
        global DATA_FRAME, FRAMES

        DATA_FRAME = None
        FRAMES = list()
        frames = list()

        video = cv2.VideoCapture(video_path)

        print('Number of frames of predicted video: {0}'.format(int(video.get(cv2.CAP_PROP_FRAME_COUNT))))

        rotate_code = Extractor.__check_rotation(video_path)

        while video.isOpened():
            frame_exists, current_frame = video.read()
            if frame_exists:
                frames.append(Extractor.__correct_rotation(current_frame, rotate_code))
            else:
                break

        data_frame = pd.DataFrame([
            {'time': 0, 'x': 0, 'y': 0, 'radius': 0, 'xPos': 0, 'yPos': 0, 'idx_frame': i, 'concat_xy': 0, 'u_idx': 0}
            for i in range(len(frames))])
        data_frame = data_frame.reset_index()
        index = data_frame['index'].values

        labels = dict()
        for i in range(len(data_frame)):
            row = data_frame.iloc[i, :]
            labels[row['index']] = [row['xPos'], row['yPos']]

        DATA_FRAME = data_frame
        FRAMES = frames

        test_set = Dataset(index, labels)
        test_generator = data.DataLoader(test_set, **PARAMS)

        return test_generator

    @staticmethod
    def __check_rotation(video_path):
        video_rotation = int(ffmpeg.probe(video_path)['streams'][0]['tags']['rotate'])
        rotate_code = None

        if video_rotation == 90:
            rotate_code = cv2.ROTATE_90_CLOCKWISE
        elif video_rotation == 180:
            rotate_code = cv2.ROTATE_180
        elif video_rotation == 270:
            rotate_code = cv2.ROTATE_90_COUNTERCLOCKWISE

        return rotate_code

    @staticmethod
    def __correct_rotation(frame, rotate_code):
        return cv2.rotate(frame, rotate_code)


class Dataset(data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, list_id, labels):
        """Initialization"""
        self.labels = labels
        self.list_id = list_id

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.list_id)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample

        label_id = self.list_id[index]

        # Load data and get label
        row = DATA_FRAME.iloc[label_id, :]
        image = FRAMES[row['idx_frame']]

        segmented = Dataset.__create_image_segment(image)

        if len(segmented) > 3:
            face = cv2.resize(segmented[1], (224, 224)) - FACE_MEAN
            left_eye = cv2.resize(segmented[2], (224, 224)) - LEFT_EYE_MEAN
            right_eye = cv2.resize(segmented[3], (224, 224)) - RIGHT_EYE_MEAN
            result = ([face / 255, left_eye / 255, right_eye / 255, segmented[4]])
        else:
            print('Could not detect face in frame index ' + str(row['idx_frame']))
            result = [np.zeros((224, 224)), np.zeros((224, 224)), np.zeros((224, 224)), np.zeros(25 * 25)]

        label = self.labels[label_id]

        return result, label

    @staticmethod
    def __create_image_segment(image):
        gray_image = Dataset.__get_gray_image(image)
        detected_rectangle = Dataset.__get_face_rectangle(gray_image)

        if len(detected_rectangle) == 0:
            # no component is detected
            return Dataset.__empty_numpy_array(image)

        rectangle, face_boundary_box = Dataset.__find_largest_face(detected_rectangle)
        np_shape = Dataset.__get_np_shape(gray_image, rectangle)

        roi_left_eye = Dataset.__get_roi_left_eye(image, np_shape)
        roi_right_eye = Dataset.__get_roi_right_eye(image, np_shape)
        roi_color = Dataset.__get_roi_color(image, np_shape, face_boundary_box)
        face_grid = Dataset.__get_face_grid(image, np_shape, face_boundary_box)

        try:
            output = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                      cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB),
                      cv2.cvtColor(roi_left_eye, cv2.COLOR_BGR2RGB),
                      cv2.cvtColor(roi_right_eye, cv2.COLOR_BGR2RGB),
                      np.reshape(cv2.resize(face_grid, (25, 25)), -1)]

            return output
        except:
            return Dataset.__empty_numpy_array()

    @staticmethod
    def __get_gray_image(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def __get_face_rectangle(gray_image):
        return DETECTOR(gray_image, 1)

    @staticmethod
    def __find_largest_face(detected_rectangle):
        rectangle = detected_rectangle[0]
        face_boundary_box = face_utils.rect_to_bb(rectangle)

        for i in range(len(detected_rectangle[1:])):
            next_rectangle = detected_rectangle[i]
            next_face = face_utils.rect_to_bb(rectangle)

            if face_boundary_box[1] * face_boundary_box[3] < next_face[1] * next_face[3]:
                rectangle = next_rectangle
                face_boundary_box = next_face

        return rectangle, face_boundary_box

    @staticmethod
    def __get_np_shape(gray_image, rectangle):
        shape = PREDICTOR(gray_image, rectangle)

        return face_utils.shape_to_np(shape)

    @staticmethod
    def __empty_numpy_array(image):
        return [cv2.cvtColor(image, cv2.COLOR_BGR2RGB), np.zeros((224, 224, 3), dtype=np.uint8),
                np.zeros((224, 224, 3), dtype=np.uint8), np.zeros((224, 224, 3), dtype=np.uint8), np.zeros(625)]

    @staticmethod
    def __get_roi_left_eye(image, np_shape):
        # Find left eye ROI
        left_eye_y = (max(min(np_shape[42:48, 1]), 0), max(np_shape[42:48, 1]))
        left_eye_x = (max(min(np_shape[42:48, 0]), 0), max(np_shape[42:48, 1]))

        x_left = np.expand_dims([(left_eye_x[1] - left_eye_x[0]) / (left_eye_y[1] - left_eye_y[0])], axis=1)

        c_eye_left_height = LREG['lr_eye_left_h'].predict(x_left)
        c_eye_left_width = LREG['lr_eye_left_w'].predict(x_left)

        left_margin_y = int(((left_eye_y[1] - left_eye_y[0]) * c_eye_left_height - (left_eye_y[1] - left_eye_y[0])) / 2)
        left_margin_x = int(((left_eye_x[1] - left_eye_x[0]) * c_eye_left_width - (left_eye_x[1] - left_eye_x[0])) / 2)

        left_eye_y = (max(left_eye_y[0] - left_margin_y, 0), left_eye_y[1] + left_margin_y)
        left_eye_x = (max(left_eye_x[0] - left_margin_x, 0), left_eye_x[1] + left_margin_x)

        return image[left_eye_y[0]:left_eye_y[1], left_eye_x[0]: left_eye_x[1], :]

    @staticmethod
    def __get_roi_right_eye(image, np_shape):
        # Find right eye ROI
        right_eye_y = (max(min(np_shape[36:42, 1]), 0), max(np_shape[36:42, 1]))
        right_eye_x = (max(min(np_shape[36:42, 0]), 0), max(np_shape[36:42, 0]))

        x_right = np.expand_dims([(right_eye_x[1] - right_eye_x[0]) / (right_eye_y[1] - right_eye_y[0])], axis=1)

        c_eye_right_height = LREG['lr_eye_right_h'].predict(x_right)
        c_eye_right_width = LREG['lr_eye_right_w'].predict(x_right)

        right_margin_y = int(
            ((right_eye_y[1] - right_eye_y[0]) * c_eye_right_height - (right_eye_y[1] - right_eye_y[0])) / 2)
        right_margin_x = int(
            ((right_eye_x[1] - right_eye_x[0]) * c_eye_right_width - (right_eye_x[1] - right_eye_x[0])) / 2)

        right_eye_y = (max(right_eye_y[0] - right_margin_y, 0), right_eye_y[1] + right_margin_y)
        right_eye_x = (max(right_eye_x[0] - right_margin_x, 0), right_eye_x[1] + right_margin_x)

        return image[right_eye_y[0]:right_eye_y[1], right_eye_x[0]: right_eye_x[1], :]

    @staticmethod
    def __get_roi_color(image, np_shape, face_boundary_box):
        face_y = (face_boundary_box[1], max(np_shape[:, 1]))
        face_x = (max(min(np_shape[:, 0]), 0), max(np_shape[:, 0]))

        return image[face_y[0]:face_y[1], face_x[0]:face_x[1], :]

    @staticmethod
    def __get_face_grid(image, np_shape, face_boundary_box):
        face_grid = np.zeros(np.shape(image)[0:2])

        face_y = (face_boundary_box[1], max(np_shape[:, 1]))
        face_x = (max(min(np_shape[:, 0]), 0), max(np_shape[:, 0]))

        face_grid[face_y[0]:face_y[1], face_x[0]:face_x[1]] = 1

        return face_grid
