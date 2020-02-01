import os

import cv2
import numpy as np
import pandas as pd

RESULT_VIDEO_FOLDER = './result'
RED_COLOR = (0, 0, 255)
DOT_RADIUS = 5
THICKNESS_FILL = -1
VIDEO_DIMENSION = (840, 720)
CODEC = cv2.VideoWriter_fourcc(*'avc1')
ANGLE = 0
START_ANGLE = 0
END_ANGLE = 360
OPACITY_STD = 0.6
OPACITY_2STD = 0.4
OPACITY_3STD = 0.2


class VideoProcessor:
    @staticmethod
    def process(screen_video_path, face_video_path, tag, result):
        result_df = VideoProcessor.__get_result_data_frame(result)

        print('Result video ({}) is processing...'.format(tag))

        VideoProcessor.__create_result_video(screen_video_path, face_video_path, tag, result_df)

    @staticmethod
    def __create_result_video(screen_video_path, face_video_path, tag, result_df):
        old_filename = screen_video_path.split('/')[-1].split('.')[0]
        new_filename = '{}_{}.mp4'.format(old_filename, tag)
        result_video_path = os.path.join(RESULT_VIDEO_FOLDER, new_filename)

        screen_video_frame_list = VideoProcessor.__get_video_frame_list(screen_video_path)
        face_video_frame_list = VideoProcessor.__get_video_frame_list(face_video_path)

        print('Result video path: ' + result_video_path)
        print('Number of frames in screen video: ' + str(len(screen_video_frame_list)))
        print('Number of frames in face video: ' + str(len(face_video_frame_list)))
        print('Result Length: ' + str(len(result_df.index)))
        print('Shape screen video frame: ' + str(screen_video_frame_list[0].shape))
        print('Shape face video frame: ' + str(face_video_frame_list[0].shape))

        VideoProcessor.__apply_result_to_screen_video(screen_video_frame_list, result_df)
        VideoProcessor.__combine_frame(screen_video_frame_list, face_video_frame_list, result_video_path)

    @staticmethod
    def __get_video_frame_list(video_path):
        video = cv2.VideoCapture(video_path)
        frame_list = list()

        while video.isOpened():
            frame_exists, current_frame = video.read()

            if frame_exists:
                frame_list.append(current_frame)
            else:
                break

        video.release()

        return frame_list

    @staticmethod
    def __apply_result_to_screen_video(screen_video_frame_list, result_df):
        for frame, result in zip(screen_video_frame_list, result_df.itertuples()):
            overlay_std = frame.copy()
            overlay_2std = frame.copy()
            overlay_3std = frame.copy()

            adapted_x_mean = max(0, min(result.x_mean_px, 720))
            adapted_y_mean = max(0, min(result.y_mean_px, 1440))

            center_coordinates = (adapted_x_mean, adapted_y_mean)
            axes_length_std = (result.x_std_px, result.y_std_px)
            axes_length_2std = (result.x_2std_px, result.y_2std_px)
            axes_length_3std = (result.x_3std_px, result.y_3std_px)

            # Center circle
            cv2.circle(frame, center_coordinates, DOT_RADIUS, RED_COLOR, THICKNESS_FILL)

            # 3 STD Ellipse
            cv2.ellipse(overlay_3std, center_coordinates, axes_length_3std,
                        ANGLE, START_ANGLE, END_ANGLE, RED_COLOR, THICKNESS_FILL)
            cv2.addWeighted(overlay_3std, OPACITY_3STD, frame, 1 - OPACITY_3STD, 0, frame)

            # 2 STD Ellipse
            cv2.ellipse(overlay_2std, center_coordinates, axes_length_2std,
                        ANGLE, START_ANGLE, END_ANGLE, RED_COLOR, THICKNESS_FILL)
            cv2.addWeighted(overlay_2std, OPACITY_2STD, frame, 1 - OPACITY_2STD, 0, frame)

            # 1 STD Ellipse
            cv2.ellipse(overlay_std, center_coordinates, axes_length_std,
                        ANGLE, START_ANGLE, END_ANGLE, RED_COLOR, THICKNESS_FILL)
            cv2.addWeighted(overlay_std, OPACITY_STD, frame, 1 - OPACITY_STD, 0, frame)

    @staticmethod
    def __get_result_data_frame(result):
        df = pd.DataFrame(result, columns=['x', 'y']).reset_index(drop=False)

        df['x_mean'] = df['index'].apply(
            lambda idx: df.loc[(df['index'] <= idx) & (df['index'] > idx - 10), 'x'].mean())
        df['y_mean'] = df['index'].apply(
            lambda idx: df.loc[(df['index'] <= idx) & (df['index'] > idx - 10), 'y'].mean())
        x_std = df['index'].apply(
            lambda idx: df.loc[(df['index'] <= idx) & (df['index'] > idx - 10), 'x'].std()).fillna(0)
        y_std = df['index'].apply(
            lambda idx: df.loc[(df['index'] <= idx) & (df['index'] > idx - 10), 'y'].std()).fillna(0)

        df['x_std'] = x_std
        df['y_std'] = y_std
        df['x_2std'] = 2 * x_std
        df['y_2std'] = 2 * y_std
        df['x_3std'] = 2 * x_std
        df['y_3std'] = 2 * y_std

        df['x_mean_px'] = ((df['x_mean'] + 1.82) * 111.282844).astype(int)
        df['y_mean_px'] = (-(df['y_mean'] + 0.28) * 111.196911).astype(int)
        df['x_std_px'] = ((df['x_std'] / 2.54) * 111.282844).astype(int)
        df['y_std_px'] = ((df['y_std'] / 2.54) * 111.196911).astype(int)
        df['x_2std_px'] = ((df['x_2std'] / 2.54) * 111.282844).astype(int)
        df['y_2std_px'] = ((df['y_2std'] / 2.54) * 111.196911).astype(int)
        df['x_3std_px'] = ((df['x_3std'] / 2.54) * 111.282844).astype(int)
        df['y_3std_px'] = ((df['y_3std'] / 2.54) * 111.196911).astype(int)

        return df

    @staticmethod
    def __combine_frame(screen_video_frame_list, face_video_frame_list, result_video_path):
        output = cv2.VideoWriter(result_video_path, CODEC, 20.0, VIDEO_DIMENSION)

        for screen_frame, face_frame in zip(screen_video_frame_list, face_video_frame_list):
            frame_1 = cv2.rotate(face_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            frame_2 = cv2.resize(screen_frame, None, fx=0.5, fy=0.5)

            combined_frame = np.zeros((720, 840, 3), dtype="uint8")

            combined_frame[0:720, 0:480] = frame_1
            combined_frame[0:720, 480:840] = frame_2

            output.write(combined_frame)

        output.release()
