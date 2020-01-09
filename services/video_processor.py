import os

import cv2

RESULT_VIDEO_FOLDER = './result'
DOT_COLOR = (0, 0, 255)
DOT_RADIUS = 10
DOT_THICKNESS = -1


class VideoProcessor:
    @staticmethod
    def process(path_to_src_screen_video, tag, result):
        converted_result = VideoProcessor.__convert_coordinate(result)

        print('Result video ({}) is processing...'.format(tag))
        VideoProcessor.__create_result_video(path_to_src_screen_video, tag, converted_result)

    @staticmethod
    def __convert_coordinate(result):
        converted_coordinate_list = list()

        for row in result:
            coordinate = (int((row[0] + 1.82) * 111.282844), -int((row[1] + 0.28) * 111.196911))
            converted_coordinate_list.append(coordinate)

        return converted_coordinate_list

    @staticmethod
    def __create_result_video(path_to_src_screen_video, tag, converted_result):
        counter = 0
        old_filename = path_to_src_screen_video.split('/')[-1].split('.')[0]
        new_filename = '{}_{}.mp4'.format(old_filename, tag)
        result_video_path = os.path.join(RESULT_VIDEO_FOLDER, new_filename)

        print('Result video path: ' + result_video_path)

        video = cv2.VideoCapture(path_to_src_screen_video)

        number_of_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video_dimension = (int(video.get(3)), int(video.get(4)))

        output = result_video_path

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output, fourcc, 20.0, video_dimension)

        print('Screen video dimension: {}'.format(video_dimension))
        print('Number of frames in screen video: ' + str(number_of_frames))
        print('Result Length: ' + str(len(converted_result)))

        while video.isOpened():
            frame_exists, current_frame = video.read()

            if frame_exists:
                index = counter // 2

                if index >= len(converted_result):
                    out.write(current_frame)
                else:
                    # print('Drawing: X = {}, Y = {}'.format(converted_result[index][0], converted_result[index][1]))
                    cv2.circle(current_frame, converted_result[index], DOT_RADIUS, DOT_COLOR, DOT_THICKNESS)
                    out.write(current_frame)
            else:
                break

            counter += 1

        video.release()
        out.release()
