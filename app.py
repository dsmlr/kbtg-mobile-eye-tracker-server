import os

import torch
from flask import Flask, render_template
from flask import request
from werkzeug.utils import secure_filename

from services.prediction import Predictor, Calibrator
from services.preparation import Extractor
from services.video_processor import VideoProcessor

FACES_FOLDER = './faces'
SCREENS_FOLDER = './screens'
CALIBRATE_FOLDER = './calibrate'
app = Flask(__name__, static_folder='result')
app.config['FACES_FOLDER'] = FACES_FOLDER
app.config['SCREENS_FOLDER'] = SCREENS_FOLDER
app.config['CALIBRATE_FOLDER'] = CALIBRATE_FOLDER

SCREEN_VIDEO_PATH = None


@app.route('/')
def index():
    results = os.listdir('result')
    results = [file for file in results if '.mp4' in file]

    return render_template('index.html', results=results)


@app.route('/check-cuda')
def check_cuda():
    print(torch.cuda.current_device())
    print(torch.cuda.device(0))
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.is_available())

    return {'message': 'Checking Successfully'}, 200


@app.route('/test-predict')
def test_predict():
    VideoProcessor.process(None, None, 'test', [[1.4, -2.1], [1.5, -3.2], [1.6, -4.3]])

    return {'message': 'Upload Successfully'}, 200


@app.route('/calibrate', methods=['POST'])
def calibrate():
    if 'video[]' not in request.files:
        return {'message': 'No video in the request'}, 400

    x_positions = request.form.get('xPositions').split(',')
    y_positions = request.form.get('yPositions').split(',')

    calibrate_video = request.files.getlist('video[]')[0]
    calibrate_video_filename = secure_filename(calibrate_video.filename)
    calibrate_video_path = os.path.join(app.config['CALIBRATE_FOLDER'], calibrate_video_filename)
    calibrate_video.save(calibrate_video_path)

    training_generator, validation_generator = Extractor.get_dataset_for_calibration(calibrate_video_path, x_positions,
                                                                                     y_positions)
    Calibrator.calibrate(training_generator, validation_generator)

    return {'message': 'Upload Successfully'}, 200


@app.route('/save-screen-video', methods=['POST'])
def save_screen_video():
    global SCREEN_VIDEO_PATH

    if 'video[]' not in request.files:
        return {'message': 'No video in the request'}, 400

    videos = request.files.getlist('video[]')

    screen_video = videos[0]
    screen_video_filename = secure_filename(screen_video.filename)
    SCREEN_VIDEO_PATH = os.path.join(app.config['SCREENS_FOLDER'], screen_video_filename)
    screen_video.save(SCREEN_VIDEO_PATH)

    return {'message': 'Upload Successfully'}, 200


@app.route('/predict', methods=['POST'])
def predict():
    global SCREEN_VIDEO_PATH

    if 'video[]' not in request.files:
        return {'message': 'No video in the request'}, 400

    videos = request.files.getlist('video[]')

    face_video = videos[0]
    face_video_filename = secure_filename(face_video.filename)
    face_video_path = os.path.join(app.config['FACES_FOLDER'], face_video_filename)
    face_video.save(face_video_path)

    test_generator = Extractor.extract_frames_from_video(face_video_path)

    # normal_result = Predictor.predict(test_generator)
    svr_result = Predictor.svr_predict(test_generator)

    print(SCREEN_VIDEO_PATH)

    if svr_result is not None:
        VideoProcessor.process(SCREEN_VIDEO_PATH, face_video_path, 'svr', svr_result)

    # VideoProcessor.process(SCREEN_VIDEO_PATH, face_video_path, 'normal', normal_result)

    return {'message': 'Success'}, 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
