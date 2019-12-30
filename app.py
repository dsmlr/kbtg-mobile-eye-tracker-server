import os

from flask import Flask, request
from werkzeug.utils import secure_filename

from services.prediction import Predictor
from services.preparation import Extractor

UPLOAD_FOLDER = './uploads'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# @app.route('/calibrate', methods=["POST"])
# def calibrate():
#     if 'image[]' not in request.files:
#         return {'message': 'No image in the request'}, 400
#
#     images = request.files.getlist("image[]")
#
#     Predictor.
#
#     for image in images:
#         image.save(os.path.join(app.config['UPLOAD_FOLDER'], image.filename + ".jpeg"))
#
#     return {'message': 'Upload Successfully'}, 200


@app.route('/save-screen-video', methods=["POST"])
def save_screen_video():
    if 'video[]' not in request.files:
        return {'message': 'No video in the request'}, 400

    videos = request.files.getlist("video[]")

    screen_video = videos[0]
    screen_video_filename = secure_filename(screen_video.filename)
    screen_video.save(os.path.join(app.config['UPLOAD_FOLDER'], screen_video_filename))

    return {'message': 'Upload Successfully'}, 200


@app.route('/predict', methods=["POST"])
def predict():
    if 'video[]' not in request.files:
        return {'message': 'No video in the request'}, 400

    videos = request.files.getlist("video[]")

    face_video = videos[0]
    face_video_filename = secure_filename(face_video.filename)
    face_video.save(os.path.join(app.config['UPLOAD_FOLDER'], face_video_filename))
    face_video_path = "./uploads/" + face_video_filename

    test_generator = Extractor.extract_frames_from_video(face_video_path)

    print(Predictor.predict(test_generator))
    # print(Predictor.svr_predict())

    return {'message': 'Upload Successfully'}, 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
