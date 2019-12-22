import os

from flask import Flask, request

UPLOAD_FOLDER = './uploads'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/calibrate', methods=["POST"])
def calibrate():
    print(request)
    if 'image[]' not in request.files:
        return {'message': 'No image in the request'}, 400

    images = request.files.getlist("image[]")

    print(images)

    for image in images:
        image.save(os.path.join(app.config['UPLOAD_FOLDER'], image.filename + ".jpeg"))

    return {'message': 'Upload Successfully'}, 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
