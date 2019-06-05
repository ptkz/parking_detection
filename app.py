from ObjectDetector import Detector
import io
import base64
import os
import cv2
import numpy as np

from flask import Flask, render_template, request, make_response, send_file
from flask_material import Material

from PIL import Image, JpegImagePlugin

import vis_utils

app = Flask(__name__, static_folder="assets")
Material(app)

detector = Detector()

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

CATEGORY_INDEX = {1: {'name': 'occupied', 'id': 1}, 2: {'name': 'empty', 'id': 2}}

# detector.detectNumberPlate('twocar.jpg')

# def url_for_static(filename, image_type):
#     root = app.config.get('STATIC_ROOT', '')
#     return join(root, filename)

def get_images(image_type):
    root = app.config.get('STATIC_ROOT', image_type)
    if image_type == 'test':
        images = os.listdir('assets/test')
        images = ['test/' + image for image in images]
        # images = [os.path.join(root, image for image in images)]        
        # print(images)
        return images
        # return render_template('report.html', hists = hists)
    else:
        
        images = os.listdir('assets/train')
        # images = [os.path.join(root, image for image in images)]
        images = ['train/' + image for image in images]
        # print(images)
        return images

def detect_image(image): 
    (boxes, scores, classes, num) = detector.get_classification(image)
    img = vis_utils.visualize_boxes_and_labels_on_image_array(
        image, 
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        CATEGORY_INDEX,
        use_normalized_coordinates=True,
        line_thickness=4,
        min_score_thresh=0.80,
        skip_scores=False,
        skip_labels=False)
        
    image_pil = Image.fromarray(np.uint8(img))
    byte_io = io.BytesIO()
    image_pil.save(byte_io, format='JPEG')
    return base64.b64encode(byte_io.getvalue()).decode('utf-8')


@app.route("/")
def index():
    if request.method == 'GET':
        test_images = get_images('test')
        train_images = get_images('train')
        return render_template('index.html', test_images=test_images, train_images=train_images)


@app.route("/", methods=['POST'])
def upload():
    if request.method == 'POST':


        upload = request.files['file']
        filename = upload.filename

        target = os.path.join(APP_ROOT, 'assets', 'uploads')
        destination = '/'.join([target, filename])

        upload.save(destination)

        file = cv2.imread(destination)

        # file = cv2.imread(request.files['file'].stream)
        # img = detector.detectObject(file)
        
        # byte_io = io.BytesIO()
        # byte_io.write(img)
        # base64image = base64.b64encode(byte_io.getvalue()).decode('utf-8')
        base64image = detect_image(file)

        test_images = get_images('test')
        train_images = get_images('train')
        # result = base64image[2:-1]

        # print(result)
        # byte_io.seek(0,0)
        # response = make_response(send_file(byte_io,mimetype='image/jpg'))
        # response.headers['Content-Transfer-Encoding']='base64'
        # return send_file(byte_io, mimetype='image/jpg', as_attachment=False)
        return render_template('index.html', detection_img=base64image, test_images=test_images, train_images=train_images)
        # return response

@app.route("/run_detection/<path:filename>", methods = ['GET'])
def run_detection(filename):
    if request.method == 'GET':
        test_images = get_images('test')
        train_images = get_images('train')

        file_path = os.path.join('assets', filename)
        print(file_path)

        # image = Image.open(file_path)
        image = cv2.imread(file_path)
        base64image = detect_image(image)
        return render_template('index.html', detection_img=base64image, test_images=test_images, train_images=train_images)

if __name__ == "__main__":
    app.run(debug=True)
