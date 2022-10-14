import io
import base64
import os
import cv2
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, make_response, send_file
from flask_material import Material
from PIL import Image, JpegImagePlugin

import vis_utils
from ObjectDetector import Detector

app = Flask(__name__, static_folder="assets")
Material(app)
detector = Detector()

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

CATEGORY_INDEX = {1: {'name': 'occupied', 'id': 1}, 2: {'name': 'empty', 'id': 2}}
MIN_SCORE_THRESH = 0.80

def calculate_accuracy(boxes, classes, scores, img_path):
    # Detected classes count
    count_occupied_det = 0
    count_empty_det = 0
    # Ground Truth classes count
    count_occupied_ground = 0
    count_empty_ground = 0

    for i in range(min(200, boxes.shape[0])):
        if scores is None or scores[i] > MIN_SCORE_THRESH:
            box = tuple(boxes[i].tolist())
            class_name = CATEGORY_INDEX[classes[i]]['name']
            if class_name == 'occupied':
                count_occupied_det += 1
            elif class_name == 'empty':
                count_empty_det += 1   

    # Get the groundtruth labels
    (path, file_name) = img_path.split('/')

    label_file = pd.read_csv("assets/labels/" + path + "_labels.csv")
    for index, row in label_file.iterrows() :
        if row['filename'] == file_name:
            if row['class'] == 'occupied':
                count_occupied_ground += 1
            elif row['class'] == 'empty':
                count_empty_ground += 1

    accuracy =( (count_empty_det + count_occupied_det) / (count_empty_ground + count_occupied_ground) ) * 100

    result = {
        'count_empty_det': count_empty_det, 
        'count_empty_ground': count_empty_ground,
        'count_occupied_det': count_occupied_det,
        'count_occupied_ground': count_occupied_ground,
        'accuracy': format(accuracy, '.2f')}

    return result


def get_images(image_type):
    root = app.config.get('STATIC_ROOT', image_type)
    if image_type == 'test':
        images = os.listdir('assets/test')
        images = ['test/' + image for image in images]
        return images
    elif image_type == 'train':
        images = os.listdir('assets/train')
        images = ['train/' + image for image in images]
        return images
    else:
        images = os.listdir('assets/uploads')
        images = ['uploads/' + image for image in images]
        return images

def visualize_image(image, boxes, scores, classes, num): 

    # Draw bounding boxes to the image based on classification results
    img = vis_utils.visualize_boxes_and_labels_on_image_array(
        image, 
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        CATEGORY_INDEX,
        use_normalized_coordinates=True,
        line_thickness=4,
        min_score_thresh=MIN_SCORE_THRESH,
        skip_scores=False,
        skip_labels=False)

    # Return the image with bounding boxes in a Base64 format    
    image_pil = Image.fromarray(np.uint8(img))
    byte_io = io.BytesIO()
    image_pil.save(byte_io, format='JPEG')

    return base64.b64encode(byte_io.getvalue())


@app.route("/")
def index():
    if request.method == 'GET':

        # Load test and train images to show 
        test_images = get_images('test')
        train_images = get_images('train')
        uploaded_images = get_images('uploaded')

        return render_template('index.html', test_images=test_images, train_images=train_images, uploaded_images=uploaded_images)


@app.route("/", methods=['POST'])
def upload():
    if request.method == 'POST':

        # Retrieve the uploaded image
        upload = request.files['file']
        filename = upload.filename

        # Specify path to save the uploaded image
        target = os.path.join(APP_ROOT, 'assets', 'uploads')
        destination = '/'.join([target, filename])

        # Save the uploaded image and read it
        upload.save(destination)
        image = cv2.imread(destination)

        # Run parking detection
        (boxes, scores, classes, num) = detector.get_classification(image)
        base64image = visualize_image(image, boxes, scores, classes, num)


        # Decode image from base64 and save to the /uploads folder
        imgdata = base64.b64decode(base64image)
        with open(destination, 'wb') as f:
            f.write(imgdata)

        base64image = base64image.decode('utf-8')

        # Load test and train images to show 
        test_images = get_images('test')
        train_images = get_images('train')
        uploaded_images = get_images('uploaded')

        return render_template('index.html', 
            detection_img=base64image, 
            test_images=test_images, 
            train_images=train_images, 
            uploaded_images=uploaded_images)

@app.route("/run_detection/<path:filename>", methods = ['GET'])
def run_detection(filename):
    if request.method == 'GET':

        # Get filepath and read the image
        file_path = os.path.join('assets', filename)
        image = cv2.imread(file_path)

        # Run parking detection
        (boxes, scores, classes, num) = detector.get_classification(image)
        base64image = visualize_image(image, boxes, scores, classes, num)

        # Calculate the accuracy
        accuracy = calculate_accuracy(
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            filename)

        # Load test and train images to show
        test_images = get_images('test')
        train_images = get_images('train')
        uploaded_images = get_images('uploaded')

        base64image = base64image.decode('utf-8')

        return render_template('index.html',
            detection_img=base64image,
            test_images=test_images, 
            train_images=train_images, 
            uploaded_images=uploaded_images,
            accuracy=accuracy)

if __name__ == "__main__":
    app.run(debug=True)
