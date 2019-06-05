from ObjectDetector import Detector
import io
import base64
import os

from flask import Flask, render_template, request, make_response
from flask_material import Material

from PIL import Image
from flask import send_file

app = Flask(__name__, static_folder="assets")
Material(app)

detector = Detector()

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# detector.detectNumberPlate('twocar.jpg')

def url_for_static(filename, image_type):
    root = app.config.get('STATIC_ROOT', '')
    return join(root, filename)

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
    img = detector.detectObject(image)
        
    byte_io = io.BytesIO()
    byte_io.write(img)
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
        print('------------------ FILE ---------------')
        print(request)
        print('------------------ End of FILE ---------------')

        file = Image.open(request.files['file'].stream)
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

        image = Image.open(file_path)

        base64image = detect_image(image)
        return render_template('index.html', detection_img=base64image, test_images=test_images, train_images=train_images)

if __name__ == "__main__":
    app.run(debug=True)
