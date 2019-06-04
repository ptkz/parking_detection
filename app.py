from ObjectDetector import Detector
import io
import base64

from flask import Flask, render_template, request, make_response
from flask_material import Material

from PIL import Image
from flask import send_file

app = Flask(__name__)
Material(app)

detector = Detector()

# detector.detectNumberPlate('twocar.jpg')


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/", methods=['POST'])
def upload():
    if request.method == 'POST':
        print('------------------ FILE ---------------')
        print(request)
        print('------------------ End of FILE ---------------')

        file = Image.open(request.files['file'].stream)
        img = detector.detectObject(file)
        
        byte_io = io.BytesIO()
        byte_io.write(img)
        base64image = base64.b64encode(byte_io.getvalue()).decode('utf-8')

        # result = base64image[2:-1]

        # print(result)
        # byte_io.seek(0,0)
        # response = make_response(send_file(byte_io,mimetype='image/jpg'))
        # response.headers['Content-Transfer-Encoding']='base64'
        # return send_file(byte_io, mimetype='image/jpg', as_attachment=False)
        return render_template('index.html', detection_img=base64image)
        # return response


if __name__ == "__main__":
    app.run(debug=True)
