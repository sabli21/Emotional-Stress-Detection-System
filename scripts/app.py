from flask import Flask, Response, render_template
from main import VideoCapture as vc

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


def gen(vc):
    """Video streaming generator function."""
    while True:
        frame = vc.get_frame()
        yield b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n'


@app.route("/predict", methods=['POST', 'GET'])
def predict():
    return Response(gen(vc()), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True, port=5000, threaded=False)
