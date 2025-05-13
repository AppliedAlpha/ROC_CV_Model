from flask import Flask, request, jsonify, render_template
from cv_model import model
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

id2name = model.names

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Receiving file
    if 'file' not in request.files:
        return jsonify({'error': 'no file'}), 400
    file = request.files['file']
    path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(path)

    # Predict image
    results = model.predict(source=path, imgsz=640, conf=0.25)

    res = results[0].cpu().numpy()
    output = [
        {
        'class_id': int(det[5]),
        'class_name': id2name[int(det[5])],
        'confidence': float(det[4]),
        'bbox': [float(det[0]), float(det[1]), float(det[2]), float(det[3])]
        }
        for det in res.boxes.data.tolist()
    ]

    return jsonify({'predictions': output})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8882)