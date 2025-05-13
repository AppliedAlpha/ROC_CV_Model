from flask import Flask, request, jsonify, render_template
from cv_model import model
from PIL import Image, ImageDraw, ImageFont
import io, os, base64
import numpy as np

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
    box_data = results[0].cpu().boxes

    # Open original image
    img = Image.open(path).convert('RGB')
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", size=16)
    except:
        font = ImageFont.load_default()

    # Draw by boxes
    xyxy = box_data.xyxy.cpu().numpy()   # [[x1,y1,x2,y2], ...]
    confs = box_data.conf.cpu().numpy()  # [conf, ...]
    cls_ids = box_data.cls.cpu().numpy() # [class_id, ...]

    for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, cls_ids):
        # Drawing Rectangle and label
        draw.rectangle([x1, y1, x2, y2], outline="green", width=3)

        label = f"{id2name[int(cls)]}:{conf:.4f}"
        
        x_min, y_min, x_max, y_max = font.getbbox(label)
        text_width  = x_max - x_min
        text_height = y_max - y_min

        text_size = [text_width, text_height]
        draw.rectangle([x1, y1 - text_size[1], x1 + text_size[0], y1], fill="green")
        draw.text((x1, y1 - text_size[1]), label, fill="white", font=font)

    # Image to Base64
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    img_b64 = base64.b64encode(buf.getvalue()).decode('ascii')

    # JSON Serialize
    return jsonify({
        'predictions': [
            {'class_id': int(c), 
             'class_name': id2name[int(c)],
             'confidence': float(f),
             'bbox': [float(x1), float(y1), float(x2), float(y2)]
            }
            for (x1, y1, x2, y2), f, c in zip(xyxy, confs, cls_ids)
        ],
        'annotated_image': f"data:image/png;base64,{img_b64}"
    })


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8882)