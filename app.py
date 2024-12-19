from flask import Flask, render_template, request, redirect, url_for, jsonify
from ultralytics import YOLO
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import io
import os
from werkzeug.utils import secure_filename

def draw_bounding_boxes(image, detections, class_names):
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection['bbox'])
        class_id = detection['class_id']
        class_name = class_names[class_id]
        confidence = detection['confidence']

        color = (0, 255, 0)
        thickness = 2
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        label = f"{class_name} {confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        cv2.putText(image, label, (x1, y1 - 10), font, font_scale, color, font_thickness)

    return image

app = Flask(__name__)

game_year_model = tf.keras.models.load_model('models/game_release_year_model.keras')
minecraft_mob_model = YOLO('models/minecraft_mob_model.pt')


@app.route('/')
def home():
    return redirect(url_for('classification'))

@app.route("/classification", methods=['POST', 'GET'])
def classification():
    if request.method == 'GET':
        return render_template('classification.html', title="Определение года выхода игры")
    if request.method == 'POST':
        class_names = [str(year) for year in range(1986, 2001)]
        file = request.files.get('file')
        if file:
            image = Image.open(io.BytesIO(file.read()))
            image = image.resize((150, 150)).convert('RGB')
            image_array = np.array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)

            predictions = game_year_model.predict(image_array)
            predicted_class = np.argmax(predictions[0])
            predicted_label = class_names[predicted_class]

            return render_template('classification.html', title="Определение года выхода игры",
                                   class_model=f"{predicted_label} год.")


UPLOAD_FOLDER = 'static/images/processed_images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/detection', methods=['GET', 'POST'])
def detection():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        image = cv2.imread(file_path)

        result = minecraft_mob_model(image)[0]
        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_id = box.cls.item()
            confidence = box.conf.item()
            detections.append({
                'class_id': int(class_id),
                'class_name': result.names[class_id],
                'confidence': float(confidence),
                'bbox': [float(x1), float(y1), float(x2), float(y2)]
            })

        image_with_boxes = draw_bounding_boxes(image, detections, result.names)
        output_filename = f"processed_{filename}"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        cv2.imwrite(output_path, image_with_boxes)

        return render_template('detection.html', image=output_filename, detections=detections)

    return render_template('detection.html')

@app.route('/api/detect', methods=['POST'])
def api_detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    result = minecraft_mob_model(image)[0]
    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        class_id = box.cls.item()
        confidence = box.conf.item()
        detections.append({
            'class_id': int(class_id),
            'class_name': result.names[class_id],
            'confidence': float(confidence),
            'bbox': [float(x1), float(y1), float(x2), float(y2)]
        })
    return jsonify({'detections': detections})

@app.route('/api_docs')
def api_docs():
    return render_template('api_docs.html')

if __name__ == '__main__':
    app.run(debug=True)