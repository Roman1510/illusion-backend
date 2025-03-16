from flask import Flask, request, jsonify
from rembg import remove
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import io
import base64
import os

app = Flask(__name__)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

if not os.path.exists('temp'):
    os.makedirs('temp')

@app.route('/upload', methods=['POST'])
def upload_images():
    if 'front' not in request.files or 'side' not in request.files:
        return jsonify({'error': 'Missing files'}), 400

    front_file = request.files['front']
    side_file = request.files['side']
    front_img = cv2.imdecode(np.frombuffer(front_file.read(), np.uint8), cv2.IMREAD_COLOR)
    side_img = cv2.imdecode(np.frombuffer(side_file.read(), np.uint8), cv2.IMREAD_COLOR)

    front_img = cv2.resize(front_img, (300, 300))
    side_img = cv2.resize(side_img, (300, 300))

    front_no_bg = remove(front_img)
    side_no_bg = remove(side_img)

    cv2.imwrite('temp/front_no_bg.png', front_no_bg)
    cv2.imwrite('temp/side_no_bg.png', side_no_bg)

    front_rgb = cv2.cvtColor(front_no_bg, cv2.COLOR_BGR2RGB)
    side_rgb = cv2.cvtColor(side_no_bg, cv2.COLOR_BGR2RGB)

    front_results = face_mesh.process(front_rgb)
    side_results = face_mesh.process(side_rgb)

    front_points = []
    side_points = []
    if front_results.multi_face_landmarks:
        landmarks = front_results.multi_face_landmarks[0].landmark
        front_points = [
            (int(landmarks[0].x * 300), int(landmarks[0].y * 300)),
            (int(landmarks[1].x * 300), int(landmarks[1].y * 300)),
            (int(landmarks[13].x * 300), int(landmarks[13].y * 300))
        ]
    if side_results.multi_face_landmarks:
        landmarks = side_results.multi_face_landmarks[0].landmark
        side_points = [
            (int(landmarks[0].x * 300), int(landmarks[0].y * 300)),
            (int(landmarks[1].x * 300), int(landmarks[1].y * 300)),
            (int(landmarks[13].x * 300), int(landmarks[13].y * 300))
        ]

    _, front_encoded = cv2.imencode('.png', front_no_bg)
    _, side_encoded = cv2.imencode('.png', side_no_bg)
    front_base64 = base64.b64encode(front_encoded).decode('utf-8')
    side_base64 = base64.b64encode(side_encoded).decode('utf-8')

    return jsonify({
        'frontImage': front_base64,
        'sideImage': side_base64,
        'frontPoints': front_points,
        'sidePoints': side_points
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)