from flask import Flask, request, jsonify
import cv2
import face_recognition
import os
import csv
from datetime import datetime

app = Flask(__name__)

def construct_path(*args):
    return os.path.join(os.path.dirname(__file__), *args)

def load_face_encodings_from_directory(directory_path):
    encodings = []
    names = []
    for filename in os.listdir(directory_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory_path, filename)
            name = os.path.splitext(filename)[0]
            image = face_recognition.load_image_file(image_path)
            face_encoding = face_recognition.face_encodings(image)
            if face_encoding:
                encodings.append(face_encoding[0])
                names.append(name)
    return encodings, names

images_directory = construct_path('images')
known_face_encodings, known_face_names = load_face_encodings_from_directory(images_directory)
csv_file_path = construct_path('attendance.csv')

if not os.path.exists(csv_file_path):
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Name', 'Time'])

@app.route('/recognize', methods=['POST'])
def recognize_faces():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    image_path = 'temp_image.jpg'
    image_file.save(image_path)

    recognized_name = None

    try:
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)

        if not face_locations:
            return jsonify({"error": "No faces detected"}), 404

        face_encodings = face_recognition.face_encodings(image, face_locations)
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
            if True in matches:
                first_match_index = matches.index(True)
                recognized_name = known_face_names[first_match_index]
                break

        if recognized_name:
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open(csv_file_path, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([recognized_name, current_time])
            return jsonify({"name": recognized_name}), 200
        else:
            return jsonify({"error": "Face not matched"}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        os.remove(image_path)

if __name__ == '__main__':
    app.run(debug=True)
