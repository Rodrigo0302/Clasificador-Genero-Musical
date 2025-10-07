
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import json
import base64
from PIL import Image
import io
import os

app = Flask(__name__)

# Cargar el modelo al iniciar la aplicación
model = None
config = None

def load_model():
    global model, config
    model = tf.keras.models.load_model('music_genre_classifier.h5')

    with open('model_config.json', 'r') as f:
        config = json.load(f)

def preprocess_image(image_data):
    """Preprocesa una imagen para la predicción"""
    # Si es base64, decodificar
    if isinstance(image_data, str):
        image_data = base64.b64decode(image_data)

    # Abrir imagen con PIL
    image = Image.open(io.BytesIO(image_data))

    # Convertir a RGB si es necesario
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Redimensionar
    image = image.resize((432, 288))  # width, height

    # Convertir a array numpy y normalizar
    image_array = np.array(image) / 255.0

    # Expandir dimensiones para batch
    image_array = np.expand_dims(image_array, axis=0)

    return image_array

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener la imagen del request
        if 'image' not in request.files:
            return jsonify({'error': 'No se proporcionó ninguna imagen'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No se seleccionó ningún archivo'}), 400

        # Leer y preprocesar la imagen
        image_data = file.read()
        processed_image = preprocess_image(image_data)

        # Hacer predicción
        predictions = model.predict(processed_image, verbose=0)

        # Obtener resultados
        predicted_index = np.argmax(predictions)
        inverse_genre_map = {int(k): v for k, v in config['inverse_genre_map'].items()}
        predicted_genre = inverse_genre_map[predicted_index]

        # Calcular probabilidades para todos los géneros
        probabilities = {
            inverse_genre_map[i]: float(predictions[0][i])
            for i in range(len(config['genres_list']))
        }

        # Top 3 predicciones
        top_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3 = [
            {
                'genre': inverse_genre_map[idx],
                'probability': float(predictions[0][idx])
            }
            for idx in top_indices
        ]

        response = {
            'predicted_genre': predicted_genre,
            'confidence': float(predictions[0][predicted_index]),
            'top_predictions': top_3,
            'all_probabilities': probabilities
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'API de Clasificación de Géneros Musicales',
        'endpoints': {
            '/predict': 'POST - Subir imagen para clasificar',
            '/health': 'GET - Estado del servicio'
        }
    })

if __name__ == '__main__':
    print("Cargando modelo...")
    load_model()
    print("Modelo cargado exitosamente!")

    print("Iniciando servidor Flask...")
    app.run(debug=True, host='0.0.0.0', port=5000)
