
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import sys
import os

class MusicGenreClassifier:
    def __init__(self, model_path='music_genre_classifier.h5', config_path='model_config.json'):
        """
        Inicializa el clasificador de géneros musicales
        """
        self.model = tf.keras.models.load_model(model_path)

        with open(config_path, 'r') as f:
            config = json.load(f)

        self.genre_map = config['genre_map']
        self.inverse_genre_map = {int(k): v for k, v in config['inverse_genre_map'].items()}
        self.input_shape = config['input_shape']
        self.genres_list = config['genres_list']
        self.img_size = tuple(config['img_size'])

    def preprocess_image(self, image_path):
        """Preprocesa una imagen de espectrograma para la predicción"""
        image = tf.image.decode_png(tf.io.read_file(image_path), channels=3)
        image = tf.image.resize(image, self.img_size)
        image = tf.cast(image, tf.float32) / 255.0
        image = np.expand_dims(image, axis=0)
        return image

    def predict_genre(self, image_path, return_probabilities=False):
        """Predice el género musical de un espectrograma"""
        processed_image = self.preprocess_image(image_path)
        predictions = self.model.predict(processed_image, verbose=0)

        predicted_index = np.argmax(predictions)
        predicted_genre = self.inverse_genre_map[predicted_index]

        if return_probabilities:
            probabilities = {
                self.inverse_genre_map[i]: float(predictions[0][i]) 
                for i in range(len(self.genres_list))
            }
            return predicted_genre, probabilities

        return predicted_genre

    def get_top_predictions(self, image_path, top_k=3):
        """Obtiene las top-k predicciones más probables"""
        processed_image = self.preprocess_image(image_path)
        predictions = self.model.predict(processed_image, verbose=0)[0]

        top_indices = np.argsort(predictions)[-top_k:][::-1]

        results = [
            (self.inverse_genre_map[idx], float(predictions[idx]))
            for idx in top_indices
        ]

        return results

def main():
    """Función principal para usar desde línea de comandos"""
    if len(sys.argv) != 2:
        print("Uso: python music_classifier_app.py <ruta_imagen_espectrograma>")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"Error: No se encontró el archivo {image_path}")
        sys.exit(1)

    # Inicializar el clasificador
    classifier = MusicGenreClassifier()

    # Hacer predicción
    predicted_genre, probabilities = classifier.predict_genre(image_path, return_probabilities=True)
    top_3 = classifier.get_top_predictions(image_path, top_k=3)

    print(f"Imagen analizada: {image_path}")
    print(f"Género predicho: {predicted_genre}")
    print(f"Confianza: {probabilities[predicted_genre]:.2%}")
    print("\nTop 3 predicciones:")
    for i, (genre, prob) in enumerate(top_3, 1):
        print(f"{i}. {genre}: {prob:.2%}")

if __name__ == "__main__":
    main()
