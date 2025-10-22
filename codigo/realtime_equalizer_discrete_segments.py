"""
🎵 Aplicación de Ecualización en Tiempo Real con Segmentos Discretos de 30s
===========================================================================

Esta aplicación utiliza inteligencia artificial para detectar automáticamente
el género musical en segmentos discretos de 30 segundos, guardando cada audio
y espectrograma generado.

Características:
- Segmentos discretos de 30 segundos (sin solapamiento)
- Guardado automático de audio y espectrograma
- Colormap magma con dimensiones 432x288
- Detección independiente por segmento

Autor: Sistema de IA Musical
Versión: 2.0 - Segmentos Discretos
"""

import os
import sys
import numpy as np
import tensorflow as tf
import sounddevice as sd
import librosa
import threading
import time
import queue
import json
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image
import io
import soundfile as sf
from datetime import datetime
import scipy.signal as signal

class DiscreteAudioSegmentProcessor:
    """Procesador de audio en segmentos discretos de 3 segundos"""

    def __init__(self, sample_rate=22050, segment_duration=3.0, hop_length=512, n_mels=128, device_id=None):
        self.sample_rate = sample_rate
        self.segment_duration = segment_duration
        self.segment_size = int(sample_rate * segment_duration)
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.device_id = device_id
        
        # Buffer para el segmento actual
        self.current_segment = []
        self.is_recording = False
        self.segment_complete = False
        self.latest_spectrogram = None
        self.latest_audio = None
        self.lock = threading.Lock()
        
        # Para guardar archivos
        self.output_dir = self.create_output_directory()
        self.segment_counter = 0
        
        # Obtener información del dispositivo de audio
        self.current_device_info = self.get_current_device_info()
        print(f"🎤 Dispositivo de audio: {self.current_device_info}")
        print(f"📁 Directorio de salida: {self.output_dir}")
    
    def create_output_directory(self):
        """Crea directorio para guardar los archivos"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"captured_segments_{timestamp}"
        
        # Crear directorios
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/audio", exist_ok=True)
        os.makedirs(f"{output_dir}/spectrograms", exist_ok=True)
        
        return output_dir
    
    @staticmethod
    def get_available_audio_devices():
        """Obtiene lista de dispositivos de audio disponibles"""
        try:
            devices = sd.query_devices()
            input_devices = []
            
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:  # Solo dispositivos de entrada
                    input_devices.append({
                        'id': i,
                        'name': device['name'],
                        'channels': device['max_input_channels'],
                        'sample_rate': device['default_samplerate'],
                        'is_default': i == sd.default.device[0]
                    })
            
            return input_devices
        except Exception as e:
            print(f"Error obteniendo dispositivos: {e}")
            return []
    
    def get_current_device_info(self):
        """Obtiene información del dispositivo actual"""
        try:
            if self.device_id is not None:
                device = sd.query_devices(self.device_id)
                return f"{device['name']} (ID: {self.device_id})"
            else:
                default_device = sd.query_devices(kind='input')
                return f"{default_device['name']} (Predeterminado)"
        except Exception as e:
            return f"Dispositivo desconocido (Error: {e})"
    
    def set_audio_device(self, device_id):
        """Cambia el dispositivo de audio"""
        if self.is_recording:
            print("⚠️ Detén la grabación antes de cambiar el dispositivo")
            return False
        
        try:
            # Verificar que el dispositivo existe
            device = sd.query_devices(device_id)
            if device['max_input_channels'] > 0:
                self.device_id = device_id
                self.current_device_info = self.get_current_device_info()
                print(f"✅ Dispositivo cambiado a: {self.current_device_info}")
                return True
            else:
                print(f"❌ El dispositivo {device_id} no soporta entrada de audio")
                return False
        except Exception as e:
            print(f"❌ Error cambiando dispositivo: {e}")
            return False

    def clear_audio_buffer(self):
        """Limpia completamente el buffer de audio"""
        with self.lock:
            self.current_segment = []
            self.segment_complete = False
            self.latest_spectrogram = None
            self.latest_audio = None
        print("🧹 Buffer de audio limpiado completamente")

    def audio_callback(self, indata, frames, time, status):
        """Callback para capturar audio en segmentos discretos"""
        if status:
            print(f"Audio status: {status}")
        
        # Obtener datos de audio
        audio_data = indata[:, 0] if indata.shape[1] > 1 else indata.flatten()
        
        with self.lock:
            # Agregar datos al segmento actual
            self.current_segment.extend(audio_data)
            
            # Verificar si el segmento está completo
            if len(self.current_segment) >= self.segment_size:
                # Tomar exactamente la cantidad de muestras necesarias
                segment_audio = np.array(self.current_segment[:self.segment_size])
                self.latest_audio = segment_audio
                self.segment_complete = True
                
                # Limpiar para el siguiente segmento
                self.current_segment = self.current_segment[self.segment_size:]
                
                print(f"📦 Segmento {self.segment_counter + 1} completo ({self.segment_duration}s)")

    def generate_mel_spectrogram_magma(self, audio_data):
        """Genera espectrograma mel con colormap magma y dimensiones 256x192"""
        try:
            # Verificar que tenemos suficientes datos
            if len(audio_data) < self.segment_size:
                print(f"⚠️ Audio insuficiente: {len(audio_data)} < {self.segment_size}")
                return None
                
            # Generar espectrograma mel
            mel_spec = librosa.feature.melspectrogram(
                y=audio_data, 
                sr=self.sample_rate,
                hop_length=self.hop_length,
                n_mels=self.n_mels
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            # Crear figura con dimensiones específicas para 256x192
            fig, ax = plt.subplots(figsize=(2.56, 1.92), dpi=100)
            ax.axis('off')
            
            # Usar colormap magma
            librosa.display.specshow(
                mel_spec_db, 
                sr=self.sample_rate, 
                hop_length=self.hop_length,
                x_axis='time', 
                y_axis='mel', 
                ax=ax, 
                cmap='magma'  # Colormap magma como solicitado
            )

            # Guardar como imagen con dimensiones exactas
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
            buf.seek(0)
            img = Image.open(buf).resize((256, 192))  # Dimensiones exactas 256x192
            plt.close(fig)
            buf.close()
            
            return img
        except Exception as e:
            print(f"Error generando espectrograma: {e}")
            return None

    def save_segment_files(self, audio_data, spectrogram_img, genre=None, confidence=None):
        """Guarda el audio y espectrograma del segmento"""
        try:
            timestamp = datetime.now().strftime("%H%M%S")
            segment_name = f"segment_{self.segment_counter:03d}_{timestamp}"
            
            # Información adicional para el nombre
            if genre and confidence:
                segment_name += f"_{genre}_{confidence:.0%}"
            
            # Guardar archivo de audio
            audio_filename = f"{self.output_dir}/audio/{segment_name}.wav"
            sf.write(audio_filename, audio_data, self.sample_rate)
            
            # Guardar espectrograma
            spectrogram_filename = f"{self.output_dir}/spectrograms/{segment_name}.png"
            spectrogram_img.save(spectrogram_filename)
            
            print(f"💾 Guardado: {segment_name}")
            print(f"   🎵 Audio: {audio_filename}")
            print(f"   📊 Espectrograma: {spectrogram_filename}")
            
            return audio_filename, spectrogram_filename
            
        except Exception as e:
            print(f"❌ Error guardando archivos: {e}")
            return None, None

    def get_completed_segment(self):
        """Obtiene el segmento completado si está disponible"""
        with self.lock:
            if self.segment_complete and self.latest_audio is not None:
                audio_copy = self.latest_audio.copy()
                self.segment_complete = False
                self.segment_counter += 1
                return audio_copy
            return None

    def get_segment_progress(self):
        """Obtiene el progreso del segmento actual"""
        with self.lock:
            progress = len(self.current_segment) / self.segment_size
            return min(progress, 1.0)

    def start_recording(self):
        """Inicia la grabación con buffer limpio"""
        self.clear_audio_buffer()  # Limpiar antes de empezar
        self.is_recording = True
        self.segment_counter = 0
        print(f"🎤 Iniciando grabación con: {self.current_device_info}")
        print(f"⏱️  Capturando segmentos discretos de {self.segment_duration} segundos")
        
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self.audio_callback,
            blocksize=1024,
            device=self.device_id
        )
        self.stream.start()
        print("✅ Grabación iniciada")

    def stop_recording(self):
        """Detiene la grabación y limpia el buffer"""
        self.is_recording = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        
        # Limpiar completamente el buffer
        self.clear_audio_buffer()
        print("🛑 Grabación detenida y buffer limpiado")

class RealTimeGenreClassifier:
    """Clasificador de género musical en tiempo real"""

    def __init__(self, model_path=None, config_path='model_config.json'):
        if model_path is None:
            model_path = self._find_best_model()

        self.model = tf.keras.models.load_model(model_path)

        with open(config_path, 'r') as f:
            config = json.load(f)

        self.genre_map = config['genre_map']
        self.inverse_genre_map = {int(k): v for k, v in config['inverse_genre_map'].items()}
        self.genres_list = config['genres_list']
        self.img_size = tuple(config['img_size'])

    def _find_best_model(self):
        candidates = ['music_genre_classifier.keras', 'music_genre_classifier.h5', 'music_genre_model_savedmodel']
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError("No se encontró ningún modelo guardado")

    def preprocess_pil_image(self, pil_image):
        img_array = np.array(pil_image)
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[-1] == 4:
            img_array = img_array[:, :, :3]

        if img_array.shape[:2] != self.img_size:
            pil_resized = pil_image.resize(self.img_size)
            img_array = np.array(pil_resized)

        img_array = img_array.astype(np.float32) / 255.0
        return np.expand_dims(img_array, axis=0)

    def predict_genre_discrete_segment(self, spectrogram_image):
        """Predice género para un segmento discreto (sin smoothing entre segmentos)"""
        try:
            processed_image = self.preprocess_pil_image(spectrogram_image)
            predictions = self.model.predict(processed_image, verbose=0)[0]

            predicted_index = np.argmax(predictions)
            predicted_genre = self.inverse_genre_map[predicted_index]
            confidence = float(predictions[predicted_index])

            probabilities = {
                self.inverse_genre_map[i]: float(predictions[i])
                for i in range(len(self.genres_list))
            }

            return predicted_genre, confidence, probabilities
        except Exception as e:
            print(f"Error en predicción: {e}")
            return "unknown", 0.0, {}

class AutoEqualizer:
    """Sistema de ecualización automática basado en género musical"""

    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        self.current_genre = None
        self.current_eq_settings = None

        self.eq_presets = {
            'blues': {'name': 'Blues Preset', 'gains': [2, 1, 0, 1, 2, 1, -1, 0, 1, 0]},
            'classical': {'name': 'Classical Preset', 'gains': [0, 0, 0, 1, 2, 3, 2, 1, 1, 1]},
            'jazz': {'name': 'Jazz Preset', 'gains': [1, 0, 1, 2, 2, 1, 0, 1, 1, 0]},
            'metal': {'name': 'Metal Preset', 'gains': [3, 2, 0, -1, 1, 2, 3, 2, 1, 1]},
            'reggae': {'name': 'Reggae Preset', 'gains': [3, 2, 1, 0, 0, 1, 0, 1, 0, 0]},
            'rock': {'name': 'Rock Preset', 'gains': [2, 1, 0, 1, 1, 2, 2, 1, 1, 0]}
        }

        self.eq_frequencies = [60, 170, 310, 600, 1000, 3000, 6000, 12000, 14000, 16000]

    def get_eq_settings_for_genre(self, genre):
        genre_lower = genre.lower()
        if genre_lower in self.eq_presets:
            return self.eq_presets[genre_lower]
        return {'name': 'Neutral', 'gains': [0] * 10}

class DiscreteSegmentEqualizerApp:
    """Aplicación principal de ecualización con segmentos discretos"""

    def __init__(self, root):
        self.root = root
        self.root.title("🎵 Ecualizador por Segmentos Discretos de 30s")
        self.root.geometry("1200x800")

        # Inicializar componentes
        try:
            self.audio_processor = DiscreteAudioSegmentProcessor()
            self.genre_classifier = RealTimeGenreClassifier()
            self.equalizer = AutoEqualizer()
        except Exception as e:
            messagebox.showerror("Error de Inicialización", 
                               f"Error cargando modelo: {e}\n\nAsegúrate de que los archivos del modelo estén en el directorio actual.")
            root.destroy()
            return

        self.is_processing = False
        self.processing_thread = None
        self.update_queue = queue.Queue()

        # Variables de la GUI
        self.current_genre_var = tk.StringVar(value="Esperando segmento...")
        self.confidence_var = tk.StringVar(value="0%")
        self.eq_preset_var = tk.StringVar(value="Ninguno")
        self.current_device_var = tk.StringVar(value="Cargando...")
        self.segment_progress_var = tk.StringVar(value="0%")
        self.segments_processed_var = tk.StringVar(value="0")
        self.output_dir_var = tk.StringVar(value=self.audio_processor.output_dir)
        
        # Obtener dispositivos de audio disponibles
        self.audio_devices = DiscreteAudioSegmentProcessor.get_available_audio_devices()
        self.update_device_info()

        self.create_gui()
        self.update_gui_periodically()
    
    def update_device_info(self):
        """Actualiza la información del dispositivo actual"""
        current_info = self.audio_processor.get_current_device_info()
        self.current_device_var.set(current_info)

    def create_gui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        # Control frame
        control_frame = ttk.LabelFrame(main_frame, text="🎤 Control de Segmentos Discretos", padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        # Fila 0: Botones de control
        self.start_button = ttk.Button(control_frame, text="▶️ Iniciar Segmentación", command=self.start_processing)
        self.start_button.grid(row=0, column=0, padx=(0, 10))

        self.stop_button = ttk.Button(control_frame, text="⏹️ Detener", command=self.stop_processing, state="disabled")
        self.stop_button.grid(row=0, column=1)
        
        # Botón para refrescar dispositivos
        refresh_button = ttk.Button(control_frame, text="🔄 Refrescar", command=self.refresh_audio_devices)
        refresh_button.grid(row=0, column=2, padx=(10, 0))
        
        # Botón de información de dispositivos
        info_button = ttk.Button(control_frame, text="ℹ️ Info", command=self.show_audio_devices_info)
        info_button.grid(row=0, column=3, padx=(5, 0))

        # Fila 1: Selección de dispositivo de audio
        ttk.Label(control_frame, text="🎤 Dispositivo:").grid(row=1, column=0, sticky=tk.W, pady=(10, 5))
        
        self.device_combo = ttk.Combobox(control_frame, width=50, state="readonly")
        self.device_combo.grid(row=1, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 5), padx=(10, 0))
        self.device_combo.bind('<<ComboboxSelected>>', self.on_device_selected)
        
        # Fila 2: Dispositivo actual
        ttk.Label(control_frame, text="📍 Actual:").grid(row=2, column=0, sticky=tk.W, pady=(0, 5))
        device_current_label = ttk.Label(control_frame, textvariable=self.current_device_var, foreground="blue")
        device_current_label.grid(row=2, column=1, columnspan=2, sticky=tk.W, pady=(0, 5), padx=(10, 0))
        
        # Configurar expansión de columnas
        control_frame.columnconfigure(1, weight=1)
        
        # Llenar el combo con dispositivos
        self.refresh_audio_devices()

        # Progress frame
        progress_frame = ttk.LabelFrame(main_frame, text="📊 Progreso de Segmentación", padding="10")
        progress_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(progress_frame, text="⏱️ Progreso del Segmento:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.progress_bar = ttk.Progressbar(progress_frame, length=300, mode='determinate')
        self.progress_bar.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 10))
        ttk.Label(progress_frame, textvariable=self.segment_progress_var).grid(row=0, column=2, sticky=tk.W)

        ttk.Label(progress_frame, text="📦 Segmentos Procesados:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Label(progress_frame, textvariable=self.segments_processed_var, font=("Arial", 10, "bold")).grid(row=1, column=1, sticky=tk.W, padx=(10, 0))

        ttk.Label(progress_frame, text="📁 Directorio de Salida:").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Label(progress_frame, textvariable=self.output_dir_var, foreground="green").grid(row=2, column=1, columnspan=2, sticky=tk.W, padx=(10, 0))

        progress_frame.columnconfigure(1, weight=1)

        # Info frame (izquierda)
        info_frame = ttk.LabelFrame(main_frame, text="📊 Última Detección", padding="10")
        info_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N), padx=(0, 10))

        ttk.Label(info_frame, text="🎵 Género:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Label(info_frame, textvariable=self.current_genre_var, font=("Arial", 12, "bold")).grid(row=0, column=1, sticky=tk.W, padx=(10, 0))

        ttk.Label(info_frame, text="📈 Confianza:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Label(info_frame, textvariable=self.confidence_var).grid(row=1, column=1, sticky=tk.W, padx=(10, 0))

        ttk.Label(info_frame, text="🎛️ Preset EQ:").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Label(info_frame, textvariable=self.eq_preset_var).grid(row=2, column=1, sticky=tk.W, padx=(10, 0))

        # Visualization frame (derecha)
        viz_frame = ttk.LabelFrame(main_frame, text="📊 Ecualizador", padding="10")
        viz_frame.grid(row=2, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        viz_frame.columnconfigure(0, weight=1)
        viz_frame.rowconfigure(0, weight=1)

        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.ax_equalizer = self.fig.add_subplot(1, 1, 1)
        self.ax_equalizer.set_title("Configuración del Ecualizador")
        self.ax_equalizer.set_xlabel("Frecuencia (Hz)")
        self.ax_equalizer.set_ylabel("Ganancia (dB)")

        # Info frame
        info_bottom_frame = ttk.LabelFrame(main_frame, text="ℹ️ Información del Sistema", padding="10")
        info_bottom_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))

        info_text = tk.Text(info_bottom_frame, height=3, state="disabled")
        info_text.grid(row=0, column=0, sticky=(tk.W, tk.E))

        info_text.config(state="normal")
        info_text.insert(tk.END, "🎯 Sistema de Segmentación Discreta:\n")
        info_text.insert(tk.END, "• Cada segmento es independiente (30 segundos exactos)\n")
        info_text.insert(tk.END, "• Espectrogramas con colormap 'magma' y dimensiones 432x288\n")
        info_text.insert(tk.END, "• Audio y espectrograma guardados automáticamente para cada segmento")
        info_text.config(state="disabled")

        info_bottom_frame.columnconfigure(0, weight=1)
    
    def refresh_audio_devices(self):
        """Refresca la lista de dispositivos de audio"""
        try:
            self.audio_devices = DiscreteAudioSegmentProcessor.get_available_audio_devices()
            
            # Limpiar y llenar el combobox
            device_names = []
            for device in self.audio_devices:
                name = f"{device['name']} (ID: {device['id']})"
                if device['is_default']:
                    name += " [PREDETERMINADO]"
                device_names.append(name)
            
            self.device_combo['values'] = device_names
            
            # Seleccionar el dispositivo actual
            if self.audio_devices:
                current_device_id = self.audio_processor.device_id
                if current_device_id is None:
                    # Buscar el dispositivo predeterminado
                    for i, device in enumerate(self.audio_devices):
                        if device['is_default']:
                            self.device_combo.current(i)
                            break
                else:
                    # Buscar el dispositivo por ID
                    for i, device in enumerate(self.audio_devices):
                        if device['id'] == current_device_id:
                            self.device_combo.current(i)
                            break
            
            print(f"🔄 Dispositivos de audio actualizados: {len(self.audio_devices)} encontrados")
            
        except Exception as e:
            print(f"❌ Error refrescando dispositivos: {e}")
            messagebox.showerror("Error", f"Error obteniendo dispositivos de audio: {e}")
    
    def on_device_selected(self, event=None):
        """Maneja la selección de un nuevo dispositivo de audio"""
        try:
            if self.is_processing:
                messagebox.showwarning("Advertencia", 
                                     "Detén el análisis antes de cambiar el dispositivo de audio")
                return
            
            selected_index = self.device_combo.current()
            if selected_index >= 0 and selected_index < len(self.audio_devices):
                selected_device = self.audio_devices[selected_index]
                device_id = selected_device['id']
                
                # Cambiar dispositivo
                if self.audio_processor.set_audio_device(device_id):
                    self.update_device_info()
                    messagebox.showinfo("Éxito", 
                                      f"Dispositivo cambiado a:\n{selected_device['name']}")
                else:
                    messagebox.showerror("Error", "No se pudo cambiar el dispositivo de audio")
                    # Revertir selección
                    self.refresh_audio_devices()
                    
        except Exception as e:
            print(f"❌ Error cambiando dispositivo: {e}")
            messagebox.showerror("Error", f"Error cambiando dispositivo: {e}")
    
    def show_audio_devices_info(self):
        """Muestra información detallada de los dispositivos de audio"""
        info_window = tk.Toplevel(self.root)
        info_window.title("🎤 Información de Dispositivos de Audio")
        info_window.geometry("600x400")
        
        # Texto con scroll
        text_frame = ttk.Frame(info_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Llenar con información
        text_widget.insert(tk.END, "🎤 DISPOSITIVOS DE AUDIO DISPONIBLES\n")
        text_widget.insert(tk.END, "=" * 50 + "\n\n")
        
        for i, device in enumerate(self.audio_devices):
            text_widget.insert(tk.END, f"📱 Dispositivo {i + 1}:\n")
            text_widget.insert(tk.END, f"   Nombre: {device['name']}\n")
            text_widget.insert(tk.END, f"   ID: {device['id']}\n")
            text_widget.insert(tk.END, f"   Canales: {device['channels']}\n")
            text_widget.insert(tk.END, f"   Frecuencia: {device['sample_rate']} Hz\n")
            text_widget.insert(tk.END, f"   Predeterminado: {'Sí' if device['is_default'] else 'No'}\n")
            text_widget.insert(tk.END, "\n")
        
        text_widget.config(state=tk.DISABLED)

    def start_processing(self):
        try:
            if not self.is_processing:
                self.is_processing = True
                self.start_button.config(state="disabled")
                self.stop_button.config(state="normal")
                print("🎵 Iniciando segmentación discreta de audio...")
                print("⏱️  Cada segmento durará exactamente 30 segundos")
                print(f"📁 Archivos se guardarán en: {self.audio_processor.output_dir}")
                self.audio_processor.start_recording()
                self.processing_thread = threading.Thread(target=self.processing_loop, daemon=True)
                self.processing_thread.start()
        except Exception as e:
            messagebox.showerror("Error", f"Error al iniciar: {e}")
            self.stop_processing()

    def stop_processing(self):
        self.is_processing = False
        self.audio_processor.stop_recording()
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.current_genre_var.set("Esperando segmento...")
        self.confidence_var.set("0%")
        self.eq_preset_var.set("Ninguno")
        self.progress_bar['value'] = 0
        self.segment_progress_var.set("0%")
        print("🛑 Segmentación detenida")

    def processing_loop(self):
        """Bucle principal de procesamiento de segmentos discretos"""
        while self.is_processing:
            try:
                # Obtener segmento completado
                audio_data = self.audio_processor.get_completed_segment()
                
                if audio_data is not None:
                    print(f"🔄 Procesando segmento {self.audio_processor.segment_counter}...")
                    
                    # Generar espectrograma con colormap magma
                    spectrogram_img = self.audio_processor.generate_mel_spectrogram_magma(audio_data)
                    
                    if spectrogram_img is not None:
                        # Clasificar género (sin smoothing entre segmentos)
                        genre, confidence, probabilities = self.genre_classifier.predict_genre_discrete_segment(spectrogram_img)
                        
                        # Obtener configuración de ecualizador
                        eq_settings = self.equalizer.get_eq_settings_for_genre(genre)
                        
                        # Guardar archivos del segmento
                        audio_file, spec_file = self.audio_processor.save_segment_files(
                            audio_data, spectrogram_img, genre, confidence
                        )
                        
                        # Preparar datos para actualizar GUI
                        update_data = {
                            'genre': genre,
                            'confidence': confidence,
                            'eq_settings': eq_settings,
                            'probabilities': probabilities,
                            'segment_number': self.audio_processor.segment_counter,
                            'audio_file': audio_file,
                            'spec_file': spec_file
                        }
                        
                        self.update_queue.put(update_data)
                        
                        print(f"✅ Segmento {self.audio_processor.segment_counter} procesado:")
                        print(f"   🎵 Género: {genre} ({confidence:.1%})")
                        print(f"   🎚️  EQ: {eq_settings['name']}")
                
                # Actualizar progreso del segmento actual
                progress = self.audio_processor.get_segment_progress()
                progress_data = {
                    'progress': progress,
                    'segments_count': self.audio_processor.segment_counter
                }
                self.update_queue.put(progress_data)
                
                # Pequeña pausa para no sobrecargar
                time.sleep(0.1)
                
            except Exception as e:
                print(f"❌ Error en procesamiento: {e}")
                time.sleep(1.0)

    def update_gui_periodically(self):
        """Actualiza la GUI periódicamente"""
        try:
            while True:
                update_data = self.update_queue.get_nowait()
                
                # Verificar si es actualización de progreso o de resultado
                if 'genre' in update_data:
                    # Actualización de resultado de segmento
                    self.current_genre_var.set(update_data['genre'].capitalize())
                    self.confidence_var.set(f"{update_data['confidence']:.1%}")
                    self.eq_preset_var.set(update_data['eq_settings']['name'])
                    self.segments_processed_var.set(str(update_data['segment_number']))
                    self.update_equalizer_visualization(update_data['eq_settings'])
                    
                elif 'progress' in update_data:
                    # Actualización de progreso
                    progress_percent = update_data['progress'] * 100
                    self.progress_bar['value'] = progress_percent
                    self.segment_progress_var.set(f"{progress_percent:.1f}%")
                    self.segments_processed_var.set(str(update_data['segments_count']))
                
        except queue.Empty:
            pass
        except Exception as e:
            print(f"❌ Error actualizando GUI: {e}")

        self.root.after(50, self.update_gui_periodically)  # Actualización más frecuente

    def update_equalizer_visualization(self, eq_settings):
        """Actualiza la visualización del ecualizador"""
        try:
            self.ax_equalizer.clear()
            frequencies = self.equalizer.eq_frequencies
            gains = eq_settings['gains']

            bars = self.ax_equalizer.bar(range(len(frequencies)), gains, 
                                       color=['red' if g < 0 else 'green' if g > 0 else 'gray' for g in gains])

            self.ax_equalizer.set_xticks(range(len(frequencies)))
            self.ax_equalizer.set_xticklabels([f"{f//1000}K" if f >= 1000 else f"{f}" for f in frequencies], rotation=45)
            self.ax_equalizer.set_ylabel("Ganancia (dB)")
            self.ax_equalizer.set_title(f"EQ: {eq_settings['name']}")
            self.ax_equalizer.grid(True, alpha=0.3)
            self.ax_equalizer.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            self.canvas.draw()
        except Exception as e:
            print(f"❌ Error actualizando visualización: {e}")

def main():
    """Función principal"""
    print("🎵 Iniciando Aplicación de Segmentación Discreta")
    print("=" * 60)
    
    # Obtener el directorio donde está el script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)  # Cambiar al directorio del script
    
    print(f"📁 Directorio de trabajo: {script_dir}")
    
    # Verificar archivos necesarios
    required_files = ['model_config.json']
    model_files = ['music_genre_classifier.keras', 'music_genre_classifier.h5', 'music_genre_model_savedmodel']
    
    print("🔍 Verificando archivos del modelo...")
    found_models = []
    for model_file in model_files:
        if os.path.exists(model_file):
            found_models.append(model_file)
            print(f"   ✅ {model_file}")
        else:
            print(f"   ❌ {model_file}")
    
    if not found_models:
        print("❌ Error: No se encontró ningún archivo de modelo.")
        print("   Archivos buscados:", model_files)
        print("   Asegúrate de haber entrenado y guardado el modelo primero.")
        return
    
    if not os.path.exists('model_config.json'):
        print("❌ Error: No se encontró 'model_config.json'")
        return
    else:
        print("   ✅ model_config.json")
    
    try:
        root = tk.Tk()
        app = DiscreteSegmentEqualizerApp(root)

        def on_closing():
            if app.is_processing:
                app.stop_processing()
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", on_closing)
        print("✅ Aplicación iniciada exitosamente")
        print()
        print("🎯 CARACTERÍSTICAS PRINCIPALES:")
        print("   📦 Segmentos discretos de 30 segundos (sin solapamiento)")
        print("   💾 Guardado automático de audio + espectrograma")
        print("   🌈 Colormap 'magma' con dimensiones 432x288")
        print("   🎵 Detección independiente por segmento")
        print()
        
        # Mostrar dispositivos de audio disponibles
        print("🎤 Dispositivos de audio disponibles:")
        print("-" * 50)
        devices = DiscreteAudioSegmentProcessor.get_available_audio_devices()
        for i, device in enumerate(devices):
            default_mark = " [PREDETERMINADO]" if device['is_default'] else ""
            print(f"  {i+1}. {device['name']} (ID: {device['id']}){default_mark}")
            print(f"     📊 Canales: {device['channels']}, Frecuencia: {device['sample_rate']} Hz")
        
        if not devices:
            print("  ❌ No se encontraron dispositivos de entrada de audio")
        print("-" * 50)
        
        root.mainloop()

    except Exception as e:
        print(f"❌ Error iniciando aplicación: {e}")

if __name__ == "__main__":
    main()