import flet as ft
#from flet import Icons
import sounddevice as sd
import numpy as np
import speech_recognition as sr
from scipy.io.wavfile import write
from pyannote.audio import Inference, Model
from scipy.spatial.distance import pdist
import os
from threading import Thread
from pydub import AudioSegment
import soundfile as sf
import random

class SpeakerIdentification(ft.Control):
    def __init__(self, page: ft.Page):
        super().__init__()
        self.page = page
        self.status = False
        self.samplerate = 16000
        self.audio_data = np.array([], dtype=np.float32)
        self.device = None
        self.channels = 1
        self.recognizer = sr.Recognizer()

        # Reference folders for speaker identification
        self.reference_speaker_folders = {
            "Kaviya": r"C:\Users\mukes\Desktop\Biometrics\voice identification\voice identification\dataset\kaviya",
            #"Vamsi": r"C:\Users\mukes\Desktop\Biometrics\voice identification\voice identification\dataset\vamsi",
            #"Lavanya": r"C:\Users\mukes\Desktop\Biometrics\voice identification\voice identification\dataset\lavanya",
            "Mukesh": r"C:\Users\mukes\Desktop\Biometrics\voice identification\voice identification\dataset\Mukesh",
            "Bharath": r"C:\Users\mukes\Desktop\Biometrics\voice identification\voice identification\dataset\Bharath",

        }

    def build(self):
        # UI elements
        self.status_text = ft.Text(value='', font_family="Poppins", size=15, weight='bold', color='Red')
        self.progress_indicator = ft.ProgressRing(width=25, height=25, stroke_width=2)

        # Record button
        self.record_button = ft.IconButton(
            icon=ft.Icons.MIC_ROUNDED,
            selected_icon=ft.Icons.MIC_OFF_ROUNDED,
            selected=False,
            on_click=self.toggle_recording,
        )

        # Play button to playback recorded audio
        self.play_button = ft.IconButton(
            icon=ft.Icons.PLAY_CIRCLE_OUTLINED,
            selected_icon=ft.Icons.STOP_CIRCLE_OUTLINED,
            on_click=self.toggle_playback,
            disabled=True,
        )

        # Button for speaker identification
        self.identify_button = ft.TextButton(
            text="Identify Speaker",
            on_click=self.identify_speaker,
            style=ft.ButtonStyle(
                color="white",
                bgcolor="blue",
                padding=ft.padding.all(10),
            ),
        )

        # UI layout
        return ft.Column(
            controls=[
                self.record_button,
                self.play_button,
                self.identify_button,
                self.status_text
            ]
        )

    def generate_liveness_numbers(self):
        """Generate a random 3-digit number for liveness detection."""
        return str(random.randint(100, 999))

    def toggle_recording(self, e):
        if not e.control.selected:
            # Generate random number phrase for liveness detection
            self.liveness_phrase = self.generate_liveness_numbers()

            # Display liveness phrase before starting recording
            self.status_text.value = f"Say the number: {self.liveness_phrase}"
            self.page.update()

            # Start recording
            self.audio_data = np.array([], dtype=np.float32)
            self.status = True
            e.control.selected = True
            self.play_button.disabled = True
            self.page.update()

            # Start recording in a separate thread
            self.record_thread = Thread(target=self.record_audio)
            self.record_thread.start()
        else:
            # Stop recording
            self.status = False
            e.control.selected = False
            self.status_text.value = "Recording stopped."
            self.play_button.disabled = False
            self.page.update()

    def toggle_playback(self, e):
        if not e.control.selected:
            # Play recorded audio
            e.control.selected = True
            sd.play(self.audio_data, self.samplerate)
            sd.wait()
            e.control.selected = False

    def record_audio(self):
        try:
            with sd.InputStream(samplerate=self.samplerate, channels=self.channels, callback=self.audio_callback):
                while self.status:
                    sd.sleep(100)
        except Exception as ex:
            self.status_text.value = f"Error: {str(ex)}"
            self.page.update()

    def audio_callback(self, indata, frames, time, status):
        # Append audio chunks
        self.audio_data = np.append(self.audio_data, indata[:, 0])

    def save_audio_to_wav(self, file_path, data, samplerate):
        """Save audio data to a WAV file using soundfile."""
        try:
            sf.write(file_path, data, samplerate, subtype='PCM_16')
        except Exception as e:
            print(f"Error saving audio: {e}")

    def identify_speaker(self, e):
        if len(self.audio_data) == 0:
            self.status_text.value = "No audio recorded."
            self.page.update()
            return

        # Update status text to "Identifying..."
        self.status_text.value = "Identifying..."
        self.page.update()

        # Save recorded audio to a file
        test_audio_path = "test_audio.wav"
        self.save_audio_to_wav(test_audio_path, self.audio_data, self.samplerate)

        # Convert reference MP3 files to WAV for each speaker
        wav_paths = {}
        for speaker, folder_path in self.reference_speaker_folders.items():
            mp3_files = self.get_audio_files(folder_path)
            wav_paths[speaker] = [self.convert_mp3_to_wav(mp3) for mp3 in mp3_files]

        # Verify liveness before proceeding with speaker identification
        if not self.verify_liveness(test_audio_path):
            self.status_text.value = "Liveness check failed. Access Denied."
            self.page.update()
            return

        # Load the pyannote.audio model for speaker embedding
        try:
            model = Model.from_pretrained(
                "pyannote/embedding", use_auth_token="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"
            )
            inference = Inference(model, window="whole")
        except Exception as ex:
            self.status_text.value = f"Model loading error: {str(ex)}"
            self.page.update()
            return

        # Perform speaker identification
        try:
            # Compute embedding of the test audio
            test_embedding = inference(test_audio_path)

            # Compare against each reference speaker
            min_distance = float("inf")
            closest_speaker = None

            for speaker, ref_audio_wav_list in wav_paths.items():
                for ref_audio_wav in ref_audio_wav_list:
                    ref_embedding = inference(ref_audio_wav)
                    distance = pdist([test_embedding, ref_embedding], metric="cosine")[0]

                    if distance < min_distance:
                        min_distance = distance
                        closest_speaker = speaker

            # Check if distance is below a threshold for identification
            if min_distance < 0.6:
                self.status_text.value = f"Access Granted for {closest_speaker}"
            else:
                self.status_text.value = "Access Denied"

            os.remove(test_audio_path)  # Cleanup test audio file
        except Exception as ex:
            self.status_text.value = f"Identification error: {str(ex)}"

        self.page.update()

    def get_audio_files(self, folder_path):
        """Retrieve all MP3 files from the specified folder."""
        return [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".wav")]

    def convert_mp3_to_wav(self, mp3_path):
        wav_path = mp3_path.replace(".mp3", ".wav")
        if not os.path.exists(wav_path):
            try:
                sound = AudioSegment.from_mp3(mp3_path)
                sound = sound.set_channels(1)  # Convert to mono
                sound = sound.set_frame_rate(self.samplerate)  # Convert to 16kHz
                sound.export(wav_path, format="wav")
                print(f"Converted {mp3_path} to {wav_path}")
            except Exception as e:
                print(f"Error converting {mp3_path} to WAV: {e}")
        return wav_path

    def verify_liveness(self, audio_path):
        """Verify if the user correctly repeated the liveness phrase."""
        try:
            with sr.AudioFile(audio_path) as source:
                audio_data = self.recognizer.record(source)

            # Use speech recognition to transcribe the audio
            transcription = self.recognizer.recognize_google(audio_data)
            print(f"Transcription: {transcription}")
            # Check if the transcription matches the liveness number
            return self.liveness_phrase in transcription.replace(" ", "")
        except sr.UnknownValueError:
            return False
        except Exception as e:
            print(f"Error during liveness verification: {e}")
            return False

def main(page: ft.Page):
    page.title = "Speaker Identification"
    page.window_width = 500  # Set window width
    page.window_height = 250  # Set window height
    app = SpeakerIdentification(page)
    page.add(app.build())
    page.update()

# Run the Flet app
ft.app(target=main)
