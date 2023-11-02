from boto3 import Session
from botocore.exceptions import BotoCoreError, ClientError
from rgbmatrix import RGBMatrix, RGBMatrixOptions
from PIL import Image
from pydub import AudioSegment
from pydub.playback import play
import openai
import json
import time
import threading
import io
import pyaudio
import numpy as np
import wave
import os

openai.api_key = os.environ["OPENAI_API_KEY"]
messages = [
        {"role": "system", "content": "Your name is Kevin, and you are a conversational chatbot that a companion for a lonely person. If innaprporiate language is used, do not mention it. Respond accordingly and briefly as if you were in a casual dinner seting. Don't say numbered lists and avoid making responses longer than 3-4 sentences."}
]

face1 = 'faces/face1.jpg'
face2 = 'faces/face2.jpg'
face3 = 'faces/face3.jpg'
face4 = 'faces/face4.jpg'
face5 = 'faces/face5.jpg'
face6 = 'faces/face6.jpg'

mappings = {
        'k': face1,
        'E': face1,
        'i': face2,
        '@': face2,
        'a': face2,
        'e': face2,
        'O': face2,
        't': face3,
        's': face3,
        'S': face3,
        'T': face3,
        'f': face4,
        'sil': face5,
        'p': face5,
        'r': face6,
        'u': face6,
        'o': face6,
}

options = RGBMatrixOptions()
options.rows = 32
options.chain_length = 1
options.parallel = 1
options.hardware_mapping = 'adafruit-hat'

matrix = RGBMatrix(options = options)

session = Session(profile_name="default")
polly = session.client("polly")

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
THRESHOLD = 100 
SILENCE_TIMEOUT = 2


def gpt_response(prompt):
        messages.append({"role": "user", "content": prompt})
        response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages
        )
        content = response['choices'][0]['message']['content']
        messages.append({"role": "assistant", "content": content})
        return content



def tts_lipsync(text):
        audio_response = polly.synthesize_speech(
                Engine="neural",
                Text=text, 
                OutputFormat="mp3", 
                VoiceId="Stephen")

        visemes_response = polly.synthesize_speech(
                Engine="neural",
                Text=text, 
                OutputFormat="json", 
                SpeechMarkTypes=["viseme"],
                VoiceId="Stephen")

        audio = audio_response["AudioStream"].read()
        visemes = [json.loads(v) for v in visemes_response["AudioStream"].read().decode().split() if v]

        audio_thread = threading.Thread(target=lambda: play(AudioSegment.from_mp3(io.BytesIO(audio))))
        audio_thread.start()

        time.sleep(1)
        start_time = time.time() * 1000

        for viseme in visemes:
                while time.time() * 1000 - start_time < viseme["time"]:
                        time.sleep(1/1000)
                matrix.SetImage(Image.open(mappings.get(viseme["value"])).rotate(-90))
        
        matrix.SetImage(Image.open(face5).rotate(-90))
        time.sleep(1)


def main():
        response = gpt_response("Address the user as Ryan. Make an upbeat conversation starter question.")
        print(response)
        tts_lipsync(response)

        audio = pyaudio.PyAudio()
        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

        recording = False
        recording_data = []

        silence_timer = None

        print("Listening...")

        while True:
                data = stream.read(CHUNK)
                audio_signal = np.frombuffer(data, dtype=np.int16)

                if max(audio_signal) > THRESHOLD and not recording:
                        recording = True
                        silence_timer = None
                        print("Recording...")

                if recording:
                        recording_data.append(data)

                        if max(audio_signal) <= THRESHOLD:
                                if silence_timer is None:
                                        silence_timer = time.time()
                                elif time.time() - silence_timer >= SILENCE_TIMEOUT:
                                        print("Silence detected, stopping recording")

                                        stream.stop_stream()
                                        stream.close()
                                        
                                        with wave.open('recording.wav', "wb") as wf:
                                                wf.setnchannels(CHANNELS)
                                                wf.setsampwidth(audio.get_sample_size(FORMAT))
                                                wf.setframerate(RATE)
                                                wf.writeframes(b"".join(recording_data))

                                        transcript = openai.Audio.transcribe("whisper-1", open('recording.wav', 'rb'))["text"]           
                                        print("Transcription:", transcript)
                                        os.remove('recording.wav')
                                        
                                        tts_lipsync(gpt_response(transcript))

                                        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
                                        recording = False
                                        recording_data = []
                                        print("Listening...")
                        else:
                                silence_timer = None
                        


if __name__ == '__main__':
        main()
