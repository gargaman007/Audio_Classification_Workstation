import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from acrcloud.recognizer import ACRCloudRecognizer
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydub import AudioSegment
from tensorflow.keras.models import load_model

app = FastAPI()

templates = Jinja2Templates(directory=".")
model = load_model('./models/neural_networks.h5')
# ACRCloud Configuration using SDK
ACRCLOUD_CONFIG = {
    'host': 'identify-ap-southeast-1.acrcloud.com',
    'access_key': '',
    'access_secret': '',
    'timeout': 1  # seconds
}
acr_recognizer = ACRCloudRecognizer(ACRCLOUD_CONFIG) 

# Load YAMNet model and labels
yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)

with open("yamnet_class_map.csv", "r") as f:
    yamnet_classes = [line.strip().split(",")[2] for line in f.readlines()[1:]]

# Set up ffmpeg path
FFMPEG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffmpeg-master-latest-win64-gpl", "bin")
if os.path.exists(FFMPEG_PATH):
    os.environ["PATH"] = FFMPEG_PATH + os.pathsep + os.environ["PATH"]
    AudioSegment.converter = os.path.join(FFMPEG_PATH, "ffmpeg.exe")
    AudioSegment.ffmpeg = os.path.join(FFMPEG_PATH, "ffmpeg.exe")
    AudioSegment.ffprobe = os.path.join(FFMPEG_PATH, "ffprobe.exe")
    
def extract_features(audio_path, max_length=100):
    y, sr = librosa.load(audio_path, sr=None)
    y_normalized = librosa.util.normalize(y)
    segments = librosa.effects.split(y_normalized, top_db=20)

    mfccs = []
    for start, end in segments:
        segment = y[start:end]
        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
        if mfcc.shape[1] > max_length:
            mfcc = mfcc[:, :max_length]
        else:
            pad_width = max_length - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        mfccs.append(mfcc)
    
    return mfccs

def predict_vehicle_class(audio_path):
    features = extract_features(audio_path)
    
    # Normalize using training distribution (consider saving stats during training if accuracy matters)
    features = np.array(features)
    features = (features - np.mean(features)) / np.std(features)

    # Average predictions across all segments
    predictions = model.predict(features)
    averaged_prediction = np.mean(predictions, axis=0)
    predicted_class = int(np.argmax(averaged_prediction))  # Convert numpy.int64 to Python int

    return predicted_class

def convert_audio_to_wav(src_path: str, dst_path: str) -> bool:
    """Convert any audio file to WAV format using pydub."""
    try:
        # Get the file extension
        ext = os.path.splitext(src_path)[1].lower().lstrip('.')
        
        # Load the audio file with specific parameters
        audio = AudioSegment.from_file(
            src_path,
            format=ext,
            parameters=["-ar", "16000", "-ac", "1"]  # Set sample rate to 16kHz and mono
        )
        
        # Export as WAV with specific parameters
        audio.export(
            dst_path,
            format="wav",
            parameters=["-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le"]
        )
        return True
    except Exception as e:
        logging.error(f"Error converting audio file: {str(e)}")
        return False

def classify_audio_with_yamnet(file_path):
    try:
        # Create a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            temp_wav_path = temp_wav.name

        # Convert the input file to WAV if needed
        if not convert_audio_to_wav(file_path, temp_wav_path):
            return {
                "success": False,
                "message": "Failed to convert audio file to WAV format"
            }

        try:
            # Load and process the audio
            waveform, sr = librosa.load(temp_wav_path, sr=16000)  # YAMNet expects 16kHz
            scores, embeddings, spectrogram = yamnet_model(waveform)
            scores_np = scores.numpy().mean(axis=0)  # average over time

            top5_i = np.argsort(scores_np)[::-1][:5]
            top_labels = [(yamnet_classes[i], float(scores_np[i])) for i in top5_i]  # Convert scores to Python float

            return {
                "success": True,
                "top_classes": top_labels
            }
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_wav_path):
                os.unlink(temp_wav_path)

    except Exception as e:
        logging.exception("YAMNet classification failed:")
        return {
            "success": False,
            "message": f"Audio classification failed: {str(e)}"
        }

def is_vehicle_sound(yamnet_classes):
    """
    Check if any of the top YAMNet classifications are vehicle-related.
    Returns True if a vehicle sound is detected, along with the matched class and score.
    """
    vehicle_keywords = [
        # General vehicle terms
        'vehicle', 'automobile', 'motor vehicle',
        # Specific vehicle types
        'car', 'truck', 'bus', 'van', 'motorcycle', 'scooter',
        # Vehicle components
        'engine', 'motor', 'horn', 'siren', 'tire', 'wheel',
        # Vehicle sounds
        'revving', 'acceleration', 'braking', 'idling',
        # Transportation
        'transport', 'traffic', 'road'
    ]
    
    # Log the top classifications for debugging
    logging.info("Top YAMNet classifications:")
    for class_name, score in yamnet_classes:
        logging.info(f"- {class_name}: {score:.2f}")
    
    # Check each classification against vehicle keywords
    for class_name, score in yamnet_classes:
        class_name_lower = class_name.lower()
        for keyword in vehicle_keywords:
            if keyword in class_name_lower:
                logging.info(f"Vehicle sound detected: '{class_name}' (score: {score:.2f})")
                return True, class_name, score
    
    logging.info("No vehicle sounds detected in the audio")
    return False, None, 0.0

@app.post("/classify/")
async def classify_audio(file: UploadFile = File(...)):
    temp_filename = f"temp_classify_{file.filename}"
    file_content = await file.read()

    try:
        with open(temp_filename, "wb") as f:
            f.write(file_content)

        # First try music recognition
        result_json_str = acr_recognizer.recognize_by_file(temp_filename, 0)
        music_result = format_acrcloud_response(result_json_str)

        if music_result["success"]:
            # If music recognition was successful, return that result
            return {
                "success": True,
                "type": "music",
                "music_result": music_result
            }
        else:
            # If music recognition failed, try YAMNet classification
            yamnet_result = classify_audio_with_yamnet(temp_filename)
            if yamnet_result["success"]:
                # Check if the sound is vehicle-related
                is_vehicle, vehicle_class, vehicle_score = is_vehicle_sound(yamnet_result["top_classes"])
                if is_vehicle:
                    # If it's a vehicle sound, use the neural network for specific classification
                    vehicle_class = predict_vehicle_class(temp_filename)
                    vehicle_type = "Car" if vehicle_class == 0 else "Truck"
                    
                    return {
                        "success": True,
                        "type": "vehicle",
                        "vehicle_result": {
                            "vehicle_type": vehicle_type,
                            "detected_sound": vehicle_class,
                            "confidence": float(vehicle_score) * 100
                        }
                    }
                
                # If not a vehicle sound, return YAMNet classification
                return {
                    "success": True,
                    "type": "sound",
                    "sound_result": yamnet_result
                }
            else:
                return {
                    "success": False,
                    "message": "No music, vehicle, or sound patterns recognized."
                }

    except Exception as e:
        logging.exception("Error during classification:")
        return {"success": False, "message": str(e)}

    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/recognize/")
async def recognize_song_acr(file: UploadFile = File(...)):
    temp_filename = f"temp_recognize_{file.filename}"
    file_content = await file.read()

    try:
        with open(temp_filename, "wb") as buffer:
            buffer.write(file_content)
        
        result_json_str = acr_recognizer.recognize_by_file(temp_filename, 0) 
        
        return format_acrcloud_response(result_json_str)
    except Exception as e:
        logging.exception("Error during SDK ACRCloud recognition:")
        return {"success": False, "message": f"Recognition failed: {str(e)}"}
    finally:
        # Changed: Ensure temp file is cleaned up
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

@app.post("/upload/")
async def upload_song_acr(file: UploadFile = File(...), song_name: str = Form(None)): 
    temp_filename = f"temp_upload_{file.filename}"
    file_content = await file.read()

    try:
        with open(temp_filename, "wb") as buffer:
            buffer.write(file_content)
            
        result_json_str = acr_recognizer.recognize_by_file(temp_filename, 0)
        
        response_data = format_acrcloud_response(result_json_str)
        if song_name and response_data.get("success"):
            response_data["message_context"] = f"Recognition for (originally uploaded as '{song_name}')"
        elif song_name and not response_data.get("success"):
             response_data["message"] = f"Recognition for (originally uploaded as '{song_name}') failed: {response_data.get('message')}"

        return response_data
    except Exception as e:
        logging.exception("Error during SDK ACRCloud upload/recognition:")
        return {"success": False, "message": f"Upload/Recognition failed: {str(e)}"}
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

@app.post("/recognize-live-chunk/")
async def recognize_live_chunk(file: UploadFile = File(...)):
    file_content = await file.read()

    if not file_content:
        return {"success": False, "message": "Empty audio chunk received."}

    try:
        logging.info(f"Received live chunk, size: {len(file_content)} bytes, filename: {file.filename}")
        
        # First try music recognition
        result_json_str = acr_recognizer.recognize_by_filebuffer(file_content, 0)
        music_result = format_acrcloud_response(result_json_str)

        # Check if we got a valid music result
        if music_result["success"] and music_result.get("song_name"):
            # If we have a valid song name, return the music result
            return {
                "success": True,
                "type": "music",
                "music_result": music_result
            }
        
        # If no valid music result, try YAMNet classification
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_file:
            temp_filename = temp_file.name
            temp_file.write(file_content)

        try:
            # Convert to WAV first
            wav_filename = temp_filename.replace('.webm', '.wav')
            if convert_audio_to_wav(temp_filename, wav_filename):
                yamnet_result = classify_audio_with_yamnet(wav_filename)
                
                if yamnet_result["success"]:
                    # Check if the sound is vehicle-related
                    is_vehicle, vehicle_class, vehicle_score = is_vehicle_sound(yamnet_result["top_classes"])
                    if is_vehicle:
                        # If it's a vehicle sound, use the neural network for specific classification
                        vehicle_class = predict_vehicle_class(wav_filename)
                        vehicle_type = "Car" if vehicle_class == 0 else "Truck"
                        
                        return {
                            "success": True,
                            "type": "vehicle",
                            "vehicle_result": {
                                "vehicle_type": vehicle_type,
                                "detected_sound": str(vehicle_class),  # Convert to string
                                "confidence": float(vehicle_score) * 100  # Convert to Python float
                            }
                        }
                    
                    # If not a vehicle sound, return YAMNet classification
                    return {
                        "success": True,
                        "type": "sound",
                        "sound_result": {
                            "top_classes": [(str(label), float(score)) for label, score in yamnet_result["top_classes"]]
                        }
                    }
            
            # If we get here, all recognition attempts failed
            return {
                "success": False,
                "message": "No music, vehicle, or sound patterns recognized."
            }
        finally:
            # Clean up temporary files
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            if os.path.exists(wav_filename):
                os.remove(wav_filename)

    except Exception as e:
        logging.exception("Error during audio processing:")
        return {"success": False, "message": f"Processing failed: {str(e)}"}

def format_acrcloud_response(result_json_str: str):
    """
    Parses the JSON string response from ACRCloud and formats it.
    """
    try:
        result = json.loads(result_json_str)
        logging.info(f"ACRCloud raw response: {result}")
        
        # Check if we have a valid music result
        if result.get("status", {}).get("code") == 0 and "metadata" in result and "music" in result["metadata"]:
            # Ensure 'music' list is not empty
            if not result["metadata"]["music"]:
                return {"success": False, "message": "No music metadata found in response."}
            
            music_info = result["metadata"]["music"][0]
            title = music_info.get("title")
            
            # If no title, it's not a valid music result
            if not title:
                return {"success": False, "message": "No song title found in response."}
            
            artists_list = music_info.get("artists", [])
            artists = ", ".join([artist["name"] for artist in artists_list if "name" in artist])
            album = music_info.get("album", {}).get("name")
            
            offset_seconds = music_info.get("play_offset_ms", 0) / 1000.0
            if offset_seconds == 0 and "sample_begin_time_offset_ms" in music_info:
                 offset_seconds = music_info.get("sample_begin_time_offset_ms", 0) / 1000.0
            
            confidence = music_info.get("score", 0)
            if confidence == 0 and "result_type" in result: 
                 confidence = result.get("result_type",0) * 25 

            return {
                "success": True,
                "song_name": title,
                "artists": artists,
                "album": album,
                "confidence": confidence,
                "offset_seconds": offset_seconds,
                "raw_acr_response": result
            }
        else:
            return {"success": False, "message": result.get("status", {}).get("msg", "Song not recognized or error in response.")}
    except json.JSONDecodeError:
        logging.error(f"Failed to decode ACRCloud JSON response: {result_json_str}")
        return {"success": False, "message": "Error parsing recognition server response."}
    except Exception as e:
        logging.error(f"Error processing ACRCloud response: {e} -- Response was: {result_json_str}")
        return {"success": False, "message": f"An unexpected error occurred: {str(e)}"}

@app.post("/predict/")
async def predict_audio(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # Predict using the neural network
        predicted_class = predict_vehicle_class(tmp_path)
        return {"filename": file.filename, "predicted_class": int(predicted_class)}
    finally:
        os.remove(tmp_path)

logging.basicConfig(level=logging.INFO)
