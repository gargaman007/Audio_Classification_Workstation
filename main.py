import json
import logging
import os
import shutil

from acrcloud.recognizer import ACRCloudRecognizer
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()

templates = Jinja2Templates(directory=".")

# ACRCloud Configuration using SDK
ACRCLOUD_CONFIG = {
    'host': 'identify-ap-southeast-1.acrcloud.com',
    'access_key': 'c529996b7457352ca72e2ccb1fcbc4dd',
    'access_secret': 'MQitmw327GTfkoLhCzk90Uwcf2dL0DGhUvQvQwS0',
    'timeout': 1  # seconds
}
acr_recognizer = ACRCloudRecognizer(ACRCLOUD_CONFIG) 

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def format_acrcloud_response(result_json_str: str):
    """
    Parses the JSON string response from ACRCloud and formats it.
    (This function is largely the same as it processes the JSON string)
    """
    try:
        result = json.loads(result_json_str)
        logging.info(f"ACRCloud raw response: {result}")
        if result.get("status", {}).get("code") == 0 and "metadata" in result and "music" in result["metadata"]:
            # Ensure 'music' list is not empty
            if not result["metadata"]["music"]:
                return {"success": False, "message": "No music metadata found in response."}
            
            music_info = result["metadata"]["music"][0]
            title = music_info.get("title")
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
        result_json_str = acr_recognizer.recognize_by_filebuffer(file_content, 0)
        
        return format_acrcloud_response(result_json_str)
    except Exception as e:
        logging.exception("Error during SDK ACRCloud live chunk recognition:")
        return {"success": False, "message": f"Live recognition failed: {str(e)}"}

logging.basicConfig(level=logging.INFO)
