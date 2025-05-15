import argparse
import os

import requests


def recognize_song_from_file(server_url: str, file_path: str):
    """
    Sends a local audio file to the song recognition server and prints the response.

    Args:
        server_url (str): The base URL of the FastAPI server (e.g., http://127.0.0.1:8000).
        file_path (str): The local path to the audio file.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    recognize_endpoint = f"{server_url.rstrip('/')}/recognize/"
    
    try:
        with open(file_path, 'rb') as f:
            files = {
                'file': (os.path.basename(file_path), f, 'audio/mpeg') # Adjust content type if needed
            }
            print(f"Sending {file_path} to {recognize_endpoint}...")
            response = requests.post(recognize_endpoint, files=files, timeout=30)
            response.raise_for_status() 
            
            print("Response from server:")
            print(response.json())

    except requests.exceptions.RequestException as e:
        print(f"Error connecting to server or during request: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Client script to recognize a song by sending an audio file to the recognition server.")
    parser.add_argument("file_path", type=str, help="The path to the audio file to recognize.")
    parser.add_argument("--server_url", type=str, default="http://127.0.0.1:8000", help="The base URL of the FastAPI recognition server.")
    
    args = parser.parse_args()
    
    recognize_song_from_file(args.server_url, args.file_path) 