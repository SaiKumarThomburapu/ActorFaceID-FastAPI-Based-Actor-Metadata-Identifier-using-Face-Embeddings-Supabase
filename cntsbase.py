import json
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
import requests
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from supabase import create_client, Client
import time

# ------------------ Supabase Setup ------------------
# This is like logging into our online database
SUPABASE_URL = "https://ydsieurelklvbeyxqvso.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inlkc2lldXJlbGtsdmJleXhxdnNvIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0OTYzNTEwMywiZXhwIjoyMDY1MjExMTAzfQ.Ng7ffsah_z0Gt-ulkHzbKdiZ8ytTEAvIlWORgM0g14Y"  # replace this key when you share code
TABLE_NAME = "face detection dataset"
IMAGE_URL_PREFIX = "https://ixnbfvyeniksbqcfdmdo.supabase.co/"

# We create a connection to Supabase so we can read/write data
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ------------------ Model Setup ------------------
# If we have a GPU, we use it. If not, we fall back to CPU.
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# MTCNN is the face detector — finds all faces in the image
mtcnn = MTCNN(keep_all=True, device=device, post_process=True)

# InceptionResnetV1 is the face recognizer — converts face to numbers (embeddings)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# ------------------ Track Seen Images ------------------
# This dictionary remembers which image has been processed before, and which face index we're on
seen_images = {}

# ------------------ Function: Download Image ------------------
def download_image(image_path):
    """
    Downloads an image from Supabase storage using the image path.
    Returns a PIL Image object in RGB format.
    """
    try:
        url = IMAGE_URL_PREFIX + image_path  # Full URL of the image
        response = requests.get(url, timeout=10)  # Download it
        img = Image.open(BytesIO(response.content)).convert("RGB")  # Convert to RGB
        return img
    except Exception as e:
        print(f"[ERROR] Could not download image {image_path}: {e}")
        return None

# ------------------ Function: Get Embedding ------------------
def get_embedding(img, face_index):
    """
    Takes an image and a face index, detects all faces,
    and returns the embedding (list of numbers) for that specific face.
    """
    faces = mtcnn(img)  # Detect all faces
    if faces is None or len(faces) <= face_index:
        return None  # If no face or index out of range, skip
    face = faces[face_index]  # Pick the right face
    if face.ndim == 3:
        face = face.unsqueeze(0)  # Add batch dimension if needed
    emb = resnet(face.to(device)).detach().cpu().numpy()[0]  # Get face embedding
    return emb.tolist()  # Convert to list so Supabase can store it

# ------------------ Function: Fetch Batch ------------------
def fetch_batch_from_supabase(offset, batch_size):
    """
    Fetches a batch of rows from Supabase starting at 'offset'.
    Returns a list of rows with id, image_path, and embedding fields.
    """
    print(f"[INFO] Fetching rows {offset} to {offset + batch_size - 1}...")
    response = supabase.table(TABLE_NAME)\
        .select("id, image_path, embedding")\
        .range(offset, offset + batch_size - 1)\
        .execute()
    return response.data

# ------------------ Function: Update Embedding ------------------
def update_embedding_in_supabase(record_id, embedding):
    """
    Updates a specific row in Supabase with the computed face embedding.
    """
    try:
        supabase.table(TABLE_NAME)\
            .update({"embedding": embedding})\
            .eq("id", record_id)\
            .execute()
        print(f"[INFO] Updated embedding for ID: {record_id}")
    except Exception as e:
        print(f"[ERROR] Failed to update embedding for ID: {record_id} - {e}")
        time.sleep(1)

# ------------------ Function: Process One Row ------------------
def process_row(row):
    """
    Processes a single row from the Supabase table:
    - Skips if embedding already exists
    - Downloads the image
    - Gets the correct face
    - Computes embedding
    - Updates the database
    """
    if row.get("embedding"):
        print(f"[SKIP] ID {row['id']} already has embedding.")
        return False

    image_path = row.get("image_path")
    if not image_path:
        print(f"[SKIP] No image path for ID: {row['id']}")
        return False

    # Determine which face index to use for this image
    face_index = seen_images.get(image_path, 0)

    # Download and process the image
    img = download_image(image_path)
    if img is None:
        return False

    # Get embedding for the detected face
    embedding = get_embedding(img, face_index)
    if embedding is None:
        print(f"[WARNING] No face #{face_index+1} found in image: {image_path}")
        return False

    # Save embedding to Supabase
    update_embedding_in_supabase(row["id"], embedding)

    # Mark that we’ve seen this image and move to next face for future rows
    seen_images[image_path] = face_index + 1
    return True

# ------------------ Function: Run Main Pipeline ------------------
def run_embedding_pipeline(batch_size=1000):
    """
    This is the main function that runs everything:
    - Loads rows in batches
    - Processes each row
    - Tracks how many were processed
    - Stops when all rows are done
    """
    offset = 0
    total_processed = 0

    while True:
        data = fetch_batch_from_supabase(offset, batch_size)
        if not data:
            print("[DONE] No more records to process.")
            break

        for row in data:
            if process_row(row):
                total_processed += 1

        offset += batch_size
        time.sleep(1)  # To avoid hitting rate limits

    print(f"[DONE] Processed {total_processed} records.")

# ------------------ Start Everything ------------------
if __name__ == "__main__":
    run_embedding_pipeline()




