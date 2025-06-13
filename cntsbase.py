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

# --- Supabase Setup ---
SUPABASE_URL = "https://ydsieurelklvbeyxqvso.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inlkc2lldXJlbGtsdmJleXhxdnNvIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0OTYzNTEwMywiZXhwIjoyMDY1MjExMTAzfQ.Ng7ffsah_z0Gt-ulkHzbKdiZ8ytTEAvIlWORgM0g14Y"  # Use actual service key here
TABLE_NAME = "face detection dataset"
IMAGE_URL_PREFIX = "https://ixnbfvyeniksbqcfdmdo.supabase.co/"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Torch & Model Setup ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=True, device=device, post_process=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# --- Helper Functions ---
def download_image(image_path):
    try:
        url = IMAGE_URL_PREFIX + image_path
        response = requests.get(url, timeout=10)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return img
    except Exception as e:
        print(f"[ERROR] Could not download image {image_path}: {e}")
        return None

def get_embedding(img, face_index):
    faces = mtcnn(img)
    if faces is None or len(faces) <= face_index:
        return None
    face = faces[face_index]  # Get specified face index
    if face.ndim == 3:
        face = face.unsqueeze(0)
    emb = resnet(face.to(device)).detach().cpu().numpy()[0]
    return emb.tolist()  # For Supabase compatibility

# --- Track Seen Images ---
seen_images = {}

# --- Fetch and Process in Batches ---
BATCH_SIZE = 1000
OFFSET = 0
TOTAL_PROCESSED = 0

while True:
    print(f"[INFO] Fetching rows {OFFSET} to {OFFSET + BATCH_SIZE - 1}...")
    response = supabase.table(TABLE_NAME).select("id, image_path, embedding").range(OFFSET, OFFSET + BATCH_SIZE - 1).execute()
    data = response.data
    if not data:
        print("[DONE] No more records to process.")
        break

    for row in data:
        if row.get("embedding"):
            print(f"[SKIP] ID {row['id']} already has embedding.")
            continue

        image_path = row.get("image_path")
        if not image_path:
            print(f"[SKIP] No image path for ID: {row['id']}")
            continue

        face_index = seen_images.get(image_path, 0)  # 0 if not seen before
        img = download_image(image_path)
        if img is None:
            continue

        embedding = get_embedding(img, face_index)
        if embedding is None:
            print(f"[WARNING] No face #{face_index+1} found in image: {image_path}")
            continue

        try:
            update_resp = supabase.table(TABLE_NAME).update({"embedding": embedding}).eq("id", row["id"]).execute()
            print(f"[INFO] Updated embedding for ID: {row['id']} (face #{face_index+1})")
            seen_images[image_path] = face_index + 1  # Increment index for next occurrence
        except Exception as e:
            print(f"[ERROR] Failed to update embedding for ID: {row['id']} - {e}")
            time.sleep(1)

        TOTAL_PROCESSED += 1

    OFFSET += BATCH_SIZE
    time.sleep(1)  # Prevent hitting rate limits

print(f"[DONE] Processed {TOTAL_PROCESSED} records.")




