from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from supabase import create_client
import time

# --- Supabase Setup ---
SUPABASE_URL = "https://ydsieurelklvbeyxqvso.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inlkc2lldXJlbGtsdmJleXhxdnNvIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0OTYzNTEwMywiZXhwIjoyMDY1MjExMTAzfQ.Ng7ffsah_z0Gt-ulkHzbKdiZ8ytTEAvIlWORgM0g14Y"
TABLE_NAME = "face detection dataset"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Model Setup ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# --- FastAPI App Setup ---
app = FastAPI(title="🎭 Actor Metadata Identifier API")

# Allow CORS for frontend or Swagger testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/identify-actor/")
async def identify_actor(file: UploadFile = File(...), threshold: float = Form(0.85)):
    start_time = time.time()

    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError:
        return JSONResponse(status_code=400, content={"message": "Invalid image uploaded."})

    # Detect faces
    face_tensors = mtcnn(image)
    if face_tensors is None:
        return JSONResponse(status_code=400, content={"message": "No faces detected."})

    if isinstance(face_tensors, torch.Tensor):
        face_tensors = [face_tensors[i] for i in range(face_tensors.shape[0])]

    # Fetch embeddings from Supabase
    response = supabase.table(TABLE_NAME).select("name, dob, filmography, embedding").execute()
    raw_data = response.data
    valid_rows = [row for row in raw_data if isinstance(row["embedding"], list) and len(row["embedding"]) == 512]

    if not valid_rows:
        return JSONResponse(status_code=500, content={"message": "No valid embeddings in Supabase."})

    db_embeddings = np.array([row["embedding"] for row in valid_rows])
    db_embeddings = normalize(db_embeddings)

    results = []
    for i, face_tensor in enumerate(face_tensors):
        face_tensor = face_tensor.to(device).unsqueeze(0)
        with torch.no_grad():
            embedding = resnet(face_tensor).cpu().numpy()
        embedding = normalize(embedding)

        similarities = cosine_similarity(embedding, db_embeddings)[0]
        best_idx = int(np.argmax(similarities))
        best_score = similarities[best_idx]

        if best_score >= threshold:
            matched = valid_rows[best_idx]
            results.append({
                "face_index": i,
                "name": matched["name"],
                "dob": matched["dob"],
                "filmography": matched["filmography"],
                "similarity_score": round(float(best_score), 3)
            })
        else:
            results.append({
                "face_index": i,
                "name": "Unknown",
                "dob": "N/A",
                "filmography": "N/A",
                "similarity_score": round(float(best_score), 3)
            })

    end_time = time.time()
    execution_time = round(end_time - start_time, 2)

    return {
        "faces_detected": len(face_tensors),
        "results": results,
        "execution_time_seconds": execution_time
    }

# Optional: Customize Swagger UI to show threshold slider
@app.get("/openapi.json", include_in_schema=False)
async def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version="1.0",
        routes=app.routes,
        description="Upload an image to identify actors by matching face embeddings."
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

