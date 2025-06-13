# 🎭 ActorFaceID — FastAPI-Based Actor Metadata Identifier

ActorFaceID is a high-performance FastAPI application that detects one or more faces in an uploaded image, computes embeddings using a pretrained FaceNet model (InceptionResnetV1), and retrieves corresponding actor metadata (name, date of birth, filmography, etc.) from a Supabase database.

This system is optimized for accuracy, supports similarity thresholding, and handles multi-face detection and matching in real time.

---

## 🚀 Features

- 📸 Upload images with one or multiple faces  
- 🧠 Detect and embed facial features using MTCNN + InceptionResnetV1  
- 🧾 Retrieve rich actor metadata from Supabase  
- 🎚️ Dynamic similarity threshold (via API form parameter)  
- ⚡ Fast response with PyTorch and FAISS optimization  
- ✅ Built-in CORS support for frontend integration  
- 📦 Clean, modular, and production-ready code  

---

## 🛠️ Tech Stack

| Component     | Technology                      |
|---------------|----------------------------------|
| Backend API   | FastAPI                         |
| Face Detection| MTCNN (facenet-pytorch)         |
| Face Embedding| InceptionResnetV1 (VGGFace2)    |
| DB/Storage    | Supabase                        |
| Vector Search | FAISS + Cosine Similarity       |
| Language      | Python 3.10+                    |

---

## 📁 Folder Structure
project-root/
│
├── main.py # FastAPI app (face detection + metadata fetch)
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── .env (optional) # For local Supabase keys (if needed)






---

## 📥 API Usage

### Endpoint: `/identify-actor/`  
`POST` an image and a similarity threshold to get metadata for detected faces.

#### 📩 Request
`Content-Type: multipart/form-data`

| Field        | Type     | Description                                  |
|--------------|----------|----------------------------------------------|
| file         | file     | Uploaded image (.jpg, .jpeg, .png)           |
| threshold    | float    | Optional, similarity threshold (0.5 to 0.99) |

#### 🧪 Example with `curl`:
```bash
curl -X POST http://localhost:8000/identify-actor/ \
  -F "file=@actor_photo.jpg" \
  -F "threshold=0.85"



###Sample Responses
{
  "faces_detected": 2,
  "results": [
    {
      "face_index": 0,
      "name": "Robert Downey Jr.",
      "dob": "1965-04-04",
      "filmography": "Iron Man, Sherlock Holmes",
      "similarity_score": 0.913
    },
    {
      "face_index": 1,
      "name": "Scarlett Johansson",
      "dob": "1984-11-22",
      "filmography": "Lucy, Black Widow",
      "similarity_score": 0.887
    }
  ],
  "execution_time_seconds": 1.56
}





