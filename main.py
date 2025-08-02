# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse
# import shutil
# import os

# app = FastAPI()

# @app.get("/")
# def read_root():
#     return {"message": "Server is running!"}

# @app.post("/analyze")
# async def analyze_video(file: UploadFile = File(...)):
#     # Save the uploaded file temporarily
#     temp_filename = f"temp_{file.filename}"
#     with open(temp_filename, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)
    
#     # === Your AI/Computer Vision processing would go here ===
#     # For now, return dummy data to test the API
#     results = {
#         "made": 7,
#         "missed": 3,
#         "shots": [
#             {"x": 2.1, "y": 3.5, "made": True},
#             {"x": 1.2, "y": 4.0, "made": False},
#         ]
#     }

#     # Delete the temp file
#     os.remove(temp_filename)

#     return JSONResponse(content=results)



from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil
import os
import cv2
from PIL import Image
from inference_sdk import InferenceHTTPClient

app = FastAPI()

# Initialize Roboflow client once
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="gELFKfGqTFzYUYu4WQ40"
)

@app.get("/")
def read_root():
    return {"message": "Server is running!"}

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    # Save the uploaded file temporarily
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Run actual detection logic
    shots_made, shots_missed = process_video(temp_filename)

    # Clean up
    os.remove(temp_filename)

    return JSONResponse(content={
        "made": shots_made,
        "missed": shots_missed
    })

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    shots_made = 0
    shots_missed = 0
    hoop_box = None
    ball_boxes = []
    ball_in_air = False

    frame_count = 0
    detect_every_n = 5

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % detect_every_n == 0:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Run detection
            ball_result = CLIENT.infer(pil_image, model_id="basketball-detection-ya1fm/1")
            hoop_result = CLIENT.infer(pil_image, model_id="basketball-hoop-detect/1")

            # Hoop box
            if hoop_result["predictions"]:
                pred = hoop_result["predictions"][0]
                hx, hy, hw, hh = pred["x"], pred["y"], pred["width"], pred["height"]
                hoop_box = (int(hx - hw/2), int(hy - hh/2), int(hx + hw/2), int(hy + hh/2))

            # Ball boxes
            ball_boxes = []
            for pred in ball_result["predictions"]:
                bx, by, bw, bh = pred["x"], pred["y"], pred["width"], pred["height"]
                bx1, by1, bx2, by2 = int(bx - bw/2), int(by - bh/2), int(bx + bw/2), int(by + bh/2)
                ball_boxes.append((bx1, by1, bx2, by2, bx, by))

            # Smart shot tracking
            if hoop_box and ball_boxes:
                bx_center, by_center = ball_boxes[0][4], ball_boxes[0][5]
                hx1, hy1, hx2, hy2 = hoop_box
                in_hoop = hx1 <= bx_center <= hx2 and hy1 <= by_center <= hy2

                if by_center < hy2:
                    ball_in_air = True

                if in_hoop:
                    if ball_in_air:
                        shots_made += 1
                        ball_in_air = False
                elif by_center > hy2:
                    if ball_in_air:
                        shots_missed += 1
                        ball_in_air = False

    cap.release()
    return shots_made, shots_missed


#uvicorn main:app --host 0.0.0.0 --port 8000
