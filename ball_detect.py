# from ultralytics import YOLO
# import cv2

# # Load YOLOv8 pretrained on COCO (includes sports ball)
# model = YOLO("yolov8n.pt")  # can also try yolov8s.pt, yolov8m.pt etc

# # Open your video file
# cap = cv2.VideoCapture("temp_video.mp4")  # change to your actual video

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Run YOLO detection
#     results = model.predict(source=frame, save=False, conf=0.3, verbose=False)

#     # results[0].boxes.xyxy gives (x1,y1,x2,y2)
#     for box in results[0].boxes:
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         cls = int(box.cls[0])
#         conf = box.conf[0]

#         # Only care about sports ball class (32)
#         if cls == 32:
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
#             cv2.putText(frame, f"Ball {conf:.2f}", (x1, y1-10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

#     cv2.imshow("Ball Detection", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()






# from ultralytics import YOLO
# import cv2

# # Load YOLOv8 pretrained on COCO (which includes sports ball as class 32)
# model = YOLO("yolov8n.pt")

# # Class names mapping
# names = model.names

# # Open your video file
# cap = cv2.VideoCapture("temp_video.mp4")  # or whatever your file is

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Run YOLO detection
#     results = model.predict(source=frame, save=False, conf=0.1, verbose=False)  # lower conf to catch more

#     # Draw ALL detections with labels
#     if results[0].boxes is not None:
#         for box in results[0].boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             cls = int(box.cls[0])
#             conf = box.conf[0]

#             # Draw blue box + label
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
#             cv2.putText(frame, f"{names[cls]} {conf:.2f}", (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#             print(f"Detected: {names[cls]} with confidence: {conf:.2f}")

#     cv2.imshow("All YOLO Detections", frame)
#     if cv2.waitKey(30) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()




#model using api

#fixing frame error
# import cv2
# from PIL import Image
# from inference_sdk import InferenceHTTPClient

# CLIENT = InferenceHTTPClient(
#     api_url="https://detect.roboflow.com",
#     api_key="gELFKfGqTFzYUYu4WQ40"
# )

# cap = cv2.VideoCapture("new.mp4")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert OpenCV frame to PIL Image
#     pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#     # Inference
#     result = CLIENT.infer(pil_image, model_id="basketball-detection-ya1fm/1")

#     # Draw results
#     for pred in result["predictions"]:
#         x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
#         x1, y1 = int(x - w/2), int(y - h/2)
#         x2, y2 = int(x + w/2), int(y + h/2)
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
#         cv2.putText(frame, f"{pred['class']} {pred['confidence']:.2f}",
#                     (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

#     cv2.imshow("Roboflow Basketball Detection", frame)
#     if cv2.waitKey(30) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


#with player and hoop detecxtion
# 
# import cv2
# from PIL import Image
# from inference_sdk import InferenceHTTPClient

# CLIENT = InferenceHTTPClient(
#     api_url="https://detect.roboflow.com",  # unified endpoint
#     api_key="gELFKfGqTFzYUYu4WQ40"
# )

# cap = cv2.VideoCapture("new.mp4")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert OpenCV frame to PIL Image
#     pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#     # Run all three detections
#     ball_result = CLIENT.infer(pil_image, model_id="basketball-detection-ya1fm/1")
#     hoop_result = CLIENT.infer(pil_image, model_id="basketball-hoop-detect/1")
#     player_result = CLIENT.infer(pil_image, model_id="basketball-player-detection-v8kcy/6")

#     # Draw balls - GREEN
#     for pred in ball_result["predictions"]:
#         x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
#         x1, y1, x2, y2 = int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)
#         cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
#         cv2.putText(frame, f"Ball {pred['confidence']:.2f}", (x1,y1-10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

#     # Draw hoops - RED
#     for pred in hoop_result["predictions"]:
#         x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
#         x1, y1, x2, y2 = int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)
#         cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
#         cv2.putText(frame, f"Hoop {pred['confidence']:.2f}", (x1,y1-10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

#     # Draw players - BLUE
#     for pred in player_result["predictions"]:
#         x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
#         x1, y1, x2, y2 = int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)
#         cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
#         cv2.putText(frame, f"Player {pred['confidence']:.2f}", (x1,y1-10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

#     # Show the annotated frame
#     cv2.imshow("Basketball Tracking", frame)
#     if cv2.waitKey(30) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
 


# #with mis/make tracking
# import cv2
# from PIL import Image
# from inference_sdk import InferenceHTTPClient

# CLIENT = InferenceHTTPClient(
#     api_url="https://detect.roboflow.com",
#     api_key="gELFKfGqTFzYUYu4WQ40"
# )

# cap = cv2.VideoCapture("new.mp4")

# shots_made = 0
# shots_missed = 0
# ball_in_hoop_last_frame = False
# hoop_box = None

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert to PIL
#     pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#     # Run detections
#     ball_result = CLIENT.infer(pil_image, model_id="basketball-detection-ya1fm/1")
#     hoop_result = CLIENT.infer(pil_image, model_id="basketball-hoop-detect/1")
#     player_result = CLIENT.infer(pil_image, model_id="basketball-player-detection-v8kcy/6")

#     # Draw hoop (take first detection)
#     if hoop_result["predictions"]:
#         pred = hoop_result["predictions"][0]
#         hx, hy, hw, hh = pred["x"], pred["y"], pred["width"], pred["height"]
#         x1, y1, x2, y2 = int(hx - hw/2), int(hy - hh/2), int(hx + hw/2), int(hy + hh/2)
#         hoop_box = (x1, y1, x2, y2)
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
#         cv2.putText(frame, f"Hoop {pred['confidence']:.2f}", (x1,y1-10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

#     # Draw ball detections
#     for pred in ball_result["predictions"]:
#         bx, by, bw, bh = pred["x"], pred["y"], pred["width"], pred["height"]
#         bx1, by1, bx2, by2 = int(bx - bw/2), int(by - bh/2), int(bx + bw/2), int(by + bh/2)
#         cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0,255,0), 2)
#         cv2.putText(frame, f"Ball {pred['confidence']:.2f}", (bx1,by1-10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

#     # Draw players
#     for pred in player_result["predictions"]:
#         px, py, pw, ph = pred["x"], pred["y"], pred["width"], pred["height"]
#         px1, py1, px2, py2 = int(px - pw/2), int(py - ph/2), int(px + pw/2), int(py + ph/2)
#         cv2.rectangle(frame, (px1, py1), (px2, py2), (255,0,0), 2)
#         cv2.putText(frame, f"Player {pred['confidence']:.2f}", (px1,py1-10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

#     # Check shot made / missed
#     if hoop_box and ball_result["predictions"]:
#         bx, by = int(ball_result["predictions"][0]["x"]), int(ball_result["predictions"][0]["y"])
#         hx1, hy1, hx2, hy2 = hoop_box
#         in_hoop = hx1 <= bx <= hx2 and hy1 <= by <= hy2

#         if in_hoop and not ball_in_hoop_last_frame:
#             shots_made += 1
#             print(f"🏀 Shot made! Total made: {shots_made}")
#             ball_in_hoop_last_frame = True
#         elif not in_hoop:
#             ball_in_hoop_last_frame = False

#         # Count as missed if ball fell below hoop and didn't enter
#         if by > hy2 and not in_hoop and not ball_in_hoop_last_frame:
#             shots_missed += 1
#             print(f"❌ Shot missed. Total missed: {shots_missed}")

#     # Draw scoreboard
#     cv2.putText(frame, f"Made: {shots_made}  Missed: {shots_missed}",
#                 (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

#     cv2.imshow("Basketball Shot Tracker", frame)
#     if cv2.waitKey(30) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


# #optimising speed 



# import cv2
# from PIL import Image
# from inference_sdk import InferenceHTTPClient

# CLIENT = InferenceHTTPClient(
#     api_url="https://detect.roboflow.com",
#     api_key="gELFKfGqTFzYUYu4WQ40"
# )

# cap = cv2.VideoCapture("new.mp4")

# shots_made = 0
# shots_missed = 0
# ball_in_hoop_last_frame = False
# hoop_box = None
# ball_boxes = []

# frame_count = 0
# detect_every_n = 5  # only run detection every 5 frames

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame_count += 1

#     if frame_count % detect_every_n == 0:
#         pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#         # Run detection on ball & hoop
#         ball_result = CLIENT.infer(pil_image, model_id="basketball-detection-ya1fm/1")
#         hoop_result = CLIENT.infer(pil_image, model_id="basketball-hoop-detect/1")

#         # Update hoop box (take first detection)
#         if hoop_result["predictions"]:
#             pred = hoop_result["predictions"][0]
#             hx, hy, hw, hh = pred["x"], pred["y"], pred["width"], pred["height"]
#             hoop_box = (int(hx - hw/2), int(hy - hh/2), int(hx + hw/2), int(hy + hh/2))

#         # Update ball boxes for drawing
#         ball_boxes = []
#         for pred in ball_result["predictions"]:
#             bx, by, bw, bh = pred["x"], pred["y"], pred["width"], pred["height"]
#             bx1, by1, bx2, by2 = int(bx - bw/2), int(by - bh/2), int(bx + bw/2), int(by + bh/2)
#             ball_boxes.append((bx1, by1, bx2, by2, bx, by))

#         # Check shot made / missed (only use first detected ball)
#         if hoop_box and ball_boxes:
#             bx_center, by_center = ball_boxes[0][4], ball_boxes[0][5]
#             hx1, hy1, hx2, hy2 = hoop_box
#             in_hoop = hx1 <= bx_center <= hx2 and hy1 <= by_center <= hy2

#             if in_hoop and not ball_in_hoop_last_frame:
#                 shots_made += 1
#                 print(f"🏀 Shot made! Total made: {shots_made}")
#                 ball_in_hoop_last_frame = True
#             elif not in_hoop:
#                 ball_in_hoop_last_frame = False

#             if by_center > hy2 and not in_hoop and not ball_in_hoop_last_frame:
#                 shots_missed += 1
#                 print(f"❌ Shot missed. Total missed: {shots_missed}")

#     # Draw hoop if available
#     if hoop_box:
#         hx1, hy1, hx2, hy2 = hoop_box
#         cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0,0,255), 2)
#         cv2.putText(frame, "Hoop", (hx1, hy1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

#     # Draw ball detections
#     for bx1, by1, bx2, by2, _, _ in ball_boxes:
#         cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0,255,0), 2)
#         cv2.putText(frame, "Ball", (bx1, by1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

#     # Draw scoreboard
#     cv2.putText(frame, f"Made: {shots_made}  Missed: {shots_missed}",
#                 (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

#     cv2.imshow("Basketball Shot Tracker", frame)
#     if cv2.waitKey(30) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()



#attempting to fix shots made
import cv2
from PIL import Image
from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="gELFKfGqTFzYUYu4WQ40"
)

cap = cv2.VideoCapture("new.mp4")

shots_made = 0
shots_missed = 0
ball_in_hoop_last_frame = False
hoop_box = None
ball_boxes = []
ball_in_air = False

frame_count = 0
detect_every_n = 5  # only run detection every 5 frames

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % detect_every_n == 0:
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Run detection on ball & hoop
        ball_result = CLIENT.infer(pil_image, model_id="basketball-detection-ya1fm/1")
        hoop_result = CLIENT.infer(pil_image, model_id="basketball-hoop-detect/1")

        # Update hoop box (take first detection)
        if hoop_result["predictions"]:
            pred = hoop_result["predictions"][0]
            hx, hy, hw, hh = pred["x"], pred["y"], pred["width"], pred["height"]
            hoop_box = (int(hx - hw/2), int(hy - hh/2), int(hx + hw/2), int(hy + hh/2))

        # Update ball boxes for drawing
        ball_boxes = []
        for pred in ball_result["predictions"]:
            bx, by, bw, bh = pred["x"], pred["y"], pred["width"], pred["height"]
            bx1, by1, bx2, by2 = int(bx - bw/2), int(by - bh/2), int(bx + bw/2), int(by + bh/2)
            ball_boxes.append((bx1, by1, bx2, by2, bx, by))

        # NEW: Smart shot logic
        if hoop_box and ball_boxes:
            bx_center, by_center = ball_boxes[0][4], ball_boxes[0][5]
            hx1, hy1, hx2, hy2 = hoop_box
            in_hoop = hx1 <= bx_center <= hx2 and hy1 <= by_center <= hy2

            if by_center < hy2:
                # ball is above hoop
                ball_in_air = True

            if in_hoop:
                if ball_in_air:
                    shots_made += 1
                    print(f"🏀 Shot made! Total made: {shots_made}")
                    ball_in_air = False  # reset
            elif by_center > hy2:
                if ball_in_air:
                    shots_missed += 1
                    print(f"❌ Shot missed. Total missed: {shots_missed}")
                    ball_in_air = False  # reset

    # Draw hoop if available
    if hoop_box:
        hx1, hy1, hx2, hy2 = hoop_box
        cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0,0,255), 2)
        cv2.putText(frame, "Hoop", (hx1, hy1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    # Draw ball detections
    for bx1, by1, bx2, by2, _, _ in ball_boxes:
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0,255,0), 2)
        cv2.putText(frame, "Ball", (bx1, by1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # Draw scoreboard
    cv2.putText(frame, f"Made: {shots_made}  Missed: {shots_missed}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.imshow("Basketball Shot Tracker", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
