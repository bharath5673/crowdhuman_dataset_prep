import cv2
import numpy as np
import time
import json
from collections import deque
from ultralytics import YOLO

# # Load YOLO model
model = YOLO('crowdhuman_yolov8n.pt')  # Load a pre-trained model

# Model configuration
model.overrides['conf'] = 0.3  # NMS confidence threshold
model.overrides['iou'] = 0.4  # NMS IoU threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic
model.overrides['max_det'] = 1000  # Maximum detections per image
model.overrides['classes'] = 0  # Define specific class for detection

# Random colors for visualization
colors = np.random.randint(0, 255, size=(80, 3), dtype='uint8')

# Dictionary to store tracking trajectories
tracking_trajectories = {}
annotations = []

def process(image, frame_id, track=False):
    if track:
        results = model.track(image, verbose=False, device=0, persist=True, tracker="bytetrack.yaml")
        detected_ids = {int(bbox.id) for predictions in results if predictions is not None for bbox in predictions.boxes if bbox.id is not None}
        
        # Remove lost IDs
        tracking_trajectories.keys() & detected_ids or tracking_trajectories.clear()

        for predictions in results:
            if predictions is None or predictions.boxes is None or predictions.boxes.id is None:
                continue
            
            for bbox in predictions.boxes:
                for score, cls, bbox_coords, obj_id in zip(bbox.conf, bbox.cls, bbox.xyxy, bbox.id):
                    if obj_id is None:
                        continue
                    obj_id = int(obj_id)
                    xmin, ymin, xmax, ymax = map(int, bbox_coords)
                    
                    # Draw bounding box
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 225), 2)
                    label = f'ID: {obj_id} {predictions.names[int(cls)]} {round(float(score) * 100, 1)}%'
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.rectangle(image, (xmin, ymin - text_size[1] - 5), (xmin + text_size[0], ymin), (30, 30, 30), -1)
                    cv2.putText(image, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Compute centroid
                    centroid = ((xmin + xmax) // 2, (ymin + ymax) // 2)
                    tracking_trajectories.setdefault(obj_id, deque(maxlen=5)).append(centroid)
                    
                    # Store annotation
                    annotations.append({
                        'frame_id': frame_id,
                        'object_id': obj_id,
                        'class': predictions.names[int(cls)],
                        'bbox': [xmin, ymin, xmax, ymax],
                        'centroid': centroid
                    })
    
    return image

if __name__ == '__main__':
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture('/home/bharath/Downloads/test_codes/hcl_task1/yolo/test_videos/trimmedVideo_2.mp4')
    frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_video.mp4', fourcc, 30, (frame_width, frame_height))
    out2 = cv2.VideoWriter('output_video_3.mp4', fourcc, 30, (frame_width, frame_height))

    
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()
    
    frame_id, start_time, fps = 0, time.time(), ""
    
    while True:
        frame_id += 1
        ret, frame = cap.read()
        frame2 = frame.copy()
        if not ret:
            break
        
        frame = process(frame, frame_id, track=True)
        
        # Calculate FPS
        if frame_id % 10 == 0:
            elapsed_time = time.time() - start_time
            fps = f'FPS: {10 / elapsed_time:.2f}'
            start_time = time.time()
        
        # Display FPS and Frame ID
        text = f'{fps} | Frame: {frame_id}'
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("YOLO Object Tracking", frame)
        out.write(frame)
        out2.write(frame2)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Save annotations to JSON
    with open("output_video_3.json", "w") as f:
        json.dump(annotations, f, indent=4)
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
