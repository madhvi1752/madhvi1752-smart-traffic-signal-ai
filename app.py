import cv2
import torch
from flask import Flask, Response
import threading
import time

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
app = Flask(__name__)
traffic_state = {"signal": "RED"}
video_source = 0
cap = cv2.VideoCapture(video_source)

def count_vehicles(frame):
    results = model(frame)
    count = 0
    for *box, conf, cls in results.xyxy[0]:
        label = model.names[int(cls)]
        if label in ["car", "truck", "bus", "motorbike"]:
            count += 1
    return count, results.render()[0]

def traffic_logic(vehicle_count):
    if vehicle_count > 5:
        return "GREEN"
    elif 2 < vehicle_count <= 5:
        return "YELLOW"
    else:
        return "RED"

def ai_loop():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        vehicle_count, _ = count_vehicles(frame)
        signal = traffic_logic(vehicle_count)
        traffic_state["signal"] = signal
        time.sleep(5)

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        _, rendered_frame = count_vehicles(frame)
        cv2.putText(rendered_frame, f"Signal: {traffic_state['signal']}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if traffic_state["signal"] == "GREEN" else (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', rendered_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return f"""
    <h1>Smart Traffic Signal AI</h1>
    <img src="/video">
    <h2>Current Signal: <span style='color:green;'>{{traffic_state["signal"]}}</span></h2>
    """

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    threading.Thread(target=ai_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=5000)
