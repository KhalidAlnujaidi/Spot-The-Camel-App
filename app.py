import argparse
import io
import os
from PIL import Image
import cv2
import numpy as np

import torch
from flask import Flask, render_template, request, redirect, Response

# Load custom model
model = torch.hub.load("ultralytics/yolov5", "custom", path = "./best_camel.pt", force_reload=True)
# Set Model Settings
model.eval()
model.conf = 0.6  # confidence threshold (0-1)
model.iou = 0.45  # NMS Intersection over Union (IoU) threshold (0-1) 

app = Flask(__name__)

# Generate webcam connection
def gen():
    
    cap=cv2.VideoCapture(0)
    
    # Read until video is completed
    while(cap.isOpened()):
        
        # Capture frame-by-fram 
        success, frame = cap.read()
        if success == True:

            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
            

            img = Image.open(io.BytesIO(frame))
            results = model(img, size=640)
            
            # print results to screen
            results.print()  
            
            #convert remove single-dimensional entries from the shape of an array
            img = np.squeeze(results.render()) #RGB
            # read image as BGR
            img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #BGR

        else:
            print("error connecting to webcam")
            break
        


        # Encode BGR image to bytes so that cv2 will convert to RGB
        frame = cv2.imencode('.jpg', img_BGR)[1].tobytes()
        #print(frame)
        
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    """Video streaming route. Put this in the src attribute of an img tag."""

    return Response(gen(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

#run app
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat