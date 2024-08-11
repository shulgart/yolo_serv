# https://www.freecodecamp.org/news/how-to-detect-objects-in-images-using-yolov8/#detect

from ultralytics import YOLO
import cv2
import numpy as np
import re
from flask import request, Response, Flask
from waitress import serve
from PIL import Image
import base64
from io import BytesIO
import json

app = Flask(__name__)

@app.route("/")
def root():
    """
    Site main page handler function.
    :return: Content of index.html file
    """
    with open("index.html") as file:
        return file.read()


@app.route("/detect", methods=["POST"])
def detect():
    """
        Handler of /detect POST endpoint
        Receives uploaded file with a name "image_file", 
        passes it through YOLOv8 object detection 
        network and returns an array of bounding boxes.
        :return: a JSON array of objects bounding 
        boxes in format 
        [[x1,y1,x2,y2,object_type,probability],..]
    """
    buf = request.files["image_file"]
    boxes = detect_objects_on_image(Image.open(buf.stream))
    return Response(
      json.dumps(boxes),  
      mimetype='application/json'
    )

@app.route("/detect_base64", methods=["POST"])
def detect_base64():
    """
        receives base64 string of an encoded image,
        decodes it into PIL image,
        transmits it into detecting function that returns bbox information,
        produces image with bbox-mask with alpha channels, 
        that in turn is encoded back into base64 string and transmitted back to the sender
        :return: 
              base64 string (utf-8) of a resulting bbox-mask
    """
    # html = '<html><body><img src="data:image/jpeg;base64,' + data_base64 + '"></body></html>' # embed in html
    buf = request.get_json()['image_file']
    buf = re.sub('^data:image/.+;base64,', '', buf)     # trim the html tags if there are
    buf = Image.open(BytesIO(base64.b64decode(buf)))    # get PIL image object
    H, W = buf.size                                     # image shapes
    boxes = detect_objects_on_image(buf)                # detecting function (returns bboxes)
    blank_image = np.zeros((W, H, 4), dtype=np.uint8)   # blank transparent image to draw bboxes onto
    # blank_image = cv2.cvtColor(blank_image, cv2.COLOR_BGR2RGBA)
    for box in boxes:
        # draw bboxes
        cv2.rectangle(blank_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255, 255), 2)
    b = cv2.imencode('.jpg', blank_image)               # encode the resulting image
    b = base64.b64encode(b[1])                          # encode to base64 bytes
    base_64str = b.decode('utf-8')                      # base64 bytes -> utf-8 string for transmit
    result = {'base64_string' : base_64str}
    return Response(
      json.dumps(result),
      mimetype='application/json'
    )


def detect_objects_on_image(buf):
    """
    Function receives an image,
    passes it through YOLOv8 neural network
    and returns an array of detected objects
    and their bounding boxes
    :param buf: Input image file stream
    :return: Array of bounding boxes in format 
    [[x1,y1,x2,y2,object_type,probability],..]
    """
    model = YOLO("best.pt")                  # choose YOLO-model (best.pt - is fine-tuned one)
    results = model.predict(buf, classes=2)     # predict
    result = results[0]                         # one image only
    output = []
    for box in result.boxes:
        x1, y1, x2, y2 = [
          round(x) for x in box.xyxy[0].tolist()
        ]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        output.append([
          x1, y1, x2, y2, result.names[class_id], prob
        ])
    return output

if __name__ == '__main__':
  app.run(host="0.0.0.0", port="8080", debug=True)
  # serve(app, host='0.0.0.0', port=8080)