# Server part referenced from the Picamera2 documentation,9.1. Streaming to a network
# Obejct detection referenced from Picamera2 repo @ https://github.com/raspberrypi/picamera2/tree/main/examples/tensorflow

import io
import logging
import socketserver
import cv2
import numpy as np
from http import server
from threading import Condition, Thread
import tflite_runtime.interpreter as tflite
from libcamera import Transform
from picamera2 import Picamera2,MappedArray
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput
import requests
from time import localtime,strftime

PAGE = """\
<html>
<head>
<title>picamera2 MJPEG streaming demo</title>
</head>
<body>
<img src="stream.mjpg" width="640" height="480" />
</body>
</html>
"""

MODEL_PATH = "mobilenet_v2.tflite"
LABEL_PATH = "coco_labels.txt"
interpreter = tflite.Interpreter(model_path=MODEL_PATH, num_threads=4)


class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()


class StreamingHandler(server.http):
    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                logging.warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
        else:
            self.send_error(404)
            self.end_headers()


class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

def ReadLabelFile(file_path):
  with open(file_path, 'r') as f:
    lines = f.readlines()
  ret = {}
  for line in lines:
    pair = line.strip().split(maxsplit=1)
    ret[int(pair[0])] = pair[1].strip()
  return ret

def detect(image):
   global rectangles,interpreter,LABEL_PATH
   
   labels = ReadLabelFile(LABEL_PATH)
   interpreter.allocate_tensors()

   input_details = interpreter.get_input_details()
   output_details = interpreter.get_output_details()
   height = input_details[0]['shape'][1]
   width = input_details[0]['shape'][2]
   floating_model = False
   if input_details[0]['dtype'] == np.float32:
       floating_model = True

   rgb = cv2.cvtColor(image,cv2.COLOR_RGBA2RGB)
   initial_h, initial_w, channels = rgb.shape

   picture = cv2.resize(rgb, (width, height))

   input_data = np.expand_dims(picture, axis=0)
   if floating_model:
      input_data = (np.float32(input_data) - 127.5) / 127.5

   interpreter.set_tensor(input_details[0]['index'], input_data)

   interpreter.invoke()

   detected_boxes = interpreter.get_tensor(output_details[0]['index'])
   detected_classes = interpreter.get_tensor(output_details[1]['index'])
   detected_scores = interpreter.get_tensor(output_details[2]['index'])
   num_boxes = interpreter.get_tensor(output_details[3]['index'])

   rectangles = []
   for i in range(int(num_boxes)):
      top, left, bottom, right = detected_boxes[0][i]
      classId = int(detected_classes[0][i])
      score = detected_scores[0][i]
      if score > 0.5:
          xmin = left * initial_w
          ymin = bottom * initial_h
          xmax = right * initial_w
          ymax = top * initial_h
          box = [xmin, ymin, xmax, ymax]
          rectangles.append(box)
          print(labels[classId], 'score = ', score)
          rectangles[-1].append(labels[classId])
      return rectangles

def post_proccess(request):
    with MappedArray(request,"main") as m:
        rectangles = detect(m.array)
        if len(rectangles) > 0:
            for rect in rectangles:
                rect_start = (int(rect[0]) - 5, int(rect[1]) - 5)
                rect_end = (int(rect[2]) + 5, int(rect[3]) + 5)
                cv2.rectangle(m.array, rect_start, rect_end, (0,255,0,0))
                if len(rect) == 5:
                    text = rect[4]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(m.array, text, (int(rect[0]) + 10, int(rect[1]) + 10), font, 1, (255,255,255),2,cv2.LINE_AA)
                try:
                    obj = rect[4]
                    time_now = strftime("%a, %d %b %Y %H:%M:%S", localtime())
                    content = {"action":"send","time":time_now,"object":obj}
                    res = requests.post("http://172.16.109.23:5000/email",json=content)
                except:
                    pass

picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)},transform=Transform(vflip=True)))
output = StreamingOutput()
picam2.post_callback = post_proccess
picam2.start_recording(JpegEncoder(), FileOutput(output))

try:
    address = ('', 8000)
    server = StreamingServer(address, StreamingHandler)
    server.serve_forever()
finally:
    picam2.stop_recording()
