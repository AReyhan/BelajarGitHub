# Based on https://github.com/tensorflow/examples/blob/master/lite/examples/object_detection/raspberry_pi/README.md
from tflite_runtime.interpreter import Interpreter
import numpy as np
from flask import Flask, send_file, Response, request
import io, base64
import requests, json, torch, os, cv2
from PIL import Image
from io import BytesIO
import numpy as np
import time

def set_input_tensor(interpreter, image):
    """Sets the input tensor."""
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = np.expand_dims((image - 255) / 255, axis=0)


def get_output_tensor(interpreter, index):
    """Returns the output tensor at the given index."""
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    return tensor


def detect_objects(interpreter, image, threshold):
    """Returns a list of detection results, each a dictionary of object info."""
    set_input_tensor(interpreter, image)
    interpreter.invoke()
    # Get all output details
    boxes = get_output_tensor(interpreter, 0)
    classes = get_output_tensor(interpreter, 1)
    scores = get_output_tensor(interpreter, 2)
    count = int(get_output_tensor(interpreter, 3))

    results = []
    for i in range(count):
        if scores[i] >= threshold:
            result = {
                'bounding_box': boxes[i],
                'class_id': classes[i],
                'score': scores[i]
            }
            results.append(result)
    return results

# Create a Flask app
app = Flask(__name__)
app.config['DEBUG'] = True

# Define a route for inference
@app.route('/predict', methods = ["POST"])
def predict():
    total_waktu = time.time()
    image = request.json["Image"]
    image = io.BytesIO(base64.b64decode(image))
    image = Image.open(image)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    interpreter = Interpreter('efficientdet_d0_coco17_tpu-32.tflite')
    interpreter.allocate_tensors()
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

    # Inference
    res = detect_objects(interpreter=interpreter, image=image, threshold=0.8)
    total_waktu = time.time() - total_waktu
    print(f'Total Waktu: {total_waktu} detik')
    return Response(json.dumps(res), 200)

if __name__ == '__main__':
    app.run('0.0.0.0')