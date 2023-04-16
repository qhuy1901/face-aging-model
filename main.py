import os

retval = os.getcwd()
print("Current working directory %s" % retval)

os.chdir(retval)
CODE_DIR = 'SAM'

os.chdir(f'./{CODE_DIR}')

from argparse import Namespace
import os
import sys
import pprint
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

sys.path.append(".")
sys.path.append("..")
retval = os.getcwd()
print("Current working directory %s" % retval)
from datasets.augmentations import AgeTransformer
from utils.common import tensor2im
from models.psp import pSp

EXPERIMENT_DATA_ARGS = {
    "ffhq_aging": {
        "model_path": "../pretrained_models/sam_ffhq_aging.pt",
        "image_path": "notebooks/images/866.jpg",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    }
}

EXPERIMENT_TYPE = 'ffhq_aging'

model_path = "../pretrained_models/sam_ffhq_aging.pt"
ckpt = torch.load(model_path, map_location='cpu')
opts = ckpt['opts']
opts['checkpoint_path'] = model_path

opts = Namespace(**opts)
net = pSp(opts)
net.eval()
net.cuda()
print('Model successfully loaded!')

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# import io
# import tensorflow as tf
# from tensorflow import keras
# import numpy as np
# from PIL import Image

from flask import Flask, request, jsonify

# model = keras.models.load_model("nn.h5")


# pprint.pprint(opts)


# def transform_image(pillow_image):
#     data = np.asarray(pillow_image)
#     data = data / 255.0
#     data = data[np.newaxis, ..., np.newaxis]
#     # --> [1, x, y, 1]
#     data = tf.image.resize(data, [28, 28])
#     return data


# def predict(x):
#     predictions = model(x)
#     predictions = tf.nn.softmax(predictions)
#     pred0 = predictions[0]
#     label0 = np.argmax(pred0)
#     return label0

def run_alignment(image_path):
    import dlib
    from scripts.align_all_parallel import align_face
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    aligned_image = align_face(filepath=image_path, predictor=predictor) 
    print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image 

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        image_path = EXPERIMENT_DATA_ARGS[EXPERIMENT_TYPE]["image_path"]
        
        if image_path is None or image_path == "":
            return jsonify({"error": "no file"})

        try:
            original_image = Image.open(image_path).convert("RGB")
            original_image.resize((256, 256))

            # prediction = predict(tensor)
            data = {"prediction": 1}
            return jsonify(data)
        except Exception as e:
            return jsonify({"error": str(e)})

    return "OK"


if __name__ == "__main__":
    app.run(debug=True)