from __future__ import division, print_function
# coding=utf-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import time

# Keras
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, send_from_directory
from werkzeug.utils import secure_filename

#pywebio
from pywebio.platform.flask import webio_view
from pywebio import STATIC_PATH
from pywebio.input import *
from pywebio.output import *

model = load_model('model_vgg19.h5')
app = Flask(__name__)

import argparse
from pywebio import start_server

def model_predict():
    
    put_text('Malaria Disease Detection').style('color: black; font-size: 30px')
    put_text('This is a webapp for Malaria disease detection. The model was created on the top of VGG19 model via Transfer learning. ').style('color: black; font-size: 20px')
    put_text('Model expects a single cell image and classifies whether the person is infected or not. ').style('color: black; font-size: 20px')
    put_text('For testing the model download some cell images from here : ').style('color: black; font-size: 20px')
    put_link(name = 'https://drive.google.com/file/d/13NpIBWqxmuWlHIN8IMLLPID5y7146GDl/view',url ='https://drive.google.com/file/d/13NpIBWqxmuWlHIN8IMLLPID5y7146GDl/view', new_window=True)
    
    put_text('Sample cell images : ').style('color: black; font-size: 18px')
    put_image(open('sample_images.png', 'rb').read())

    
    file = file_upload(accept="image/*", placeholder ='Upload the image to be classified' ,max_size='100K', required=True)
    if file is None:
        put_markdown("`img = %r`" % file)
    else:
        open('uploads/' + str(file['filename']), 'wb').write(file['content'])
        img1 = image.load_img('uploads/' + str(file['filename']),target_size=(224,224))
        x = image.img_to_array(img1)
        x=x/255
        x = np.expand_dims(x, axis=0)
                
        put_processbar('bar')
        for i in range(1, 11):
            set_processbar('bar', i / 10)
            time.sleep(0.1)

        model.predict(x)

        if model.predict(x) > 0.5:
            put_text('The Person is Not Infected With Malaria').style('color: black; font-size: 25px')
        else:
            put_text('The Person is Infected With Malaria').style('color: black; font-size: 25px')



app.add_url_rule('/tool', 'webio_view', webio_view(model_predict),
            methods=['GET', 'POST', 'OPTIONS'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=8080)
    args = parser.parse_args()

    start_server(predict, port=args.port)

#app.run(host='localhost', port=80, debug=True)
#visit http://localhost/tool to open the PyWebIO application.