import pickle
import numpy as np
from flask import Flask, render_template, request, redirect
from flask import *
from flask_uploads import UploadSet, configure_uploads, IMAGES

import numpy as np
from keras.preprocessing import image

app = Flask(__name__)
app.config['SECRET_KEY'] = "ABC"

# for images
photos = UploadSet('photos', IMAGES)

app.config['UPLOADED_PHOTOS_DEST'] = 'static/img'
configure_uploads(app, photos)


def classifying(val):
    from tensorflow.keras.models import load_model

    classifier = load_model('model.h5')
    img = "static/img/" + val
    test_image = image.load_img(img, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)

    pred = np.argmax(result[0])

    if pred == 0:
        res = "Bike"
    elif pred == 1:
        res = "Car"
    elif pred == 2:
        res = "Cycle"
    return res


@app.route('/result', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        val = filename
        value = classifying(val)
        return render_template('result.html', pic=val, value=value)


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
