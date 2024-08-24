from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = load_model('brain_tumor_model.h5')

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    preds = model.predict(img_array)
    return preds

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./static/uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'static/uploads', f.filename)
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result = "Tumor Detected" if preds[0] > 0.5 else "No Tumor"

        return render_template('index.html', prediction=result, img_path=f'uploads/{f.filename}')
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
