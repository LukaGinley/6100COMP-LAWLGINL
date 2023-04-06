from flask import Flask, render_template, request
import os
import tensorflow as tf
import numpy as np
from PIL import Image
from bs4 import BeautifulSoup
import requests
import datetime
from datetime import date

app = Flask(__name__)

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(5, activation="softmax"))

model.load_weights('FlowerModel.h5')

class_names = ['Dandelion', 'Daisy', 'Tulip', 'Sunflower', 'Rose']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_flower_info(class_name):
    # Use the class name to search for flower information
    search_term = class_name
    url = f'https://en.wikipedia.org/wiki/{search_term}'
    print(f"url: {url}")
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    print(f"soup: {soup}")
    # Find the first paragraph with text
    paragraphs = soup.find_all('p')
    for paragraph in paragraphs:
        if len(paragraph.text.strip()) > 0:
            return paragraph.text.strip()
    return f"No information found for {class_name}."

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    if not allowed_file(file.filename):
        return 'Invalid file format. Allowed formats are png, jpg, jpeg.'
    img = Image.open(file)
    img = np.array(img)  # Convert PIL Image object to numpy array
    img = tf.image.resize(img, (224, 224))/255.0
    img = img.numpy()
    img = img.reshape(1, 224, 224, 3)
    prediction = model.predict(img)[0]
    class_index = np.argmax(prediction)
    class_name = class_names[class_index]
    accuracy = round(prediction[class_index]*100, 2)
    flower_info = get_flower_info(class_name)

    # Get the user input location
    location = request.form['location']
    
    if 'share_results' in request.form:
        share_results = request.form['share_results']
        if share_results == 'yes':
            # Save the results to the text file
            with open('flower_results.txt', 'a') as f:
                f.write(f"{class_name}, {accuracy}, {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}, {location}\n")
    
    return render_template('predict.html', class_name=class_name, accuracy=accuracy, flower_info=flower_info)

if __name__ == '__main__':
    app.run(debug=True)