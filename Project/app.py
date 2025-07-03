from flask import Flask, request, render_template

from tensorflow.keras.models import load_model

from tensorflow.keras.applications.xception import preprocess_input

from tensorflow.keras.preprocessing import image

from tensorflow.keras.preprocessing.image import load_img

import numpy as np

import os

import cv2

from werkzeug.utils import secure_filename
model = load_model("xcep_yoga.h5") 
app = Flask(__name__)

@app.route("/")

def home():
    return render_template("index.html")

@app.route("/home")

def homeagain():
    return render_template("index.html")

@app.route("/iimage")

def predict():
    return render_template("image.html")

@app.route("/ioutput", methods=['POST', 'GET'])

def output():
    if request.method == 'POST':
        img = request.files["file"]

        img.save("uploaded_image.png")

        img = load_img("uploaded_image.png", target_size=(224,224))

        img = image.img_to_array(img)

        img.shape

        x = np.expand_dims(img, axis=0)

        img_data=preprocess_input(x)

        pred = model.predict(img_data)

        p = np.argmax(pred)

        columns = ['Downdog', 'Goddess', 'Plank', 'Tree', 'Warrior2']

        result = str(columns[p])

        return render_template("image.html", prediction = result)
@app.route("/ioutput", methods=['POST', 'GET'])

def video():
    print("[INFO] starting video stream...")

    vs =cv2.VideoCapture(0)

    (W,H) =(None, None)

#oup over francs from the ideo vile streat

    while True:
        (grabbed, frase)= vs.read()


        if not grabbed:
            break


        if W is None or H is None:
            (H,W)=frame.shape[:2]


        output =frame.copy()

        frame= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame= cv2.resize(frame, (224, 224)) 

        x=np.expand_dims(frame, axis=0)

        result =np.argmax(model.predict(x), axis=-1)

        Index= [' Downdog' , 'Goddess', 'Plank', 'Tree', 'warrior2']

        result=str(image[result[0]])
        cv2.putText(output, "Pose: {}".format(result), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)

#playaudio("Emergency it is a disaster")

        cv2.imshow("Output", output)

        key=cv2.waitKey(1) & 0xFF



        if key==ord("q"):
             break

#release the file pointers

    print("[INFO] cleaning up...")

    vs.release()

    cv2.destroyAllWindows()

    return render_template("output.html")
if __name__=="__main__":
    app.run(debug=  False) 