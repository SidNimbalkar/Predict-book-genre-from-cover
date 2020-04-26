from flask import Flask
import werkzeug
import keras.models
import numpy
import scipy.misc

app = flask.Flask(__name__)

@app.route('/predict', methods = ['GET', 'POST'])
def handle_request():
    imagefile = flask.request.files['image']
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name : " + imagefile.filename)
    imagefile.save(filename)

    img = scipy.misc.imread(filename, mode="L")
    img = img.reshape(784)
    loaded_model = keras.models.load_model('model/model.h5')
    predicted_label = loaded_model.predict_classes(numpy.array([img]))[0]
    print(predicted_label)

    return str(predicted_label)



if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
