import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import json
import requests
import pickle
import tensorflow as tf

# load model
json_file = open('lstm_model_1.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = tf.keras.models.model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights('lstm_model_1_weights.h5')
print()
print('Model loaded from desk')
print()

def prepareDataForLSTM(data):
    seq_len = 1
    # data is a datafram
    data_reshaped = []
    for i in range(seq_len, data.shape[0]):
        data_reshaped.append(data.values[i-seq_len:i, :])
    # Convert to Numpy array
    data_reshaped = np.array(data_reshaped)
    return data_reshaped

# json_file = open('data_test.json','r')
# data = json.load(json_file)
# json_file.close()
# print(type(data))
# app
app = Flask(__name__)
# routes
@app.route('/', methods=['POST'])
def predict():
    # get test data
    # data = request.get_json(force=True)
    json_file = open('data_test.json', 'r')
    data = json.load(json_file)
    json_file.close()
    # convert data into dataframe
    data_df = pd.DataFrame.from_dict(data)  # force – Ignore the mimetype and always try to parse JSON.
    # reshape data
    data_reshaped = prepareDataForLSTM(data_df)
    # predictions
    result = loaded_model.predict(data_reshaped)
    # convert numpy array to a list so it can be jsonify
    result = result.tolist()
    # send back to browser
    output = {'results': result}
    # return data
    return jsonify(results=output)


if __name__ == "__main__":
    app.run(debug=True)











# @app.route('/')
# def predict():
#     '''
#     For rendering results on HTML GUI
#     '''
#     int_features = [int(x) for x in request.form.values()]
#     final_features = [np.array(int_features)]
#     prediction = model.predict(final_features)

#     output = round(prediction[0], 2)

#     return render_template('index.html', 'expected stream flow is {}'.format(output))

# if __name__ == "__main__":
#     app.run(debug=True)



