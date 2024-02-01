from flask import Flask , render_template ,request,jsonify
import requests
# import last_logic_backup
import Feature_extraction_ff1 as fex
import pickle
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)
model = tf.keras.models.load_model("nn.h5")
# model = pickle.load(open("model.h5","rb"))



@app.route('/', methods=['POST'])
def index():
        domain = request.form.get('URL', '')
        print(domain)
        if "https://" in domain:
            domain = domain[8:]
        if "http://" in domain:
            domain = domain[7:]
        if "." not in domain:
            return render_template('real_result.html', result="invalid")
        if "www." in domain:
            domain = domain[4:]
        else:
            pass
        url = "https://" + domain
        response = requests.get(url)
        if response.status_code == 200:

            new_url_features = fex.data_set_list_creation(domain)

            new_url_features = [new_url_features]

            prediction = model.predict(new_url_features)
            print(prediction)
            if prediction >= 0.5:
                print("The URL  is predicted as a phishing URL.")
                return jsonify({"result": "phishing"})

            else:
                print("The URL  is predicted as a legitimate URL.")
                return jsonify({"result": "safe"})
        else:
            return render_template('plugin_ui.html', result="not active")   
        

    
if __name__=='__main__':
    app.run(debug=False , host='0.0.0.0')
