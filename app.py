from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask('__name__')
model= pickle.load(open('wine.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods= ["POST"])
def predict():
    fa= float(request.form.get("fixed acidity"))
    va= float(request.form.get("volatile acidity"))
    ca= float(request.form.get("citric acid"))
    rs= float(request.form.get("residual sugar"))
    chl= float(request.form.get("chlorides"))
    fsd= float(request.form.get("free sulfur dioxide"))
    tsd= float(request.form.get("total sulfur dioxide"))
    den= float(request.form.get("density"))
    pH= float(request.form.get("pH"))
    sul= float(request.form.get("sulphates"))
    alc= float(request.form.get("alcohol"))

    feature_final= np.array([[fa, va, ca, rs, chl, fsd, tsd, den, pH, sul, alc]])
    prediction= model.predict(feature_final)

    if prediction[0] == 1:
        return render_template('index.html', prediction_text= "This is a GOOD quality wine.")
    else:
        return render_template('index.html', prediction_text= "This is a BAD quality wine.")


if(__name__=='__main__'):
    app.run(host= "127.0.0.1", port= 5000, debug=True)