from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('model1.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("Cheatfit.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    print(int_features)
    print(final)
    output = model.predict(final)
    
    return render_template('CheatFit.html',pred= output)


if __name__ == '__main__':
    app.run(debug=True)
