import numpy as np
from joblib import load
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model=load('decision.pkl')

@app.route('/')
def home():
    return render_template('Chronic_Kidney.html')

@app.route('/y_predict',methods=['POST'])
def y_predict():
    '''
    For rendering results on HTML GUI
    '''
    x_test = [x for x in request.form.values()]
    print(x_test);
    if(x_test[5]=='abnormal'):
        x_test[5]=0
    else:
        x_test[5]=1
    if(x_test[6]=='abnormal'):
        x_test[6]=0
    else:
        x_test[6]=1
    if(x_test[7]=='notpresent'):
        x_test[7]=0
    else:
        x_test[7]=1
    if(x_test[8]=='notpresent'):
        x_test[8]=0
    else:
        x_test[8]=1
    if(x_test[18]=='no'):
        x_test[18]=0
    else:
        x_test[18]=1
    if(x_test[19]=='no'):
        x_test[19]=3
    else:
        x_test[19]=4
    if(x_test[20]=='no'):
        x_test[20]=1
    else:
        x_test[20]=2
    if(x_test[21]=='good'):
        x_test[21]=0
    else:
        x_test[21]=1
    if(x_test[22]=='no'):
        x_test[22]=0
    else:
        x_test[22]=1
    if(x_test[23]=='no'):
        x_test[23]=0
    else:
        x_test[23]=1
    for i in range(len(x_test)):
        x_test[i]=float(x_test[i])
    print(x_test)
    l=np.array(x_test,dtype=object)
    l=l.reshape(1,-1)
    prediction = model.predict(l)
    print(prediction)
    d={0:'have chronic kidney disease',1:'not have chronic kidney disease'}
    k=prediction.tolist()
    for i in k:
        output=d[i]
    print(output)
    return render_template('Chronic_Kidney.html', prediction_text='Person does {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.y_predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
