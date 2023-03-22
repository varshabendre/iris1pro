from flask import Flask,render_template, request
import pickle
import json
import numpy as np

with open(r'artifacts/model.pkl','rb') as file:
    model = pickle.load(file)
 
with open (r'artifacts/asset.json','r') as file:
    asset=json.load(file)

col=asset['columns']
app=Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/get_data", methods=["POST","GET"])
def data():
    input_data = request.form
    print(input_data)
   # model.predict(input_data)

    data=np.zeros(len(col))
    data[0]=input_data['Sepal_lenght']
    data[1]=input_data['Sepal_width']
    data[2]=input_data['Petal_lenght']
    data[3]=input_data['Petal_width']
    
    result=model.predict([data])
    print(result)

    if result[0] == 0:
        iris_value= "Setosa"
    if result[0] == 1:
        iris_value="Versicolor"
    if result[0] == 2:
        iris_value="Virginica"

    return iris_value

if __name__=="__main__":
    app.run(debug=True, host="127.0.0.1",port=5000)
