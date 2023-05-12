from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline


application = Flask(__name__)

app = application

# Route for a Home page

@app.route('/')
def index(): 
    return render_template('index.html')

@app.route('/predictdata', methods = ['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
         data= CustomData(
             cloneSize = float(request.form.get('cloneSize')),
        honeybee = float(request.form.get('honeybee')),
        bumbles = float(request.form.get('bumbles')),
        andrena =float(request.form.get('andrena')),
        osmia = float(request.form.get('osmia')),
        maxUpperTRange = float(request.form.get('maxUpperTRange')),
        rainingDays = float(request.form.get('rainingDays')),
        fruitset = float(request.form.get('fruitset'))
        )
    pred_df = data.get_data_as_data_frame()
    print(pred_df)

    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(pred_df)
    return render_template('home.html', results =results[0])

if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)