from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

from src.exception import CustomException



application = Flask(__name__)

app = application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods = ['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            name = request.form.get("name"),
            mileage = int(request.form.get("mileage")),
            rating = float(request.form.get("rating")),
            review_count = int(request.form.get("review_count")),
        )
        

        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        
        return render_template('home.html', results = results[0])
    
    
    

if __name__=="__main__":
    app.run(host = "0.0.0.0", debug = True)       