from flask import Flask,request,render_template
import pandas as pd
import numpy as np

from src.pipeline.predict_pipeline import CustomDataClass,Predict_pipline

application=Flask(__name__) # Entry point for the app
app=application

# Route for a home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_data():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomDataClass(
            Date=request.form.get('Date'),
            Hour=request.form.get('Hour'),
            Temperature=request.form.get('Temperature'),
            Humidity=request.form.get('Humidity'),
            Wind_speed=request.form.get('Wind_speed'),
            Visibility=request.form.get('Visibility'),
            Dew_point_temperature=request.form.get('Dew_point_temperature'),
            Solar_Radiation=request.form.get('Solar_Radiation'),
            Rainfall=request.form.get('Rainfall'),
            Snowfall=request.form.get('Snowfall'),
            Seasons=request.form.get('Seasons'),
            Holiday=request.form.get('Holiday'),
            Functioning_Day=request.form.get('Functioning_Day')
        )
        pred_df=data.get_data_as_dataframe()
        print(pred_df)

        predict_pipline=Predict_pipline()
        results=predict_pipline.predict(pred_df)
        return render_template('home.html',results=round(results[0]))
    
if __name__=='__main__':
    app.run(host="0.0.0.0",debug=True)