from flask import render_template, request
from app import app
import numpy as np
import pandas as pd
import requests
import json
import tools
import logging

@app.route("/")
def homepage():
    logging.info('Loading homepage')
    
    return render_template('homepage.html')
    
@app.route("/", methods=['POST'])
def homepage_post():
    logging.info('Handling user input')
    
    successful_input, restaurant_name, restaurant_address, forecast_length = tools.handle_input(request.form)

    if successful_input == False:
        return render_template('error_in_model.html')

    model = tools.XGBoost_model(int(forecast_length[0][0]))  
    
    html_file, context = tools.handle_backend(restaurant_name, restaurant_address, model)
    
    context['name'] = restaurant_name
    context['forecast_length'] = str(forecast_length[0])
    
    return render_template(html_file, **context)