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
    try:
        restaurant_names = [line[:-1] for line in open('app/static/VegasRestaurantNames.csv')]
        return render_template('homepage.html', restaurant_names=list( restaurant_names[i] for i in [ 16, 17, 21, 24, 29 ] ))
    except Exception as e:
        logging.error('Exception occurred: %s', e)
        return render_template('error_in_model.html')

@app.route("/", methods=['POST'])
def homepage_post():
    logging.info('Request form: %s', request.form)

    restaurant_name = request.form['restaurant_name'] if 'restaurant_name' in request.form else ''
    restaurant_address = request.form['restaurant_address'] if 'restaurant_address' in request.form else ''
    forecast_length = request.form['forecast_length'].split(',') if 'forecast_length' in request.form else ''

    logging.info('User input his/her own restaurant and address')
    logging.info('Restaurant name: %s', restaurant_name)
    logging.info('Restaurant address: %s', restaurant_address)
    logging.info('Forecast length: %s', forecast_length)

    model = tools.XGBoost_model(int(forecast_length[0][0]))

    HEADERS = {'Authorization':'Bearer %s' %'t2POjvVZfc64zVMRRDEvhFA_ffRJpB_MJvk0oqiOcJvyBtu_42soOy-m6JQo0JSZyqESd56-bE41ZxXRv8qmSXs01Pb05hCU-UocJXlOLFytEpodpjFWNZWkypgoXnYx'}
    URL_business = 'https://api.yelp.com/v3/businesses/matches'
    PARAMS_business = {'name' : restaurant_name, 'address1' : restaurant_address, 'city' : 'Las Vegas', 'state' : 'NV', 'country' : 'US'}

    logging.info('Doing business request')
    yelp_request = requests.get(URL_business, params = PARAMS_business, headers = HEADERS)
    if yelp_request.status_code == 200:
        try:
            response_blob = json.loads(yelp_request.text)['businesses'][0]
            business_id = response_blob['id']
            logging.info('Response blob: %s', response_blob)
            logging.info('Type: %s', type(response_blob))
        except:
            logging.error("Error in yelp business request. Yelp page doesn't exist")
            logging.error("Params: %s", PARAMS_business)
            logging.error("Status Code: %s", yelp_request.status_code)
            logging.error(json.loads(yelp_request.text))
            return render_template('error_in_yelp_api_call.html', name=restaurant_name, address=restaurant_address)

        try:
            logging.info('Doing review request')
            review_features = tools.do_review_request(response_blob)
            if review_features is None:
                logging.error("Review request was unsuccessful")
                raise
            else:
                logging.info("Review request was successful")
                avg_review_length, avg_compound_sentiment, alias, duplicate_location = review_features
        except:
            logging.error("Error in yelp review request. Information wasn't present")
            return render_template('error_in_yelp_review_request.html', name=restaurant_name, address=restaurant_address)

        try:
            logging.info('Doing business info request')
            business_info_features = tools.do_business_info_request(response_blob)
            if business_info_features is None:
                logging.error("Business info request was unsuccessful")
                raise
            else:
                logging.info("Business info request was successful")
                is_closed, rating, review_count, cost, cost_1, cost_2, cost_3, cost_4, is_claimed = business_info_features
                if is_closed:
                    logging.warning("Restaurant is already closed")
                    return render_template('closed_restaurant.html', name=restaurant_name, address=restaurant_address)
        except:
            logging.error("Error in yelp business info request. Information wasn't present")
            return render_template('error_in_yelp_business_info_request.html', name=restaurant_name, address=restaurant_address)

        try:
            chains = [line[:-1] for line in open('app/static/chains.csv')] # TODO handle if file doesn't exist
            is_chain = 1 if str(restaurant_name.encode('utf-8')) in chains else 0

            features = np.array( [ is_chain, duplicate_location, cost_1, cost_2, cost_3, cost_4, is_claimed, avg_compound_sentiment, avg_review_length, review_count, rating ] )

            model_prob, model_output = model.predict(features)

            logging.info("Model output: %s", model_output)

            logging.info("Alias: %s", alias)

            try:
                link = u'https://www.yelp.com/biz/{alias}'.format(alias=alias)
            except Exception as e:
                logging.error("Exception occurred: %s", e)
                return render_template('error_in_model.html', name=restaurant_name, address=restaurant_address)

            logging.info("Link: %s", link)

            try:
                reasons = tools.get_reasons(features, model_output, model.get_forecast_len())

                reason_1 = reasons[0]
                reason_2 = reasons[1]
                reason_3 = reasons[2]
            except Exception as e:
                logging.error("Exception occurred: %s", e)
                return render_template('error_in_model.html', name=restaurant_name, address=restaurant_address)

            return render_template('input_restaurant_result.html',  name=restaurant_name, link=link, is_closed=bool(is_closed), is_chain=bool(is_chain), \
                                                                        duplicate_location=bool(duplicate_location), cost=cost, is_claimed=bool(is_claimed), \
                                                                        avg_compound_sentiment=str(round(avg_compound_sentiment, 2)), avg_review_length=int(avg_review_length), \
                                                                        review_count=int(review_count), rating=str(round(rating, 2)), model_prob=model_prob, model_output=model_output, \
                                                                        reason_1=reason_1, reason_2=reason_2, reason_3=reason_3, forecast_length=str(forecast_length[0]))
        except:
            logging.error('Error in model prediction or making webpage with findings')
            return render_template('error_in_model.html', name=restaurant_name, address=restaurant_address)
    else:
        logging.error(yelp_request)
        return render_template('error_in_business_input.html', name=restaurant_name, address=restaurant_address)

