import pandas as pd
import numpy as np
import requests
import json
import xgboost as xgb
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pickle
from sklearn.pipeline import Pipeline
import logging

class XGBoost_model():

    def __init__(self, forecast_len):
        with open('app/static/trained_classifier_%s_months.pkl'%(forecast_len), 'rb') as fid:
            self.gs_model = pickle.load(fid)
            logging.info("Loaded model: %s", self.gs_model)
        self.forecast_len = forecast_len

    def predict(self, features):
        logging.info("Features: %s", features)

        probabilities = self.gs_model.predict_proba( features.reshape(1, -1) )[0]

        logging.info("Probabilities: %s", probabilities)

        prob_0, prob_1 = probabilities[0], probabilities[1]
        pred = 1 if prob_1 > prob_0 else 0

        if pred == 1:
            return prob_1, pred
        else:
            return prob_0, pred

    def get_forecast_len(self):
        return self.forecast_len
    
def handle_backend(restaurant_name, restaurant_address, model):
    successful_bus_req, response_blob = handle_business_request(restaurant_name, restaurant_address)

    if successful_bus_req == False:
        return 'error_in_yelp_api_call.html', {}
    
    successful_rev_req, avg_review_length, avg_compound_sentiment, alias, duplicate_location = handle_review_request(response_blob)

    if successful_rev_req == False:
        return 'error_in_yelp_review_request.html', {}

    successful_bus_info_req, is_closed, rating, review_count, cost, cost_1, cost_2, cost_3, cost_4, is_claimed = handle_business_info_request(response_blob)

    if successful_bus_info_req == False:
        return 'error_in_yelp_business_info_request.html', {}
    elif is_closed == True:
        return 'closed_restaurant.html', {}

    successful_pred, link, model_prob, model_output, reason_1, reason_2, reason_3 = handle_prediction(restaurant_name, duplicate_location, cost_1, cost_2, cost_3, cost_4, is_claimed, avg_compound_sentiment, avg_review_length, review_count, rating, alias, model)

    if successful_pred == True:
        return 'input_restaurant_result.html', {'link': link, 'model_prob': model_prob, 'model_output': model_output, 'reason_1': reason_1, 'reason_2': reason_2, 'reason_3':reason_3}
    else:
        return 'error_in_model.html', {}

def handle_input(request_form):
    success = True
    
    try:
        logging.info('Request form: %s', request_form)

        restaurant_name = request_form['restaurant_name'] if 'restaurant_name' in request_form else ''
        restaurant_address = request_form['restaurant_address'] if 'restaurant_address' in request_form else ''
        forecast_length = request_form['forecast_length'].split(',') if 'forecast_length' in request_form else ''

        logging.info('Restaurant name: %s', restaurant_name)
        logging.info('Restaurant address: %s', restaurant_address)
        logging.info('Forecast length: %s', forecast_length)
        
        return success, restaurant_name, restaurant_address, forecast_length
    except Exception as e:
        success = False
        logging.error('Exception occurred: %s', e)
        
        return success, '',                              '',              ''

def handle_business_request(restaurant_name, restaurant_address):
    success = True
    
    logging.info('Doing business request')
    
    HEADERS = {'Authorization':'Bearer %s'%'t2POjvVZfc64zVMRRDEvhFA_ffRJpB_MJvk0oqiOcJvyBtu_42soOy-m6JQo0JSZyqESd56-bE41ZxXRv8qmSXs01Pb05hCU-UocJXlOLFytEpodpjFWNZWkypgoXnYx'}
    PARAMS = {'name': restaurant_name, 'address1': restaurant_address, 'city': 'Las Vegas', 'state': 'NV', 'country': 'US'}
    
    yelp_request = requests.get('https://api.yelp.com/v3/businesses/matches', params = PARAMS, headers = HEADERS)
    
    if yelp_request.status_code != 200:
        logging.error("Status Code: %s", yelp_request.status_code)
        logging.error(yelp_request)
        success = False
        return success, ''
    
    try:
        response_blob = json.loads(yelp_request.text)['businesses'][0]
        business_id = response_blob['id']
        logging.info('Response blob: %s', response_blob)
        logging.info('Type: %s', type(response_blob))
    except:
        logging.error("Error in yelp business request")
        logging.error(json.loads(yelp_request.text))
        success = False
        return success, ''
    
    return success, response_blob

def handle_review_request(response_blob):
    success = True
    
    logging.info("Starting review request")
    try:
        business_id = response_blob['id']
    except:
        logging.error("'id' not in response_blob")
        logging.error("Review request was unsuccessful")
        success = False
        return success, '', '', '', ''

    HEADERS = {'Authorization':'Bearer %s'%'t2POjvVZfc64zVMRRDEvhFA_ffRJpB_MJvk0oqiOcJvyBtu_42soOy-m6JQo0JSZyqESd56-bE41ZxXRv8qmSXs01Pb05hCU-UocJXlOLFytEpodpjFWNZWkypgoXnYx'}
    URL_reviews = 'https://api.yelp.com/v3/businesses/{business_id}/reviews'
    logging.info("Headers and URL_reviews")

    try:
        duplicate_locations = np.genfromtxt('app/static/duplicate_locations.csv', delimiter=',')
    except:
        logging.error("Error opening duplicate locations file")
        logging.error("Review request was unsuccessful")
        success = False
        return success, '', '', '', ''
    logging.info("Duplicate locations")

    try:
        yelp_review_request = requests.get(URL_reviews.format(business_id=business_id), headers = HEADERS)
        logging.info("Review request done, status code: %s", yelp_review_request.status_code)
    except:
        logging.error("Review request was unsuccessful")
        success = False
        return success, '', '', '', '' 

    if yelp_review_request.status_code == 200:
        logging.info("Review request text: %s", yelp_review_request.text)
        sum_review_length = 0.0
        count_of_reviews = 0
        compound_sentiment = 0.0
        try:
            analyzer = SentimentIntensityAnalyzer()
        except:
            logging.error("Error doing vader sentiment analysis")
            logging.error("Review request was unsuccessful")
            success = False
            return success, '', '', '', ''

        for review in json.loads(yelp_review_request.text)['reviews']:
            text = review['text']
            vs = analyzer.polarity_scores(text.encode('utf-8'))
            compound_sentiment += vs['compound']
            sum_review_length += len(text)
            count_of_reviews += 1
        avg_review_length = float(sum_review_length) / count_of_reviews
        avg_compound_sentiment = float(compound_sentiment) / count_of_reviews

        logging.info("Avg review length: %s", avg_review_length)
        logging.info("Avg compound sentiment: %s", avg_compound_sentiment)

        alias = response_blob['alias']

        latitude = response_blob['coordinates']['latitude']
        longitude = response_blob['coordinates']['longitude']

        logging.info("Alias: %s", alias)
        logging.info("Latitude: %s", latitude)
        logging.info("Longitude: %s", longitude)

        duplicate_location = 1 if ([ latitude, longitude ] == duplicate_locations).all(axis=1).any() else 0

        logging.info("Duplicate location: %s", duplicate_location)

        return success, avg_review_length, avg_compound_sentiment, alias, duplicate_location
    else:
        logging.error("Review request was unsuccessful")
        logging.info("Review request status code: %s", yelp_review_request.status_code)
        logging.info("Error in review_input: %s", yelp_review_request)
        success = False
        return success, '', '', '', ''

def handle_business_info_request(response_blob):
    success = True
    
    logging.info("Starting business info request")
    business_id = response_blob['id']
    HEADERS = {'Authorization':'Bearer %s' %'t2POjvVZfc64zVMRRDEvhFA_ffRJpB_MJvk0oqiOcJvyBtu_42soOy-m6JQo0JSZyqESd56-bE41ZxXRv8qmSXs01Pb05hCU-UocJXlOLFytEpodpjFWNZWkypgoXnYx'}
    URL_business_info = 'https://api.yelp.com/v3/businesses/{business_id}'

    yelp_business_info_request = requests.get(URL_business_info.format(business_id=business_id), headers = HEADERS)
    if yelp_business_info_request.status_code == 200:
        biz_info_response_blob = json.loads(yelp_business_info_request.text)
        
        is_closed = biz_info_response_blob['is_closed']
        if is_closed == True:
            logging.warning("Restaurant is already closed")
            
        rating = biz_info_response_blob['rating']
        
        review_count = biz_info_response_blob['review_count']
        
        is_claimed = biz_info_response_blob['is_claimed']

        cost = len(biz_info_response_blob['price']) # these are strings that look liked '$', '$$', '$$$', and '$$$$'
        
        cost_1 = 0
        cost_2 = 0
        cost_3 = 0
        cost_4 = 0

        if cost == 1:
            cost_1 = 1
        elif cost == 2:
            cost_2 = 1
        elif cost == 3:
            cost_3 = 1
        elif cost == 4:
            cost_4 = 1
        else:
            return None

        if is_claimed == True:
            is_claimed = 1
        else:
            is_claimed = 0

        return success, is_closed, rating, review_count, cost, cost_1, cost_2, cost_3, cost_4, is_claimed
    else:
        logging.info("Business info status code: %s", yelp_business_info_request.status_code)
        logging.info("Error in business info input: %s", yelp_business_info_request)
        success = False
        return success, '', '', '', '', '', '', '', '', ''

def handle_prediction(restaurant_name, duplicate_location, cost_1, cost_2, cost_3, cost_4, is_claimed, avg_compound_sentiment, avg_review_length, review_count, rating, alias, model):
    success = True
    
    try:
        chains = [line[:-1] for line in open('app/static/chains.csv')]
        is_chain = 1 if str(restaurant_name.encode('utf-8')) in chains else 0
    except Exception as e:
        logging.error('Exception occurred: %s', e)
        logging.error("Handle prediction failed")
        success = False
        return success, '', '', '', '', '', ''
    
    features = np.array( [ is_chain, duplicate_location, cost_1, cost_2, cost_3, cost_4, is_claimed, avg_compound_sentiment, avg_review_length, review_count, rating ] )

    try:
        model_prob, model_output = model.predict(features)
        logging.info("Model output: %s", model_output)
        logging.info("Model probability: %s", model_prob)
    except Exception as e:
        logging.error("Exception occurred: %s", e)
        logging.error("Handle prediction failed")
        success = False
        return success, '', '', '', '', '', ''

    try:
        link = u'https://www.yelp.com/biz/{alias}'.format(alias=alias)
        logging.info("Link: %s", link)
    except Exception as e:
        logging.error("Exception occurred: %s", e)
        logging.error("Handle prediction failed")
        success = False
        return success, '', '', '', '', '', ''

    try:
        reasons = get_reasons(features, model_output, model.get_forecast_len())

        reason_1 = reasons[0]
        reason_2 = reasons[1]
        reason_3 = reasons[2]
    except Exception as e:
        logging.error("Exception occurred: %s", e)
        logging.error("Handle prediction failed")
        success = False
        return success, '', '', '', '', '', ''

    return success, link, model_prob, model_output, reason_1, reason_2, reason_3

def get_reasons(features, model_output, forecast_len):
    """
    claimed, avg review length, rating, sentiment, duplicate location, review count, chain, cost are feature importances (1 month model)
    """

    logging.info("Figuring out the reasons")

    # negative signs are for inversely correlated features
    features_sorted_by_importance = [ features[6], -features[8], features[10], features[7], features[1], features[9], features[0], features[2], features[3], features[4], features[5] ]

    #                   Close reasons                                                               Open reasons
    feature_names = [   ('The restaurant is not claimed by the owner',                              'The restaurant is claimed by the owner',                                   0.5     ),
                        ('Yelpers have left long reviews',                                          'Yelpers have left short reviews',                                          -500    ),
                        ('The restaurant has a low rating',                                         'The restaurant has a high rating',                                         3.25    ),
                        ('Yelpers have left harsh sounding reviews',                                'Yelpers have left positive sounding reviews',                              0.0     ),
                        ('The restaurant is in a location that other restaurants have been in',     'The restaurant is not in a location that other restaurants have been in',  0.5     ),
                        ('The restaurant has too few reviews',                                      'The restaurant has too many reviews',                                      325     ),
                        ('The restuarant is not a chain',                                           'The restuarant is a chain',                                                0.5     ),
                        ('The restaurant is not in the cheapest Yelp category',                     'The restaurant is in the cheapest Yelp category',                          0.5     ),
                        ('The restaurant is not in the second cheapest Yelp category',              'The restaurant is in the second cheapest Yelp category',                   0.5     ),
                        ('The restaurant is not in the second most expensive Yelp category',        'The restaurant is in the second most expensive Yelp category',             0.5     ),
                        ('The restaurant is not in the most expensive Yelp category',               'The restaurant is in the most expensive Yelp category',                    0.5     )   ]

    reasons = []
    max_reasons = 3

    for feature_val, feature_name in zip(features_sorted_by_importance, feature_names):
        logging.info("Feature name and value: %s (%s)", feature_name, feature_val)
        if model_output == True and feature_val < feature_name[2]: # will close and feature_val implies closing
            reasons.append(feature_name[0])
        elif model_output == False and feature_val > feature_name[2]: # will remain open and feature_val implies staying open
            reasons.append(feature_name[1])

        if len(reasons) >= max_reasons:
            break

    for reason in reasons:
        logging.info("Reasons: %s", reason)

    return reasons