import pandas as pd
import numpy as np
import requests
import json
import xgboost as xgb
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pickle
from sklearn.pipeline import Pipeline
import logging

def do_business_request():
    HEADERS = {'Authorization':'Bearer %s'%'t2POjvVZfc64zVMRRDEvhFA_ffRJpB_MJvk0oqiOcJvyBtu_42soOy-m6JQo0JSZyqESd56-bE41ZxXRv8qmSXs01Pb05hCU-UocJXlOLFytEpodpjFWNZWkypgoXnYx'}
    URL_business = 'https://api.yelp.com/v3/businesses/matches'
    PARAMS_business = {'name' : restaurant_name, 'address1' : restaurant_address, 'city' : 'Las Vegas', 'state' : 'NV', 'country' : 'US'}

    URL_reviews = 'https://api.yelp.com/v3/businesses/{business_id}/reviews'
    URL_business_info = 'https://api.yelp.com/v3/businesses/{business_id}'

    yelp_request = requests.get(URL_business, params = PARAMS_business, headers = HEADERS)


def do_review_request(response_blob):
    logging.info("Starting review request")
    try:
        business_id = response_blob['id']
    except:
        logging.info("'id' not in response_blob")
        return

    HEADERS = {'Authorization':'Bearer %s'%'t2POjvVZfc64zVMRRDEvhFA_ffRJpB_MJvk0oqiOcJvyBtu_42soOy-m6JQo0JSZyqESd56-bE41ZxXRv8qmSXs01Pb05hCU-UocJXlOLFytEpodpjFWNZWkypgoXnYx'}
    URL_reviews = 'https://api.yelp.com/v3/businesses/{business_id}/reviews'
    logging.info("Headers and URL_reviews")

    try:
        duplicate_locations = np.genfromtxt('app/static/duplicate_locations.csv',delimiter=',')
    except:
        logging.info("Error opening duplicate locations file")
        return
    logging.info("Duplicate locations")

    try:
        yelp_review_request = requests.get(URL_reviews.format(business_id=business_id), headers = HEADERS)
        logging.info("Review request done, status code: %s", yelp_review_request.status_code)
    except:
        logging.info("Error doing review request")
        return

    if yelp_review_request.status_code == 200:
        logging.info("Review request text: %s", yelp_review_request.text)
        sum_review_length = 0.0
        count_of_reviews = 0
        compound_sentiment = 0.0
        try:
            analyzer = SentimentIntensityAnalyzer()
        except:
            logging.info("Error for vader")
            return

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

        return ( avg_review_length, avg_compound_sentiment, alias, duplicate_location )
    else:
        logging.info("Review request status code: %s", yelp_review_request.status_code)
        logging.info("Error in review_input: %s", yelp_review_request)
        return None

def do_business_info_request(response_blob):
    business_id = response_blob['id']
    HEADERS = {'Authorization':'Bearer %s' %'t2POjvVZfc64zVMRRDEvhFA_ffRJpB_MJvk0oqiOcJvyBtu_42soOy-m6JQo0JSZyqESd56-bE41ZxXRv8qmSXs01Pb05hCU-UocJXlOLFytEpodpjFWNZWkypgoXnYx'}
    URL_business_info = 'https://api.yelp.com/v3/businesses/{business_id}'

    yelp_business_info_request = requests.get(URL_business_info.format(business_id=business_id), headers = HEADERS)
    if yelp_business_info_request.status_code == 200:
        biz_info_response_blob = json.loads(yelp_business_info_request.text)
        is_closed = biz_info_response_blob['is_closed'] # TODO if closed, cut out and render a template that says restaurant is currently closed
        rating = biz_info_response_blob['rating']
        review_count = biz_info_response_blob['review_count']
        cost = len(biz_info_response_blob['price']) # these are strings that look liked '$', '$$', '$$$', and '$$$$'
        is_claimed = biz_info_response_blob['is_claimed']

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

        return ( is_closed, rating, review_count, cost, cost_1, cost_2, cost_3, cost_4, is_claimed )
    else:
        logging.info("Business info status code: %s", yelp_business_info_request.status_code)
        logging.info("Error in business info input: %s", yelp_business_info_request)
        return None

class LR_model():

    # TODO add decorators
    def __init__(self): # include weights for other forecast lengths
        self.weights = np.array([[  -7.99999848e-01,  8.21156318e-01, -7.40526739e-01,
                                    -3.00875588e-01, -2.64969982e-01,  5.03601493e-01,
                                    -6.88070682e+00,  1.79273077e-01,  5.07445374e-04,
                                    3.22390460e-04,  7.66727829e-01]])
        self.intercept = -0.89443562
        self.threshold = 0.5

    def set_new_model_params(weights=None, intercept=None, threshold=None):
        if weights is not None:
            self.weights = weights
        if intercept is not None:
            self.intercept = intercept
        if threshold is not None:
            self.threshold = threshold

    def predict(self, features):
        logging.info("Features: %s", features)

        log_odds = np.dot(self.weights, features) + self.intercept
        logging.info("Log odds: %s", log_odds[0])

        model_prob = 1.0 / ( 1.0 + np.exp(-log_odds) ) #convert_to_prob(log_odds)
        logging.info("Model_prob: %s", model_prob[0])

        model_output = True if model_prob > self.threshold else False
        logging.info("Model_output: %s", model_output)

        return model_prob[0], model_output

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
