import pandas as pd
import numpy as np
import requests
import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def do_business_request():
    HEADERS = {'Authorization':'Bearer %s' %'t2POjvVZfc64zVMRRDEvhFA_ffRJpB_MJvk0oqiOcJvyBtu_42soOy-m6JQo0JSZyqESd56-bE41ZxXRv8qmSXs01Pb05hCU-UocJXlOLFytEpodpjFWNZWkypgoXnYx'}
    URL_business = 'https://api.yelp.com/v3/businesses/matches'
    PARAMS_business = {'name' : restaurant_name, 'address1' : restaurant_address, 'city' : 'Las Vegas', 'state' : 'NV', 'country' : 'US'}

    URL_reviews = 'https://api.yelp.com/v3/businesses/{business_id}/reviews'
    URL_business_info = 'https://api.yelp.com/v3/businesses/{business_id}'

    yelp_request = requests.get(URL_business, params = PARAMS_business, headers = HEADERS)


def do_review_request(response_blob):
    business_id = response_blob['id']
    HEADERS = {'Authorization':'Bearer %s' %'t2POjvVZfc64zVMRRDEvhFA_ffRJpB_MJvk0oqiOcJvyBtu_42soOy-m6JQo0JSZyqESd56-bE41ZxXRv8qmSXs01Pb05hCU-UocJXlOLFytEpodpjFWNZWkypgoXnYx'}
    URL_reviews = 'https://api.yelp.com/v3/businesses/{business_id}/reviews'
    duplicate_locations = np.genfromtxt('static/duplicate_locations.csv',delimiter=',')

    yelp_review_request = requests.get(URL_reviews.format(business_id=business_id), headers = HEADERS)
    if yelp_review_request.status_code == 200:
        sum_review_length = 0.0
        count_of_reviews = 0
        compound_sentiment = 0.0
        analyzer = SentimentIntensityAnalyzer()
        for review in json.loads(yelp_review_request.text)['reviews']:
            text = review['text']
            vs = analyzer.polarity_scores(text.encode('utf-8'))
            compound_sentiment += vs['compound']
            sum_review_length += len(text)
            count_of_reviews += 1
        avg_review_length = float(sum_review_length) / count_of_reviews
        avg_compound_sentiment = float(compound_sentiment) / count_of_reviews
        
        alias = response_blob['alias']
        
        latitude = response_blob['coordinates']['latitude']
        longitude = response_blob['coordinates']['longitude']
        
        duplicate_location = 1 if ([ latitude, longitude ] == duplicate_locations).all(axis=1).any() else 0
        
        return ( avg_review_length, avg_compound_sentiment, alias, duplicate_location ) 
    else:
        print("Review request status code:", yelp_review_request.status_code)
        print("Error in review_input: ", yelp_review_request)
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
        print("Business info status code:", yelp_business_info_request.status_code)
        print("Error in business info input: ", yelp_business_info_request)
        return None

class LR_model():
   
    # TODO add decorators
    def __init__(self):
        self.weights = np.array([[  -7.99999848e-01,  8.21156318e-01, -7.40526739e-01,
                                    -3.00875588e-01, -2.64969982e-01,  5.03601493e-01,
                                    -6.88070682e+00,  1.79273077e-01,  5.07445374e-04,
                                    3.22390460e-04,  7.66727829e-01]])
        self.intercept = -0.89443562
        self.threshold = 0.5

    #@staticmethod
    #def convert_to_prob(x):
    #    return 1.0/(1 + np.exp(-x))

    def set_new_model_params(weights=None, intercept=None, threshold=None):
        if weights is not None:
            self.weights = weights
        if intercept is not None:
            self.intercept = intercept
        if threshold is not None:
            self.threshold = threshold

    def predict(self, features):
        print("Features:", features)

        log_odds = np.dot(self.weights, features) + self.intercept
        print("Log odds:", log_odds[0])

        model_prob = 1.0 / ( 1.0 + np.exp(-log_odds) ) #convert_to_prob(log_odds)
        print("Model_prob:", model_prob[0])

        model_output = True if model_prob > self.threshold else False
        print("Model_output:", model_output)

        return model_prob[0], model_output
