from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import requests
import json
import tools

app = Flask(__name__)

@app.route("/")
def homepage():
    restaurant_names = [line[:-1] for line in open('static/VegasRestaurantNames.csv')] # TODO handle if file doesn't exit
    return render_template('homepage.html', restaurant_names=restaurant_names[15:25])

@app.route("/", methods=['POST'])
def homepage_post():
    
    model = tools.LR_model()

    if 'submit_button' in request.form:
        name = request.form['submit_button']

        with open('static/name_to_id_dict.json', 'r') as filepath:
            name_to_id_dict = json.load(filepath)
        
        business_id = name_to_id_dict[name]
   
        with open('static/id_to_features_dict.json', 'r') as filepath:
            id_to_features_dict = json.load(filepath)

        cols = ['is_chain', 'duplicate_location', 'cost_1', 'cost_2', 'cost_3', 'cost_4', 'is_claimed', 'sentiment', 'avg_review_length', 'review_count_before_date', 'rating_before_date'] 
        try:
            features = pd.DataFrame(data=id_to_features_dict[business_id], index = [business_id], columns = cols).values[0]
            print("features:", features)
            model_prob, model_output = model.predict(features)
            return render_template('cached_restaurant_result.html', name=name, model_prob=model_prob, model_output=model_output)
        except:
            # Restaurant is closed
            return render_template('closed_restaurant.html', name=name) # make this file
    else:
        restaurant_name = request.form['restaurant_name'] if 'restaurant_name' in request.form else ''
        restaurant_address = request.form['restaurant_address'] if 'restaurant_address' in request.form else ''
        
        # do something now with input i.e. query yelp with input, make sure to handle errors
        HEADERS = {'Authorization':'Bearer %s' %'t2POjvVZfc64zVMRRDEvhFA_ffRJpB_MJvk0oqiOcJvyBtu_42soOy-m6JQo0JSZyqESd56-bE41ZxXRv8qmSXs01Pb05hCU-UocJXlOLFytEpodpjFWNZWkypgoXnYx'}
        URL_business = 'https://api.yelp.com/v3/businesses/matches'
        PARAMS_business = {'name' : restaurant_name, 'address1' : restaurant_address, 'city' : 'Las Vegas', 'state' : 'NV', 'country' : 'US'}
        
        yelp_request = requests.get(URL_business, params = PARAMS_business, headers = HEADERS)
        if yelp_request.status_code == 200:
            try:
                response_blob = json.loads(yelp_request.text)['businesses'][0]
                business_id = response_blob['id']
                
                review_features = tools.do_review_request(response_blob)
                if review_features is None:
                    return render_template('error_in_review_input.html') # TODO make this html file
                else:
                    avg_review_length, avg_compound_sentiment, alias, duplicate_location = review_features
               
                business_info_features = tools.do_business_info_request(response_blob)
                if business_info_features is None:
                    return render_template('error_in_business_info_input.html') # TODO make this html file
                else:
                    is_closed, rating, review_count, cost, cost_1, cost_2, cost_3, cost_4, is_claimed = business_info_features
              
                if is_closed:
                    return render_template('closed_restaurant.html', name=name) # make this file

                chains = [line[:-1] for line in open('static/chains.csv')] # TODO handle if file doesn't exist
                is_chain = 1 if str(restaurant_name.encode('utf-8')) in chains else 0

                print("Is chain:", is_chain, type(str(restaurant_name.encode('utf-8'))))

                features = np.array( [ is_chain, duplicate_location, cost_1, cost_2, cost_3, cost_4, is_claimed, avg_compound_sentiment, avg_review_length, review_count, rating ] )
                
                model_prob, model_output = model.predict(features)

                link = "https://www.yelp.com/biz/{alias}".format(alias=alias)

                return render_template('input_restaurant_result.html',  name=restaurant_name, address=restaurant_address, link=link, is_closed=is_closed, is_chain=is_chain, \
                                                                        duplicate_location=duplicate_location, cost=cost, is_claimed=is_claimed, \
                                                                        avg_compound_sentiment=avg_compound_sentiment, avg_review_length=avg_review_length, review_count=review_count, \
                                                                        rating=rating, model_prob=model_prob, model_output=model_output)
            except: 
                print("Params: ", PARAMS_business)
                print("Status Code: ", yelp_request.status_code) 
                print(json.loads(yelp_request.text))
                return render_template('error_in_yelp_api_call.html')
        else:
            # TODO handle errors in input here
            print(yelp_request)
            return render_template('error_in_business_input.html') # TODO make this html file

if __name__ == "__main__":
    app.run(port=2988,debug=True)
