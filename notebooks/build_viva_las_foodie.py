import pandas as pd
import numpy as np
from datetime import date
import foodie_features
import geopy.distance
from scipy.spatial.distance import pdist, squareform
from dateutil.relativedelta import relativedelta

NOV_14_2018 = date(2018, 11, 14)

VEGAS_VISITORS_BY_YEAR = {'2018' : 42.12, '2017': 39.01, '2016': 42.94, '2015' : 42.31, '2014' : 41.13, \
                          '2013' : 39.67, '2012' : 39.73, '2011' : 38.93, '2010' : 37.34, '2009' : 36.35}
VEGAS_VISITORS_2018_BY_MONTH = {'1' : 3393900, '2' : 3130400, '3' : 3749800, '4'  : 3548000, '5'  : 3630400, '6'  : 3565400,\
                                '7' : 3659600, '8' : 3555200, '9' : 3457500, '10' : 3680600, '11' : 3478500, '12' : 3267600 }
VEGAS_POPULATION_BY_YEAR = {'2018' : 42.12, '2017': 39.01, '2016': 42.94, '2015' : 42.31, '2014' : 41.13,\
                            '2013' : 39.67, '2012' : 39.73, '2011' : 38.93, '2010' : 37.34, '2009' : 36.35}

def calculate_additional_features(businesses_df, reviews_df):
    
    open_restaurants_yelp_api_data_df = pd.read_pickle('saved_data/open_restaurants_yelp_api_data_df.pkl').rename(columns={'price':'cost'})
    open_restaurants_yelp_api_data_df['cost'] = open_restaurants_yelp_api_data_df['cost'].map(lambda x: len(x))
    businesses_df['latitude'].update(open_restaurants_yelp_api_data_df['latitude'])
    businesses_df['longitude'].update(open_restaurants_yelp_api_data_df['longitude'])
    
    # is restaurant claimed
    businesses_df['is_claimed'] = [False for i in range(businesses_df.shape[0])]
    businesses_df['is_claimed'].update(open_restaurants_yelp_api_data_df['is_claimed'])
    businesses_df['is_claimed'] = businesses_df['is_claimed'].apply(lambda x: 1 if x is True else 0)
    
    # compute actual review counts since the dataset is wrong
    businesses_df['actual_review_count'] = reviews_df['business_id'].value_counts()
    
    # compute actual star rating based on actual reviews since the dataset might be wrong and yelp doesn't compute precise ratings
    businesses_df['actual_stars'] = reviews_df.groupby('business_id')['stars'].mean()
    
    # is restaurant a chain (find if business name is not unique)
    chains = businesses_df[businesses_df.name.duplicated(keep=False)].sort_values(by='name').name.unique()
    businesses_df['is_chain'] = businesses_df.name.isin(chains)
    businesses_df['is_chain'] = businesses_df['is_chain'].apply(lambda x: 1 if x is True else 0)
    
    # check if location has multiple closures (TODO fix the fact that the first duplicate location shouldn't matter)
    duplicate_lat_filter = businesses_df.latitude.duplicated(keep=False)
    duplicate_long_filter = businesses_df.longitude.duplicated(keep=False)
    duplicate_locations_df = businesses_df[ (duplicate_lat_filter) & (duplicate_long_filter) ].sort_values(by='latitude')[['latitude','longitude']]
    duplicate_locations = businesses_df.isin(duplicate_locations_df)
    
    businesses_df['duplicate_location'] = duplicate_locations.latitude & duplicate_locations.longitude
    businesses_df['duplicate_location'] = businesses_df['duplicate_location'].apply(lambda x: 1 if x is True else 0)
        
    # add if business has parking
    parking_attrbs = [(key, value[u'BusinessParking']) \
                      if value is not None and 'BusinessParking' in value else (key, 'False') \
                      for key,value in businesses_df.attributes.iteritems()]
    has_parking_tuples = [(attrbs[0], 'True' in attrbs[1]) for attrbs in parking_attrbs]
    business_ids, has_parking = zip(*has_parking_tuples)
    has_parking_series = pd.Series(has_parking, business_ids)

    businesses_df['has_parking'] = has_parking_series
    businesses_df['has_parking'] = businesses_df['has_parking'].apply(lambda x: 1 if x is True else 0)
    
    # add in cost of restaurant (TODO fix the fact that some restaurants had incomplete data so have 0 under price)
    price_attrbs = [(key, int(value[u'RestaurantsPriceRange2'])) \
                    if value is not None and 'RestaurantsPriceRange2' in value and \
                    value[u'RestaurantsPriceRange2'] != u'None' else (key, 1) \
                    for key, value in businesses_df.attributes.iteritems() ]
    business_ids, cost_rating = zip(*price_attrbs)
    cost = pd.Series(cost_rating, business_ids)
    
    businesses_df['cost'] = cost
    businesses_df['cost'].update(open_restaurants_yelp_api_data_df['cost'])
    
    # proxy for when restaurant open/closed, make sure to include updated dates
    bus_date_df = reviews_df[reviews_df.business_id.duplicated(keep=False)].sort_values(by=['business_id','date'])
    open_dates = bus_date_df.drop_duplicates(subset=['business_id'],keep='first').set_index('business_id')
    closed_dates = bus_date_df.drop_duplicates(subset=['business_id'],keep='last').set_index('business_id')

    open_closed_dates_df = pd.DataFrame(data={'open_dates' : open_dates.date, 'closed_dates' : closed_dates.date, 'is_open' : businesses_df.is_open}, index=open_dates.index)
    open_closed_dates_df['open_dates'] = open_closed_dates_df['open_dates'].apply(lambda x: x.date())
    open_closed_dates_df['closed_dates'] = open_closed_dates_df['closed_dates'].apply(lambda x: x.date())

    businesses_df['open_dates'] = open_closed_dates_df.open_dates
    businesses_df['closed_dates'] = open_closed_dates_df.closed_dates
    
    businesses_df.loc[businesses_df['is_open'] == 1, 'days_since_closed'] = 0
    businesses_df.loc[businesses_df['is_open'] == 1, 'closed_dates'] = NOV_14_2018 # 2018-11-14 is the last day in this dataset
    
    closed_ages = NOV_14_2018 - pd.to_datetime(closed_dates.date, format='%Y%m%d').dt.date
    open_ages = businesses_df['closed_dates'] - pd.to_datetime(open_dates.date, format='%Y%m%d').dt.date
    
    businesses_df['age (in days)'] = open_ages.dt.days
    businesses_df['days_since_closed'] = closed_ages.dt.days
    
    # relative review count, rating, price

    return chains, duplicate_locations_df

def build_X_and_y(businesses_df, reviews_df, date, load_NLP=True, forecast_months=[1, 3, 6, 9], ignore_distance=False, do_distance=False, features=None):
    """
    businesses_df: dataframe of businesses 
    date: needs to be in a date object from the datetime library
    """
    date_str = date.strftime("%Y-%m-%d")
    
    before_date_filter = reviews_df.date <= pd.Timestamp(date)
    after_date_filter = reviews_df.date > pd.Timestamp(date)
        
    reviews_before_date_df = reviews_df[before_date_filter] # all reviews before date (i.e. they were open before date)
    reviews_after_date_df = reviews_df[after_date_filter] # all reviews after date (i.e. they were open after date)
    
    # restaurants with reviews both before and after date
    restaurant_ids = list( set(reviews_before_date_df.business_id.values) & set(reviews_after_date_df.business_id.values) )
    reviews_for_open_businesses = reviews_df[reviews_df.business_id.isin(restaurant_ids)]
    reviews_for_open_businesses_before_date = reviews_for_open_businesses[before_date_filter]
    
    # find number of other restaurants with same yelp categories -- TODO think about whether it should be at the time of date 
    business_category_counts = foodie_features.calculate_category_counts(businesses_df)
    
    if forecast_months is not None:
        # NLP score -- grab 3 most recent reviews since that's what allowed by yelp
        sentiment, avg_review_length = foodie_features.calculate_review_sentiment_and_length(reviews_for_open_businesses_before_date, date_str=date_str, load_NLP=load_NLP)
    
        # compute review count and rating before date
        review_count_on_date = reviews_for_open_businesses_before_date['business_id'].value_counts()
        rating_on_date = reviews_for_open_businesses_before_date.groupby('business_id')['stars'].mean()
        
        # compute smart rating (using if the reviews were useful)
        smart_rating_on_date = foodie_features.calculate_smart_ratings(reviews_for_open_businesses_before_date)
    else:
        # NLP score -- grab 3 most recent reviews since that's what allowed by yelp
        sentiment, avg_review_length = foodie_features.calculate_review_sentiment_and_length(reviews_df, date_str=date_str, load_NLP=load_NLP)
    
        # compute review count and rating before date
        review_count_on_date = businesses_df['actual_review_count']
        rating_on_date = businesses_df['actual_stars']
        
        # compute smart rating (using if the reviews were useful)
        smart_rating_on_date = foodie_features.calculate_smart_ratings(reviews_df)
    
    # get number of businesses within fixed distance
    if ignore_distance == False:
        count_of_businesses_within_a_tenth_mile = pd.Series()
        count_of_businesses_within_a_quarter_mile = pd.Series()
        count_of_businesses_within_a_half_mile = pd.Series()
        count_of_businesses_within_1_mile = pd.Series()
        count_of_businesses_within_5_miles = pd.Series()
        count_of_businesses_within_10_miles = pd.Series()
    
        for _, city_df in businesses_df.reindex(restaurant_ids).groupby('city'):
            if do_distance == True:        
                coords = city_df[['latitude','longitude']].values
                geopy_distance_func = lambda x,y: geopy.distance.distance(x,y).miles
                distances = pdist(coords, metric=geopy_distance_func)
                np.savetxt('saved_data/distances_all_%s_%s.csv'%(city_df.city.unique()[0], date_str), distances, delimiter=',')
            else:
                distances = np.genfromtxt('saved_data/distances_all_%s_%s.csv'%(city_df.city.unique()[0], date_str), delimiter=',')
            dist_matrix = squareform(distances)
            count_of_businesses_within_a_tenth_mile = count_of_businesses_within_a_tenth_mile.append( pd.Series( np.sum((dist_matrix < 0.1),axis=1) - 1, index = city_df.index ) )
            count_of_businesses_within_a_quarter_mile = count_of_businesses_within_a_quarter_mile.append ( pd.Series( np.sum((dist_matrix < 0.25),axis=1) - 1, index = city_df.index ) )
            count_of_businesses_within_a_half_mile = count_of_businesses_within_a_half_mile.append ( pd.Series( np.sum((dist_matrix < 0.5),axis=1) - 1, index = city_df.index ) )
            count_of_businesses_within_1_mile = count_of_businesses_within_1_mile.append ( pd.Series( np.sum((dist_matrix < 1),axis=1) - 1, index = city_df.index ) )
            count_of_businesses_within_5_miles = count_of_businesses_within_5_miles.append ( pd.Series( np.sum((dist_matrix < 5),axis=1) - 1, index = city_df.index ) )
            count_of_businesses_within_10_miles = count_of_businesses_within_10_miles.append ( pd.Series( np.sum((dist_matrix < 10),axis=1) - 1, index = city_df.index ) )
    
    data = businesses_df.copy()
    if forecast_months is not None:
        data = data.reindex(restaurant_ids)
    
    data['review_count_before_date'] = review_count_on_date
    data['rating_before_date'] = rating_on_date
    data['smart_rating_before_date'] = smart_rating_on_date
    data['age_at_date'] = (date - data['open_dates']).dt.days  
    data['sentiment'] = sentiment
    data['avg_review_length'] = avg_review_length
    data['business_category_counts'] = business_category_counts
    if ignore_distance == False:
        data['num_within_a_tenth_mile_at_date'] = count_of_businesses_within_a_tenth_mile
        data['num_within_a_quarter_mile_at_date'] = count_of_businesses_within_a_quarter_mile
        data['num_within_a_half_mile_at_date'] = count_of_businesses_within_a_half_mile
        data['num_within_1_mile_at_date'] = count_of_businesses_within_1_mile
        data['num_within_5_miles_at_date'] = count_of_businesses_within_5_miles
        data['num_within_10_miles_at_date'] = count_of_businesses_within_10_miles
    data = data.join(pd.get_dummies(data['cost'],prefix = 'cost')).drop(['cost'], axis = 1 )
    data = data.join(pd.get_dummies(data['city'],prefix = 'city')).drop(['city'], axis = 1 )
    if forecast_months is None:
            data['is_open'] = businesses_df['is_open']#.replace({0:1, 1:0})
        
    if features is not None:
        data = data[features]
    
    #### TARGET VARIABLE -- Is a restaurant open 6 months after the input date ####
    if forecast_months is not None:
        for forecast_month in forecast_months:
            forecast_filter = reviews_df.date > pd.Timestamp(date + relativedelta(months=forecast_month))
            data['closed_forecast_%s_months'%forecast_month] = foodie_features.calculate_future_restaurant_closure(businesses_df, restaurant_ids, reviews_for_open_businesses, forecast_filter)

    return data