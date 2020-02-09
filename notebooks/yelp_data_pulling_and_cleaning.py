import pandas as pd
import numpy as np
import os
import glob

def pull_raw_business_data():
    return pd.read_json('yelp_reviews/business.json',lines=True)

def clean_business_data(businesses_df,
                        review_threshold=10,
                        required_num_of_closed_thresh_in_city=1000, 
                        city_filter_list=None,
                        type_of_business_list=None,
                        remove_hours=False):
    
    # capitalize first letter in city names
    businesses_df['city'] = businesses_df['city'].apply(lambda x: x.title())
    
    # remove cities which don't have enough closed businesses
    num_of_closed_businesses_by_city_df = businesses_df.groupby('city')['is_open'].count() - businesses_df.groupby('city')['is_open'].sum()
    filtered_out_cities = list(num_of_closed_businesses_by_city_df[num_of_closed_businesses_by_city_df > required_num_of_closed_thresh_in_city].index)
    cleaned_businesses_df = businesses_df[ businesses_df['city'].isin(filtered_out_cities) ]
    
    # remove business that don't have enough reviews 
    cleaned_businesses_df = cleaned_businesses_df[ cleaned_businesses_df['review_count'] >= review_threshold ]
    
    # filter only for a particular city if you so choose
    if city_filter_list is not None:
        if type(city_filter_list) is not list: 
            return "Please pass in the city filters as a list"
        else:
            cleaned_businesses_df = cleaned_businesses_df[ cleaned_businesses_df['city'].isin(city_filter_list) ]
        
    # filter only for a particular type of business if you so choose
    if type_of_business_list is not None:
        if type(type_of_business_list) is not list: 
            return "Please pass in the type of business filters as a list"
        else:
            for type_of_business in type_of_business_list:
                cleaned_businesses_df = cleaned_businesses_df[ cleaned_businesses_df['categories'].str.contains(type_of_business, na=False) ]
    
    # get all categories
    raw_categories = [category for category in cleaned_businesses_df.categories.unique() if category is not None]
    categories = np.unique([x[1:] if x[0] == ' ' else x for category in raw_categories for x in category.split(',') ])
    
    # drop useless columns
    cleaned_businesses_df.drop(['state','postal_code'],axis=1,inplace=True)
    if remove_hours is True:
        cleaned_businesses_df.drop(['hours'],axis=1, inplace=True)

    cleaned_businesses_df.set_index('business_id',inplace=True)
    
    return cleaned_businesses_df, categories

def clean_reviews_data(business_ids):
    yelp_review_path = os.path.abspath(os.getcwd()) + '/yelp_reviews/'
    review_files = glob.glob1(yelp_review_path, "split_reviews*")
    reviews_df = pd.DataFrame()
    for review_file in review_files:
        review_df = pd.read_json(yelp_review_path + review_file, lines=True)
        
        reviews_df = pd.concat([reviews_df, review_df[review_df['business_id'].isin(business_ids)]], ignore_index=True)
        
        del review_df
        
    return reviews_df

def clean_users_data(review_user_ids):
    yelp_review_path = os.path.abspath(os.getcwd()) + '/yelp_reviews/'
    user_files = glob.glob1(yelp_review_path, "split_users*")
    users_df = pd.DataFrame()
    for user_file in user_files:
        user_df = pd.read_json(yelp_review_path + user_file, lines=True)
    
        users_df = pd.concat([users_df, user_df[user_df['user_id'].isin(review_user_ids)]], ignore_index=True)
        
        del user_df
        
    return users_df