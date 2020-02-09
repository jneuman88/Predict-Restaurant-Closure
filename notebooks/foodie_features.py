import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def calculate_category_counts(df):
    
    raw_categories = [category for category in df.categories.unique() if category is not None]
    categories = np.unique([x[1:] if x[0] == ' ' else x for category in raw_categories for x in category.split(',') ])
    
    category_counts = []
    cleaned_categories = []
    for category in categories:
        if category != u'Restaurants': # ignore this category since every restaurant has it
            cleaned_categories.append ( category )
            category_counts.append( df[df.categories.str.contains(category)].shape[0] )
    category_totals_df = pd.DataFrame(data = {'counts' : category_counts}, index = cleaned_categories )
    
    category_counts_df = pd.DataFrame(data={'cat_counts' : np.zeros(df.shape[0])}, index = df.index)
    for row in category_totals_df.itertuples():
        category_counts_df.cat_counts = category_counts_df.cat_counts + df.categories.apply(lambda x: row.counts if x.find(row.Index) > -1 else 0)
    
    return category_counts_df

def calculate_smart_ratings(df):
    def mul_two_cols(df):
        return (df['useful'] * df['stars']).sum()

    valid_reviews_groupby_id = df.groupby('business_id')
    smart_rating_on_date = valid_reviews_groupby_id.apply( mul_two_cols ) / valid_reviews_groupby_id['useful'].sum()
    smart_rating_on_date = smart_rating_on_date.fillna(0)
    
    return smart_rating_on_date


def calculate_future_restaurant_closure(df, restaurant_ids, reviews_for_open_businesses, after_date_filter):
    reviews_after_date_df = reviews_for_open_businesses[after_date_filter]
    
    restaurants_open_after_date = reviews_after_date_df.business_id.unique()
    
    is_closed_after_date = [ (restaurant_id, 0) if restaurant_id in restaurants_open_after_date \
                             or df.loc[restaurant_id,'is_open'] == 1 else \
                             (restaurant_id, 1) for restaurant_id in restaurant_ids ]
    final_restaurant_ids, is_closed = zip(*is_closed_after_date)
    is_closed_after_date_ser = pd.Series(is_closed, index=final_restaurant_ids).reindex(restaurant_ids)

    return is_closed_after_date_ser

def calculate_review_sentiment_and_length(reviews_for_open_businesses_before_date, date_str, load_NLP=False):
    last_three_reviews_per_rest_df = reviews_for_open_businesses_before_date.sort_values(by='date', ascending=True).groupby('business_id').tail(3)
    
    if load_NLP == False:
        negative_sentiment = np.zeros(last_three_reviews_per_rest_df.shape[0])
        neutral_sentiment = np.zeros(last_three_reviews_per_rest_df.shape[0])
        positive_sentiment = np.zeros(last_three_reviews_per_rest_df.shape[0])
        compound_sentiment = np.zeros(last_three_reviews_per_rest_df.shape[0])
        review_length = np.zeros(last_three_reviews_per_rest_df.shape[0])
        
        analyzer = SentimentIntensityAnalyzer()
        for idx, row in enumerate(last_three_reviews_per_rest_df.itertuples()):
            vs = analyzer.polarity_scores(row.text.encode('utf-8'))
            negative_sentiment[idx] = vs['neg']
            neutral_sentiment[idx] = vs['neu']
            positive_sentiment[idx] = vs['pos']
            compound_sentiment[idx] = vs['compound']
            review_length[idx] = len(row.text)
            
            if idx % 10000 == 0:
                print "Checkpoint :", idx
        np.savetxt('saved_data/NLP_negative_sentiment_%s_%s.csv'%(last_three_reviews_per_rest_df.shape[0], date_str), negative_sentiment, delimiter=',')
        np.savetxt('saved_data/NLP_neutral_sentiment_%s_%s.csv'%(last_three_reviews_per_rest_df.shape[0], date_str), neutral_sentiment, delimiter=',')
        np.savetxt('saved_data/NLP_positive_sentiment_%s_%s.csv'%(last_three_reviews_per_rest_df.shape[0], date_str), positive_sentiment, delimiter=',')
        np.savetxt('saved_data/NLP_compound_sentiment_%s_%s.csv'%(last_three_reviews_per_rest_df.shape[0], date_str), compound_sentiment, delimiter=',')
        np.savetxt('saved_data/Avg_review_length_%s_%s.csv'%(last_three_reviews_per_rest_df.shape[0], date_str), review_length, delimiter=',')
    else:
        negative_sentiment = np.genfromtxt('saved_data/NLP_negative_sentiment_%s_%s.csv'%(last_three_reviews_per_rest_df.shape[0], date_str), delimiter=',')
        neutral_sentiment  = np.genfromtxt('saved_data/NLP_neutral_sentiment_%s_%s.csv'%(last_three_reviews_per_rest_df.shape[0], date_str), delimiter=',')
        positive_sentiment = np.genfromtxt('saved_data/NLP_positive_sentiment_%s_%s.csv'%(last_three_reviews_per_rest_df.shape[0], date_str), delimiter=',')
        compound_sentiment = np.genfromtxt('saved_data/NLP_compound_sentiment_%s_%s.csv'%(last_three_reviews_per_rest_df.shape[0], date_str), delimiter=',')
        review_length = np.genfromtxt('saved_data/Avg_review_length_%s_%s.csv'%(last_three_reviews_per_rest_df.shape[0], date_str), delimiter=',')
        
    reviews_sentiment_df = pd.DataFrame(data = {'negative_sentiment' : negative_sentiment, \
                                                'neutral_sentiment' : neutral_sentiment, \
                                                'positive_sentiment' : positive_sentiment, \
                                                'compound_sentiment' : compound_sentiment, \
                                                'review_length'      : review_length}, 
                                        index = last_three_reviews_per_rest_df.business_id )
    
    # for now we'll only use the compound sentiment
    review_stats = reviews_sentiment_df.groupby('business_id').mean()[['compound_sentiment','review_length']]

    sentiment = review_stats['compound_sentiment']
    avg_review_len = review_stats['review_length']
    
    
    return sentiment, avg_review_len