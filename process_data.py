import os
import re
import argparse
import pickle
import pandas as pd
import numpy as np
import json
import scipy.sparse as ssp
import torch
import torchtext
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction import FeatureHasher
from input_features import CategoricalFeature, MultiCategoricalFeature
from utils import train_test_split_by_time
user_data_path = 'data/yelp_academic_dataset_user.json'
review_data_path = 'data/yelp_academic_dataset_review.json'
item_data_path = 'data/yelp_academic_dataset_business.json'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('encoder_path', type=str)
    parser.add_argument('catogory_encoder_path', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('is_feature_hasher', type=bool, default=False)
    parser.add_argument('hash_dim', type=int, default=64)
    args = parser.parse_args()
    catogory_encoder_path = args.catogory_encoder_path
    sklearn_encoder_path = args.encoder_path
    output_path = args.output_path
    hash_dim = args.hash_dim
    is_feature_hasher = args.is_feature_hasher

   
    # =========================================================
    #               Load user item review into dataframe
    # ========================================================= 
    user_column_names = ["user_id", "review_count", "average_stars"] 
    items_info = [] 
    with open(user_data_path, 'r') as fh:
        for single_item in fh.readlines():
            item_dict = json.loads(single_item)
            data_dict = {
                        "user_id": item_dict["user_id"], \
                        "review_count":int(item_dict["review_count"]), \
                        "average_stars": float(item_dict["average_stars"])
                        }
            items_info.append(data_dict)
                
    user_df = pd.DataFrame(items_info).astype({'user_id': 'category'})
    # Discrete Bin
    user_df['review_count_interval'] = pd.qcut(user_df.review_count, q=10, duplicates='drop')
    user_id2reviewCountInterval = dict( zip( user_df['review_count_interval'].cat.codes, user_df['review_count_interval'] ) )
    user_reviewCountInterval2id = dict( zip( user_df['review_count_interval'] , user_df['review_count_interval'].cat.codes ) )
    
    
    # item data
    exclude_genre_column_names = ["business_id", "name", "city", "stars", \
               "review_count", "is_open", "categories"]
    items_info = [] 
    with open(item_data_path, 'r') as fh:
        for single_item in fh.readlines():
            item_dict = json.loads(single_item)
            if not item_dict["categories"]:
                cat_set = set(['others'])
            else:
                cat_set = set([item.strip() for item in item_dict["categories"].split(',') ])
            data_dict = {"business_id":item_dict["business_id"], \
                        "name": item_dict["name"], 
                        "city":item_dict["city"], \
                        "stars": float(item_dict["stars"]), \
                        "review_count": int(item_dict["review_count"]), \
                        "is_open": int(item_dict["is_open"]), \
                        "categories": ','.join(item for item in cat_set )
                        }
            # encode genres
            #for genre in cat_set:
                #data_dict[genre] = True
            items_info.append(data_dict)
                
    item_df = pd.DataFrame(items_info).astype({'business_id': 'category', \
                                        'city': 'category', \
                                        'is_open': 'category'})
    # bin and discrete
    item_df['review_count_interval'] = pd.qcut(item_df.review_count, q=10, duplicates='drop')
    item_id2reviewCountInterval = dict( zip( item_df['review_count_interval'].cat.codes, item_df['review_count_interval'] ) )
    item_reviewCountInterval2id = dict( zip( item_df['review_count_interval'] , item_df['review_count_interval'].cat.codes ) )
    item_city_id2name = dict( zip( item_df['city'].cat.codes, item_df['city'] ) )  # 0 for other city
    item_city_name2id = dict( zip(  item_df['city'], item_df['city'].cat.codes ) )


    # Review data
    review_column_names = ["business_id", "user_id", "stars", "date"]             
    items_info = [] 
    with open(review_data_path, 'r') as fh:
        for single_item in fh.readlines():
            item_dict = json.loads(single_item)
            date_str = item_dict["date"].split(' ')[0]  # YYYY-MM-DD
            date_int = int(date_str.replace('-','') )
            data_dict = {
                        "user_id": item_dict["user_id"], \
                        "business_id":item_dict["business_id"], \
                        "stars": int(item_dict["stars"]), \
                        "date": date_int
                        }
            items_info.append(data_dict)
                
    review_df = pd.DataFrame(items_info).astype({'business_id': 'category', \
                                        'user_id': 'category'
                                        })

    
    # Filter the users and items that never appear in the rating table.
    distinct_users_in_ratings = review_df['user_id'].unique()
    distinct_items_in_ratings = review_df['business_id'].unique()
    user_df = user_df[user_df['user_id'].isin(distinct_users_in_ratings)]
    item_df = item_df[item_df['business_id'].isin(distinct_items_in_ratings)]

    # deal with high cardinarity feature : category encoding
    # TODO: PCA, linear transformation, mean encoder
    all_item_categories = set([ item for cat_str in item_df['categories'].values for item in cat_str.split(',')])
    all_item_categories.add('others')
    mlb = MultiLabelBinarizer()
    mlb.fit([all_item_categories])
    encodings = mlb.transform( [ cat_str.split(',') for cat_str in item_df['categories'].values ] )
    # store mlb for inference
    with open(sklearn_encoder_path, 'wb') as f:
        pickle.dump(mlb, f)
    if is_feature_hasher:
        print('feature hasing ...')
        fea_hasher = FeatureHasher(n_features=hash_dim)
        # wrap 'encodings' into dict
        all_categories = list(mlb.classes_)
        encode_dict_list = [ dict(zip(all_categories, list(instance_encoding)))  for instance_encoding in encodings] 
        encodings = fea_hasher.transform(encode_dict_list).toarray()
    cate_encoding_dict = dict(zip(item_df['business_id'].values, encodings))
    # store for inference
    with open(catogory_encoder_path, 'wb') as f:
        pickle.dump(cate_encoding_dict, f)

    # create feature objects
    print('Creating feature objects ...')
    user_feat_columns = ["review_count_interval", "average_stars"] 
    item_feat_colums = ["city", "stars", "review_count_interval", "is_open", "categories"]
    user_feat_objects = []
    item_feat_objects = []
    user_feat_objects.append(CategoricalFeature("review_count_interval",10,16) )
    user_feat_objects.append(MultiCategoricalFeature("average_stars",1,64) )

    item_feat_objects.append(CategoricalFeature("city",len(item_city_id2name),16) )
    item_feat_objects.append(MultiCategoricalFeature("stars",1,64) )
    item_feat_objects.append(CategoricalFeature("review_count_interval",10,16) )
    item_feat_objects.append(CategoricalFeature("is_open",2,32) )
    if is_feature_hasher:
        item_feat_objects.append(MultiCategoricalFeature("categories",hash_dim,64) )
    else: 
        item_feat_objects.append(MultiCategoricalFeature("categories",len(mlb.classes_),64) )
    
    
    # Train-validation-test split
    #print('split train-test-val ...')
    #train_indices, val_indices, test_indices = train_test_split_by_time(review_df, 'date', 'user_id')

    ## Dump the graph and the datasets

    dataset = {
        'user_df': user_df,
        'item-df': item_df,
        'review_df': review_df,
        'user_feat_obj': user_feat_objects,
        'item_feat_obj': item_feat_objects,
            }

    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)
