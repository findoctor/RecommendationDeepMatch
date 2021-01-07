import os
import re
import argparse
import pickle
import pandas as pd
import numpy as np
import json
import scipy.sparse as ssp
import dgl
import torch
import torchtext
from builder import PandasGraphBuilder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction import FeatureHasher
from data_utils import *
user_data_path = 'data/yelp_academic_dataset_user.json'
review_data_path = 'data/yelp_academic_dataset_review.json'
item_data_path = 'data/yelp_academic_dataset_business.json'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_path', type=str)
    parser.add_argument('is_feature_hasher', type=bool, default=False)
    parser.add_argument('hash_dim', type=int, default=64)
    args = parser.parse_args()
    output_path = args.output_path
    hash_dim = args.hash_dim
    is_feature_hasher = args.is_feature_hasher

    ## Build heterogeneous graph

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

    exclude_genre_column_names = ["business_id", "name", "city", "stars", \
               "review_count", "is_open", "categories"]
    
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
                        "name": item_dict["name"], "city":item_dict["city"], \
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
    if is_feature_hasher:
        print('feature hasing ...')
        mlb = MultiLabelBinarizer()
        encodings = mlb.fit_transform( [ cat_str.split(',') for cat_str in item_df['categories'].values ] )
        fea_hasher = FeatureHasher(n_features=hash_dim)
        # wrap 'encodings' into dict
        all_categories = list(mlb.classes_)
        encode_dict_list = [ dict(zip(all_categories, list(instance_encoding)))  for instance_encoding in encodings] 
        hash_encodings = fea_hasher.transform(encode_dict_list).toarray()
    else:
        mlb = MultiLabelBinarizer()
        hash_encodings = mlb.fit_transform( [ cat_str.split(',') for cat_str in item_df['categories'].values ] )

    # Build graph
    print('building graph ...')
    graph_builder = PandasGraphBuilder()
    graph_builder.add_entities(user_df, 'user_id', 'user')
    graph_builder.add_entities(item_df, 'business_id', 'item')
    graph_builder.add_binary_relations(review_df, 'user_id', 'business_id', 'reviewed')
    graph_builder.add_binary_relations(review_df, 'business_id', 'user_id', 'reviewed-by')

    g = graph_builder.build()

    print('Assigning feature ...')
    # Assign features.
    g.nodes['user'].data['review_count'] = torch.FloatTensor(user_df['review_count'].values)
    g.nodes['user'].data['average_stars'] = torch.FloatTensor(user_df['average_stars'].values)

    g.nodes['item'].data['city'] = torch.LongTensor(item_df['city'].cat.codes.values)
    g.nodes['item'].data['is_open'] = torch.LongTensor(item_df['is_open'].cat.codes.values)
    g.nodes['item'].data['stars'] = torch.FloatTensor(item_df['stars'].values)
    g.nodes['item'].data['review_count'] = torch.FloatTensor(item_df['review_count'].values)
    g.nodes['item'].data['categories'] = torch.FloatTensor(hash_encodings)

    g.edges['reviewed'].data['rating'] = torch.LongTensor(review_df['stars'].values)
    g.edges['reviewed-by'].data['rating'] = torch.LongTensor(review_df['stars'].values)
    g.edges['reviewed'].data['date'] = torch.LongTensor(review_df['date'].values)
    g.edges['reviewed-by'].data['date'] = torch.LongTensor(review_df['date'].values)

    # Train-validation-test split
    # This is a little bit tricky as we want to select the last interaction for test, and the
    # second-to-last interaction for validation.
    print('split train-test-val ...')
    train_indices, val_indices, test_indices = train_test_split_by_time(review_df, 'date', 'user_id')

    # Build the graph with training interactions only.
    print('build train graph ... ')
    train_g = build_train_graph(g, train_indices, 'user', 'item', 'reviewed', 'reviewed-by')
    assert train_g.out_degrees(etype='reviewed').min() > 0

    # Build the user-item sparse matrix for validation and test set.
    print('build val matrix ... ')
    val_matrix, test_matrix = build_val_test_matrix(g, val_indices, test_indices, 'user', 'item', 'reviewed')

    ## Build title set

    # movie_textual_dataset = {'title': movies['title'].values}

    # The model should build their own vocabulary and process the texts.  Here is one example
    # of using torchtext to pad and numericalize a batch of strings.
    #     field = torchtext.data.Field(include_lengths=True, lower=True, batch_first=True)
    #     examples = [torchtext.data.Example.fromlist([t], [('title', title_field)]) for t in texts]
    #     titleset = torchtext.data.Dataset(examples, [('title', title_field)])
    #     field.build_vocab(titleset.title, vectors='fasttext.simple.300d')
    #     token_ids, lengths = field.process([examples[0].title, examples[1].title])

    ## Dump the graph and the datasets

    dataset = {
        'train-graph': train_g,
        'val-matrix': val_matrix,
        'test-matrix': test_matrix,
        #'item-texts': movie_textual_dataset,
        'item-images': None,
        'user-type': 'user',
        'item-type': 'item',
        'user-to-item-type': 'reviewed',
        'item-to-user-type': 'reviewed-by',
        'timestamp-edge-column': 'date'}

    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)
