import numpy as np
import dgl
import torch
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
from input_features import build_features

"""
1. sample from user_df -> list of 64 user_id
2. sample pos & neg  (user_id, item_id) from review_df

"""

def get_posneg_samples(df_review, df_user, df_item, ukey, ikey, skey, datekey, \
                        user_seeds, ufeats, ifeats, cate_encoding_dict, flag='pos', score_thres=3.5):
    """
        param df: dataframe
        param ukey / ikey: user/item id key in df
        param skey: score key 
        param user_seeds: list of user ids
        param ufeats/ifeats: list of feature object. e.g. [CategoricalFeature]
        param score_thres: if score > thres,  -> postive samples
        
        Return
            tuple (user_features, item_features)
            where user_features is a dict (key: feature_name, value: batch_size vals) 
                  e.g. {'gender': [0,1,0]}
    """
    sub_mask = df_review[ukey].isin(user_seeds)
    sub_review_df = df_review[sub_mask]
    if flag=='pos':
        sub_review_df = sub_review_df[sub_review_df[skey] >= score_thres]
    else: 
        sub_review_df = sub_review_df[sub_review_df[skey] < score_thres]
    # for each user, select the single highest score item as the postive sample
    # 避免一些用户评价过多 有倾向性
    # highest score
    score_idx = sub_review_df.groupby([ukey])[skey].transform(max)== sub_review_df[skey]
    sub_review_df = sub_review_df[score_idx]
    # most recent and drop duplicate
    date_idx = sub_review_df.groupby([ukey])[datekey].transform(max)== sub_review_df[datekey]
    sub_review_df = sub_review_df[date_idx]
    sub_review_df = sub_review_df.drop_duplicates([ukey], keep='first')
    

    # sampled user & item ids 
    sub_user_ids = sub_review_df[ukey].tolist()
    sub_item_ids = sub_review_df[ikey].tolist()  # may contain duplicates

    # build user & item features
    sub_user_indices = [  df_user.loc[df_user[ukey] == user_id ].index[0] for user_id in sub_user_ids ]
    sub_user_df = df_user.iloc[sub_user_indices]
    user_features = build_features(sub_user_df, ufeats, cate_encoding_dict)
    
    # build sub item df from list of item_ids
    sub_item_indices = [  df_item.loc[df_item[ikey] == item_id ].index[0] for item_id in sub_item_ids ]
    sub_item_df = df_item.iloc[sub_item_indices]
    # sub_item_df = df_item[df_item[ikey].isin(sub_item_ids) ]
    item_features = build_features(sub_item_df, ifeats, cate_encoding_dict, sub_item_ids)


    return user_features, item_features

class SampleCollator(object):
    def __init__(self, df_review, df_user, df_item, ukey, ikey, skey, datekey, \
                    ufeats, ifeats, cate_encoding_dict, flag='pos', score_thres=3.5):
        self.df_review = df_review
        self.df_user = df_user
        self.df_item = df_item
        self.ukey = ukey
        self.ikey = ikey
        self.skey = skey
        self.datekey = datekey
        self.ufeats = ufeats
        self.ifeats = ifeats
        self.flag = flag
        self.score_thres = score_thres
        self.cate_encoding_dict = cate_encoding_dict
        #self.textset = textset

    def collate_samples(self, batches):
        user_seeds = batches
        user_features, item_features = get_posneg_samples(self.df_review, self.df_user,\
            self.df_item, self.ukey, self.ikey, self.skey, self.datekey, user_seeds,  \
                self.ufeats, self.ifeats, self.cate_encoding_dict, self.flag, self.score_thres)

        return [user_features, item_features]


    


