import numpy as np
import faiss
import torch
import torch.nn as nn
import pandas as pd 
import pickle
import logging

class evaluate_DSSM(object):
    def __init__(self, feat_dim, user_ids, user_embedding_path, item_embedding_path, k, args):
        """
        param user_ids: ids of test users
        param user_embedding_path: path that storing trained user embedding, dict{key is user id, val is embedding}
        param item_embedding_path: path that storing trained item embedding
        
        """
        self.user_embedding_path = user_embedding_path
        self.item_embedding_path = item_embedding_path
        self.user_ids = user_ids
        self.args = args
        self.k = k
        self.feat_dim = feat_dim
    
    # Load embeddings
    def get_user_embeddings(self):
        user_embeddings = {}
        with open(self.user_embedding_path, 'rb') as f:
            all_user_embeddings = pickle.load(f)
        for uid in self.user_ids:
            if uid not in all_user_embeddings.keys():
                logging.error('User {}, embedding not found!'.format(uid))
            user_embeddings[uid] = all_user_embeddings[uid]
        return  user_embeddings
    def get_item_embeddings(self):
        with open(self.item_embedding_path, 'rb') as f:
            all_item_embeddings = pickle.load(f)
        return all_item_embeddings
    
    def get_topk_item(self, user_emb_dict, item_emb_dict):
        k = self.k  # recommend k items
        feat_dim = self.feat_dim
        # convert into np arrays and record id-index mapping
        user_embeddings = np.array([emb for emb in user_emb_dict.values()]).astype('float32')
        item_embeddings = np.array([emb for emb in item_emb_dict.values()]).astype('float32')
        item_index2id = { _k:_v for _k, _v in enumerate(item_emb_dict.keys())}
        user_index2id = { _k:_v for _k, _v in enumerate(user_emb_dict.keys())}
        hash_bucket_len = 16
        assert feat_dim%hash_bucket_len ==  0
        hash_bucket_num = feat_dim/hash_bucket_len
        cluster_num = len(item_emb_dict)/100  # 100 samples each cluster
        quantizer = faiss.IndexFlatL2(feat_dim)
        index = faiss.IndexIVFPQ(quantizer, feat_dim, cluster_num, hash_bucket_num, 8)
        index.train(item_embeddings)
        index.add(item_embeddings)
        dis, ind = index.search(user_embeddings, k)
        # ind: [n_user, k]
        topk_items = []
        for _indices in ind:
            recmd_item_ids = [item_index2id[_index] for _index in _indices]
            topk_items.append(recmd_item_ids)
        return topk_items  # [n_user, k]

    def get_accuracy(self, df_review, utype, itype, score_type):
        # 在yelp数据集中，如果score超过了一定阈值 就认为是正样本 选出评分最高的k个
        sub_df_review = df_review[df_review[utype].isin(self.user_ids) ]
        sub_df_review = sub_df_review[sub_df_review[score_type]>self.args.score_thres]
        sub_df_review = sub_df_review.groupby([utype]).apply(lambda x: x.nlargest(self.k,[score_type])).reset_index(drop=True)
        gt_items = []
        for user_id in self.user_ids:
            gt_items.append( sub_df_review[ sub_df_review[utype]==user_id ][itype].values.tolist()  )
        # get predictions
        user_embeddig_dict = self.get_user_embeddings()
        item_embedding_dict = self.get_item_embeddings()
        topk_items = self.get_topk_item(user_embeddig_dict, item_embedding_dict)
        # compare predictions to groud truth
        hit = 0.0
        num_samples = 0.0
        for gt_list, pd_list in zip(gt_items, topk_items):
            num_samples += len(gt_list)
            hit += sum(x == y for x, y in zip(gt_list, pd_list))
        if num_samples == 0:
            return 0.0
        else:
            return hit/num_samples 
