from input_features import CategoricalFeature, MultiCategoricalFeature, \
                            get_embedding_dict, get_input_features
from base_modules import DNN
import torch
import torch.nn as nn
from utils import *
import logging

# TODO: 对双塔两侧输出的Embedding进行L2标准化；

def build_DNN_module(feat_objects, dnn_dims, dnn_activation, l2_reg_dnn):
    input_dim=0
    for feat in feat_objects:
        input_dim += feat.embedding_dim
    return DNN(input_dim, dnn_dims, \
                dnn_activation, l2_reg_dnn)
    

class DSSM(nn.Module):
    def __init__(self, user_features, item_features, user_dnn_dims=[128, 64], item_dnn_dims=[128,64] ,\
                 dnn_activation='relu', l2_reg_dnn=0, dnn_use_bn=False, dropout=False):
        """
        param user_features(item_features): list of feature class object
        param l2_reg_dnn: L2 regularizer strength applied to DNN
        """
        super().__init__()
        self.user_features = user_features
        self.item_features = item_features
        self.user_dnn_dims = user_dnn_dims
        self.item_dnn_dims = item_dnn_dims
        self.dnn_activation = dnn_activation
        self.l2_reg_dnn = l2_reg_dnn
        self.user_DNN_module = build_DNN_module(user_features, user_dnn_dims, dnn_activation, l2_reg_dnn)
        self.item_DNN_module = build_DNN_module(item_features, item_dnn_dims, dnn_activation, l2_reg_dnn)
        # build embedding layers
        self.user_embedding_dict = get_embedding_dict(self.user_features)
        self.item_embedding_dict = get_embedding_dict(self.item_features)
        self.dnn_out_dim = None
        # BN, dropout
        if dropout:
            self.dropout_rate = dropout
    
    def inference_embedding(self, feature_dict, type):
        """
        param input_data: tuple(dict,dict), key: feature name; value: feature values(if batch mode)
        return 
        """
        if type == 'user':
            # user side feature
            user_dnn_input = get_input_features(feature_dict, self.user_embedding_dict)
            user_dnn_out = self.user_DNN_module(user_dnn_input)
            return user_dnn_out
        elif type == 'item':
            # item side feature
            item_dnn_input = get_input_features(feature_dict, self.item_embedding_dict)
            item_dnn_out = self.item_DNN_module(item_dnn_input)
            return item_dnn_out
        else:
            logging.error('Wrong type! should either be user or item')
            return None

    def forward(self, user_feature_dict, item_feature_dict, target):
        """
        param input_data: tuple(dict,dict), key: feature name; value: feature values(if batch mode)
        param target: [batch_size]
        """
        # user side feature
        user_dnn_input = get_input_features(user_feature_dict, self.user_embedding_dict)
        # item side feature
        item_dnn_input = get_input_features(item_feature_dict, self.item_embedding_dict)
        
        user_dnn_out = self.user_DNN_module(user_dnn_input)
        item_dnn_out = self.item_DNN_module(item_dnn_input)
        assert user_dnn_out.shape == item_dnn_out.shape
        self.dnn_out_dim = user_dnn_out.shape[1]
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        sim_score = cos(user_dnn_out, item_dnn_out)
        # feed to sigmoid, then compute log-loss
        sim_sigmoid = torch.sigmoid(sim_score)
        loss_func = nn.BCELoss()
        loss = loss_func(sim_sigmoid, target) 
        return loss  # scalar




   