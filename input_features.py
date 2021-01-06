import torch
import torch.nn as nn
import numpy as np
from collections import namedtuple, defaultdict

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class CategoricalFeature(namedtuple('CategoricalFeature',
                            ['name', 'vocabulary_size', 'embedding_dim', 'use_hash', 'dtype',\
                              'embeddings_initializer',
                             ])):
    __slots__ = ()

    def __new__(cls, name, vocabulary_size, embedding_dim=64, use_hash=False, dtype="int32",\
                 embeddings_initializer=None):

        if embedding_dim == "auto":
            embedding_dim = 6 * int(pow(vocabulary_size, 0.25))
        return super(CategoricalFeature, cls).__new__(cls, name, vocabulary_size, embedding_dim, use_hash, dtype,
                                              embeddings_initializer)

    def __hash__(self):
        return self.name.__hash__()

# binary multi-hot encoding feature OR dense feature(e.g. score,count)
class MultiCategoricalFeature(namedtuple('MultiCategoricalFeature',
                            ['name', 'vector_dim', 'embedding_dim', 'use_hash', 'dtype',\
                              'embeddings_initializer',
                             ])):
    __slots__ = ()

    def __new__(cls, name, vocabulary_size, embedding_dim=64, use_hash=False, dtype="float32",\
                 embeddings_initializer=None):

        if embedding_dim == "auto":
            embedding_dim = 6 * int(pow(vocabulary_size, 0.25))
        
        return super(MultiCategoricalFeature, cls).__new__(cls, name, vocabulary_size, embedding_dim, use_hash, dtype,
                                              embeddings_initializer)

    def __hash__(self):
        return self.name.__hash__()

# TODO : Text feature using word embeddings

def build_features(df, feature_columns, cate_encoding_dict, item_ids=None):
    feat_dict = {}
    for feat in feature_columns:
        if isinstance(feat, CategoricalFeature):
            feat_dict[feat.name] = torch.LongTensor(df[feat.name].cat.codes.values)
        if isinstance(feat, MultiCategoricalFeature):
            if feat.name == "categories" and item_ids:
                cate_features = [cate_encoding_dict[_id] for _id in item_ids ]
                feat_dict[feat.name] = torch.FloatTensor(cate_features)
            else:
                feat_dict[feat.name] = torch.FloatTensor(df[feat.name].values).reshape((-1,1))
    return feat_dict 
            

def get_embedding_dict(feature_columns):
    #  user item的要分开两个dict 因为有common key
    """
    params:
        feature_columns: list of feature object(CategoricalFeature, MultiCategoricalFeature, ... )
    """

    embedding_dict = {}
    for feat in feature_columns:
        feat_name = feat.name
        if isinstance(feat, CategoricalFeature):
            emb_layer = nn.Embedding(feat.vocabulary_size, feat.embedding_dim)
            nn.init.xavier_uniform_(emb_layer.weight)
            embedding_dict[feat_name] = emb_layer
        if isinstance(feat, MultiCategoricalFeature):
            emb_layer = nn.Linear(feat.vector_dim, feat.embedding_dim)
            nn.init.xavier_uniform_(emb_layer.weight)
            embedding_dict[feat_name] = emb_layer
    return embedding_dict

def get_input_features(input_data, embedding_dict):
    """
    params:
        input_data: dict, key: feat_name  
    """
    input_feature  = []
    for _k, _v in input_data.items():
        embed_layer = embedding_dict[_k]
        feat_embedding = embed_layer(_v)   # [batch_size, hidden_dim]
        input_feature.append(feat_embedding)
    return torch.cat(input_feature, dim=1).to(device)
