import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchtext
import tqdm
from sampler import get_posneg_samples, SampleCollator
from models.DSSM import DSSM
from input_features import build_features
from evaluation import evaluate_DSSM

def divide_chunks(l, n): 
    # looping till length l 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 


def train(dataset, cate_encoding_dict, args):
    df_user = dataset['user_df']
    df_item = dataset['item-df']
    df_review = dataset['review_df']
    user_feat_objects = dataset['user_feat_obj']
    item_feat_objects = dataset['item_feat_obj']
    ukey = 'user_id'
    ikey = 'business_id'
    skey = 'stars'
    datekey = 'date'
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Sampler
    pos_collator = SampleCollator(df_review, df_user, df_item, ukey, ikey, skey, datekey, \
                    user_feat_objects, item_feat_objects, cate_encoding_dict, 'pos', 3.5)
    neg_collator = SampleCollator(df_review, df_user, df_item, ukey, ikey, skey, datekey, \
                    user_feat_objects, item_feat_objects, cate_encoding_dict, 'neg', 3.5)
    
    dataloader_pos = DataLoader(
        df_user[ukey].values.tolist(),
        batch_size=args.batch_size,
        collate_fn=pos_collator.collate_samples,
        num_workers=args.num_workers)
    dataloader_pos_it = iter(dataloader_pos)

    dataloader_neg = DataLoader(
        df_user[ukey].values.tolist(),
        batch_size=args.batch_size,
        collate_fn=neg_collator.collate_samples,
        num_workers=args.num_workers)
    dataloader_neg_it = iter(dataloader_neg)

    # Model
    model = DSSM(user_feat_objects, item_feat_objects)
    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # For each batch of head-tail-negative triplets...
    model.to(device)

    user_batch_ids = list(divide_chunks(df_user[ukey].values.tolist(), args.batch_size)) 
    item_batch_ids = list(divide_chunks(df_item[ikey].values.tolist(), args.batch_size)) 
    for epoch_id in range(args.num_epochs):
        model.train()
        batch_count = 0
        batch_loss = 0.0
        for i, batch_id in enumerate(user_batch_ids):
            pos_user_features, pos_item_features, n_pos_samples = get_posneg_samples(df_review, df_user, df_item, ukey, ikey, skey, datekey, \
                                            batch_id, user_feat_objects, item_feat_objects, cate_encoding_dict, 'pos', args.score_thres)
            neg_user_features, neg_item_features, n_neg_samples = get_posneg_samples(df_review, df_user, df_item, ukey, ikey, skey, datekey, \
                                            batch_id, user_feat_objects, item_feat_objects, cate_encoding_dict, 'neg', args.score_thres)
            
            if n_pos_samples > 0:
                pos_target = torch.ones(n_pos_samples)
                pos_loss = model(pos_user_features, pos_item_features, pos_target)
                pos_loss.backward()
                opt.step()
                batch_loss += pos_loss.item()
                batch_count += n_pos_samples
            
            if n_neg_samples > 0:
                neg_target = torch.zeros(n_neg_samples)
                neg_loss = model(neg_user_features, neg_item_features, neg_target)
                opt.zero_grad()
                neg_loss.backward()
                opt.step()
                batch_loss += neg_loss.item()
                batch_count += n_neg_samples
            if i % 100 == 0 and batch_count > 0:
                print("Epoch {}, Loss: {} ".format(epoch_id+1, batch_loss/batch_count ))
                batch_count = 0
                batch_loss = 0.0

        # inference user embeddings and store them
        all_user_embeddings = {}
        for batch_id in user_batch_ids:
            sub_df_user = df_user[ df_user[ukey].isin(batch_id) ]
            user_feature_dict = build_features(sub_df_user, user_feat_objects, cate_encoding_dict, item_ids=None)
            user_feature_output = model.inference_embedding(user_feature_dict, 'user').numpy()
            assert len(batch_id) == user_feature_output.shape[0]
            for u_id, u_emb in zip(batch_id, user_feature_output):
                all_user_embeddings[u_id] = u_emb
        
        # inference item embeddings and store them
        all_item_embeddings = {}
        for batch_id in item_batch_ids:
            sub_df_item = df_item[ df_item[ikey].isin(batch_id) ]
            item_feature_dict = build_features(sub_df_item, item_feat_objects, cate_encoding_dict, item_ids=batch_id)
            item_feature_output = model.inference_embedding(item_feature_dict, 'item').numpy()
            assert len(batch_id) == item_feature_output.shape[0]
            for _id, _emb in zip(batch_id, item_feature_output):
                all_item_embeddings[_id] = _emb
        with open(args.user_embedding_path, 'wb') as f:
            pickle.dump(all_user_embeddings, f)
        with open(args.item_embedding_path, 'wb') as f:
            pickle.dump(all_item_embeddings, f)

        # Evaluate
        model.eval()
        df_user_test = df_user.sample(n=args.num_test_samples)
        test_user_ids = df_user_test[ukey].values.tolist()
        with torch.no_grad():
            dnn_out_dim = model.dnn_out_dim  # user embedding dimensions 
            eva_class = evaluate_DSSM(dnn_out_dim, test_user_ids, \
                        args.user_embedding_path, args.item_embedding_path, args.k, args)
            print("Recommendation Acc = {}".format( \
                    eva_class.get_accuracy(df_review, ukey, ikey, skey) ))

 

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--catogory_encoding_path', type=str)
    
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_test_samples', type=int, default=1024)
    parser.add_argument('--device', type=str, default='cpu')        # can also be "cuda:0"
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--batches_per_epoch', type=int, default=20000)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('-k', type=int, default=10)
    parser.add_argument('--score_thres', type=float, default=3.5)
    parser.add_argument('--user_embedding_path', type=str, default='saved/user_embeddings.pkl')
    parser.add_argument('--item_embedding_path', type=str, default='saved/item_embeddings.pkl')
    args = parser.parse_args()

    # Load dataset
    with open(args.dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    with open(args.catogory_encoding_path, 'rb') as f:
        cate_encodings = pickle.load(f)

    train(dataset,cate_encodings, args)
