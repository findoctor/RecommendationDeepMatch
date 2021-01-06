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
    for epoch_id in range(args.num_epochs):
        model.train()
        batch_count = 0
        batch_loss = 0.0
        for batch_id in user_batch_ids:
            batch_count += 1
            pos_user_features, pos_item_features = get_posneg_samples(df_review, df_user, df_item, ukey, ikey, skey, datekey, \
                                            batch_id, user_feat_objects, item_feat_objects, cate_encoding_dict, 'pos', args.score_thres)
            neg_user_features, neg_item_features = get_posneg_samples(df_review, df_user, df_item, ukey, ikey, skey, datekey, \
                                            batch_id, user_feat_objects, item_feat_objects, cate_encoding_dict, 'neg', args.score_thres)
            
            
            pos_score = model(pos_user_features, pos_item_features).sum()
            neg_score = model(neg_user_features, neg_item_features).sum()
            loss = neg_score-pos_score
            opt.zero_grad()
            loss.backward()
            opt.step()
            batch_loss += loss.item()
            if batch_count % 100 == 0:
                print("Epoch {}, Loss: {} ".format(epoch_id+1, batch_loss/batch_count ))
                batch_count = 0
                batch_loss = 0.0
        

           

        # Evaluate
        """
        model.eval()
        with torch.no_grad():
            item_batches = torch.arange(g.number_of_nodes(item_ntype)).split(args.batch_size)
            h_item_batches = []
            for blocks in dataloader_test:
                for i in range(len(blocks)):
                    blocks[i] = blocks[i].to(device)

                h_item_batches.append(model.get_repr(blocks))
            h_item = torch.cat(h_item_batches, 0)

            print(evaluation.evaluate_nn(dataset, h_item, args.k, args.batch_size))
        """

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str)
    parser.add_argument('catogory_encoding_path', type=str)
    
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cpu')        # can also be "cuda:0"
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--batches_per_epoch', type=int, default=20000)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('-k', type=int, default=10)
    parser.add_argument('--score_thres', type=float, default=3.5)
    args = parser.parse_args()

    # Load dataset
    with open(args.dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    with open(args.catogory_encoding_path, 'rb') as f:
        cate_encodings = pickle.load(f)

    train(dataset,cate_encodings, args)
