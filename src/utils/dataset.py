# coding: utf-8
# @email: enoche.chow@gmail.com
#
# updated: Mar. 25, 2022
# Filled non-existing raw features with non-zero after encoded from encoders

"""
Data pre-processing
##########################
"""
from logging import getLogger
from collections import Counter
import os
import pandas as pd
import numpy as np
import torch
from utils.data_utils import (ImageResize, ImagePad, image_to_tensor, load_decompress_img_from_lmdb_value)
import lmdb
import torch.nn.functional as F

class RecDataset(object):
    def __init__(self, config, df=None):
        self.config = config
        self.logger = getLogger()

        # data path & files
        self.dataset_name = config['dataset']
        self.dataset_path = os.path.abspath(config['data_path']+self.dataset_name)

        # dataframe
        self.uid_field = self.config['USER_ID_FIELD']
        self.iid_field = self.config['ITEM_ID_FIELD']
        self.splitting_label = self.config['inter_splitting_label']

        if df is not None:
            self.df = df
            return
        # if all files exists
        check_file_list = [self.config['inter_file_name']]
        for i in check_file_list:
            file_path = os.path.join(self.dataset_path, i)
            if not os.path.isfile(file_path):
                raise ValueError('File {} not exist'.format(file_path))

        # load rating file from data path?
        self.load_inter_graph(config['inter_file_name'])
        self.item_num = int(max(self.df[self.iid_field].values)) + 1
        self.user_num = int(max(self.df[self.uid_field].values)) + 1

        if config['modal_augment']:
            self.load_modal_feat(config)
            self.modal_augment(config)
        if config['modal_denoise']:
            self.load_modal_feat(config)
            self.modal_denoise(config)

    def load_inter_graph(self, file_name):
        inter_file = os.path.join(self.dataset_path, file_name)
        cols = [self.uid_field, self.iid_field, self.splitting_label]
        self.df = pd.read_csv(inter_file, usecols=cols, sep=self.config['field_separator'])
        if not self.df.columns.isin(cols).all():
            raise ValueError('File {} lost some required columns.'.format(inter_file))

    def split(self):
        dfs = []
        # splitting into training/validation/test
        for i in range(3):
            temp_df = self.df[self.df[self.splitting_label] == i].copy()
            temp_df.drop(self.splitting_label, inplace=True, axis=1)        # no use again
            dfs.append(temp_df)
        if self.config['filter_out_cod_start_users']:
            # filtering out new users in val/test sets
            train_u = set(dfs[0][self.uid_field].values)
            for i in [1, 2]:
                dropped_inter = pd.Series(True, index=dfs[i].index)
                dropped_inter ^= dfs[i][self.uid_field].isin(train_u)
                dfs[i].drop(dfs[i].index[dropped_inter], inplace=True)

        # wrap as RecDataset
        full_ds = [self.copy(_) for _ in dfs]
        return full_ds

    def load_modal_feat(self, config):
                # load modal features
        v_feat_file_path = os.path.join(self.dataset_path, config['vision_feature_file'])
        t_feat_file_path = os.path.join(self.dataset_path, config['text_feature_file'])
        if os.path.isfile(v_feat_file_path):
            self.v_feat = torch.from_numpy(np.load(v_feat_file_path, allow_pickle=True)).type(torch.FloatTensor).to(
                    config['device'])
        if os.path.isfile(t_feat_file_path):
            self.t_feat = torch.from_numpy(np.load(t_feat_file_path, allow_pickle=True)).type(torch.FloatTensor).to(
                    config['device'])
    
    def modal_augment(self, config):
        # augmenting the dataset with modal features
        self.logger.info('Augmenting the dataset with modal features...')

        if self.v_feat is None and self.t_feat is None:
            self.logger.warning('No modal features loaded.')
            return
        # 统计训练集df中每个用户交互的物品数以及交互的所有物品
        all_items_per_user = self.df.groupby(self.uid_field)[self.iid_field].apply(list).to_dict()

        # 将每个用户交互的所有物品的t_feat进行平均，作为每个用户的多模态特征
        user_v_feat = {}
        user_t_feat = {}
        # aug_df = pd.DataFrame(columns=[self.uid_field, self.iid_field, self.splitting_label])
        for user, items in all_items_per_user.items():
            if len(all_items_per_user[user]) <= config['augment_threshold']:
                user_v_feat[user] = self.v_feat[items].mean(dim=0)
                user_t_feat[user] = self.t_feat[items].mean(dim=0)
        # 利用每个用户的多模态特征去召回cosine最相近的k个item id, 同时这k个item 不包括在原始训练交互数据中
                user_v = user_v_feat[user].unsqueeze(0)
                user_t = user_t_feat[user].unsqueeze(0)
                # 计算每个物品与用户的cosine相似度
                v_sim = F.cosine_similarity(self.v_feat, user_v, dim=1)
                t_sim = F.cosine_similarity(self.t_feat, user_t, dim=1)
                # 对两个模态的相似度进行加权
                sim = v_sim * config['v_weight'] + t_sim * config['t_weight']
                k = int(config['modal_augment_num'])
                # 取出最相近的k个物品, 但应排除掉原本存在于训练集中的物品
                topk_item = sim.argsort(descending=True)
                mask = torch.tensor([i not in all_items_per_user[user] for i in range(len(topk_item))]).to(config['device'])
                topk_item = topk_item[mask][:k].cpu().numpy()
                # 将topk_item 里面的每个item 拼成一个user-item 对， 加入到self.df 中
                self.df = pd.concat([self.df, pd.DataFrame({self.uid_field: [user] * k, self.iid_field: topk_item, self.splitting_label: [0] * k})], ignore_index=True)

        self.df = self.df.sort_values(by=[self.uid_field], ascending=True).reset_index(drop=True)
        
        self.logger.info('Augmenting the dataset with modal features done.')

    def modal_denoise(self, config):
        # denoising the dataset with modal features
        self.logger.info('Denoising the dataset with modal features...')

        if self.v_feat is None and self.t_feat is None:
            self.logger.warning('No modal features loaded.')
            return
        # 统计训练集df中每个用户交互的物品数以及交互的所有物品
        all_items_per_user = self.df.groupby(self.uid_field)[self.iid_field].apply(list).to_dict()

        # 将每个用户交互的所有物品的t_feat进行平均，作为每个用户的多模态特征
        user_v_feat = {}
        user_t_feat = {}
        for user, items in all_items_per_user.items():
            if len(all_items_per_user[user]) >= config['denoise_threshold']:   # 当交互数过多时，可能存在噪声, 因此使用多模态特征进行降噪
                user_v_feat[user] = self.v_feat[items].mean(dim=0)
                user_t_feat[user] = self.t_feat[items].mean(dim=0)
                user_v = user_v_feat[user].unsqueeze(0)
                user_t = user_t_feat[user].unsqueeze(0) 
                v_sim = F.cosine_similarity(self.v_feat[all_items_per_user[user]], user_v, dim=1)
                t_sim = F.cosine_similarity(self.t_feat[all_items_per_user[user]], user_t, dim=1)
                sim = v_sim * config['v_weight'] + t_sim * config['t_weight']
                k = int(config['modal_denoise_ratio'] * len(all_items_per_user[user]))
                # 建立一个原本物品id 和 sim相似度的映射
                sim = pd.Series(sim.cpu().numpy(), index=all_items_per_user[user])
                # 取出语义最不相近的 k 个 物品
                topk_item = sim.argsort(kind='stable')[:k]   # 最不相近的k个物品id
                # 从原始的训练集中删除掉这 k 个 user-item 
                self.df = self.df[~((self.df[self.uid_field] == user) & (self.df[self.iid_field].isin(topk_item)))]    

        self.logger.info('Denoising the dataset with modal features done.')

    def copy(self, new_df):
        """Given a new interaction feature, return a new :class:`Dataset` object,
                whose interaction feature is updated with ``new_df``, and all the other attributes the same.

                Args:
                    new_df (pandas.DataFrame): The new interaction feature need to be updated.

                Returns:
                    :class:`~Dataset`: the new :class:`~Dataset` object, whose interaction feature has been updated.
                """
        nxt = RecDataset(self.config, new_df)

        nxt.item_num = self.item_num
        nxt.user_num = self.user_num
        return nxt

    def get_user_num(self):
        return self.user_num

    def get_item_num(self):
        return self.item_num

    def shuffle(self):
        """Shuffle the interaction records inplace.
        """
        self.df = self.df.sample(frac=1, replace=False).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Series result
        return self.df.iloc[idx]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        info = [self.dataset_name]
        self.inter_num = len(self.df)
        uni_u = pd.unique(self.df[self.uid_field])
        uni_i = pd.unique(self.df[self.iid_field])
        tmp_user_num, tmp_item_num = 0, 0
        if self.uid_field:
            tmp_user_num = len(uni_u)
            avg_actions_of_users = self.inter_num/tmp_user_num
            info.extend(['The number of users: {}'.format(tmp_user_num),
                         'Average actions of users: {}'.format(avg_actions_of_users)])
        if self.iid_field:
            tmp_item_num = len(uni_i)
            avg_actions_of_items = self.inter_num/tmp_item_num
            info.extend(['The number of items: {}'.format(tmp_item_num),
                         'Average actions of items: {}'.format(avg_actions_of_items)])
        info.append('The number of inters: {}'.format(self.inter_num))
        if self.uid_field and self.iid_field:
            sparsity = 1 - self.inter_num / tmp_user_num / tmp_item_num
            info.append('The sparsity of the dataset: {}%'.format(sparsity * 100))
        return '\n'.join(info)
    

