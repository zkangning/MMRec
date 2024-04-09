# coding: utf-8
# @email: enoche.chow@gmail.com
r"""
VBPR -- Recommended version
################################################
Reference:
VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback -Ruining He, Julian McAuley. AAAI'16
"""
import numpy as np
import os
import torch
import torch.nn as nn

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss
from common.init import xavier_normal_initialization
import torch.nn.functional as F
from DiffRec.models.DNN import DNN
import DiffRec.models.gaussian_diffusion as gd


class VBPR_DIFF(GeneralRecommender):
    r"""BPR is a basic matrix factorization model that be trained in the pairwise way.
    """
    def __init__(self, config, dataloader):
        super(VBPR_DIFF, self).__init__(config, dataloader)

        # load parameters info
        self.u_embedding_size = self.i_embedding_size = config['embedding_size']
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalizaton

        # define layers and loss
        self.u_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_users, self.u_embedding_size * 2)))
        self.i_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_items, self.i_embedding_size)))
        if self.v_feat is not None and self.t_feat is not None:
            self.item_raw_features = torch.cat((self.t_feat, self.v_feat), -1)
        elif self.v_feat is not None:
            self.item_raw_features = self.v_feat
        else:
            self.item_raw_features = self.t_feat

        self.item_linear = nn.Linear(self.item_raw_features.shape[1], self.i_embedding_size)
        self.loss = BPRLoss()
        self.reg_loss = EmbLoss()

        self.projector = DNN([self.item_raw_features.shape[1], 1000], [1000, self.item_raw_features.shape[1]], emb_size=10)
        self.diffusion = gd.GaussianDiffusion(
            mean_type=gd.ModelMeanType.START_X,
            noise_schedule='linear-var',
            noise_scale=0.1,
            noise_min=0.0001,
            noise_max=0.02,
            steps=100,
            device=config['device']
        )
        self.diff_weight = 0.1
        # parameters initialization
        self.apply(xavier_normal_initialization)

    def get_user_embedding(self, user):
        r""" Get a batch of user embedding tensor according to input user's id.

        Args:
            user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of user, shape: [batch_size, embedding_size]
        """
        return self.u_embedding[user, :]

    def get_item_embedding(self, item):
        r""" Get a batch of item embedding tensor according to input item's id.

        Args:
            item (torch.LongTensor): The input tensor that contains item's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of item, shape: [batch_size, embedding_size]
        """
        return self.item_embedding[item, :]

    def forward(self, user, pos_item, neg_item, dropout=0.0):
        pos_item_embeddings = self.item_linear(self.diffusion.p_sample(self.projector, self.item_raw_features[pos_item], steps=1))
        pos_item_embeddings = torch.cat((self.i_embedding[pos_item], pos_item_embeddings), -1)

        neg_item_embeddings = self.item_linear(self.diffusion.p_sample(self.projector, self.item_raw_features[neg_item], steps=1))
        neg_item_embeddings = torch.cat((self.i_embedding[neg_item], neg_item_embeddings), -1)

        user_e = F.dropout(self.u_embedding[user], dropout)
        pos_item_e = F.dropout(pos_item_embeddings, dropout)
        neg_item_e = F.dropout(neg_item_embeddings, dropout)

        return user_e, pos_item_e, neg_item_e

    def calculate_loss(self, interaction):
        """
        loss on one batch
        :param interaction:
            batch data format: tensor(3, batch_size)
            [0]: user list; [1]: positive items; [2]: negative items
        :return:
        """
        user = interaction[0]
        pos_item = interaction[1]
        neg_item = interaction[2]

        user_e, pos_e, neg_e = self.forward(user, pos_item, neg_item)

        pos_item_score, neg_item_score = torch.mul(user_e, pos_e).sum(dim=1), torch.mul(user_e, neg_e).sum(dim=1)
        mf_loss = self.loss(pos_item_score, neg_item_score)
        reg_loss = self.reg_loss(user_e, pos_e, neg_e)
        diff_loss = self.diffusion.training_losses(self.projector, self.item_raw_features[pos_item], reweight=True)['loss'].mean() + \
                    self.diffusion.training_losses(self.projector, self.item_raw_features[neg_item], reweight=True)['loss'].mean()
        loss = mf_loss + self.reg_weight * reg_loss + self.diff_weight * diff_loss
        return loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_embeddings, item_embeddings = self.forward()
        user_e = user_embeddings[user, :]
        all_item_e = item_embeddings
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score
