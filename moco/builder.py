# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
import pdb
import torchvision.models as models

################################ MoCo-V3 ################################
class MoCo2(nn.Module):
    """
    Build a MoCo model with a base encoder, a momentum encoder, and two MLPs
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=256, mlp_dim=4096, T=1.0):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(MoCo, self).__init__()

        self.T = T

        # build encoders
        self.base_encoder = base_encoder(num_classes=mlp_dim)
        self.momentum_encoder = base_encoder(num_classes=mlp_dim)

        self._build_projector_and_predictor_mlps(dim, mlp_dim)

        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def contrastive_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        
        # gather all targets
        k = concat_all_gather(k)
        
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
                        
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

    def forward(self, x1, x2, m):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: moco momentum
        Output:
            loss
        """

        # compute features
        q1 = self.predictor(self.base_encoder(x1))
        q2 = self.predictor(self.base_encoder(x2))

        with torch.no_grad():  # no gradient
            self._update_momentum_encoder(m)  # update the momentum encoder

            # compute momentum features as targets
            k1 = self.momentum_encoder(x1)
            k2 = self.momentum_encoder(x2)

        return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)


class MoCo(nn.Module):
    """
    Build a MoCo model with a base encoder, a momentum encoder, and two MLPs.
    Adapted to support VGG-like architectures.
    """
    def __init__(self, base_encoder, dim=256, mlp_dim=4096, T=1.0):
        super(MoCo, self).__init__()
        self.T = T
        
        # build encoders
        self.base_encoder = base_encoder(num_classes=mlp_dim)
        self.momentum_encoder = base_encoder(num_classes=mlp_dim)
        
        self._build_projector_and_predictor_mlps(dim, mlp_dim)

        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)
            param_m.requires_grad = False

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))
            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        raise NotImplementedError

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def contrastive_loss(self, q, k):
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)

        k = concat_all_gather(k)
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]
        labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)
    

    def forward_train(self, x1, x2, m):

        q1 = self.predictor(self.base_encoder(x1))
        q2 = self.predictor(self.base_encoder(x2))
     
        with torch.no_grad():
            self._update_momentum_encoder(m)
            k1 = self.momentum_encoder(x1)
            k2 = self.momentum_encoder(x2)

        return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)
    
    def forward_test(self, x1, x2, m):

        q1 = self.base_encoder(x1)
        q2 = self.base_encoder(x2)
     
        return q1, q2

    def forward(self, x1, x2, m):
        if self.training:
            return self.forward_train(x1, x2, m)
        else:
            return self.forward_test(x1, x2, m)




################################ UniCLR--SimAffinity ################################
class UniCLR(nn.Module):
    """
    Build a UniCLR model with a base encoder, a momentum encoder, and two MLPs
    """
    def __init__(self, base_encoder, dim=256, mlp_dim=4096, T=1.0):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(UniCLR, self).__init__()

        self.T = T
        self.affinity_gamma = 0.01
        self.eps = 0.01

        # build encoders
        self.base_encoder = base_encoder(num_classes=mlp_dim)
        self.momentum_encoder = base_encoder(num_classes=mlp_dim)

        self._build_projector_and_predictor_mlps(dim, mlp_dim)

        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    
    def cal_affinity(self, z1, z2):
        N, D   = z1.shape
        device = z1.device
        
        z1    = nn.functional.normalize(z1, p=2, dim=-1)
        z2    = nn.functional.normalize(z2, p=2, dim=-1)
        z_cat = torch.cat([z1, z2], dim=0)
        
  
        mean   = z_cat.mean(dim=0, keepdim=True)
        z_cat_ = z_cat - mean               # N x C
                
        z_cat_np = z_cat_.detach().cpu().numpy()
        cov      = np.cov(z_cat_np.T)
        
        U, S, V = np.linalg.svd(cov)
        
        # zca_matrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + self.eps)), U.T))
        cov = np.dot(U, np.dot(np.diag(S + self.eps), U.T))
        zca_matrix = np.linalg.solve(np.eye(D), np.linalg.cholesky(cov))
        
        inv_sqrt = torch.from_numpy(zca_matrix).to(z1.device)
        
        decorrelated = torch.matmul(z_cat_, inv_sqrt.detach())
        decorrelated = torch.chunk(decorrelated, 2)
        z1 = decorrelated[0]
        z2 = decorrelated[1]
    
        z1 = nn.functional.normalize(z1, p=2, dim=-1)
        z2 = nn.functional.normalize(z2, p=2, dim=-1)
        
        affinity_matrix = torch.matmul(z1, z2.T)
        
        return affinity_matrix / self.T
    
    def cal_covariance(self, z):
        z_mean = torch.mean(z, 0, True)
        z = z - z_mean
        z_cova = torch.matmul(z.permute(1, 0), z)
        z_cova = torch.div(z_cova, z.size(0) - 1)
        z_cova = z_cova + 0.01*torch.eye(z_cova.size(1)).cuda()
        return z_mean, z_cova
    
    def simaffinity_loss(self, q, k):

        # normalize
        z1 = nn.functional.normalize(q, dim=1)
        z2 = nn.functional.normalize(k, dim=1)
        
        # calculate affinity matrix
        affinity_matrix = torch.matmul(z1, z2.T) / self.T

        # symmetric loss
        symmetric_loss = 0.0
        if self.affinity_gamma > 0.0:
            symmetric_loss = torch.norm(affinity_matrix - affinity_matrix.T)
            
        # CE loss
        labels = torch.arange(z1.size(0), device=z1.device, dtype=torch.long)
        ce_loss = nn.functional.cross_entropy(affinity_matrix, labels)
        
        return ce_loss + self.affinity_gamma * symmetric_loss
    

    def forward_train(self, x1, x2, m):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: moco momentum
        Output:
            loss
        """
        # compute features
        q1 = self.predictor(self.base_encoder(x1))
        q2 = self.predictor(self.base_encoder(x2))
        
        with torch.no_grad():  # no gradient
            self._update_momentum_encoder(m)  # update the momentum encoder

            # compute momentum features as targets
            k1 = self.momentum_encoder(x1)
            k2 = self.momentum_encoder(x2)

        return self.simaffinity_loss(q1, k2) + self.simaffinity_loss(q2, k1) 

    def forward_test(self, x1, x2, m):

        q1 = self.base_encoder(x1)
        q2 = self.base_encoder(x2)
     
        return q1, q2

    def forward(self, x1, x2, m):
        if self.training:
            return self.forward_train(x1, x2, m)
        else:
            return self.forward_test(x1, x2, m)


################################ UniCLR--SimTrace ################################
class SimTrace(nn.Module):
    """
    Build a UniCLR model with a base encoder, a momentum encoder, and two MLPs
    """
    def __init__(self, base_encoder, dim=256, mlp_dim=4096, T=1.0):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(SimTrace, self).__init__()

        self.T = T
        self.affinity_gamma = 0.01
        self.eps = 0.01

        # build encoders
        self.base_encoder = base_encoder(num_classes=mlp_dim)
        self.momentum_encoder = base_encoder(num_classes=mlp_dim)
        

        self._build_projector_and_predictor_mlps(dim, mlp_dim)

        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def cal_covariance(self, z):
        z_mean = torch.mean(z, 0, True)
        z = z - z_mean
        z_cova = torch.matmul(z.permute(1, 0), z)
        z_cova = torch.div(z_cova, z.size(0) - 1)
        z_cova = z_cova + 0.01*torch.eye(z_cova.size(1)).cuda()
        return z_mean, z_cova
    
    def simtrace_loss(self, q, k):

        # normalize    
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        
        z = torch.cat((q, k), 0)
        z_mean, z_cova = self.cal_covariance(z.detach())  
        z_cova_inv = torch.inverse(z_cova.detach())
        z1 = q - z_mean
        z2 = k - z_mean
        z1 = torch.matmul(z1, z_cova_inv)
        affinity_matrix = torch.matmul(z1, z2.permute(1, 0))
        
        simtrace_loss = -torch.trace(affinity_matrix) / z1.size(0)
        
        return simtrace_loss
        

    def forward_train(self, x1, x2, m):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: moco momentum
        Output:
            loss
        """
        # compute features
        q1 = self.predictor(self.base_encoder(x1))
        q2 = self.predictor(self.base_encoder(x2))
        
        with torch.no_grad():  # no gradient
            self._update_momentum_encoder(m)  # update the momentum encoder

            # compute momentum features as targets
            k1 = self.momentum_encoder(x1)
            k2 = self.momentum_encoder(x2)

        return self.simtrace_loss(q1, k2) + self.simtrace_loss(q2, k1)

    
    def forward_test(self, x1, x2, m):

        q1 = self.base_encoder(x1)
        q2 = self.base_encoder(x2)
     
        return q1, q2

    def forward(self, x1, x2, m):
        if self.training:
            return self.forward_train(x1, x2, m)
        else:
            return self.forward_test(x1, x2, m)




################################ UniCLR--SimTrace ################################
class SimTrace2(nn.Module):
    """
    Build a UniCLR model with a base encoder, a momentum encoder, and two MLPs
    """
    def __init__(self, base_encoder, dim=256, mlp_dim=4096, T=1.0):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(SimTrace2, self).__init__()

        self.T = T
        self.affinity_gamma = 0.01
        self.eps = 0.01

        # build encoders
        self.base_encoder = base_encoder(num_classes=mlp_dim)
        self.momentum_encoder = base_encoder(num_classes=mlp_dim)
        
        # Learnable Mean and Covariance Matrix
        self.streaming_mean = nn.Parameter(torch.zeros(1, dim), requires_grad=False)
        self.streaming_cova = nn.Parameter(torch.zeros(dim, dim), requires_grad=False)
        self.samples_num = 0

        self._build_projector_and_predictor_mlps(dim, mlp_dim)

        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def cal_covariance(self, z):
        z_mean = torch.mean(z, 0, True)
        z = z - z_mean
        z_cova = torch.matmul(z.permute(1, 0), z)
        z_cova = torch.div(z_cova, z.size(0) - 1)
        z_cova = z_cova + 0.01*torch.eye(z_cova.size(1)).cuda()
        return z_mean, z_cova
    
    @torch.no_grad()
    def _update_streaming_data(self, q, k):
        """Momentum update of the mean and covariance matrix"""
        # normalize    
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        
        z = torch.cat((q, k), 0)
        z_mean, z_cova = self.cal_covariance(z)
        
        # Momentum mean and covariance matrix 
        Mom_mean = (self.streaming_mean * self.samples_num + z_mean * z.size(0)) / (self.samples_num + z.size(0))
        Mom_cova = (self.streaming_cova * self.samples_num + z_cova * z.size(0)) / (self.samples_num + z.size(0))
        self.streaming_mean.data = Mom_mean.detach()
        self.streaming_cova.data = Mom_cova.detach()
        self.samples_num = self.samples_num + z.size(0)
        
    
    def simtrace_loss(self, q, k):

        # normalize    
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        
        # Affinity matrix and simtrace loss                    
        z_cova_inv = torch.inverse(self.streaming_cova)
        z1 = q - self.streaming_mean
        z2 = k - self.streaming_mean
        z1 = torch.matmul(z1, z_cova_inv)
        affinity_matrix = torch.matmul(z1, z2.permute(1, 0))
        simtrace_loss = -torch.trace(affinity_matrix) / z1.size(0)
        
        return simtrace_loss


    def forward(self, x1, x2, m):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: moco momentum
        Output:
            loss
        """
        # compute features
        q1 = self.predictor(self.base_encoder(x1))
        q2 = self.predictor(self.base_encoder(x2))
        
        with torch.no_grad():  # no gradient
            self._update_momentum_encoder(m)  # update the momentum encoder
            
            # compute momentum features as targets
            k1 = self.momentum_encoder(x1)
            k2 = self.momentum_encoder(x2)
            
            self._update_streaming_data(q1, k2)  # update the streaming data

        return self.simtrace_loss(q1, k2) + self.simtrace_loss(q2, k1)


################################ UniCLR--SimWhitening ################################
class SimWhitening(nn.Module):
    """
    Build a UniCLR model with a base encoder, a momentum encoder, and two MLPs
    """
    def __init__(self, base_encoder, dim=256, mlp_dim=4096, T=1.0):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(SimWhitening, self).__init__()

        self.T = T
        self.affinity_gamma = 0.01
        self.eps = 0.01

        # build encoders
        self.base_encoder = base_encoder(num_classes=mlp_dim)
        self.momentum_encoder = base_encoder(num_classes=mlp_dim)

        self._build_projector_and_predictor_mlps(dim, mlp_dim)

        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def cal_covariance(self, z):
        z_mean = torch.mean(z, 0, True)
        z = z - z_mean
        z_cova = torch.matmul(z.permute(1, 0), z)
        z_cova = torch.div(z_cova, z.size(0) - 1)
        z_cova = z_cova + 0.01*torch.eye(z_cova.size(1)).cuda()
        return z_mean, z_cova
    
    def simwhitening_loss(self, q, k):

        # normalize    
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        
        z = torch.cat((q, k), 0)
        z_mean, z_cova = self.cal_covariance(z)
        z_cova_inv = torch.inverse(z_cova)
        z1 = q - z_mean
        z2 = k - z_mean
        z1 = torch.matmul(z1, z_cova_inv)
        
        # calculate affinity matrix
        affinity_matrix = torch.matmul(z1, z2.permute(1, 0)) / z1.size(0) / self.T
                
        # symmetric loss
        symmetric_loss = 0.0
        if self.affinity_gamma > 0.0:
            symmetric_loss = torch.norm(affinity_matrix - affinity_matrix.T)
            
        # CE loss
        labels = torch.arange(z1.size(0), device=z1.device, dtype=torch.long)
        ce_loss = nn.functional.cross_entropy(affinity_matrix, labels)
                
        return ce_loss + self.affinity_gamma * symmetric_loss
        

    def forward_train(self, x1, x2, m):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: moco momentum
        Output:
            loss
        """
        # compute features
        q1 = self.predictor(self.base_encoder(x1))
        q2 = self.predictor(self.base_encoder(x2))
        
        
        with torch.no_grad():  # no gradient
            self._update_momentum_encoder(m)  # update the momentum encoder

            # compute momentum features as targets
            k1 = self.momentum_encoder(x1)
            k2 = self.momentum_encoder(x2)

        return self.simwhitening_loss(q1, k2) + self.simwhitening_loss(q2, k1) 


    def forward_test(self, x1, x2, m):

        q1 = self.base_encoder(x1)
        q2 = self.base_encoder(x2)
     
        return q1, q2

    def forward(self, x1, x2, m):
        if self.training:
            return self.forward_train(x1, x2, m)
        else:
            return self.forward_test(x1, x2, m)


################################ UniCLR--SimWhitening ################################
class SimWhitening2(nn.Module):
    """
    Build a UniCLR model with a base encoder, a momentum encoder, and two MLPs
    """
    def __init__(self, base_encoder, dim=256, mlp_dim=4096, T=1.0):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(SimWhitening2, self).__init__()

        self.T = T
        self.affinity_gamma = 0.01
        self.eps = 0.01

        # build encoders
        self.base_encoder = base_encoder(num_classes=mlp_dim)
        self.momentum_encoder = base_encoder(num_classes=mlp_dim)

        # Learnable Mean and Covariance Matrix
        self.streaming_mean = nn.Parameter(torch.zeros(1,   dim), requires_grad=False)
        self.streaming_cova = nn.Parameter(torch.zeros(dim, dim), requires_grad=False)
        self.samples_num = 0

        self._build_projector_and_predictor_mlps(dim, mlp_dim)

        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    @torch.no_grad()
    def _update_streaming_data(self, q, k):
        """Momentum update of the mean and covariance matrix"""
        # normalize    
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        
        z = torch.cat((q, k), 0)
        z_mean, z_cova = self.cal_covariance(z)
        
        # Momentum mean and covariance matrix 
        Mom_mean = (0.1 * self.streaming_mean * self.samples_num + z_mean * z.size(0)) / (0.1 * self.samples_num + z.size(0))
        Mom_cova = (0.1 * self.streaming_cova * self.samples_num + z_cova * z.size(0)) / (0.1 * self.samples_num + z.size(0))
        self.streaming_mean.data = Mom_mean.detach()
        self.streaming_cova.data = Mom_cova.detach()
        self.samples_num = self.samples_num + z.size(0)


    def cal_covariance(self, z):
        z_mean = torch.mean(z, 0, True)
        z = z - z_mean
        z_cova = torch.matmul(z.permute(1, 0), z)
        z_cova = torch.div(z_cova, z.size(0) - 1)
        z_cova = z_cova + 0.01*torch.eye(z_cova.size(1)).cuda()
        return z_mean, z_cova
    
    
    def simwhitening_loss(self, q, k):

        # normalize    
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        
        z_cova_inv = torch.inverse(self.streaming_cova)
        z1 = q - self.streaming_mean
        z2 = k - self.streaming_mean
        z1 = torch.matmul(z1, z_cova_inv)
        
        # calculate affinity matrix
        affinity_matrix = torch.matmul(z1, z2.permute(1, 0)) / z1.size(0) / self.T
                
        # symmetric loss
        symmetric_loss = 0.0
        if self.affinity_gamma > 0.0:
            symmetric_loss = torch.norm(affinity_matrix - affinity_matrix.T)
            
        # CE loss
        labels = torch.arange(z1.size(0), device=z1.device, dtype=torch.long)
        ce_loss = nn.functional.cross_entropy(affinity_matrix, labels)
                
        return ce_loss + self.affinity_gamma * symmetric_loss
        

    def forward(self, x1, x2, m):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: moco momentum
        Output:
            loss
        """
        # compute features
        q1 = self.predictor(self.base_encoder(x1))
        q2 = self.predictor(self.base_encoder(x2))
        
        
        with torch.no_grad():  # no gradient
            self._update_momentum_encoder(m)  # update the momentum encoder

            # compute momentum features as targets
            k1 = self.momentum_encoder(x1)
            k2 = self.momentum_encoder(x2)

            self._update_streaming_data(q1, k2)  # update the streaming data

        return self.simwhitening_loss(q1, k2) + self.simwhitening_loss(q2, k1) 



class MoCo_ResNet(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.fc.weight.shape[1]
        del self.base_encoder.fc, self.momentum_encoder.fc # remove original fc layer

        # projectors
        self.base_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False)


class MoCo_ViT(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.head.weight.shape[1]
        del self.base_encoder.head, self.momentum_encoder.head # remove original fc layer

        # projectors
        self.base_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim)


class UniCLR_ResNet(UniCLR):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.fc.weight.shape[1]
        del self.base_encoder.fc, self.momentum_encoder.fc # remove original fc layer

        # projectors
        self.base_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False)


class UniCLR_ViT(UniCLR):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.head.weight.shape[1]
        del self.base_encoder.head, self.momentum_encoder.head # remove original fc layer

        # projectors
        self.base_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim)


class SimTrace_ResNet(SimTrace):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.fc.weight.shape[1]
        del self.base_encoder.fc, self.momentum_encoder.fc # remove original fc layer

        # projectors
        self.base_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False)


class SimTrace_ViT(SimTrace):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.head.weight.shape[1]
        del self.base_encoder.head, self.momentum_encoder.head # remove original fc layer

        # projectors
        self.base_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim)


class SimWhitening_ResNet(SimWhitening):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.fc.weight.shape[1]
        del self.base_encoder.fc, self.momentum_encoder.fc # remove original fc layer

        # projectors
        self.base_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False)


class SimWhitening_ViT(SimWhitening):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.head.weight.shape[1]
        del self.base_encoder.head, self.momentum_encoder.head # remove original fc layer

        # projectors
        self.base_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim)

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
