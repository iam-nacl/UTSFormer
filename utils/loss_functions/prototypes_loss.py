from cmath import exp
import torch
from torch import nn
import numpy as np
import math

class Intra_Prototypes_Loss(nn.Module):
    def __init__(self, n_classes=2, alpha= 6):
        super(Intra_Prototypes_Loss, self).__init__()
        self.n_classes = n_classes
        self.alpha = alpha # default = 0.5
    def forward(self, x):
        B, C, K, D = x.shape
        x = x.reshape(B*C, K, D)
        x_c = torch.mean(x, dim=1) #[BC D]
        dist_map = x - x_c[:, None, :] #[BC K D]
        dist_map = dist_map**2
        dist_map = torch.sqrt(torch.sum(dist_map, dim=-1)) # [BC K]
        dist_map = dist_map.reshape(B, C, K)
        loss = torch.mean(dist_map, dim=-1) # [B C]
        loss = torch.mean(loss, dim=-1) # [B]
        loss = torch.mean(loss)-self.alpha
        return max(loss, loss*0)


class Cross_Prototypes_Loss(nn.Module):
    def __init__(self, n_classes=2, alpha= 100):
        super(Cross_Prototypes_Loss, self).__init__()
        self.n_classes = n_classes
        self.alpha = alpha # default = 50
    def forward(self, x):
        B, C, K, D = x.shape
        x = x.reshape(B*C, K, D)
        x_c = torch.mean(x, dim=1) #[BC D]
        x_c = x_c.reshape(B, C, D)
        x_c_c = torch.mean(x_c, dim=1) # [B D]

        dist_map = x_c - x_c_c[:, None, :] #[B C D]
        dist_map = dist_map**2
        dist_map = torch.sqrt(torch.sum(dist_map, dim=-1)) # [B C]
        loss = torch.mean(dist_map, dim=-1) # [B]
        loss = -loss + self.alpha
        loss = torch.mean(loss, dim=-1)
        #return max(loss, torch.tensor(0).type_as(loss))
        return max(loss, loss*0)



class CrossIntra_Prototypes_Loss(nn.Module):
    def __init__(self, n_classes=2, alpha= 60, beta=5):
        super(CrossIntra_Prototypes_Loss, self).__init__()
        self.n_classes = n_classes
        self.alpha = alpha # default = 60
        self.beta = beta # default = 5
    def forward(self, x):
        B, C, K, D = x.shape
        x = x.reshape(B*C, K, D)
        x_c = torch.mean(x, dim=1) #[BC D]

        #--- intra class
        intra_dist_map = x - x_c[:, None, :] #[BC K D]
        intra_dist_map = intra_dist_map**2
        intra_dist_map = torch.sqrt(torch.sum(intra_dist_map, dim=-1)) # [BC K]
        intra_dist_map = intra_dist_map.reshape(B, C, K)
        intra_dist_map = torch.mean(intra_dist_map, dim=-1) # [B C]
        #intra_dist_map[intra_dist_map< self.beta] = self.beta

        #--- cross class
        x_c = x_c.reshape(B, C, D).permute(0, 2, 1) #[B D C]
        cross_dist_map = x_c[:, :, :, None] - x_c[:, :, None, :] + 1e-4 #[B D C C]
        cross_dist_map = cross_dist_map**2
        cross_dist_map = torch.sqrt(torch.sum(cross_dist_map, dim=1)) # [B C C]
        cross_dist_map = torch.mean(cross_dist_map, dim=-1) # [B C]
        
        #--- combine
        loss = self.alpha + intra_dist_map - cross_dist_map
        loss = torch.mean(loss, dim=-1) #[B]
        loss = torch.mean(loss, dim=-1)
        return max(loss, loss*0)


class CrossIntra_Prototypes_Loss2(nn.Module):
    def __init__(self, n_classes=2, alpha=2, beta=0.5, k_class0=1.0):
        super(CrossIntra_Prototypes_Loss2, self).__init__()
        self.n_classes = n_classes
        self.alpha = alpha # default = 1
        self.beta = beta # default = 0.3

    def forward(self, x):
        B, C, K, D = x.shape
        x = x.reshape(B*C, K, D)
        x_c = torch.mean(x, dim=1) #[BC D]

        #--- intra class
        intra_dist_map = x - x_c[:, None, :] #[BC K D]
        intra_dist_map = intra_dist_map**2
        intra_dist_map = torch.sqrt(torch.sum(intra_dist_map, dim=-1)) # [BC K]
        intra_dist_map = intra_dist_map.reshape(B, C, K)
        intra_dist_map = torch.mean(intra_dist_map, dim=-1) # [B C]
        intra_dist_map[intra_dist_map< self.beta] = self.beta

        #--- cross class
        x_c = x_c.reshape(B, C, D)
        x_c_c = torch.mean(x_c, dim=1) # [B D]
        cross_dist_map = x_c - x_c_c[:, None, :] #[B C D]
        cross_dist_map = cross_dist_map**2
        cross_dist_map = torch.sqrt(torch.sum(cross_dist_map, dim=-1)) # [B C]
        
        #--- combine
        loss = self.alpha + intra_dist_map - cross_dist_map
        loss = torch.mean(loss, dim=-1) #[B]
        loss = torch.mean(loss, dim=-1)
        return max(loss, loss*0)
        

class CrossIntra_Prototypes_Loss2NP(nn.Module):
    def __init__(self, n_classes=2, alpha= 1, beta=0.3, k_class0=1/2, n_prototypes=32):
        super(CrossIntra_Prototypes_Loss2NP, self).__init__()
        self.n_classes = n_classes
        self.alpha = alpha # default = 1
        self.beta = beta # default = 0.3
        self.k_class0 = k_class0
        self.n_pprototypes = n_prototypes

    def forward(self, x):
        B, cluster_num, D = x.shape
        n_nprototypes = cluster_num - self.n_pprototypes
        xf = x[:, n_nprototypes:, :].reshape(B, 1, self.n_pprototypes, D) #[B c p D]
        xf_c =  torch.mean(xf, dim=2)#[B C D]
        xb = x[:, :n_nprototypes, :].reshape(B, 1, n_nprototypes, D)
        xb_c = torch.mean(xb, dim=2) #[B 1 D]
        x_c = torch.cat([xb_c, xf_c], dim=1).reshape(B*2, D)#[BC D]

        #--- intra class
        intra_dist_map_f =  xf.reshape(B, self.n_pprototypes, D)- xf_c.reshape(B*1, D)[:, None, :] #[BC K D]
        intra_dist_map_f = intra_dist_map_f**2
        intra_dist_map_f = torch.sqrt(torch.sum(intra_dist_map_f, dim=-1)+1e-5) # [BC K]
        intra_dist_map_f = intra_dist_map_f.reshape(B, 1, self.n_pprototypes)
        intra_dist_map_f = torch.mean(intra_dist_map_f, dim=-1) # [B C]

        intra_dist_map_b =  xb.reshape(B*1, n_nprototypes, D)- xb_c.reshape(B*1, D)[:, None, :] #[BC K D]
        intra_dist_map_b = intra_dist_map_b**2
        intra_dist_map_b = torch.sqrt(torch.sum(intra_dist_map_b, dim=-1)+1e-5) # [BC K]
        intra_dist_map_b = intra_dist_map_b.reshape(B, 1, n_nprototypes)
        intra_dist_map_b = torch.mean(intra_dist_map_b, dim=-1) # [B 1]

        intra_dist_map = torch.cat([intra_dist_map_b, intra_dist_map_f], dim=1) #[B C]
        intra_dist_map[intra_dist_map< self.beta] = self.beta

        #--- cross class
        x_c = x_c.reshape(B, 2, D)
        x_c_c = torch.mean(x_c, dim=1) # [B D]
        cross_dist_map = x_c - x_c_c[:, None, :] #[B C D]
        cross_dist_map = cross_dist_map**2
        cross_dist_map = torch.sqrt(torch.sum(cross_dist_map, dim=-1)) # [B C]
        
        #--- combine
        loss = self.alpha + intra_dist_map - cross_dist_map
        loss = torch.mean(loss, dim=-1) #[B]
        loss = torch.mean(loss, dim=-1)
        return max(loss, loss*0)


class CrossIntra_Prototypes_Loss2COS(nn.Module):
    def __init__(self, n_classes=2, alpha= 0.2, beta=0.002, k_class0=1.0):
        super(CrossIntra_Prototypes_Loss2COS, self).__init__()
        self.n_classes = n_classes
        self.alpha = alpha # default = 0.2
        self.beta = beta # default = 0.002

    def forward(self, x):
        B, C, K, D = x.shape
        x = x.reshape(B*C, K, D)
        x_c = torch.mean(x, dim=1) #[BC D]

        #--- intra class
        intra_cos_map = (torch.cosine_similarity(x, x_c[:, None, :], dim=-1)+1)/2 #[BC K]
        intra_cos_map = intra_cos_map.reshape(B, C, K)
        intra_cos_map = torch.mean(intra_cos_map, dim=-1) # [B C]
        intra_cos_map[intra_cos_map< self.beta] = self.beta

        #--- cross class
        x_c = x_c.reshape(B, C, D)
        x_c_c = torch.mean(x_c, dim=1) # [B D]
        cross_cos_map = (torch.cosine_similarity(x_c, x_c_c[:, None, :], dim=-1)+1)/2 #[B C]
        
        #--- combine
        loss = self.alpha + intra_cos_map - cross_cos_map
        loss = torch.mean(loss, dim=-1) #[B]
        loss = torch.mean(loss, dim=-1)
        return max(loss, loss*0)