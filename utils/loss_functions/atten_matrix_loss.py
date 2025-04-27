from torch import nn
import torch
from torch.nn import functional as F
import numpy as np
from einops import rearrange
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
from torchvision.utils import save_image
import math


class selfsupervise_loss(nn.Module):

    def __init__(self, heads=6):
        super(selfsupervise_loss, self).__init__()
        self.heads = heads
        self.smoothl1 = torch.nn.SmoothL1Loss()

    def forward(self, attns, epoch, step, smooth=1e-40):
        layer = len(attns)

        entropy_mins = []
        smoothl1s = []

        for i in range(layer):
            attni = attns[i]  # b h n n
            b, h, n, d = attni.shape
            if n == d:
                #attentionmap_visual(attni)
                # entropy loss
                log_attni = torch.log2(attni + smooth)
                entropy = -1 * torch.sum(attni * log_attni, dim=-1) / torch.log2(torch.tensor(n*1.0)) # b h n
                entropy_min = torch.min(entropy, dim=-1)[0]  # b h
                p_loss = (entropy_min-0.9).clamp_min(0)*(1/0.1)

                entropy_mins.append(entropy_min)
                # print("Layer:",i,"entropy_min:",entropy_min)

                # symmetry loss
                attni_t = attni.permute(0, 1, 3, 2)
                #distance = torch.abs(attni_t*n - attni*n)  # b h n n
                #distance = torch.sum(distance, dim=-1)/n  # b h n
                #s_loss = torch.sum(distance, dim=-1)/n  # b h
                s_loss = self.smoothl1(attni*n, attni_t*n).clamp_min(0.1)

                smoothl1s.append(self.smoothl1(attni*n, attni_t*n))
                # print("Layer:",i,"smoothl1:",self.smoothl1(attni*n, attni_t*n))


                if i == 0:
                    loss = 0.8 * torch.mean(s_loss) + 0.2 * torch.mean(p_loss)
                else:
                    loss += 0.8 * torch.mean(s_loss) + 0.2 * torch.mean(p_loss)
            elif n != d:
              layer = layer-1

        if step % 1000 == 0:
            print("epoch:", epoch, "  step:", step, "entropy_min", entropy_mins)
            print("epoch:", epoch, "  step:", step, "smoothl1s", smoothl1s)
        return loss / layer

class selfsupervise_loss2(nn.Module):

    def __init__(self, heads=6):
        super(selfsupervise_loss2, self).__init__()
        self.heads = heads
        self.smoothl1 = torch.nn.SmoothL1Loss()

    def forward(self, attns, smooth=1e-40):
        layer = len(attns)
        for i in range(layer):
            attni = attns[i]  # b h n n
            b, h, n, d = attni.shape
            if n == d:
                #attentionmap_visual(attni)
                # entropy loss
                log_attni = torch.log2(attni-attni.min() + smooth)
                entropy = -1 * torch.sum(attni * log_attni, dim=-1) / torch.log2(torch.tensor(n*1.0)) # b h n
                entropy_min = torch.min(entropy, dim=-1)[0]  # b h
                p_loss = (entropy_min-0.9).clamp_min(0)*(1/0.1)

                # symmetry loss
                attni_t = attni.permute(0, 1, 3, 2)
                #distance = torch.abs(attni_t*n - attni*n)  # b h n n
                #distance = torch.sum(distance, dim=-1)/n  # b h n
                #s_loss = torch.sum(distance, dim=-1)/n  # b h
                s_loss = self.smoothl1(attni*n, attni_t*n).clamp_min(0.1)

                if i == 0:
                    loss = 0.8 * torch.mean(s_loss) + 0.2 * torch.mean(p_loss)
                else:
                    loss += 0.8 * torch.mean(s_loss) + 0.2 * torch.mean(p_loss)
            elif n != d:
              layer = layer-1


        return loss / layer

class selfsupervise_loss3(nn.Module):

    def __init__(self, heads=6):
        super(selfsupervise_loss3, self).__init__()
        self.heads = heads
        self.smoothl1 = torch.nn.SmoothL1Loss()

    # def forward(self, attns, epoch, step, smooth=1e-40):
    def forward(self, attns, smooth=1e-40):

        layer = len(attns)
        entropy_mins = []
        smoothl1s = []
        for i in range(layer):
            attni = attns[i]  # b h n n
            b, h, n, d = attni.shape
            if n == d:
                #attentionmap_visual(attni)
                # entropy loss
                log_attni = torch.log2(attni + smooth)
                entropy = -1 * torch.sum(attni * log_attni, dim=-1) / torch.log2(torch.tensor(n*1.0)) # b h n

                entropy_min = torch.min(entropy, dim=-1)[0]  # b h
                p_loss = (entropy_min-0.9).clamp_min(0) * (1 / 0.1)
                p_loss = p_loss * p_loss * 2

                # print("Layer:", i, "entropy_min:", entropy_min)
                # print("Layer:", i, "p_loss:", torch.mean(p_loss))


                # symmetry loss
                attni_t = attni.permute(0, 1, 3, 2)
                #distance = torch.abs(attni_t*n - attni*n)  # b h n n
                #distance = torch.sum(distance, dim=-1)/n  # b h n
                #s_loss = torch.sum(distance, dim=-1)/n  # b h
                s_loss = (self.smoothl1(attni*n, attni_t*n)).clamp_min(0.1)

                # print("Layer:",i,"smoothl1:",self.smoothl1(attni*n, attni_t*n))
                # print("Layer:", i, "s_loss:", torch.mean(s_loss))

                if i == 0:
                    loss = 0.8 * torch.mean(s_loss) + 0.2 * torch.mean(p_loss)
                    # loss = 0.8 * torch.mean(s_loss) + 0.2 * torch.min(p_loss)
                    # print("Layer:", i, "loss:", 0.8 * torch.mean(s_loss) + 0.2 * torch.mean(p_loss))
                else:
                    loss += 0.8 * torch.mean(s_loss) + 0.2 * torch.mean(p_loss)
                    # loss += 0.8 * torch.mean(s_loss) + 0.2 * torch.min(p_loss)
                    # print("Layer:", i, "loss:", 0.8 * torch.mean(s_loss) + 0.2 * torch.mean(p_loss))

            elif n != d:
              layer = layer-1

        # if step % 1000 == 0:
        #     print("epoch:", epoch, "  step:", step, "entropy_min", entropy_mins)
        #     print("epoch:", epoch, "  step:", step, "smoothl1s", smoothl1s)

        return loss / layer


