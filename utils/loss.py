import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Function
import torch.nn as nn


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x,lambd):
        ctx.lambd=lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * -1.0*ctx.lambd,None


def grad_reverse(x,lambd=1.0):
    return GradReverse.apply(x,lambd)


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) /
                    (1.0 + np.exp(-alpha * iter_num / max_iter)) -
                    (high - low) + low)




def Each_labelloss(F1, feat_source, feat_untarget,source_label,lamda_w,lambda_cla,thred, temp,eta=1.0):
    out_t1 = F1(feat_source, reverse=False, eta=eta)
    out_t2=F1(feat_untarget, reverse=True, eta=eta)
    pred_target = torch.where(F.sigmoid(out_t2).data >= thred, 1, 0)
    Label_num=out_t2.shape[1]
    loss_CLA=0
    for i in range(Label_num):
        source_index=feat_source[source_label[:,i]==1]
        target_index=feat_untarget[pred_target[:,i]==1]
        if (source_index.shape[0]>0) and (target_index.shape[0]>0):
            mean_source = torch.mean(source_index, 0)
            mean_target=torch.mean(target_index,0)
            loss_CLA+=torch.exp(F.cosine_similarity(mean_source,mean_target,dim=0)/temp)
    loss_CLA=torch.log(loss_CLA+torch.tensor(1e-6))*lambda_cla
    loss_WAL = lamda_w * torch.mean(torch.mean(F.sigmoid(out_t1),0)-torch.mean(F.sigmoid(out_t2),0))

    return loss_WAL-loss_CLA


