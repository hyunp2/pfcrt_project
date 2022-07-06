import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#https://github.com/clcarwin/focal_loss_pytorch/blob/e11e75bad957aecf641db6998a1016204722c1bb/focalloss.py#L30-L31
#Overall, however, the benefit of changing γ is much larger, and indeed the best α’s ranged in just [.25,.75] (we tested α ∈ [.01, .999]). 
#We use γ = 2.0 with α = .25 for all experiments but α = .5 works nearly as well (.4 AP lower). [Focal Loss for Dense Object Detection]

class FocalLoss(nn.Module):
    """SOFTMAX (only best class contrib) (not SIGMOID; adding all class contribs)"""
    def __init__(self, gamma=2., alpha=None, size_average=True, threshold=100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.threshold = threshold
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() == 2:
#             input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
#             input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
#             input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
            input = input #B,dim
        target = target.view(-1,1) #B,1
        mask = (target < self.threshold).view(-1)
        target = target[mask] #b,1
        
        logpt = F.log_softmax(input[mask])
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp().data #non-diff

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
