import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#https://github.com/clcarwin/focal_loss_pytorch/blob/e11e75bad957aecf641db6998a1016204722c1bb/focalloss.py#L30-L31
#Overall, however, the benefit of changing γ is much larger, and indeed the best α’s ranged in just [.25,.75] (we tested α ∈ [.01, .999]). 
#We use γ = 2.0 with α = .25 for all experiments but α = .5 works nearly as well (.4 AP lower). [Focal Loss for Dense Object Detection]

class FocalLoss(nn.Module):
    """SOFTMAX (only best class contrib) (not SIGMOID; adding all class contribs)"""
    def __init__(self, gamma=2., beta: float=0.9999, size_average=True, threshold=100, weight: torch.Tensor=torch.ones(3), ignore_index=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
#         self.alpha = alpha
        self.beta = beta
        self.weight = weight
        self.effective_num = torch.Tensor(list(map(lambda inp: (1 - beta) / (1 - beta**(inp)), weight ))) #nclass,
        self.threshold = threshold
#         if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
#         if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
        self.ignore_index = ignore_index

    def forward(self, input, target):
        assert input.dim() == 2 or input.dim() == 3, "input dimension should be either 2 or 3!"

        target = target.view(-1,1) #B,1
#         mask = (target < self.threshold).view(-1)
#         target = target[mask] #b,1
        
        logpt = F.log_softmax(input, dim=-1)

        if self.ignore_index is not None:
            retain_these = target.view(-1, ) != self.ignore_index
            logpt = logpt[retain_these]
            target = target.view(-1, )[retain_these].view(-1, 1)

            logpt = logpt.gather(1,target)
            logpt = logpt.view(-1)
        else:
            #target must have dropped nan (or ignore_index related values)
            logpt = logpt.gather(1,target)
            logpt = logpt.view(-1)
        
        pt = logpt.data.exp() #non-diff

        if self.effective_num is not None:
#             if self.alpha.type()!=input.data.type():
#                 self.alpha = self.alpha.type_as(input.data)
            self.effective_num = self.effective_num.to(logpt)
            at = self.effective_num.gather(0, target.data.view(-1))
            logpt = logpt * at.data

        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss #if mask.any() else 0. #b,
        if self.size_average: return loss.mean()
        else: return loss.sum()
