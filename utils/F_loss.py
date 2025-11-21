
from torch import nn
import torch
import torch.nn.functional as F

def focal_loss(input_values, gamma):
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):

        self.weight = torch.as_tensor(self.weight, device=input.device)
        # print(1111111111, input,22222222222222222,focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma))
        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)


class KL_Loss(nn.Module):
    def __init__(self, temperature=1, reduction='batchmean'):
        super(KL_Loss, self).__init__()
        self.T = temperature
        self.reduction = reduction

    def forward(self, output_batch, teacher_outputs):
        # output_batch  -> B X num_classes
        # teacher_outputs -> B X num_classes

        # loss_2 = -torch.sum(torch.sum(torch.mul(F.log_softmax(teacher_outputs,dim=1), F.softmax(teacher_outputs,dim=1)+10**(-7))))/teacher_outputs.size(0)
        # print('loss H:',loss_2)

        output_batch = F.log_softmax(output_batch / self.T, dim=1)
        teacher_outputs = F.softmax(teacher_outputs / self.T, dim=1) + 10 ** (-7)

        loss = self.T * self.T * nn.KLDivLoss(reduction=self.reduction)(output_batch, teacher_outputs)

        # Same result KL-loss implementation
        # loss = T * T * torch.sum(torch.sum(torch.mul(teacher_outputs, torch.log(teacher_outputs) - output_batch)))/teacher_outputs.size(0)
        return loss

class PCE_Loss(nn.Module):
    def __init__(self, temperature=1, reduction='none', n_classes=2):
        super(PCE_Loss, self).__init__()
        self.T = temperature
        self.reduction = reduction
        self.n_classes = n_classes

    def forward(self, output_batch, teacher_outputs, student_outputs):
        loss_t = nn.CrossEntropyLoss(reduction=self.reduction)(teacher_outputs, output_batch)
        loss_s = nn.CrossEntropyLoss(reduction=self.reduction)(student_outputs, output_batch)

        delta_loss = torch.clamp(loss_s - loss_t, min=0)
        delta_loss = torch.where(output_batch == teacher_outputs.argmax(dim=1), delta_loss, 0)

        teacher_outputs = F.softmax(teacher_outputs / self.T, dim=1)
        student_outputs = F.softmax(student_outputs / self.T, dim=1)

        loss = self.T * self.T * torch.mean(delta_loss * nn.CrossEntropyLoss(reduction=self.reduction)(teacher_outputs, student_outputs))

        return loss

class SP_Loss(nn.Module):
    '''
    Similarity-Preserving Knowledge Distillation
    https://arxiv.org/pdf/1907.09682.pdf
    '''

    def __init__(self):
        super(SP_Loss, self).__init__()

    def forward(self, fm_s, fm_t):
        fm_s = fm_s.reshape(fm_s.size(0), -1)
        G_s = torch.mm(fm_s, fm_s.t())
        norm_G_s = F.normalize(G_s, p=2, dim=1)

        fm_t = fm_t.reshape(fm_t.size(0), -1)
        G_t = torch.mm(fm_t, fm_t.t())
        norm_G_t = F.normalize(G_t, p=2, dim=1)

        loss = F.mse_loss(norm_G_s, norm_G_t)

        return loss