import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.5):
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, teacher_logits, targets):
        # Hard loss (ground truth)
        loss_ce = self.ce(student_logits, targets)

        # Soft loss (distillation)
        student_log_probs = F.log_softmax(student_logits / self.T, dim=1)
        teacher_probs = F.softmax(teacher_logits / self.T, dim=1)
        #labelsmoothing
        nc = teacher_probs.size(1)
        teacher_probs = label_sm(teacher_probs,targets,nc)
        loss_kl = self.kl(student_log_probs, teacher_probs)

        # Combine losses
        loss = self.alpha * loss_ce + (1 - self.alpha) * (self.T ** 2) * loss_kl
        return loss
    
@torch.no_grad()
def label_sm(teach_prob,target,nc):
        batch = teach_prob.size(0)
        
        true_idx = torch.argmax(teach_prob, dim=1)
        true_prob = torch.max(teach_prob, dim=1).values
        rest_prob = (1-true_prob) / (nc-1)

        tensor_list = []
        for b in range(batch):
          singbatch = torch.full((1, nc), rest_prob[b].detach())
          tensor_list.append(singbatch)
        final = torch.stack(tensor_list, dim=0).squeeze()
        #broadcasting
        final[torch.arange(batch), true_idx] = true_prob
        return final