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

        loss_kl = self.kl(student_log_probs, teacher_probs)

        # Combine losses
        loss = self.alpha * loss_ce + (1 - self.alpha) * (self.T ** 2) * loss_kl
        return loss