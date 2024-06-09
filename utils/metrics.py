import torch


class DiceScore(torch.nn.Module):
    def __init__(self, smooth=1):
        super(DiceScore, self).__init__()
        self.name = "DiceScore"
        self.smooth = smooth

    def forward(self, logits, targets):
        num = targets.size(0)

        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1) > 0.5
        m2 = targets.view(num, -1) > 0.5
        intersection = m1 * m2

        score = (
            2.0
            * (intersection.sum(1) + self.smooth)
            / (m1.sum(1) + m2.sum(1) + self.smooth)
        )
        score = score.sum() / num
        return score

class PrecisionScore(torch.nn.Module):
    def __init__(self, smooth=1):
        super(PrecisionScore, self).__init__()
        self.name = "Precision"
        self.smooth = smooth

    def forward(self, logits, targets):
        num = targets.size(0)

        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1) > 0.5
        m2 = targets.view(num, -1) > 0.5
        intersection = m1 * m2

        score = (
            (intersection.sum(1) + self.smooth)
            / (m1.sum(1) + self.smooth)
        )
        score = score.sum() / num
        return score
class RecallScore(torch.nn.Module):
    def __init__(self, smooth=1):
        super(RecallScore, self).__init__()
        self.name = "Recall"
        self.smooth = smooth

    def forward(self, logits, targets):
        num = targets.size(0)

        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1) > 0.5
        m2 = targets.view(num, -1) > 0.5
        intersection = m1 * m2

        score = (
            (intersection.sum(1) + self.smooth)
            / (m2.sum(1) + self.smooth)
        )
        score = score.sum() / num
        return score
class F1Score(torch.nn.Module):
    def __init__(self, smooth=1):
        super(F1Score, self).__init__()
        self.name = "F1Score"
        self.smooth = smooth

    def forward(self, logits, targets):
        num = targets.size(0)

        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1) > 0.5
        m2 = targets.view(num, -1) > 0.5
        intersection = m1 * m2

        R = (
            (intersection.sum(1) + self.smooth)
            / (m2.sum(1) + self.smooth)
        )
        R = R.sum() / num
        P = (
            (intersection.sum(1) + self.smooth)
            / (m1.sum(1) + self.smooth)
        )
        P = P.sum() / num
        return (2*R*P)/(R+P)
class mIoUScore(torch.nn.Module):
    def __init__(self, smooth=1):
        super(mIoUScore, self).__init__()
        self.name = "mIoUScore"
        self.smooth = smooth

    def forward(self, logits, targets):
        num = targets.size(0)

        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1) > 0.5
        m2 = targets.view(num, -1) > 0.5
        intersection = m1 * m2

        score = (
            (intersection.sum(1) + self.smooth)
            / (m1.sum(1) + m2.sum(1) - intersection.sum(1) + self.smooth)
        )
        score = score.sum() / num
        return score