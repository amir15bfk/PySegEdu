import torch


# The `DiceScore` class calculates the Dice score for binary segmentation tasks using sigmoid
# activation and smooth parameter.
class DiceScore(torch.nn.Module):
    def __init__(self, smooth=1):
        """
        The function initializes an object with the name "DiceScore" and a specified smoothing
        parameter.
        
        :param smooth: The `smooth` parameter in the `__init__` method is a parameter that has a default
        value of 1. This parameter is used to set the smoothing factor for the DiceScore class. It
        allows the user to adjust the level of smoothing applied when calculating the Dice score,
        defaults to 1 (optional)
        """
        super(DiceScore, self).__init__()
        self.name = "DiceScore"
        self.smooth = smooth

    def forward(self, logits, targets):
        num = targets.size(0)

        # Convert the logits and targets to probabilities
        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1) > 0.5
        m2 = targets.view(num, -1) > 0.5
        # Calculate the intersection of the probabilities and targets
        intersection = m1 * m2

        # Calculate the Dice score
        score = (
            2.0
            * (intersection.sum(1) + self.smooth)
            / (m1.sum(1) + m2.sum(1) + self.smooth)
        )
        score = score.sum() / num
        return score

class PrecisionScore(torch.nn.Module):
    # Initialize the class with a smooth value
    def __init__(self, smooth=1):
        super(PrecisionScore, self).__init__()
        # Set the name of the class
        self.name = "Precision"
        # Set the smooth value
        self.smooth = smooth

    # Forward function to calculate the precision score
    def forward(self, logits, targets):
        # Get the number of data points
        num = targets.size(0)

        # Calculate the probabilities from the logits
        probs = torch.sigmoid(logits)
        # Reshape the probabilities and targets
        m1 = probs.view(num, -1) > 0.5
        m2 = targets.view(num, -1) > 0.5
        # Get the intersection of the probabilities and targets
        intersection = m1 * m2

        # Calculate the precision score
        score = (
            (intersection.sum(1) + self.smooth)
            / (m1.sum(1) + self.smooth)
        )
        # Calculate the average precision score
        score = score.sum() / num
        # Return the precision score
        return score
    

class AccuracyScore(torch.nn.Module):
    # Initialize the class
    def __init__(self):
        # Call the init method of the parent class
        super(AccuracyScore, self).__init__()
        # Set the name of the class
        self.name = "Accuracy"

    # Forward method
    def forward(self, logits, targets):
        # Get the number of data points
        num = targets.size(0)

        # Calculate the probabilities using the sigmoid function
        probs = torch.sigmoid(logits)
        # Reshape the probabilities
        m1 = probs.view(num, -1) > 0.5
        # Reshape the targets
        m2 = targets.view(num, -1) > 0.5
        # Calculate the intersection
        intersection = m1 * m2

        # Calculate the accuracy score
        score = (intersection.sum(1) / m2.sum(1)).sum() / num
        # Return the accuracy score
        return score


class RecallScore(torch.nn.Module):
    # Initialize the class with a smooth value
    def __init__(self, smooth=1):
        super(RecallScore, self).__init__()
        # Set the name of the class
        self.name = "Recall"
        # Set the smooth value
        self.smooth = smooth

    # Forward function for the class
    def forward(self, logits, targets):
        # Get the number of samples
        num = targets.size(0)

        # Calculate the probabilities from the logits
        probs = torch.sigmoid(logits)
        # Reshape the probabilities and the targets
        m1 = probs.view(num, -1) > 0.5
        m2 = targets.view(num, -1) > 0.5
        # Get the intersection of the probabilities and the targets
        intersection = m1 * m2

        # Calculate the recall score
        score = (
            # Get the sum of the intersection and add the smooth value
            (intersection.sum(1) + self.smooth)
            # Get the sum of the targets and add the smooth value
            / (m2.sum(1) + self.smooth)
        )
        # Calculate the average recall score
        score = score.sum() / num
        # Return the score
        return score
    
class F1Score(torch.nn.Module):
    # Initialize the class with a smooth value
    def __init__(self, smooth=1):
        super(F1Score, self).__init__()
        # Set the name of the class
        self.name = "F1Score"
        # Set the smooth value
        self.smooth = smooth

    # Forward function for the class
    def forward(self, logits, targets):
        # Get the number of data points
        num = targets.size(0)

        # Calculate the probabilities from the logits
        probs = torch.sigmoid(logits)
        # Reshape the probabilities and targets
        m1 = probs.view(num, -1) > 0.5
        m2 = targets.view(num, -1) > 0.5
        # Get the intersection of the probabilities and targets
        intersection = m1 * m2

        # Calculate the recall
        R = (
            (intersection.sum(1) + self.smooth)
            / (m2.sum(1) + self.smooth)
        )
        # Calculate the precision
        P = (
            (intersection.sum(1) + self.smooth)
            / (m1.sum(1) + self.smooth)
        )
        # Calculate the F1 score
        return (2*R*P)/(R+P)
class mIoUScore(torch.nn.Module):
    # Initialize the class with a smooth value
    def __init__(self, smooth=1):
        super(mIoUScore, self).__init__()
        # Store the name of the class
        self.name = "mIoUScore"
        # Store the smooth value
        self.smooth = smooth

    # Calculate the intersection and union
    def forward(self, logits, targets):
        # Get the number of data points
        num = targets.size(0)

        # Calculate the probabilities
        probs = torch.sigmoid(logits)
        # Reshape the probabilities and targets
        m1 = probs.view(num, -1) > 0.5
        m2 = targets.view(num, -1) > 0.5
        # Calculate the intersection
        intersection = m1 * m2

        # Calculate the IoU score
        score = (
            (intersection.sum(1) + self.smooth)
            / (m1.sum(1) + m2.sum(1) - intersection.sum(1) + self.smooth)
        )
        # Calculate the average IoU score
        score = score.sum() / num
        # Return the score
        return score