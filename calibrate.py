import numpy as np
from scipy.special import softmax

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def binary_binning(probs, labels):
    """ Construct calibrator for the binary classifier using histogram binning 
        methods. In our setting, we only consider binning methods where each
        bin is equally spaced with number of bins equal to 10.

    Args:
        probs: output probability estimation from classifier
        labels: correct label list (list of booleans)

    Return:
        function: a function that calibrates confidence from classifier output
    """
    confidences = probs
    bins = np.linspace(0, 1, num=3, endpoint=False)
    idxs = np.digitize(confidences, bins) - 1
    cal_values = []

    for i in range(len(bins)):
        bin_idx = (idxs == i)
        bin_size = np.sum(bin_idx)
        if bin_size == 0:
            cal_values.append(bins[i])
        else:
            cal_values.append(np.sum(labels[bin_idx])/bin_size)

    def cali_func(inputs):
        confs = list(map(lambda x: cal_values[np.digitize(x, bins)-1], inputs))
        # calibrated_preds = confs > 0.5
        return confs
    
    return cali_func


def multiclass_binning(logits, labels):
    """ Construct calibrator for multiclass classifier using histogram binning
        methods. In our setting we conly consider binning methods where each
        bin is equally spaced with number of bins equal to 10.

    Args:
        logits: output logits from classifier
        labels: correct label list (list of integers)

    Return:
        function: a function that calibrates confidence from classifier output
    """
    probs = softmax(logits, axis=1)
    num_classes = probs.shape[1]
    calibrators = []

    for k in range(num_classes):
        calibrators.append(binary_binning(probs[:,k], labels==k))

    def cali_func(input_logits):
        inputs = softmax(input_logits, axis=1)
        unnorm_probs = []
        for k in range(num_classes):
            unnorm_probs.append(calibrators[k](inputs[:, k]))

        unnorm_probs = np.vstack(unnorm_probs).transpose()
        confs = unnorm_probs/(np.sum(unnorm_probs, axis=1)[:, None])
        # calibrated_preds = unnorm_probs.argmax(axis=0)
        return confs
    
    return cali_func


def matrix_scaling(logits, labels):
    """ Construct calibrator for multiclass classifier using matrix scaling 
        methods. Parameters for matrix scaling are optimized with regard to
        the negative log likelihood of the scaled probabilities.

    Args:
        logits: output logits from classifier
        labels: correct label list (list of integers)

    Return:
        function: a function that calibrates confidence from classifier output
    """
    net = nn.Linear(logits.shape[1], logits.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=1e-1)

    data, targets = torch.Tensor(logits).float(),\
                    torch.Tensor(labels).long()

    def eval():
        loss = criterion(net(data), targets)
        loss.backward()
        return loss

    optimizer.step(eval)

    def cali_func(input_logits):
        matrix_scaled = net(torch.Tensor(input_logits)).detach().numpy()
        conf = softmax(matrix_scaled, axis=1)
        # calibrated_preds = matrix_scaled.argmax(axis=1)
        return conf
    
    return cali_func


def vector_scaling(logits, labels):
    """ Construct calibrator for multiclass classifier using vector scaling 
        methods. Parameters for matrix scaling are optimized with regard to
        the negative log likelihood of the scaled probabilities.

    Args:
        logits: output logits from classifier
        labels: correct label list (list of integers)

    Return:
        function: a function that calibrates confidence from classifier output
    """
    class DiagLinear(nn.Module):
        def __init__(self, in_features):
            super().__init__()
            self.diagonal = nn.Parameter(torch.ones(in_features))
            self.bias = nn.Parameter(torch.zeros(in_features))

        def forward(self, x):
            return self.diagonal*x + self.bias

    net = DiagLinear(logits.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.LBFGS(net.parameters(), lr=1e-2, max_iter=100)

    data, targets = torch.Tensor(logits).float(),\
                    torch.Tensor(labels).long()

    def eval():
        loss = criterion(net(data), targets)
        loss.backward()
        return loss

    optimizer.step(eval)

    def cali_func(input_logits):
        matrix_scaled = net(torch.Tensor(input_logits)).detach().numpy()
        conf = softmax(matrix_scaled, axis=1)
        # calibrated_preds = matrix_scaled.argmax(axis=1)
        return conf
    
    return cali_func


def temperature_scaling(logits, labels):
    """ Construct calibrator for multiclass classifier using temperature
        scaling methods. 

    Args:
        logits: output logits from classifier
        labels: correct label list (list of integers)

    Return:
        function: a function that calibrates confidence from classifier output
    """
    class TemperatureScaler(nn.Module):
        def __init__(self):
            super().__init__()
            self.temperature = nn.Parameter(torch.ones(1))

        def forward(self, x):
            return x * self.temperature

    net = TemperatureScaler()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.LBFGS(net.parameters(), lr=1e-2, max_iter=50)

    data, targets = torch.tensor(logits).float(),\
                    torch.tensor(labels).long()

    def eval():
        loss = criterion(net(data), targets)
        loss.backward()
        return loss

    optimizer.step(eval)


    def cali_func(input_logits):
        matrix_scaled = net(torch.Tensor(input_logits)).detach().numpy()
        conf = softmax(matrix_scaled, axis=1)
        # calibrated_preds = matrix_scaled.argmax(axis=1)
        return conf
    
    return cali_func
