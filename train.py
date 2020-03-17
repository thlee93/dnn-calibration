import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

criterion = nn.CrossEntropyLoss()


def reliability_diagrams(probs, labels, mode):
    """ Draw reliability diagrams from classifier output. Save generated
        figure on the path to fname.

    Args:
        probs: output probability estimation from classifier
        labels: correct label list (list of integers)
        mode: file path to save diagram figure
    """
    preds = np.argmax(probs, axis=1)
    # confidences = np.array([probs[i, y] for i, y in enumerate(preds)])
    confidences = probs.max(axis=1)
    bins = np.linspace(0, 1, num=10, endpoint=False)
    idxs = np.digitize(confidences, bins) - 1

    acc_list = []
    for i in range(len(bins)):
        acc = 0
        bin_idx = (idxs == i)
        bin_size = np.sum(bin_idx)

        if bin_size == 0:
            acc_list.append(0)
        else :
            acc_list.append(np.sum(preds[bin_idx] == labels[bin_idx])/bin_size)

    x_list = [0.1*i + 0.05 for i in range(10)]
    legend = ['Outputs', 'Gap']

    plt.figure(figsize=(8, 8))
    plt.bar(x=x_list, height=acc_list, width=0.1, color='b', alpha=0.9, 
            edgecolor='k', linewidth=3)
    plt.bar(x=x_list, height=np.linspace(0.1, 1, num=10), width=0.1, color='r',
            alpha=0.2, edgecolor='r', linewidth=3)

    plt.xlabel('Confidence', fontsize=30)
    plt.ylabel('Accuracy', fontsize=30)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xticks([0.2 * i for i in range(1,6)], fontsize=20)
    plt.yticks([0.2 * i for i in range(1,6)], fontsize=20)
    plt.legend(legend, loc=2, fontsize=25)
    plt.tight_layout()
    plt.grid()

    plt.savefig('{}.png'.format(mode))


def expected_calibration_error(probs, labels, mode):
    """ Calculate expected calibration error from classifier output

    Args:
        probs: output probability estimation (confidences) from classifier
        labels: correct label list (list of integers)
        mode: calibration method that generated input confidences

    Returns:
        float: overall expected calibration error computed
    """
    num_data = len(labels)
    ece_score = 0

    preds = np.argmax(probs, axis=1)
    # confidences = np.array([probs[i, y] for i, y in enumerate(preds)])
    confidences = probs.max(axis=1)
    bins = np.linspace(0, 1, num=15, endpoint=False)
    idxs = np.digitize(confidences, bins) - 1

    for i in range(len(bins)):
        bin_idx = (idxs == i)
        bin_size = np.sum(bin_idx)
        if bin_size == 0:
            continue
        else:
            bin_acc = np.sum(preds[bin_idx] == labels[bin_idx])/bin_size
            bin_conf = np.sum(confidences[bin_idx])/bin_size
            ece_score += np.abs(bin_acc - bin_conf)*bin_size

    ece_score /= num_data
    print('ECE {} calibration: {}'.format(mode, ece_score))
    return ece_score


def train(model, dataset, optimizer, num_epoch, device, scheduler=None):
    model.train()
    loss_val = 0.0

    logits = []
    targets = []

    for epoch in range(num_epoch):
        for i, (data, labels) in enumerate(dataset):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_val += loss.item()
            if epoch + 1 == num_epoch:
                targets += list(labels.detach().cpu().numpy())
                logits.append(outputs.detach().cpu().numpy())

        if scheduler:
            scheduler.step()

        print('Epoch {}, Step {} loss: {:.5f}'.format(epoch, i+1, 
              loss_val/(i+1)))
        loss_val = 0.0

    logits = np.concatenate(logits, axis=0)
    return logits, np.array(targets)


def test(model, dataset, device):
    model.eval()

    logits = []
    targets = []
    correct = 0
    total = 0

    with torch.no_grad():
        for data, labels in dataset:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)

            total += len(labels)
            correct += (predicted == labels).sum().item()

            targets += list(labels.detach().cpu().numpy())
            logits.append(outputs.detach().cpu().numpy())
            
    print("Accuracy of the network : {} %%".format(100 * correct / total))

    logits = np.concatenate(logits, axis=0)
    return logits, np.array(targets)
