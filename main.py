import os

import torch
import numpy as np
from scipy.special import softmax

from utils import *
from train import *

torch.manual_seed(0)

args = argparser()
data_dir = os.path.join('/data/public_data', args.dataset.upper())
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.num_cuda)

calibration_list = ['histogram_binning', 
                    'matrix_scaling', 
                    'vector_scaling',
                    'temperature_scaling']

def main():
    if args.dataset == 'cifar10':
        num_classes = 10
    elif args.dataset == 'cifar100':
        num_classes = 100

    device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
    model = get_model(args.model_type, num_classes=num_classes)
    model.to(device)
    optimizer, scheduler = get_optim(args.optimizer, model)
    tr_dataset, val_dataset = get_train_loader(args.dataset, data_dir)

    # train the model from scratch or load trained model
    if not args.load_model:
        model_name = '{}_{}.ckpt'.format(args.model_type, args.dataset)
        train(model, tr_dataset, optimizer, args.num_epoch, device, scheduler)
        torch.save(model.state_dict(), model_name)
    else :
        model.load_state_dict(torch.load(args.load_model))

    test_dataset = get_test_loader(args.dataset, data_dir)
    test_logits, test_labels = test(model, test_dataset, device)
    test_probs = softmax(test_logits, axis=1)

    # evaluate confidence calibration of original model
    reliability_diagrams(test_probs, test_labels, mode='before')
    expected_calibration_error(test_probs, test_labels, mode='before')

    val_logits, val_labels = test(model, val_dataset, device)
    for cali in calibration_list:
        calibrator = get_calibrator(val_logits, val_labels, mode=cali)
        confidences = calibrator(test_logits)

        reliability_diagrams(confidences, test_labels, mode=cali)
        expected_calibration_error(confidences, test_labels, 
                                   mode=cali)


if __name__ == '__main__':
    main()
