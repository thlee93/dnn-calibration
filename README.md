
# Confidence Calibration for Neural Networks
Implementation of calibration methods for neural networks. Calibrators are provided as a python function that directly operates on output logits. `calibrate.py` contains implementations of calibrators. Metrics and visualizing methods for confidence outputs are contained in `train.py`.

Implementation of newly introduced calibrator will be updated continuously. 

## Execution
Train a neural network form scratch and calibrate the confidence: 
```bash
python main.py --dataset <dataset> --model_type <model> --optimizer <optim>
```

Load already trained networks and only calibrate the confidence:
```bash
python main.py --dataset <dataset> \
	       --model_type <model> \
	       --optimizer <optim> \
	       --load_model <path_to_models> 
```

Codes for models like resnet110 were copied and modified from [akamaster's repository](https://github.com/akamaster/pytorch_resnet_cifar10)

## Sample output
![Reliability Diagram](https://user-images.githubusercontent.com/12118444/76838084-04d9c900-6877-11ea-8d0f-97e22461a421.png)


## List of implemented methods
1. Histogram Binning
2. Matrix Scaling
3. Vector Scaling
4. Temperature Scaling

### Notes on hyperparameter tuning
- Histogram binning method requires user to designate adequate number of bins. This value should be carefully tuned in exploitation.
- Performance of temperature scaling reported in the paper seems to be achieved when LBFGS optimizer was used for temperature value. In our experiments, results fluctuated a lot even to the small changes in hyperparameters to the optimizer (learning rate, num_iterations).
- [Original paper](https://arxiv.org/abs/1706.04599) reported that Expected Calibration Error of matrix scaling methods for CIFAR100 is near 0.25. It seems that such results were also obtained when LBFGS optimizer is used in training calibrator. When SGD or Adam is used, error values reduce to 0.15. 

## Reference papers
- [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) ICML 2017
