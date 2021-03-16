#! /usr/bin/env python3
import sys
sys.path.append("/root/pytorch_influence_functions/")

import pytorch_influence_functions as ptif
from train_mnist import load_model, load_data

if __name__ == "__main__":
    #     config = {
    #     'outdir': 'outdir',
    #     'seed': 42,
    #     'gpu': 0,
    #     'dataset': 'CIFAR10',
    #     'num_classes': 10,
    #     'test_sample_num': 1,
    #     'test_start_index': 0,
    #     'recursion_depth': 1,
    #     'r_averaging': 1,
    #     'scale': None,
    #     'damp': None,
    #     'calc_method': 'img_wise',
    #     'log_filename': None,
    # }
    config = ptif.get_default_config()
    config['dataset'] = 'MNIST'
    config['recursion_depth'] = 5000
    config['scale'] = 10
    config['damp'] = 0.02
    config['test_sample_id'] = 6558
    config['calc_method'] = 'img_wise_single'
    config['test_sample_num'] = 1

    model = load_model("output/small_mnist_all_cnn_c_99999.pth")
    trainloader, testloader = load_data()

    ptif.init_logging('logfile.log')
    ptif.calc_img_wise_on_single(config, model, trainloader, testloader)


