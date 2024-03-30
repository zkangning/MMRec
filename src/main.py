# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Main entry
# UPDATED: 2022-Feb-15
##########################
"""

import os
import argparse
from utils.quick_start import quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '48'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='SELFCFED_LGN', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='name of datasets')
    parser.add_argument('--mg', action="store_true", help='whether to use Mirror Gradient, default is False')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')


    args, _ = parser.parse_known_args()

    config_dict = {
        'gpu_id': args.gpu_id,
    }

    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True, mg=args.mg)


