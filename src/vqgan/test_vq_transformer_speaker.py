import argparse
import json
import logging
import numpy as np
import os
import scipy.io as sio

import torch
from torch import nn
from torch.autograd import Variable
import torchvision
from torch.utils.tensorboard import SummaryWriter

from vqmodules.gan_models import setup_vq_transformer, calc_vq_loss
import sys
sys.path.append('../')
from utils.load_utils_detail import *




def generator_val_step(config, epoch, generator, g_optimizer, test_X,
                       currBestLoss, prev_save_epoch, tag, writer):
    """ Function that validates training of VQ-VAE

    see generator_train_step() for parameter definitions
    """

    generator.eval()
    batchinds = np.arange(test_X.shape[0] // config['batch_size'])
    totalSteps = len(batchinds)
    testLoss = testDLoss = 0
    for bii, bi in enumerate(batchinds):
        idxStart = bi * config['batch_size']
        gtData_np = test_X[idxStart:(idxStart + config['batch_size']), :, :]
        gtData = Variable(torch.from_numpy(gtData_np),
                          requires_grad=False).cuda()
        with torch.no_grad():
            prediction, quant_loss = generator(gtData, None)
        g_loss = calc_vq_loss(prediction, gtData, quant_loss)
        testLoss += g_loss.detach().item()
        print('Test [{}/{}] step loss:{:.4f}'.format(bii, totalSteps, testLoss/(bii+1)))
    testLoss /= totalSteps
    print('Total Test, Loss: {:.4f}, Perplexity: {:5.4f}'\
                    .format( testLoss, np.exp(testLoss)))
    print('----------------------------------')
    writer.add_scalar('Loss/val_totalLoss', testLoss / totalSteps, epoch)

    ## save model if curr loss is lower than previous best loss
    # if testLoss < currBestLoss:
    #     prev_save_epoch = epoch
    #     checkpoint = {'config': args.config,
    #                   'state_dict': generator.state_dict(),
    #                   'optimizer': {
    #                     'optimizer': g_optimizer._optimizer.state_dict(),
    #                     'n_steps': g_optimizer.n_steps,
    #                   },
    #                   'epoch': epoch}
    #     fileName = config['model_path'] + \
    #                     '{}{}_best.pth'.format(tag, config['pipeline'])
    #     print('>>>> saving best epoch {}'.format(epoch), testLoss)
    #     currBestLoss = testLoss
    #     torch.save(checkpoint, fileName)
    return currBestLoss, prev_save_epoch, testLoss


def main(args):
    """ full pipeline for training the Predictor model """

    rng = np.random.RandomState(23456)
    torch.manual_seed(23456)
    torch.cuda.manual_seed(23456)
    print('using config', args.config)
    with open(args.config) as f:
      config = json.load(f)
    tag = config['tag']
    pipeline = config['pipeline']
    currBestLoss = 1e3
    ## can modify via configs, these are default for released model
    seq_len = 32
    prev_save_epoch = 0
    writer = SummaryWriter('runs/debug_{}{}'.format(tag, pipeline))

    ## setting up models
    fileName = config['model_path'] + \
                '{}{}_best.pth'.format(tag, config['pipeline'])
    load_path = fileName if os.path.exists(fileName) else None
    generator, g_optimizer, start_epoch = setup_vq_transformer(args, config,
                                            version=None, load_path=load_path)
    generator.train()

    ## training/validation process
    test_speaker, test_listener, _, _, _ = \
                    load_test_data(config, pipeline, tag, vqconfigs=config,
                              segment_tag=config['segment_tag'], smooth=True, speaker='trevor',vqvae=True)
    test_X = np.concatenate((test_speaker[:,:seq_len,:],
                             test_speaker[:,seq_len:,:]), axis=0)
    print('loaded speaker...', test_X.shape)
    disc_factor = 0.0
    epoch = 50000
    currBestLoss, prev_save_epoch, g_loss = \
        generator_val_step(config, epoch, generator, g_optimizer, test_X,
                            currBestLoss, prev_save_epoch, tag, writer)
   


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--ar_load', action='store_true')
    args = parser.parse_args()
    main(args)
