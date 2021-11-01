# -*- coding: utf-8 -*-
"""
Main file for training SQuAD reading comprehension model.
"""
import os
import sys
import argparse
import math
import random
import paddle
import numpy as np
import paddle.nn as nn

import paddle.optimizer as optim
from datetime import datetime
from data_loader.SQuAD import prepro, get_loader
from model.QANet import QANet
from trainer.QANet_trainer import Trainer
# from util.visualize import Visualizer
from model.modules.ema import EMA
from util.file_utils import pickle_load_large_file
from reprod_log import ReprodLogger


data_folder = "datasets/"
parser = argparse.ArgumentParser(description='Lucy')

# dataset
parser.add_argument(
    '--processed_data',
    default=True, action='store_false',
    help='whether the dataset already processed')
parser.add_argument(
    '--train_file',
    default=data_folder + 'original/SQuAD/train-v1.1.json',
    type=str, help='path of train dataset')
parser.add_argument(
    '--dev_file',
    default=data_folder + 'original/SQuAD/dev-v1.1.json',
    type=str, help='path of dev dataset')
parser.add_argument(
    '--train_examples_file',
    default=data_folder + 'processed/SQuAD/train-v1.1-examples.pkl',
    type=str, help='path of train dataset examples file')
parser.add_argument(
    '--dev_examples_file',
    default=data_folder + 'processed/SQuAD/dev-v1.1-examples.pkl',
    type=str, help='path of dev dataset examples file')
parser.add_argument(
    '--train_meta_file',
    default=data_folder + 'processed/SQuAD/train-v1.1-meta.pkl',
    type=str, help='path of train dataset meta file')
parser.add_argument(
    '--dev_meta_file',
    default=data_folder + 'processed/SQuAD/dev-v1.1-meta.pkl',
    type=str, help='path of dev dataset meta file')
parser.add_argument(
    '--train_eval_file',
    default=data_folder + 'processed/SQuAD/train-v1.1-eval.pkl',
    type=str, help='path of train dataset eval file')
parser.add_argument(
    '--dev_eval_file',
    default=data_folder + 'processed/SQuAD/dev-v1.1-eval.pkl',
    type=str, help='path of dev dataset eval file')
parser.add_argument(
    '--val_num_batches',
    default=500, type=int,
    help='number of batches for evaluation (default: 500)')

# embedding
parser.add_argument(
    '--glove_word_file',
    default=data_folder + 'original/Glove/glove.840B.300d.txt',
    type=str, help='path of word embedding file')
parser.add_argument(
    '--glove_word_size',
    default=int(2.2e6), type=int,
    help='Corpus size for Glove')
parser.add_argument(
    '--glove_dim',
    default=300, type=int,
    help='word embedding size (default: 300)')
parser.add_argument(
    '--word_emb_file',
    default=data_folder + 'processed/SQuAD/word_emb.pkl',
    type=str, help='path of word embedding matrix file')
parser.add_argument(
    '--word_dictionary',
    default=data_folder + 'processed/SQuAD/word_dict.pkl',
    type=str, help='path of word embedding dict file')

parser.add_argument(
    '--pretrained_char',
    default=False, action='store_true',
    help='whether train char embedding or not')
parser.add_argument(
    '--glove_char_file',
    default=data_folder + "original/Glove/glove.840B.300d-char.txt",
    type=str, help='path of char embedding file')
parser.add_argument(
    '--glove_char_size',
    default=94, type=int,
    help='Corpus size for char embedding')
parser.add_argument(
    '--char_dim',
    default=64, type=int,
    help='char embedding size (default: 64)')
parser.add_argument(
    '--char_emb_file',
    default=data_folder + 'processed/SQuAD/char_emb.pkl',
    type=str, help='path of char embedding matrix file')
parser.add_argument(
    '--char_dictionary',
    default=data_folder + 'processed/SQuAD/char_dict.pkl',
    type=str, help='path of char embedding dict file')

# train
parser.add_argument(
    '-b', '--batch_size',
    default=32, type=int,
    help='mini-batch size (default: 32)')
parser.add_argument(
    '-e', '--epochs',
    default=30, type=int,
    help='number of total epochs (default: 30)')

# debug
parser.add_argument(
    '--debug',
    default=False, action='store_true',
    help='debug mode or not')
parser.add_argument(
    '--debug_batchnum',
    default=2, type=int,
    help='only train and test a few batches when debug (devault: 2)')

# checkpoint
parser.add_argument(
    '--resume',
    default='', type=str,
    help='path to latest checkpoint (default: none)')
parser.add_argument(
    '--verbosity',
    default=2, type=int,
    help='verbosity, 0: quiet, 1: per epoch, 2: complete (default: 2)')
parser.add_argument(
    '--save_dir',
    default='checkpoints/', type=str,
    help='directory of saved model (default: checkpoints/)')
parser.add_argument(
    '--save_freq',
    default=1, type=int,
    help='training checkpoint frequency (default: 1 epoch)')
parser.add_argument(
    '--print_freq',
    default=10, type=int,
    help='print training information frequency (default: 10 steps)')

# cuda
parser.add_argument(
    '--with_cuda',
    default=False, action='store_true',
    help='use CPU in case there\'s no GPU support')
parser.add_argument(
    '--multi_gpu',
    default=False, action='store_true',
    help='use multi-GPU in case there\'s multiple GPUs available')

# log & visualize
parser.add_argument(
    '--visualizer',
    default=False, action='store_true',
    help='use visdom visualizer or not')
parser.add_argument(
    '--log_file',
    default='log.txt',
    type=str, help='path of log file')

# optimizer & scheduler & weight & exponential moving average
parser.add_argument(
    '--lr',
    default=0.001, type=float,
    help='learning rate')
parser.add_argument(
    '--lr_warm_up_num',
    default=1000, type=int,
    help='number of warm-up steps of learning rate')
parser.add_argument(
    '--beta1',
    default=0.8, type=float,
    help='beta 1')
parser.add_argument(
    '--beta2',
    default=0.999, type=float,
    help='beta 2')
parser.add_argument(
    '--decay',
    default=0.9999, type=float,
    help='exponential moving average decay')
parser.add_argument(
    '--use_scheduler',
    default=True, action='store_false',
    help='whether use learning rate scheduler')
parser.add_argument(
    '--use_grad_clip',
    default=True, action='store_false',
    help='whether use gradient clip')
parser.add_argument(
    '--grad_clip',
    default=5.0, type=float,
    help='global Norm gradient clipping rate')
parser.add_argument(
    '--use_ema',
    default=False, action='store_true',
    help='whether use exponential moving average')
parser.add_argument(
    '--use_early_stop',
    default=True, action='store_false',
    help='whether use early stop')
parser.add_argument(
    '--early_stop',
    default=10, type=int,
    help='checkpoints for early stop')

# model
parser.add_argument(
    '--para_limit',
    default=400, type=int,
    help='maximum context token number')
parser.add_argument(
    '--ques_limit',
    default=50, type=int,
    help='maximum question token number')
parser.add_argument(
    '--ans_limit',
    default=30, type=int,
    help='maximum answer token number')
parser.add_argument(
    '--char_limit',
    default=16, type=int,
    help='maximum char number in a word')
parser.add_argument(
    '--d_model',
    default=128, type=int,
    help='model hidden size')
parser.add_argument(
    '--num_head',
    default=8, type=int,
    help='attention num head')


def main(args):
    # show configuration
    print(args)
    random_seed = None

    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        paddle.seed(random_seed)

    # set log file
    log = sys.stdout
    if args.log_file is not None:
        log = open(args.log_file, "a")

    # set device
    print(paddle.device.is_compiled_with_cuda())
    paddle.device.set_device("gpu" if paddle.device.is_compiled_with_cuda() else "cpu")
    device = ("gpu" if paddle.device.is_compiled_with_cuda() else "cpu")
    n_gpu = len(os.getenv("CUDA_VISIBLE_DEVICES", "").split(","))
    # device = "cpu"
    if paddle.device.is_compiled_with_cuda():
        print("device is cuda, # cuda is: ", n_gpu)
    else:
        print("device is cpu")

    # process word vectors and datasets
    if not args.processed_data:
        prepro(args)

    # load word vectors and datasets
    wv_tensor = paddle.to_tensor(
        np.array(pickle_load_large_file(args.word_emb_file), dtype=np.float32)).astype("float32")
    cv_tensor = paddle.to_tensor(
        np.array(pickle_load_large_file(args.char_emb_file), dtype=np.float32)).astype("float32")
    wv_word2ix = pickle_load_large_file(args.word_dictionary)

    train_dataloader = get_loader(
        args.train_examples_file, args.batch_size, shuffle=True)
    dev_dataloader = get_loader(
        args.dev_examples_file, args.batch_size, shuffle=True)

    train_dataloader = get_loader(
        args.train_examples_file, args.batch_size, shuffle=False)
    dev_dataloader = get_loader(
        args.dev_examples_file, args.batch_size, shuffle=False)

    # logger_paddle_data = ReprodLogger()

    # logger_paddle_data.add("length_train", np.array(len(train_dataloader)))
    # logger_paddle_data.add("length_dev", np.array(len(dev_dataloader)))
    # for idx, (paddle_train_batch, paddle_dev_batch) in enumerate(zip(train_dataloader,dev_dataloader)):
    #     if idx >= 5:
    #         break
    #     logger_paddle_data.add(f"traindataloader_{idx}", paddle_train_batch[idx].numpy())
    #     logger_paddle_data.add(f"devdataloader_{idx}", paddle_dev_batch[idx].numpy())
    
    # logger_paddle_data.save("data_paddle.npy")
    # exit(0)
    # construct model
    model = QANet(
        wv_tensor,
        cv_tensor,
        args.para_limit,
        args.ques_limit,
        args.d_model,
        num_head=args.num_head,
        train_cemb=(not args.pretrained_char),
        pad=wv_word2ix["<PAD>"])
    # for i,k in model.state_dict().items():
    #     print(i, k.shape)
    # exit(0)
    print(model.summary())

    # context_wids = paddle.to_tensor(np.load('context_wids.npy'))
    # context_cids = paddle.to_tensor(np.load('context_cids.npy'))
    # question_wids = paddle.to_tensor(np.load('question_wids.npy'))
    # question_cids = paddle.to_tensor(np.load('question_cids.npy'))

    input_fp = "checkpoint_epoch0.pdparams"
    paddle_dict = paddle.load(input_fp)
    model.set_state_dict(paddle_dict)
    # model.eval()
    # p1, p2 = model(context_wids,context_cids,question_wids,question_cids)
    # print("p1",p1)
    # print("p2",p2)
    # exit(0)
    # reprod_logger = ReprodLogger()
    # print(p1)
    # reprod_logger.add("p1", p1.cpu().detach().numpy())
    # print(p2)
    # reprod_logger.add("p2", p2.cpu().detach().numpy())
    # reprod_logger.save("forward_paddle.npy")
    
    # exit(0)
    # if paddle.device.is_compiled_with_cuda() > 1 and args.multi_gpu:
    #     model = paddle.DataParallel(model)
    # model.to(device)

    # exponential moving average
    ema = EMA(args.decay)
    if args.use_ema:
        for name, param in model.named_parameters():
            # print("111",type(param))
            if not param.stop_gradient:
                # print(name, param.shape)
                ema.register(name, param)

    # set optimizer and scheduler
    cr = 1.0 / math.log(args.lr_warm_up_num)
    lr_scheduler = paddle.optimizer.lr.LambdaDecay(
        learning_rate=args.lr,
        lr_lambda=lambda ee: cr * math.log(ee + 1)
        if ee < args.lr_warm_up_num else 1
    )

    parameters = filter(lambda p: not p.stop_gradient, model.parameters())
    # for i,j in enumerate(parameters):
    #     print(i, j.shape)
    # exit()
    optimizer = optim.Adam(
        parameters=parameters,
        learning_rate=lr_scheduler,
        beta1 = args.beta1,
        beta2 = args.beta2,
        epsilon=1e-8,
        weight_decay=3e-7,
        grad_clip=paddle.nn.ClipGradByNorm(clip_norm=5.0)
        )

    # set loss, metrics
    loss = paddle.nn.CrossEntropyLoss()

    # set visdom visualizer to store training process information
    # see the training process on http://localhost:8097/
    vis = None
    # if args.visualizer:
    #     os.system("python -m visdom.server")
    #     vis = Visualizer("main")

    # construct trainer
    # an identifier (prefix) for saved model
    identifier = type(model).__name__ + '_'
    trainer = Trainer(
        args, model, loss,
        train_data_loader=train_dataloader,
        dev_data_loader=dev_dataloader,
        train_eval_file=args.train_eval_file,
        dev_eval_file=args.dev_eval_file,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        epochs=args.epochs,
        with_cuda=args.with_cuda,
        save_dir=args.save_dir,
        verbosity=args.verbosity,
        save_freq=args.save_freq,
        print_freq=args.print_freq,
        resume=args.resume,
        identifier=identifier,
        debug=args.debug,
        debug_batchnum=args.debug_batchnum,
        lr=args.lr,
        lr_warm_up_num=args.lr_warm_up_num,
        grad_clip=args.grad_clip,
        decay=args.decay,
        visualizer=vis,
        logger=log,
        use_scheduler=args.use_scheduler,
        use_grad_clip=args.use_grad_clip,
        use_ema=args.use_ema,
        ema=ema,
        use_early_stop=args.use_early_stop,
        early_stop=args.early_stop)

    # start training!
    start = datetime.now()
    trainer.train()
    print("Time of training model ", datetime.now() - start)


if __name__ == '__main__':
    main(parser.parse_args())
