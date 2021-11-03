# -*- coding: utf-8 -*-
"""
Trainer file for SQuAD dataset.
"""
import os
import shutil
import time
import paddle
import paddle.nn.functional as F
from datetime import datetime
from .metric import convert_tokens, evaluate_by_dict
from util.file_utils import pickle_load_large_file
import numpy as np


class Trainer(object):

    def __init__(self, args, model, loss,
                 train_data_loader, dev_data_loader,
                 train_eval_file, dev_eval_file,
                 optimizer, scheduler, epochs, with_cuda,
                 save_dir, verbosity=2, save_freq=1, print_freq=10,
                 resume=False, identifier='',
                 debug=False, debug_batchnum=2,
                 visualizer=None, logger=None,
                 grad_clip=5.0, decay=0.9999,
                 lr=0.001, lr_warm_up_num=1000,
                 use_scheduler=False, use_grad_clip=False,
                 use_ema=False, ema=None,
                 use_early_stop=False, early_stop=10):
        self.device = ("gpu" if with_cuda else "cpu")
        self.args = args

        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.epochs = epochs
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.print_freq = print_freq
        self.verbosity = verbosity
        self.identifier = identifier
        self.visualizer = visualizer
        self.with_cuda = with_cuda

        self.train_data_loader = train_data_loader
        self.dev_data_loader = dev_data_loader
        self.dev_eval_dict = pickle_load_large_file(dev_eval_file)
        self.is_debug = debug
        self.debug_batchnum = debug_batchnum
        self.logger = logger
        self.unused = True  # whether scheduler has been updated

        self.lr = lr
        self.lr_warm_up_num = lr_warm_up_num
        self.decay = decay
        self.use_scheduler = use_scheduler
        self.scheduler = scheduler
        self.use_grad_clip = use_grad_clip
        self.grad_clip = grad_clip
        self.use_ema = use_ema
        self.ema = ema
        self.use_early_stop = use_early_stop
        self.early_stop = early_stop

        self.start_time = datetime.now().strftime('%b-%d_%H-%M')
        self.start_epoch = 1
        self.step = 0
        self.best_em = 0
        self.best_f1 = 0
        if resume:
            self._resume_checkpoint(resume)
            # self.model = self.model.to(self.device)
            # for state in self.optimizer.state.values():
            #     for k, v in state.items():
            #         if isinstance(v, paddle.Tensor):
            #             state[k] = v

    def train(self):
        patience = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            if self.use_early_stop:
                if result["f1"] < self.best_f1 and result["em"] < self.best_em:
                    patience += 1
                    if patience > self.early_stop:
                        print("Perform early stop!")
                        break
                else:
                    patience = 0

            is_best = False
            if result["f1"] > self.best_f1:
                is_best = True
            if result["f1"] == self.best_f1 and result["em"] > self.best_em:
                is_best = True
            self.best_f1 = max(self.best_f1, result["f1"])
            self.best_em = max(self.best_em, result["em"])

            # if epoch % self.save_freq == 0:
            #     self._save_checkpoint(
            #         epoch, result["f1"], result["em"], is_best)

    def _train_epoch(self, epoch):
        self.model.train()
        # self.model.to(self.device)

        # initialize
        global_loss = 0.0
        last_step = self.step - 1
        last_time = time.time()

        # self.model.train()
        # train over batches

        # reprod_logger = ReprodLogger()
        # with paddle.no_grad():
        fake_batch = None
        for batch_idx, batch in enumerate(self.train_data_loader):
            # paddle.save(self.model.state_dict(), f"{batch_idx}.data")
            # if fake_batch is None:
            #     fake_batch = batch
            # if batch_idx >= 10:
            #     reprod_logger.save("bp_align_paddle.npy")
            #     exit(0)
            # get batch
            (context_wids,
                context_cids,
                question_wids,
                question_cids,
                y1,
                y2,
                y1s,
                y2s,
                id,
                answerable) = batch
            batch_num, question_len = question_wids.shape
            _, context_len = context_wids.shape
            context_wids = context_wids
            context_cids = context_cids
            question_wids = question_wids
            question_cids = question_cids
            # y1 = y1
            # y2 = y2
            # id = id
            # answerable = answerable

            # calculate loss
            # self.model.zero_grad
            # print(context_wids)
            # print(context_cids)
            # print(question_wids)
            # print(question_cids)

            p1, p2 = self.model(
                context_wids,
                context_cids,
                question_wids,
                question_cids)
            # print("context_wids",context_wids.gradient)
            # print("context_cids",context_cids.gradient)
            # print("question_wids",question_wids.gradient)
            # print("question_cids",question_cids.gradient)
            # log_p1 = F.log_softmax(p1, dim=1)
            # log_p2 = F.log_softmax(p2, dim=1)
            loss1 = self.loss(p1, y1)
            loss2 = self.loss(p2, y2)
            # print("loss1", loss1)
            # print("loss2", loss2)
            # reprod_logger = ReprodLogger()
            # reprod_logger.add("loss1", loss1.numpy())
            # reprod_logger.add("loss2", loss2.numpy())
            # reprod_logger.save("loss_paddle.npy")
            # exit(0)

            loss = paddle.mean(loss1 + loss2)
            loss.backward()

            # for name, param in self.model.named_parameters():
            #     print(name, param)
            #     np.savetxt(str(name)+str(batch_idx)+".txt",param.numpy())
            global_loss += loss.item()
            # print("loss"+str(batch_idx), loss.item())
            # print("lr"+str(batch_idx), np.array(self.scheduler.get_lr()))
            # reprod_logger.add("loss"+str(batch_idx), np.array(loss.item()))
            # reprod_logger.add("lr"+str(batch_idx), np.array(self.scheduler.get_lr()))

            # gradient clip
            clip = None
            if self.use_grad_clip:
                clip = paddle.nn.ClipGradByNorm(clip_norm=self.grad_clip)
            # update model
            self.optimizer.step()
            

            # update learning rate
            if self.use_scheduler:
                self.scheduler.step()
            self.optimizer.clear_grad()
            # exponential moving avarage
            if self.use_ema and self.ema is not None:
                self.ema(self.model, self.step)

            # print training info
            if self.step % self.print_freq == self.print_freq - 1:
                used_time = time.time() - last_time
                step_num = self.step - last_step
                speed = self.train_data_loader.batch_size * \
                    step_num / used_time
                batch_loss = global_loss / step_num
                print(("step: {}/{} \t "
                        "epoch: {} \t "
                        "lr: {} \t "
                        "loss: {} \t "
                        "speed: {} examples/sec").format(
                            batch_idx, len(self.train_data_loader),
                            epoch,
                            self.optimizer.get_lr(),
                            batch_loss,
                            speed))
                global_loss = 0.0
                last_step = self.step
                last_time = time.time()
            self.step += 1

            if self.is_debug and batch_idx >= self.debug_batchnum:
                break

        metrics = self._valid_eopch(self.dev_eval_dict, self.dev_data_loader)
        
        # reprod_logger = ReprodLogger() 

        # reprod_logger.add("em", np.array(metrics["exact_match"]))
        # reprod_logger.add("f1", np.array(metrics["f1"]))
        # print(metrics)
        # reprod_logger.save("metric_paddle.npy")
        # exit(0)
        print("dev_em: %f \t dev_f1: %f" % (
              metrics["exact_match"], metrics["f1"]))

        result = {}
        result["em"] = metrics["exact_match"]
        result["f1"] = metrics["f1"]
        if result["em"] >= 65.0 and result["f1"] >= 75.0:
            self._save_checkpoint(epoch, result["f1"], result["em"], False)

        return result

    def _valid_eopch(self, eval_dict, data_loader):
        """
        Evaluate model over development dataset.
        Return the metrics: em, f1.
        """
        if self.use_ema and self.ema is not None:
            self.ema.assign(self.model)

        self.model.eval()
        answer_dict = {}
        with paddle.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                # print(batch)
                (context_wids,
                 context_cids,
                 question_wids,
                 question_cids,
                 y1,
                 y2,
                 y1s,
                 y2s,
                 id,
                 answerable) = batch
                context_wids = context_wids
                context_cids = context_cids
                question_wids = question_wids
                question_cids = question_cids
                # y1 = y1
                # y2 = y2
                # answerable = answerable
                # print("context_wids",context_wids)
                # print("context_cids",context_cids)
                # print("question_wids",question_wids)
                # print("question_cids",question_cids)
                p1, p2 = self.model(
                    context_wids,
                    context_cids,
                    question_wids,
                    question_cids)
                # print("p1",p1)
                # print("p2",p2)
                p1 = F.softmax(p1, axis=1)
                p2 = F.softmax(p2, axis=1)

                # print("p1",p1)
                # print("p2",p2)
                outer = paddle.matmul(p1.unsqueeze(2), p2.unsqueeze(1))
                for j in range(outer.shape[0]):
                    outer[j] = paddle.triu(outer[j])
                    # outer[j] = torch.tril(outer[j], self.args.ans_limit)
                #     print(j,outer[j])
                # exit(0)
                a1 = paddle.max(outer, axis=2)
                a2 = paddle.max(outer, axis=1)
                ymin = paddle.argmax(a1, axis=1)
                ymax = paddle.argmax(a2, axis=1)
                answer_dict_, _ = convert_tokens(
                    eval_dict, id.tolist(), ymin.tolist(), ymax.tolist())
                answer_dict.update(answer_dict_)
                # print(answer_dict)
                

                if((batch_idx + 1) == self.args.val_num_batches):
                    break

                if self.is_debug and batch_idx >= self.debug_batchnum:
                    break
        # print("eval_dict",eval_dict)
        # print("answer_dict",answer_dict)
        metrics = evaluate_by_dict(eval_dict, answer_dict)

        if self.use_ema and self.ema is not None:
            self.ema.resume(self.model)
        self.model.train()
        return metrics

    def _save_checkpoint(self, epoch, f1, em, is_best):
        if self.use_ema and self.ema is not None:
            self.ema.assign(self.model)
        arch = type(self.model).__name__
        state = {
            'epoch': epoch,
            'arch': arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_f1': self.best_f1,
            'best_em': self.best_em,
            'step': self.step + 1,
            'start_time': self.start_time}
        filename = os.path.join(
            self.save_dir,
            self.identifier +
            'checkpoint_epoch{:02d}_f1_{:.5f}_em_{:.5f}.pth.tar'.format(
                epoch, f1, em))
        print("Saving checkpoint: {} ...".format(filename))
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        paddle.save(state, filename)
        if is_best:
            shutil.copyfile(
                filename, os.path.join(self.save_dir, 'model_best.pth.tar'))
        if self.use_ema and self.ema is not None:
            self.ema.resume(self.model)
        return filename

    def _resume_checkpoint(self, resume_path):
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = paddle.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.model.set_state_dict(checkpoint['state_dict'])
        self.optimizer.set_state_dict(checkpoint['optimizer'])
        self.best_f1 = checkpoint['best_f1']
        self.best_em = checkpoint['best_em']
        self.step = checkpoint['step']
        self.start_time = checkpoint['start_time']
        if self.use_scheduler:
            self.scheduler.last_epoch = checkpoint['epoch']
        print("Checkpoint '{}' (epoch {}) loaded".format(
            resume_path, self.start_epoch))
