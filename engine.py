import datetime
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F
from torch import nn
from datetime import datetime
import glob
import time
import sys
import numpy as np
import torch
from scipy import stats


class SoftCrossEntropyLoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights

    def forward(self, y_hat, y):
        batch_losses = -torch.sum(F.log_softmax(y_hat, dim=1) * y, dim=1)
        mask = (y == -1).any(dim=1)
        batch_losses[mask] *= 0
        mean_loss = batch_losses.sum() / ((batch_losses != 0).sum() + 1e-6)
        return mean_loss


class CustomF1Loss(nn.Module):
    def __init__(self, weights=None):
        super().__init__()

    def forward(self, output, target):
        mask = (target == -1)
        losses = abs(output - target)
        losses[mask] *= 0
        mean_loss = torch.sum(losses) / ((~mask).sum() + 1e-6)
        return mean_loss


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Fitter:
    def __init__(self, model, targets, config, modality, fold):
        self.config = config
        self.epoch = 0
        self.modality = modality
        self.base_dir = f'{config.folder}/{modality}/{fold}'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        self.log_path = f'{self.base_dir}/log.txt'
        self.best_summary_loss = 10 ** 5
        self.device = config.device
        self.model = model
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr,
                                           weight_decay=0.005)  # , weight_decay=0.0005
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config.lr, weight_decay=0.0001)
        self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)

        self.l1loss = CustomF1Loss()
        self.loss = SoftCrossEntropyLoss(
            torch.ones(len(targets.keys()), device=config.device))  # torch.nn.CrossEntropyLoss()

        self.targets = targets
        self.best_auc = {key: 0 if targets[key]["isClassification"] else None for key in targets.keys()}
        self.best_sublosses = {key: 1000 for key in targets.keys()}
        self.patience_counter = {key: 0 for key in self.targets.keys()}
        #  self.best_loss = 1000  # for eexperiment 22
        # self.aws = AutomaticWeightedLoss(num=len(targets))
        #  self.aws = MultiTaskLoss(is_regression=~torch.tensor([targets[key]["isClassification"] for key in targets.keys()]), reduction="mean")
        self.log(f'Fitter prepared. Device is {self.device}. Optimizer is {self.optimizer}.')

    def fit(self, train_loader, validation_loader, subfold):

        _tr = []
        _val = []

        _train_sublosses = {key: [] for key in self.targets.keys()}
        _val_sublosses = {key: [] for key in self.targets.keys()}
        _val_aucs = {key: [] for key in self.targets.keys()}

        for e in range(self.config.n_epochs):

            if self.config.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.utcnow().isoformat()
                self.log(f'\n{timestamp}\nLR: {lr}')

            t = time.time()
            train_summary_loss, train_summarylosses = self.train_one_epoch(train_loader)
            _tr.append(train_summary_loss.avg)
            for i, what in enumerate(self.targets.keys()):
                _train_sublosses[what].append(train_summarylosses[i].avg)

            self.log(
                f'[RESULT]: Train. Epoch: {self.epoch}, train_summary_loss: {train_summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')

            t = time.time()
            val_summary_loss, val_summarylosses, aucs = self.validation(validation_loader)
            _val.append(val_summary_loss.avg)
            for i, what in enumerate(self.targets.keys()):
                _val_sublosses[what].append(val_summarylosses[i].avg)
                _val_aucs[what].append(aucs[what])

            self.log(
                f'[RESULT]: Val. Epoch: {self.epoch}, val_summary_loss: {val_summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')

            if val_summary_loss.avg < self.best_summary_loss:
                print("saving best model")
                self.best_summary_loss = val_summary_loss.avg
                self.model.eval()
                self.save(f'{self.base_dir}/best-checkpoint-subfold{subfold}-{str(self.epoch).zfill(3)}epoch.bin')
                for path in sorted(glob.glob(f'{self.base_dir}/best-checkpoint-subfold{subfold}-*epoch.bin'))[:-1]:
                    os.remove(path)

            """for j, what in enumerate(self.targets.keys()):
                if val_summarylosses[j].avg < self.best_sublosses[what]:
                    self.best_sublosses[what] = val_summarylosses[j].avg
                    self.patience_counter[what] = 0
                    self.model.eval()
                    self.save(f'{self.base_dir}/{what}-sub-{subfold}-{str(self.epoch).zfill(3)}epoch.bin')
                    for path in sorted(glob.glob(f'{self.base_dir}/{what}-sub-{subfold}-*epoch.bin'))[:-1]:
                        os.remove(path)
                    else:
                        self.patience_counter[what] += 1"""

            # print("Epoch done, patience:", np.array([self.patience_counter[key] for key in self.targets.keys()]))

            if self.config.validation_scheduler:
                self.scheduler.step(metrics=val_summary_loss.avg)

            fig, ax = plt.subplots(nrows=3, figsize=(12, 8))
            ax[0].plot(np.arange(len(_tr)), np.array(_tr), label="train loss", linewidth=2)
            ax[0].plot(np.arange(len(_val)), np.array(_val), label="val loss", linewidth=2)
            # ax[0].set_title(self.model.backbone.name)
            ax[0].set_xlim([0, self.config.n_epochs])
            ax[0].legend()
            ax[0].grid()

            for j, what in enumerate(self.targets.keys()):
                ax[1].plot(np.arange(len(_val_sublosses[what])), np.array(_val_sublosses[what]), label="val" + what,
                           alpha=1)
            # ax[1].set_ylim([0, None if e < 3 else 1])
            ax[1].set_xlim([0, self.config.n_epochs])
            ax[1].legend()
            ax[1].grid()

            for j, what in enumerate(self.targets.keys()):
                ax[2].plot(np.arange(len(_val_aucs[what])), np.array(_val_aucs[what]), label="val" + what)
            ax[2].set_ylim([0, 1])
            ax[2].set_xlim([0, self.config.n_epochs])
            ax[2].legend()
            ax[2].grid()

            plt.savefig(f'{self.base_dir}/hist_subfold{subfold}.jpg', dpi=144)
            plt.close(fig)
            # np.save(self.base_dir+"/log.npy", np.array([_tr, _val, _auc]))
            self.epoch += 1

    def validation(self, val_loader):
        self.model.eval()
        summary_loss = AverageMeter()
        summarylosses = [AverageMeter() for i in range(len(self.targets.keys()))]
        t = time.time()

        gts = {key: [] for key in self.targets.keys()}
        pes = {key: [] for key in self.targets.keys()}

        for step, (images, y, _) in enumerate(val_loader):
            if self.modality == "face":
                img = images["face"].to(self.device, dtype=torch.float)
            elif self.modality == "body":
                if np.random.random() > 0.5:
                    img = torch.cat([images["front"], images["back"]], dim=1).to(self.device, dtype=torch.float)
                else:
                    img = torch.cat([images["back"], images["front"]], dim=1).to(self.device, dtype=torch.float)
            else:
                print("Modality not supported:", self.modality)

            sys.stdout.write('\r' +
                             f'Val Step {step}/{len(val_loader)}, ' + \
                             f'summary_loss: {summary_loss.avg:.5f}, ' + \
                             f'time: {(time.time() - t):.5f}')

            with torch.no_grad():
                batch_size = img.shape[0]
                output, _ = self.model(img)

            losses = []
            for i in range(len(self.targets.keys())):
                targetname = list(self.targets.keys())[i]
                _y = y[targetname].to(self.device)
                if self.targets[targetname]["isClassification"]:
                    ls = self.loss(output[i], _y) * self.targets[targetname]["loss_weight"]
                    pes[targetname].extend(
                        output[i].squeeze(1).softmax(dim=1).detach().cpu().numpy())  # .softmax(dim=1)
                    gts[targetname].extend(_y.detach().cpu().numpy())
                else:
                    ls = self.l1loss(output[i].reshape(-1), _y) * self.targets[targetname]["loss_weight"]
                    pes[targetname].extend(output[i].squeeze(1).detach().cpu().numpy())
                    gts[targetname].extend(_y.detach().cpu().numpy())
                losses.append(ls)
                summarylosses[i].update(ls.detach().item(), batch_size)

            loss = torch.stack(losses).sum()
            summary_loss.update(loss.detach().item(), batch_size)

        aucs = {}
        for i in range(len(self.targets.keys())):
            name = list(self.targets.keys())[i]
            if self.targets[name]["isClassification"]:
                gt = np.array(gts[name])
                mask = (gt.mean(axis=1) != -1)

                y_true = np.argmax(np.array(gt), axis=1)[mask]
                y_score = np.array(pes[name])[mask, 1]
                print("")
                print("VALIDATION, how many? :", name, len(y_true))  # 1432

                if self.targets[name]["out_dim"] == 2:
                    aucs[name] = roc_auc_score(y_true, y_score)
                elif self.targets[name]["out_dim"] > 2:
                    aucs[name] = roc_auc_score(y_true, y_score, multi_class="ovr")
            else:
                gt = np.array(gts[name])
                pe = np.array(pes[name])
                pe = pe[gt != -1]
                gt = gt[gt != -1]
                aucs[name] = stats.pearsonr(pe, gt)[0] ** 2
        print("")
        return summary_loss, summarylosses, aucs

    def train_one_epoch(self, train_loader):
        self.model.train()
        summary_loss = AverageMeter()
        summarylosses = [AverageMeter() for i in range(len(self.targets.keys()))]
        t = time.time()
        for step, (images, y, _) in enumerate(train_loader):
            if self.modality == "face":
                img = images["face"].to(self.device, dtype=torch.float)
            elif self.modality == "body":
                if np.random.random() > 0.5:
                    img = torch.cat([images["front"], images["back"]], dim=1).to(self.device, dtype=torch.float)
                else:
                    img = torch.cat([images["back"], images["front"]], dim=1).to(self.device, dtype=torch.float)
            else:
                print("Modality not supported:", self.modality)

            lr = self.optimizer.param_groups[0]['lr']
            sys.stdout.write('\r' +
                             f'Train Step {step}/{len(train_loader)}, ' + \
                             f'summary_loss: {summary_loss.avg:.5f}, ' + \
                             f'time: {(time.time() - t):.5f}, ' + \
                             f'lr: {lr:.7f}'
                             )

            batch_size = img.shape[0]
            self.optimizer.zero_grad()
            output, _ = self.model(img)

            losses = []
            for i in range(len(self.targets.keys())):
                targetname = list(self.targets.keys())[i]
                _y = y[targetname].to(self.device)
                if self.targets[targetname]["isClassification"]:
                    ls = self.loss(output[i], _y) * self.targets[targetname]["loss_weight"]
                else:
                    ls = self.l1loss(output[i].reshape(-1), _y) * self.targets[targetname]["loss_weight"]
                losses.append(ls)
                summarylosses[i].update(ls.detach().item(), batch_size)

            loss = torch.stack(losses).sum()
            loss.backward()

            summary_loss.update(loss.detach().item(), batch_size)

            self.optimizer.step()

            if self.config.step_scheduler:
                self.scheduler.step()
        print("")
        return summary_loss, summarylosses

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_summary_loss = checkpoint["best_summary_loss"]
        self.epoch = checkpoint["epoch"] + 1
        self.model.eval()
        print("Checkpoint loaded for", self.epoch - 1)

    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
        }, path)

    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')
