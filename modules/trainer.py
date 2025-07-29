import os
from abc import abstractmethod
import ast
import time
import torch
import pandas as pd
from numpy import inf
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import torch.distributed as dist
from .metrics_own import cal_metrics
import csv
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support,roc_auc_score,f1_score,average_precision_score

class FocalCELoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalCELoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * CE_loss
        # reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class BaseTrainer(object):
    def __init__(self, model, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,test_dataloader):

        self.args = args

        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.model = model
        self.criterion = FocalCELoss(alpha=0.25, gamma=2.0)
        self.optimizer = optimizer
        self.epochs = self.args.epochs
        self.epochs_val = self.args.epochs_val
        self.start_val = self.args.start_val
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + "accuracy"
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1

        self.checkpoint_dir = args.save_dir
        self.num_classes = self.args.n_classes
        self.result_file = os.path.join(args.save_dir, 'test_results_new.csv')
        self.metrics_file = os.path.join(args.save_dir, 'metrice_results_new.csv')

        if args.resume is not None:
            self._resume_checkpoint(args.resume)

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best}}

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self,rank):

        self.epoch_results = []
        epoch_result_file = os.path.join(self.args.save_dir, 'epoch_results.csv')
        fieldnames = ['epoch', 'train_loss', 'train_accuracy', 'val_accuracy']
        if rank == 0 and not os.path.exists(epoch_result_file):
            with open(epoch_result_file, mode='w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):

            self.train_dataloader.sampler.set_epoch(epoch)
            result = self._train_epoch(epoch)

            best = False

            if epoch % self.epochs_val== 0 and epoch >= self.start_val : #validation
                val_result = self._val_epoch(rank, result)

                # save logged informations into log dict
                log = {'epoch': epoch}
                log.update(val_result)

                # Save the log to epoch_results list
                self.epoch_results.append(log)
                self._record_best(log)

                # print logged informations to the screen
                for key, value in log.items():
                    print('\t{:15s}: {}'.format(str(key), value))

                # evaluate model performance according to configured metric, save best checkpoint as model_best
                if self.mnt_mode != 'off':
                    try:
                        # check whether model performance improved or not, according to specified metric(mnt_metric)
                        improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                                   (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                    except KeyError:
                        print("Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                            self.mnt_metric))
                        self.mnt_mode = 'off'
                        improved = False

                    if improved:
                        self.mnt_best = log[self.mnt_metric]
                        not_improved_count = 0
                        best = True
                    else:
                        not_improved_count += 1

                    if not_improved_count > self.early_stop:
                        print("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
                            self.early_stop))
                        break

                    
            if epoch % self.save_period == 0 and rank==0:
                self._save_checkpoint(epoch, save_best=best)

                with open(epoch_result_file, mode='a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writerow({
                        'epoch': epoch,
                        'train_loss': log['train_loss'],
                        'train_accuracy': log['train_accuracy'],
                        'val_accuracy': log['val_accuracy']
                    })
            # import torch
            torch.cuda.empty_cache()

        if rank == 0:
            self._print_best()
            self._print_best_to_file()

    def test(self, rank):
        self.model.eval()
        results = []
        all_labels = []
        all_predictions = []
        all_probs = []
        with torch.no_grad():
            for batch_idx, (images_id, images, labels, text_features) in enumerate(
                    tqdm(self.test_dataloader, desc='Testing', mininterval=300)):
                images = images.cuda()
                text_features = text_features.cuda()
                labels = torch.tensor(labels).cuda()
                output = self.model(images, text_features, mode='train')
                predicted = output[1]
                probabilities = output[2]
                for i in range(len(images_id)):
                    results.append({
                        "image_id": images_id[i],
                        "true": labels[i].cpu().numpy(),
                        "predict": predicted[i].cpu().numpy(),
                        "probs": probabilities[i].cpu().numpy().tolist()
                    })
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probs.extend((probabilities.cpu().numpy()))
        with open(self.result_file, mode='w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["image_id", "true", "predict", "probs"])
            writer.writeheader()
            writer.writerows(results)
        self.metrics_print(self.result_file)

    def metrics_print(self, result_file):
        df = pd.read_csv(result_file)

        true_labels = np.array([ast.literal_eval(true) for true in df['true']])  # 真实标签
        predicted_labels = np.array([ast.literal_eval(predict)[0] for predict in df['predict']])  # 预测的标签
        pred_scores = np.array([ast.literal_eval(probs) for probs in df['probs']])  # 预测分数

        num_labels = true_labels.shape[1]

        # AUROC,PRAUC
        auc_per_class = roc_auc_score(true_labels, pred_scores, average=None)
        pr_auc_per_class = average_precision_score(true_labels, pred_scores, average=None)

        # ACC , F1-score
        acc_per_class = []
        f1_per_class = []
        for i in range(num_labels):
            acc = accuracy_score(true_labels[:, i], predicted_labels[:, i])
            f1 = f1_score(true_labels[:, i], predicted_labels[:, i])
            acc_per_class.append(acc)
            f1_per_class.append(f1)

        macro_auc = roc_auc_score(true_labels, pred_scores, average='macro')
        macro_pr_auc = average_precision_score(true_labels, pred_scores, average='macro')
        micro_f1 = f1_score(true_labels, predicted_labels, average='micro')
        micro_acc = accuracy_score(true_labels.flatten(), predicted_labels.flatten())
        results_str = []
        for i in range(num_labels):
            results_str.append(
                f"Class {i + 1}: AUROC = {auc_per_class[i]:.4f}, PRAUC = {pr_auc_per_class[i]:.4f}, "
                f"ACC = {acc_per_class[i]:.4f}, F1-score = {f1_per_class[i]:.4f}"
            )

        results_str.append("\nResults:")
        results_str.append(f"ACC: {micro_acc:.4f}, F1-score: {micro_f1:.4f}")
        results_str.append(f"AUROC: {macro_auc:.4f}, PRAUC: {macro_pr_auc:.4f}")
        for line in results_str:
            print(line)
        os.makedirs(self.args.save_dir, exist_ok=True)
        metrics_file = os.path.join(self.args.save_dir, "metrics_best.txt")
        with open(metrics_file, "w") as f:
            f.write("\n".join(results_str))
        print(f"\nMetrics saved to {metrics_file}")


    def _print_best_to_file(self):
        crt_time = time.asctime(time.localtime(time.time()))
        self.best_recorder['val']['time'] = crt_time
        self.best_recorder['val']['seed'] = self.args.seed
        self.best_recorder['val']['best_model_from'] = 'val'

        if not os.path.exists(self.args.record_dir):
            os.makedirs(self.args.record_dir)
        record_path = os.path.join(self.args.record_dir, self.args.dataset_name+'.csv')
        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)
        new_row = pd.DataFrame([self.best_recorder['val']])
        record_table = pd.concat([record_table, new_row], ignore_index=True)
        # record_table = record_table.append(self.best_recorder['val'], ignore_index=True)
        record_table.to_csv(record_path, index=False)

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        if epoch % 20 ==0:
            filename = os.path.join(self.checkpoint_dir, f'current_checkpoint_{epoch}.pth')
            torch.save(state, filename)
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            print("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _record_best(self, log):

        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
            self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            self.best_recorder['val'].update(log)


    def _print_best(self):
        print('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['val'].items():
            print('\t{:15s}: {}'.format(str(key), value))


class Trainer(BaseTrainer):
    def __init__(self, model, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader):
        super(Trainer, self).__init__(model, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader)

    def _train_epoch(self, rank):
        dist.barrier()
        train_loss = 0
        correct = 0
        total = 0
        self.model.train()
        for batch_idx, (images_id, images, labels, text_features) in enumerate(tqdm(self.train_dataloader, desc='Training', mininterval=300)):
            images = images.cuda()
            labels = torch.tensor(labels).cuda()
            text_features = text_features.cuda()
            output = self.model(images,text_features, mode='train')
            logist = output[0]

            loss = self.criterion(logist, labels)
            predicted = output[1]
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            train_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
        accuracy = correct / total * 100
        log = {'train_loss': train_loss / len(self.train_dataloader), 'train_accuracy': accuracy}
        self.lr_scheduler.step()

        return log
    
    def _val_epoch(self, rank, log):
        dist.barrier()
        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (images_id, images, labels, text_features) in enumerate(tqdm(self.val_dataloader,desc='Validating', mininterval=300)):
                images = images.cuda()
                labels = torch.tensor(labels).cuda()
                text_features = text_features.cuda()
                # print('start eval...')
                output = self.model(images, text_features, mode='train')
                predicted = output[1]
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            accuracy = correct / total * 100  # 计算准确率
            log.update({'val_accuracy': accuracy})
        return log


def distributed_concat(tensor, num_total_examples):
    output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    # truncate the dummy elements added by SequentialDistributedSampler
    return concat[:num_total_examples]
