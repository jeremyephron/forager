import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sklearn.metrics
import os
import os.path
import logging
import time
from typing import Dict, List, Any, Callable
from torch.utils.data import DataLoader
from torchvision import transforms

from config import NUM_WORKERS
from model import Model
from dataset import AuxiliaryDataset
from warmup_scheduler import GradualWarmupScheduler
from util import download, EMA

logger = logging.getLogger("bgsplit")
logger.setLevel(logging.DEBUG)

class TrainingLoop():
    def __init__(
            self,
            model_kwargs: Dict[str, Any],
            train_positive_paths: List[str],
            train_negative_paths: List[str],
            train_unlabeled_paths: List[str],
            val_positive_paths: List[str],
            val_negative_paths: List[str],
            val_unlabeled_paths: List[str],
            notify_callback: Callable[[Dict[str, Any]], None]=lambda x: None):
        '''The training loop for background splitting models.'''
        self.model_kwargs = model_kwargs
        batch_size = model_kwargs.get('batch_size', 512)
        num_workers = NUM_WORKERS
        self.val_frequency = model_kwargs.get('val_frequency', 1)
        self.checkpoint_frequency = model_kwargs.get('checkpoint_frequency', 1)
        self.use_cuda = model_kwargs.get('use_cuda', True)
        assert 'model_dir' in model_kwargs
        self.model_dir = model_kwargs['model_dir']
        assert 'aux_labels' in model_kwargs
        aux_labels = model_kwargs['aux_labels']
        self.aux_weight = model_kwargs.get('aux_weight', 0.1)
        self.notify_callback = notify_callback

        # Setup dataset
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.train_dataloader = DataLoader(
            AuxiliaryDataset(
                positive_paths=train_positive_paths,
                negative_paths=train_negative_paths,
                unlabeled_paths=train_unlabeled_paths,
                auxiliary_labels=aux_labels,
                transform=train_transform),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers)
        self.val_dataloader = DataLoader(
            AuxiliaryDataset(
                positive_paths=val_positive_paths,
                negative_paths=val_negative_paths,
                unlabeled_paths=val_unlabeled_paths,
                auxiliary_labels=aux_labels,
                transform=val_transform),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers)

        # Setup model
        num_classes = 2
        num_aux_classes = self.train_dataloader.dataset.num_auxiliary_classes
        self.model_kwargs['num_aux_classes'] = num_aux_classes
        self.model = Model(num_main_classes=num_classes,
                           num_aux_classes=num_aux_classes)
        if self.use_cuda:
            self.model = self.model.cuda()
        self.model = nn.DataParallel(self.model)
        self.main_loss = nn.CrossEntropyLoss()
        self.auxiliary_loss = nn.CrossEntropyLoss()
        self.start_epoch = 0
        self.end_epoch = 1
        self.current_epoch = 0

        # Setup optimizer
        lr = model_kwargs.get('lr', 0.1)
        endlr = model_kwargs.get('endlr', 0.0)
        optim_params = dict(
            lr=lr,
            momentum=model_kwargs.get('momentum', 0.9),
            weight_decay=model_kwargs.get('weight_decay', 0.0001),
        )
        self.optimizer = optim.SGD(self.model.parameters(), **optim_params)
        max_epochs = model_kwargs.get('max_epochs', 90)
        warmup_epochs = model_kwargs.get('warmup_epochs', 0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, max_epochs - warmup_epochs,
            eta_min=endlr)
        self.optimizer_scheduler = GradualWarmupScheduler(
            optimizer=self.optimizer,
            multiplier=1.0,
            warmup_epochs=warmup_epochs,
            after_scheduler=scheduler)

        # Resume if requested
        resume_from = model_kwargs.get('resume_from', None)
        if resume_from:
            self.load_checkpoint(resume_from)

        # Variables for estimating run-time
        self.train_batch_time = EMA(0)
        self.val_batch_time = EMA(0)
        self.train_batches_per_epoch = (
            len(self.train_dataloader.dataset) /
            self.train_dataloader.batch_size)
        self.val_batches_per_epoch = (
            len(self.val_dataloader.dataset) /
            self.val_dataloader.batch_size)
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        self.train_epoch_loss = 0
        self.train_epoch_main_loss = 0
        self.train_epoch_aux_loss = 0

    def _notify(self):
        epochs_left = self.end_epoch - self.current_epoch - 1
        num_train_batches_left = (
            epochs_left * self.train_batches_per_epoch +
            max(0, self.train_batches_per_epoch - self.train_batch_idx - 1)
        )
        num_val_batches_left = (
            (1 + round(epochs_left / self.val_frequency)) * self.val_batches_per_epoch +
            max(0, self.val_batches_per_epoch - self.val_batch_idx - 1)
        )
        time_left = (
            num_train_batches_left * self.train_batch_time.value +
            num_val_batches_left * self.val_batch_time.value)
        logger.debug('Send notify')
        self.notify_callback(**{"training_time_left": time_left})

    def load_checkpoint(self, path: str, restart: bool=False):
        checkpoint_state = torch.load(path)
        self.model.load_state_dict(checkpoint_state['state_dict'])
        if not restart:
            self.start_epoch = checkpoint_state['epoch']
            self.current_epoch = self.start_epoch
            self.end_epoch = self.start_epoch + 1

    def save_checkpoint(self, epoch, checkpoint_path: str):
        kwargs = dict(self.model_kwargs)
        del kwargs['aux_labels']
        state = dict(
            model_kwargs=kwargs,
            epoch=epoch,
            state_dict=self.model.state_dict(),
        )
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(state, checkpoint_path)

    def _validate(self, dataloader):
        self.model.eval()
        loss_value = 0
        main_gts = []
        aux_gts = []
        main_preds = []
        aux_preds = []
        for batch_idx, (images, main_labels, aux_labels) in enumerate(
                dataloader):
            batch_start = time.perf_counter()
            self.val_batch_idx = batch_idx
            if self.use_cuda:
                images = images.cuda()
                main_labels = main_labels.cuda()
                aux_labels = aux_labels.cuda()
            main_logits, aux_logits = self.model(images)
            main_loss_value = self.main_loss(main_logits, main_labels)
            aux_loss_value = self.auxiliary_loss(aux_logits, aux_labels)
            loss_value += (main_loss_value + aux_loss_value).item()
            main_pred = F.softmax(main_logits)
            aux_pred = F.softmax(main_logits)
            main_preds += list(main_pred.argmax(dim=1)[:].cpu().numpy())
            aux_preds += list(aux_pred.argmax(dim=1)[:].cpu().numpy())
            main_gts += list(main_labels[:].cpu().numpy())
            aux_gts += list(main_labels[:].cpu().numpy())
            batch_end = time.perf_counter()
            self.val_batch_time += (batch_end - batch_start)
        # Compute F1 score
        if len(dataloader) > 0:
            loss_value /= (len(dataloader) + 1e-10)
            main_prec, main_recall, _, _ = \
                sklearn.metrics.precision_recall_fscore_support(
                    main_gts, main_preds)
            aux_prec, aux_recall, _, _ = \
                sklearn.metrics.precision_recall_fscore_support(
                    main_gts, main_preds)
        else:
            loss_value = 0
            main_prec = -1
            main_recall = -1
            aux_prec = -1
            aux_recall = -1
        logger.info(f'main: prec: {main_prec:.3f}, recall: {main_recall:.3f}')
        logger.info(f'aux:  prec: {aux_prec:.3f}, recall: {aux_recall:.3f}')

    def validate(self):
        self._validate(self.val_dataloader)

    def train(self):
        self.model.train()
        logger.info('Starting train epoch')
        load_start = time.perf_counter()
        self.train_epoch_loss = 0
        self.train_epoch_main_loss = 0
        self.train_epoch_aux_loss = 0
        for batch_idx, (images, main_labels, aux_labels) in enumerate(
                self.train_dataloader):
            load_end = time.perf_counter()
            batch_start = time.perf_counter()
            self.train_batch_idx = batch_idx
            logger.debug('Train batch')
            if self.use_cuda:
                images = images.cuda()
                main_labels = main_labels.cuda()
                aux_labels = aux_labels.cuda()

            main_logits, aux_logits = self.model(images)
            # Compute loss
            main_loss_value = self.main_loss(
                main_logits[main_labels != -1], main_labels[main_labels != -1])
            aux_loss_value = self.auxiliary_loss(
                aux_logits[aux_labels != -1], aux_labels[aux_labels != -1])
            loss_value = main_loss_value + self.aux_weight * aux_loss_value
            self.train_epoch_loss += loss_value.item()
            self.train_epoch_main_loss += main_loss_value.item()
            self.train_epoch_aux_loss += aux_loss_value.item()
            # Update gradients
            self.optimizer.zero_grad()
            loss_value.backward()
            self.optimizer.step()
            self.optimizer_scheduler.step()
            batch_end = time.perf_counter()
            total_batch_time = (batch_end - batch_start)
            total_load_time = (load_end - load_start)
            self.train_batch_time += total_batch_time + total_load_time
            logger.debug(f'Train batch time: {self.train_batch_time.value}, '
                         f'this batch time: {total_batch_time}, '
                         f'this load time: {total_load_time}, '
                         f'batch epoch loss: {loss_value.item()}, '
                         f'main loss: {main_loss_value.item()}, '
                         f'aux loss: {aux_loss_value.item()}')
            self._notify()
            load_start = time.perf_counter()
        logger.debug(f'Train epoch loss: {self.train_epoch_loss}, '
                     f'main loss: {self.train_epoch_main_loss}, '
                     f'aux loss: {self.train_epoch_aux_loss}')

    def run(self):
        last_checkpoint_path = None
        for i in range(self.start_epoch, self.end_epoch):
            logger.info(f'Train: Epoch {i}')
            self.current_epoch = i
            self.train()
            if i % self.val_frequency == 0 or i == self.end_epoch - 1:
                logger.info(f'Validate: Epoch {i}')
                self.validate()
            if i % self.checkpoint_frequency == 0 or i == self.end_epoch - 1:
                logger.info(f'Checkpoint: Epoch {i}')
                last_checkpoint_path = os.path.join(
                    self.model_dir, f'checkpoint_{i:03}.pth')
                self.save_checkpoint(i, last_checkpoint_path)
        return last_checkpoint_path
