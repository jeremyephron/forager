import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sklearn.metrics
import os.path
import logging
from typing import Dict, List, Any
from torch.utils.data import DataLoader
from torchvision import transforms

from config import BATCH_SIZE, NUM_WORKERS
from model import Model
from dataset import AuxiliaryDataset
from warmup_scheduler import GradualWarmupScheduler
from util import download

logger = logging.getLogger("bgsplit")
logger.setLevel(logging.DEBUG)

class TrainingLoop(nn.Module):
    def __init__(
            self,
            model_kwargs: Dict[str, Any],
            train_positive_paths: List[str],
            train_negative_paths: List[str],
            train_unlabeled_paths: List[str],
            val_positive_paths: List[str],
            val_negative_paths: List[str],
            val_unlabeled_paths: List[str],
            aux_labels: Dict[str, int]):
        '''The training loop for background splitting models.'''
        batch_size = BATCH_SIZE
        num_workers = NUM_WORKERS
        self.val_frequency = model_kwargs.get('val_frequency', 1)
        self.checkpoint_frequency = model_kwargs.get('checkpoint_frequency', 1)
        assert 'model_dir' in model_kwargs
        self.model_dir = model_kwargs['model_dir']
        assert 'aux_labels' in model_kwargs
        aux_labels = model_kwargs['aux_labels']

        # Setup dataset
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
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
        self.validate_dataloader = DataLoader(
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
        num_aux_classes = len(torch.unique(list(aux_labels.values())))
        self.model = Model(num_main_classes=num_classes,
                           num_aux_classes=num_aux_classes)
        self.model = self.model.cuda()
        self.model = nn.DataParallel(self.model)
        self.main_loss = nn.CrossEntropyLoss()
        self.auxiliary_loss = nn.CrossEntropyLoss()

        # Setup optimizer
        optim_params = dict(
            lr=model_kwargs.get('lr', 0.1),
            momentum=model_kwargs.get('momentum', 0.9),
            weight_decay=model_kwargs.get('weight_decay', 0.0001),
        )
        self.optimizer = optim.SGD(self.model.parameters(), **optim_params)
        self.optimizer_scheduler = GradualWarmupScheduler(
            optimizer=self.optimizer,
            multiplier=1.0,
            warmup_epochs=model_kwargs.get('warmup_epochs', 0))

        # Resume if requested
        resume_from = model_kwargs.get('resume_from', None)
        if resume_from:
            self.load_checkpoint(resume_from)

    def load_checkpoint(self, path: str, restart: bool=True):
        checkpoint_state = torch.load(path)
        self.model.load_state_dict(checkpoint_state['state_dict'])
        if not restart:
            self.start_epoch = checkpoint_state['epoch']

    def save_checkpoint(self, checkpoint_path: str):
        state = dict(
            epoch=self.epoch,
            state_dict=self.model.state_dict,
        )
        torch.save(state, checkpoint_path)

    def _validate(self, dataloader):
        self.model.eval()
        loss_value = 0
        main_gts = []
        aux_gts = []
        main_preds = []
        aux_preds = []
        for images, main_labels, aux_labels in dataloader:
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
        # Compute F1 score
        loss_value /= len(dataloader)
        main_prec, main_recall, _, _ = \
            sklearn.metrics.precision_recall_fscore_support(
                main_gts, main_preds)
        aux_prec, aux_recall, _, _ = \
            sklearn.metrics.precision_recall_fscore_support(
                main_gts, main_preds)
        print(f'main: prec: {main_prec:.3f}, recall: {main_recall:.3f}')
        print(f'aux:  prec: {aux_prec:.3f}, recall: {aux_recall:.3f}')

    def validate(self):
        self._validate(self.val_dataloader)

    def train(self):
        self.model.train()
        self.optimizer_scheduler.step()
        logger.log('Starting train epoch')
        for images, main_labels, aux_labels in self.train_dataloader:
            logger.log('Train batch')
            images = images.cuda()
            main_labels = main_labels.cuda()
            aux_labels = aux_labels.cuda()

            main_logits, aux_logits = self.model(images)
            # Compute loss
            main_loss_value = self.main_loss(
                main_logits[main_labels != -1], main_labels[main_labels != -1])
            aux_loss_value = self.auxiliary_loss(
                aux_logits[aux_labels != -1], aux_labels[aux_labels != -1])
            loss_value = main_loss_value + aux_loss_value
            # Update gradients
            self.optimizer.zero_grad()
            loss_value.backward()
            self.optimizer.step()

    def run(self, num_epochs=1):
        last_checkpoint_path = None
        for i in range(self.start_epoch, num_epochs):
            print(f'Train: Epoch {i}')
            self.train()
            if i % self.val_frequency == 0 or i == num_epochs - 1:
                print(f'Validate: Epoch {i}')
                self.validate()
            if i % self.checkpoint_frequency == 0 or i == num_epochs - 1:
                print(f'Checkpoint: Epoch {i}')
                last_checkpoint_path = os.path.join(
                    self.model_dir, f'checkpoint_{i:03}.pth')
                self.save_checkpoint(last_checkpoint_path)
        return last_checkpoint_path
