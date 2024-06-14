import torch
import time
import os
import numpy as np

from mvit.logging_utils.logger import logger
module_logger = logger.getChild(__name__)

class Trainer():
  '''
  Super class implimenting basic training functionality
  '''
  def __init__(self, dataloaders_train, dataloaders_test, device, metrics, model,
               save_model_param_path=None, loss_fn=torch.nn.CrossEntropyLoss(),
               lr_scheduler=None, optimizer_fn=torch.optim.Adam, 
               optimizer_params={'lr':0.001}, print_epoch_time=False, 
               autoencoder=False, save_during_training=False,
               stop_epoch_count=5, model_name = ''):
    module_logger.info('Trainer Initializing')
    self.autoencoder = autoencoder # Loss function receive output and features
    self.metrics = metrics
    self.device = device
    self.save_model_param_path = save_model_param_path
    self.dataloaders_train = dataloaders_train
    self.dataloaders_test = dataloaders_test
    self.loss_fn = loss_fn
    self.print_epoch_time = print_epoch_time
    self.model = model.to(device)
    self.optimizer = self.get_optimizer(optimizer_fn, optimizer_params)
    self.best_metric_value = float('-inf')
    self.save_during_training = save_during_training
    self.stop_epoch_count = stop_epoch_count ## If performance is low for consecutive times, end training
    self.low_performance_rounds = 0
    self.model_name = model_name


  
  def get_optimizer(self, optimizer_fn, optimizer_params):
    module_logger.debug('Using optimizer {}'.format(optimizer_fn))
    trainable_params = []
    for param in self.model.parameters():
      if param.requires_grad:
        trainable_params.append(param)
    return optimizer_fn(trainable_params, **optimizer_params)

  def save_model(self):
    '''
    Saving model parameters in self.save_model_param_path
    '''
    module_logger.info('Saving model')
    print('Saving Model to {}'.format(self.save_model_param_path))
    torch.save(self.model.state_dict(), self.save_model_param_path)

  def load_model(self):
    '''
    For loading models parameters
    '''
    module_logger.info('Loading model')
    if os.path.exists(self.save_model_param_path):
      print('Loading model from {}'.format(self.save_model_param_path))
      self.model.load_state_dict(torch.load(self.save_model_param_path))


  def _test_step(self, dataloader):
    '''
    Used for evaluation time. During this time the gradients are disabled.
    '''
    self.model.eval()
    for features, labels in dataloader:
      # For transfering data to GPU
      if self.device != features.device:
        features = features.to(self.device)
        labels = labels.to(self.device)

      with torch.set_grad_enabled(False):
        outputs = self.model(features)
        if self.autoencoder:
          loss = self.loss_fn(outputs, features)
        else:
          loss = self.loss_fn(outputs, labels)
        for metric in self.metrics:
          if hasattr(metric, 'required_x') and metric.required_x:
            metric.update(outputs=outputs, labels=labels, 
                          features=features, loss=loss, phase='test')
          else:
            metric.update(outputs, labels, phase='test')


  def _train_step(self, dataloader):
    '''
    It will train a single dataset a single epoch
    Returning Average Loss, Accuracy and Time taken for execution
    '''
    self.model.train()
    # Iterate over data.
    for features, labels in dataloader:

      # For transfering data to GPU 
      if self.device != features.device:
        features = features.to(self.device)
        labels = labels.to(self.device)

      # Training
      self.optimizer.zero_grad()
      with torch.set_grad_enabled(True):
        outputs = self.model(features)
        if self.autoencoder:
          loss = self.loss_fn(outputs, features)
        else:
          loss = self.loss_fn(outputs, labels)
      loss.backward()
      self.optimizer.step()

      with torch.set_grad_enabled(False):
        for metric in self.metrics:
          if hasattr(metric, 'required_x') and metric.required_x:
            metric.update(outputs=outputs, labels=labels, 
                          features=features, loss=loss, phase='train')            
          else:
            metric.update(outputs, labels, phase='train')


  def _train_stage(self, progress=True, training=True):
    '''
    Function to train entire dataset one epoch
    '''

    for dataloader in self.dataloaders_train:
      self._train_step(dataloader)
      if progress:
        print('.', end='')
    print()

  def _test_stage(self, progress=True, training=False):
    '''
    Function to train entire dataset one epoch
    '''

    for dataloader in self.dataloaders_test:
      self._test_step(dataloader)
      if progress:
        print('.', end='')
    print()

  def train(self, epochs):
    print('*'*20)
    print(f'Training - {self.model_name}')
    for i in range(1, epochs+1):
      last = time.time()
      print('Epoch: {}'.format(i))
      ## Training and logging 
      print('\tTraining', end='')
      self._train_stage()
      for metric in self.metrics:
        metric.compute(phase='train')
        print('\t\t{}'.format(str(metric)))

      ## Testing and logging
      print('\tTesting', end='')
      self._test_stage()
      for metric in self.metrics:
        metric.compute(phase='test')
        print('\t\t{}'.format(str(metric)))
      time_taken = time.time() - last
      if self.print_epoch_time:
        print('\t\tTime Taken: {:4.2f}'.format(time_taken))

      master_metric_value = self.metrics[0].get_metric_value()
      if master_metric_value > self.best_metric_value and self.save_during_training:
        self.save_model()
      else:
        self.low_performance_rounds += 1

      if self.low_performance_rounds >= self.stop_epoch_count:
        module_logger.info('Testing Accuracy is dropping, Exiting the training process')
        break
      






