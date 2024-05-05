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
  def __init__(self, dataset_manager, device, metrics, model=None,
               save_model_param_path=None, loss_fn=torch.nn.CrossEntropyLoss(),
               lr_scheduler=None, optimizer_fn=torch.optim.Adam, num_test_videos=16,
               optimizer_params={'lr':0.001}, print_epoch_time=False, random_train_test=False):
    module_logger.info('Trainer Initializing')
    self.metrics = metrics
    self.device = device
    self.save_model_param_path = save_model_param_path
    self.dataset_manager = dataset_manager
    self.loss_fn = loss_fn
    self.print_epoch_time = print_epoch_time
    self.model = model.to(device)
    self.optimizer = self.get_optimizer(optimizer_fn, optimizer_params)
    if random_train_test:
      indices = range(1, 81)
      self.validation_video_index = np.random.choice(indices, num_test_videos, replace=False)
      self.training_video_index = np.setxor1d(indices, self.training_video_index)
    else:
      self.validation_video_index = range(1, num_test_videos)
      self.training_video_index = range(num_test_videos, 81)

  
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
      features = features.to(self.device)
      labels = labels.to(self.device)

      with torch.set_grad_enabled(False):
        outputs = self.model(features)
        for metric in self.metrics:
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
      features = features.to(self.device)
      labels = labels.to(self.device)

      # Training
      self.optimizer.zero_grad()
      with torch.set_grad_enabled(True):
        outputs = self.model(features)
        loss = self.loss_fn(outputs, labels)
      loss.backward()
      self.optimizer.step()

      with torch.set_grad_enabled(False):
        for metric in self.metrics:
          metric.update(outputs, labels, phase='train')


  def _train_stage(self, feature_keys=None, label_key=None, progress=True):
    '''
    Function to train entire dataset one epoch
    '''

    for idx in self.training_video_index:
      if feature_keys is None:
        dataloader = self.dataset_manager.get_dataloader(idx)
      else:
        dataloader = self.dataset_manager.get_dataloader(idx, feature_keys, label_key)
      self._train_step(dataloader)
      if progress:
        print('.', end='')
    print()

  def _test_stage(self, feature_keys=None, label_key=None, progress=True):
    '''
    Function to train entire dataset one epoch
    '''

    for idx in self.validation_video_index:
      if feature_keys is None:
        dataloader = self.dataset_manager.get_dataloader(idx)
      else:
        dataloader = self.dataset_manager.get_dataloader(idx, feature_keys, label_key)
      self._test_step(dataloader)
      if progress:
        print('.', end='')
    print()

  def train(self, epochs, feature_keys=None, label_key=None):
    for i in range(1, epochs+1):
      last = time.time()
      print('Epoch: {}'.format(i))
      ## Training and logging 
      print('\tTraining', end='')
      self._train_stage(feature_keys, label_key)
      for metric in self.metrics:
        metric.compute(phase='train')
        print('\t\t{}'.format(str(metric)))

      ## Testing and logging
      print('\tTesting', end='')
      self._test_stage(feature_keys, label_key)
      for metric in self.metrics:
        metric.compute(phase='test')
        print('\t\t{}'.format(str(metric)))
      time_taken = time.time() - last
      if self.print_epoch_time:
        print('\t\tTime Taken: {:4.2f}'.format(time_taken))





