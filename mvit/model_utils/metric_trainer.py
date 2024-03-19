import torch
import time
import os

from mvit.logging_utils.logger import logger
module_logger = logger.getChild(__name__)

class Trainer():
  '''
  Super class implimenting basic training functionality
  '''
  def __init__(self, dataset_manager, device, metrics, model=None,
               save_model_param_path=None, loss_fn=torch.nn.CrossEntropyLoss(),
               lr_scheduler=None, optimizer_fn=torch.optim.Adam, 
               optimizer_params={'lr':0.001}):
    module_logger.info('Trainer Initializing')
    self.metrics = metrics
    self.device = device
    self.save_model_param_path = save_model_param_path
    self.dataset_manager = dataset_manager
    self.loss_fn = loss_fn
    self.model = model.to(device)
    self.optimizer = self.get_optimizer(optimizer_fn, optimizer_params)
    self.lr_scheduler = lr_scheduler
    self.dataset_video_count = len(self.dataset_manager)
    validation_video_count = int(self.dataset_video_count * 0.1 )
    self.validation_video_index = range(1, validation_video_count)
    self.training_video_index = range(validation_video_count, self.dataset_video_count+1)


  
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
      for idx in features:
        features[idx] = features[idx].to(self.device)
      labels = labels.to(self.device)

      with torch.set_grad_enabled(False):
        outputs = self.model(features)
        for metric in self.metrics:
          metric.update(outputs, labels, training=False)


  def _train_step(self, dataloader):
    '''
    It will train a single dataset a single epoch
    Returning Average Loss, Accuracy and Time taken for execution
    '''
    self.model.train()
    # Iterate over data.
    for features, labels in dataloader:

      # For transfering data to GPU
      for idx in features:
        features[idx] = features[idx].to(self.device)
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
          metric.update(outputs, labels, training=True)


  def _train_stage(self, progress=True):
    '''
    Function to train entire dataset one epoch
    '''

    for idx in self.training_video_index:
      dataloader = self.dataset_manager.get_dataloader(idx)
      self._train_step(dataloader)
      if progress:
        print('.', end='')

  def _test_stage(self, progress=True):
    '''
    Function to train entire dataset one epoch
    '''

    for idx in self.validation_video_index:
      dataloader = self.dataset_manager.get_dataloader(idx)
      self._train_step(dataloader)
      if progress:
        print('.', end='')


  def train(self, epochs):
    for i in range(1, epochs+1):
      last = time.time
      print('Epoch: {}'.format(i))
      ## Training and logging 
      print('\tTraining')
      self._train_stage()
      for metric in self.metrics:
        print('\t\t{}'.format(str(metric)))

      ## Testing and logging
      print('\tTesting')
      self._test_stage()
      for metric in self.metrics:
        print('\t\t{}'.format(str(metric)))

      print('\t\tTime Taken{}'.format(time.time() - last))

