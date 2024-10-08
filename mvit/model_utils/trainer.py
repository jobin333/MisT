import torch
import time
import os

from mvit.logging_utils.logger import logger
module_logger = logger.getChild(__name__)

class Trainer():
  '''
  Super class implimenting basic training functionality
  '''
  def __init__(self, dataset_manager, device, create_model_fn=None, 
                model_creation_params=None, model_outs_save_location = None,  retain_graph=False,
                train_step_callback=None, model=None, enable_training=True,
               save_model_param_path=None, loss_fn=torch.nn.CrossEntropyLoss(),
               lr_scheduler=None, optimizer_fn=torch.optim.Adam, 
               optimizer_params={'lr':0.001}, save_during_training=False,
               tool_training=False,):
    module_logger.info('Trainer Initializing')
    self.tool_training = tool_training
    self.label_index = (2 if tool_training else 1) ## Index 1 - Phase info ; Index-2: Tool info
    self.retain_graph=retain_graph
    self.device = device
    self.save_model_param_path = save_model_param_path
    self.dataset_manager = dataset_manager
    self.loss_fn = loss_fn
    model = model if model is not None else create_model_fn(*model_creation_params)
    self.model = model.to(device)
    if enable_training:
      self.optimizer = self.get_optimizer(optimizer_fn, optimizer_params)
    self.lr_scheduler = lr_scheduler
    self.dataset_video_count = len(self.dataset_manager)
    self.train_step_callback = train_step_callback # Execute after each train_step
    validation_video_count = int(self.dataset_video_count * 0.1 )
    self.validation_video_index = range(1, validation_video_count)
    self.training_video_index = range(validation_video_count, self.dataset_video_count+1)
    self.model_outs_save_location = model_outs_save_location
    self.save_during_training = save_during_training

  
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

  def save_model_outs(self, dataloader, filename):
    '''
    Saving model output
    '''
    module_logger.info('Saving model output')
    data = []
    self.model.eval()
    for video_index in self.training_video_index:
      dataloader = self.dataset_manager.get_dataloader(video_index)      
      for x,y in dataloader:
        x = x.to(self.device)
        feature_x = self.model(x)
        for item in zip(feature_x, y):
            data.append(item)
      torch.save(data, filename)

  def make_scheduler_step(self):
    if self.lr_scheduler is not None:
      self.lr_scheduler.step()

  def eval_step(self, dataloader):
    '''
    Used for evaluation time. During this time the gradients are disabled.
    '''
    module_logger.info('Evaluating module performance')
    self.model.eval()
    running_loss = 0.0
    accurate_classifications = 0
    datapoints_seen = 0
    since = time.time()
    for data in dataloader:
      inputs = data[0].to(self.device)
      labels = data[self.label_index].to(self.device)
      datapoints_seen += labels.shape[0]

      with torch.set_grad_enabled(False):
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, labels)
        pred = (outputs if self.tool_training else torch.argmax(outputs, 1) )
        running_loss += loss.item()
        if not self.tool_training:
          accurate_classifications += sum(pred==labels).item()
        else:
          accurate_classifications += torch.sum(torch.eq(pred>0,labels>0)).item() / 7
          ### 7 is the number of tool types

    average_loss = running_loss / datapoints_seen
    accuracy = accurate_classifications / datapoints_seen
    time_elapsed = time.time() - since
    return {'average_loss':average_loss, 'accuracy':accuracy, 'time_elapsed':time_elapsed}


  def _train_step(self, dataloader):
    '''
    It will train a single dataset a single epoch
    Returning Average Loss, Accuracy and Time taken for execution
    '''
    self.model.train()
    running_loss = 0.0
    accurate_classifications = 0
    since = time.time()
    datapoints_seen = 0

    # Iterate over data.
    for data in dataloader:
      inputs = data[0].to(self.device)
      labels = data[self.label_index].to(self.device)
      datapoints_seen += labels.shape[0]
      if not self.retain_graph:
        self.optimizer.zero_grad()

      with torch.set_grad_enabled(True):
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, labels)
        loss.backward(retain_graph=self.retain_graph)
        self.optimizer.step()
      pred = (outputs if self.tool_training else torch.argmax(outputs, 1) )

      running_loss += loss.item()
      if not self.tool_training:
        accurate_classifications += sum(pred==labels).item()
      else:
        accurate_classifications += torch.sum(torch.eq(pred>0,labels>0)).item() / 7
        ### 7 is the number of tool types

    average_loss = running_loss / datapoints_seen
    accuracy = accurate_classifications / datapoints_seen
    time_elapsed = time.time() - since
    if self.train_step_callback is not None:
      self.train_step_callback()
    return {'average_loss':average_loss, 'accuracy':accuracy, 'time_elapsed':time_elapsed}


  def _train_stage(self, param_save_per_epochs, summary_only, run_evaluation):
    '''
    Function to train entire dataset one epoch
    '''
    module_logger.info('Started an epoch')

    loss_list = []
    accuracy_list = []
    time_list = []
    for i, video_index in enumerate(self.training_video_index):
      dataloader = self.dataset_manager.get_dataloader(video_index)
      statistics = self._train_step(dataloader)
      accuracy = statistics['accuracy']
      loss = statistics['average_loss']
      time_elapsed = statistics['time_elapsed']
      accuracy_list.append(accuracy)
      loss_list.append(loss)
      time_list.append(time_elapsed)
      if summary_only:
        print('.', end='')
      else:
        print('    Trained Video No: {};  Accuracy: {:.2f}; Loss: {:.6f}'.format(video_index, accuracy, loss))
      ## For saving while training
      if self.save_during_training:
        if (i+1) % int(self.dataset_video_count / param_save_per_epochs) == 0:
         self.save_model()
    average_training_loss = sum(loss_list) / len(loss_list)
    average_training_accuracy = sum(accuracy_list) / len(accuracy_list)
    time_taken = sum(time_list)
    print('')
    print('Training Summary;  Average Accuracy: {:.2f}; Average Loss: {:.6f}; Execution Time {:.2f}'.format(average_training_accuracy, average_training_loss, time_taken))
    if run_evaluation:
      self.evaluate_model(summary_only=summary_only)

  def evaluate_model(self, summary_only=True):
    '''
    Function to evaluate model performance
    '''
    module_logger.info('Started evaluating model')
    print('Starting Evaluating model', end=' ')
    accuracy_list = []
    loss_list = []
    time_list = []
    for i, video_index in enumerate(self.validation_video_index):
      dataloader = self.dataset_manager.get_dataloader(video_index)
      statistics = self.eval_step(dataloader)
      accuracy = statistics['accuracy']
      loss = statistics['average_loss']
      time_elapsed = statistics['time_elapsed']
      accuracy_list.append(accuracy)
      loss_list.append(loss)
      time_list.append(time_elapsed)
      if summary_only:
        print('.', end='')
      else:
        print('    Evaluation Video No: {};  Accuracy: {:.2f}; Loss: {:.6f}'.format(video_index, accuracy, loss))
    average_evaluation_loss = sum(loss_list) / len(loss_list)
    average_evaluation_accuracy = sum(accuracy_list) / len(accuracy_list)
    total_time_taken = sum(time_list)
    print('')
    print('Evaluation Summary;  Average Accuracy: {:.2f}; Average Loss: {:.6f}; Execution Time {:.2f}'.format(average_evaluation_accuracy, average_evaluation_loss, total_time_taken))


  def train_model(self, epochs, param_save_per_epochs=5, summary_only=True, run_evaluation=False):
    for i in range(1, epochs+1):
      print('Training Epoch: {}'.format(i), end=' ')
      self._train_stage(param_save_per_epochs=param_save_per_epochs, summary_only=summary_only, run_evaluation=run_evaluation)


  def _save_model_outs_of_dataloader(self, dataloader, filename, enable_progress=False):
    module_logger.info('Saving model output of dataloader {}'.format(filename))
    data = []
    self.model.eval()
    cpu = torch.device('cpu')
    for i, (x,y) in enumerate(dataloader):
      x = x.to(self.device)
      if type(y) == torch.Tensor:
        y = y.to(cpu)
        y = (y,) ## To make it a tuple
      else:
        y = tuple(item.to(cpu) for item in y)
      feature_x = self.model(x)
      feature_x = feature_x.to(cpu)
      for item in zip(feature_x, *y):
          data.append(item)
      if enable_progress:
        if i%100 == 0:
          print('.', end='')
    print()
    torch.save(data, filename)

  def save_model_outs_of_dataset_manager(self, enable_progress=False, start_video_index=1):
      if not os.path.exists(self.model_outs_save_location):
          os.makedirs(self.model_outs_save_location)

      for i in range(start_video_index, 81):
          model_out_path = os.path.join(self.model_outs_save_location, 'tensors_{}.pt'.format(i))
          data_loader = self.dataset_manager.get_dataloader(i)
          print('Creating {}'.format(model_out_path))
          self._save_model_outs_of_dataloader(data_loader, model_out_path, enable_progress=enable_progress)