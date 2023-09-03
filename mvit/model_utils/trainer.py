import torch
import time

class Trainer():
  '''
  Super class implimenting basic training functionality
  '''
  def __init__(self, cholec80_dataset_manager, device):
    self.device = device
    self.model_param_path = None
    self.cholec80_dataset_manager = cholec80_dataset_manager
    self.loss_fn = torch.nn.CrossEntropyLoss()
    self.optimizer = None
    self.lr_scheduler = None
    self.model = None # Created by subclass
    self.dataset_video_count = len(self.cholec80_dataset_manager)

    validation_video_count = int(self.dataset_video_count * 0.1 )
    self.validation_video_index = range(1, validation_video_count)
    self.training_video_index = range(validation_video_count, self.dataset_video_count+1)

  def save_model(self):
    '''
    Saving model parameters in self.model_param_path
    '''
    torch.save(self.model.state_dict(), self.model_param_path)

  def make_scheduler_step(self):
    if self.lr_scheduler is not None:
      self.lr_scheduler.step()

  def eval_step(self, dataloader):
    '''
    Used for evaluation time. During this time the gradients are disabled.
    '''

    self.model.eval()
    running_loss = 0.0
    accurate_classifications = 0
    since = time.time()
    for inputs, labels in dataloader:
      inputs = inputs.to(self.device)
      labels = labels.to(self.device)
      with torch.set_grad_enabled(False):
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, labels)
        pred = torch.argmax(outputs, 1)
        running_loss += loss.item()
        accurate_classifications += sum(pred==labels).item()

    average_loss = running_loss / len(dataloader) / inputs.shape[0]
    accuracy = accurate_classifications / len(dataloader) / inputs.shape[0]
    time_elapsed = time.time() - since
    return {'average_loss':average_loss, 'accuracy':accuracy, 'time_elapsed':time_elapsed}


  def train_step(self, dataloader):
    '''
    It will train a single dataset a single epoch
    Returning Average Loss, Accuracy and Time taken for execution
    '''
    self.model.train()
    running_loss = 0.0
    accurate_classifications = 0
    since = time.time()

    # Iterate over data.
    for inputs, labels in dataloader:
      inputs = inputs.to(self.device)
      labels = labels.to(self.device)
      self.optimizer.zero_grad()

      with torch.set_grad_enabled(True):
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, labels)
        loss.backward()
        self.optimizer.step()
      pred = torch.argmax(outputs, 1)

      running_loss += loss.item()
      accurate_classifications += sum(pred==labels).item()

    average_loss = running_loss / len(dataloader) / inputs.shape[0]
    accuracy = accurate_classifications / len(dataloader) / inputs.shape[0]
    time_elapsed = time.time() - since

    return {'average_loss':average_loss, 'accuracy':accuracy, 'time_elapsed':time_elapsed}


  def train_stage(self, param_save_per_epochs):
    '''
    Function to train entire dataset one epoch
    '''
    for i, video_index in enumerate(self.training_video_index):
      dataloader = self.cholec80_dataset_manager.get_dataloader(video_index)
      statistics = self.train_step(dataloader)
      accuracy = statistics['accuracy']
      loss = statistics['average_loss']
      print('Trained Video No: {};  Accuracy: {}; Loss: {}'.format(video_index, accuracy, loss))
      if (i+1) % int(self.dataset_video_count / param_save_per_epochs) == 0:
        self.save_model()
    print('------------------------------------------------------')

  def evaluate_model(self):
    '''
    Function to evaluate model performance
    '''
    print('Starting Evaluating model-----------------------------')
    for i, video_index in enumerate(self.validation_video_index):
      dataloader = self.cholec80_dataset_manager.get_dataloader(video_index)
      statistics = self.eval_step(dataloader)
      accuracy = statistics['accuracy']
      loss = statistics['average_loss']
      print('Evaluation Video No: {};  Accuracy: {}; Loss: {}'.format(video_index, accuracy, loss))

  def train_model(self, epochs):
    for i in range(1, epochs+1):
      print('Training Epoch: {}.................................'.format(i))
      self.train_stage(param_save_per_epochs=5)
