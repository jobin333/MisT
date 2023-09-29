import torch
import os
from mvit.model_utils.trainer import Trainer
from mvit.models.video_classifier_trainer import create_classifier_model
import time


class TubeletClassifierDrivenLSTM(torch.nn.Module):
  def __init__(self, classifier_backbone_model, classes, hidden_features):
    '''
    Batch Size = 1
    '''
    super().__init__()
    self.classifier_backbone_model = classifier_backbone_model
    self.lstm = torch.nn.LSTM(classes, hidden_features)
    self.linear = torch.nn.Linear(hidden_features, classes)
    self.hidden_features = hidden_features
    self.clear_lstm_states()
    print('Warning: Please ensure to keep the batch size to 1')


  def clear_lstm_states(self):
      h0 = c0 = torch.zeros(1, self.hidden_features)
      self.lstm_state = (h0, c0)


  def forward(self, x):
    x = self.classifier_backbone_model(x)
    x, self.lstm_state = self.lstm(x)
    x = self.linear(x)
    return x

class LSTMTrainer(Trainer):
  '''
  Subclass of Trainer class. It is used for training LSTM network. 
  It need a pretrained tublet classification model
  '''
  def __init__(self, cholec80_dataset_manager, device, classifier_backbone_model=None, 
               classifier_backbone_name=None, classifier_backbone_head=None, 
               backbone_model_save_dir='./', lstm_save_file_name='lstm.pt', fresh_backbone=False, 
               enable_finetune=False, lstm_model_save_dir='./',
                  learning_rate=0.01, optimizer_name='adam', delete_existing_model=False,
                hidden_features=20, backbone_classes=7):
    '''
    args:
        model_name: 'video_resnet', 'swin3d', 's3d', 'mvit'
        model_head: video_resnet:1,2; swin3d:1,2; s3d:1, mvit:1
    '''
    super().__init__(cholec80_dataset_manager, device)
    
    self.model_param_path = os.path.join(lstm_model_save_dir, lstm_save_file_name)

    if cholec80_dataset_manager.shuffle:
      raise ('Training on shuffled dataset is not implemented yet')
    
    if classifier_backbone_model is None:
      complete_model_name = 'save_{}_{}.pt'.format(classifier_backbone_name, classifier_backbone_head)
      model_param_path = os.path.join(backbone_model_save_dir, complete_model_name)
      self.classifier_backbone_model = create_classifier_model(model_name=classifier_backbone_name, 
                                                               model_head=classifier_backbone_head, 
                                                               fresh_model=fresh_backbone, delete_existing_model=False,
                                                               model_param_path=model_param_path)
    else:
      self.classifier_backbone_model = classifier_backbone_model

    self.model = TubeletClassifierDrivenLSTM(classifier_backbone_model=self.classifier_backbone_model, 
                                                classes=backbone_classes, hidden_features=20)

    
    if os.path.exists(model_param_path):
      print('Loading LSTM model from {}'.format(model_param_path))
      self.model.load_state_dict(torch.load(model_param_path))

    
    if enable_finetune:
      for param in self.model.parameters():
        param.requires_grad = True
    else:
      for param in self.classifier_backbone_model.parameters():
        param.requires_grad = False

    trainable_params = [param for param in self.model.parameters() if param.requires_grad ]

    if optimizer_name == 'adam':
      self.optimizer = torch.optim.Adam(trainable_params, learning_rate)
    elif optimizer_name == 'sgd':
      self.optimizer = torch.optim.SGD(trainable_params, learning_rate, momentum=0.9)


  def train_step(self, dataloader):
    '''
    It will train a single dataset a single epoch
    Returning Average Loss, Accuracy and Time taken for execution
    '''
    self.model.train()
    self.model.clear_lstm_states() ## Important
    running_loss = 0.0
    accurate_classifications = 0
    since = time.time()
    datapoints_seen = 0

    # Iterate over data.
    for inputs, labels in dataloader:
      inputs = inputs.to(self.device)
      labels = labels.to(self.device)
      datapoints_seen += inputs.shape[0]
      self.optimizer.zero_grad()

      with torch.set_grad_enabled(True):
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, labels)
        loss.backward()
        self.optimizer.step()
      pred = torch.argmax(outputs, 1)

      running_loss += loss.item()
      accurate_classifications += sum(pred==labels).item()

    average_loss = running_loss / datapoints_seen
    accuracy = accurate_classifications / datapoints_seen
    time_elapsed = time.time() - since

    return {'average_loss':average_loss, 'accuracy':accuracy, 'time_elapsed':time_elapsed}

    