import os 
import torchvision
import torch
from mvit.model_utils.trainer import Trainer

class MvitTrainer(Trainer):
  '''
  Subclass of Trainer class. It is used for training Mvit deeplearning model
  The __init__ model should set the data self.model and self.model_param_path
  '''
  def __init__(self, cholec80_dataset_manager, device, save_dir='./', fresh_model=False, enable_finetune=False, learning_rate=0.01):
    super(MvitTrainer, self).__init__(cholec80_dataset_manager, device)
    self.model_param_path = os.path.join(save_dir, 'best_model_params.pt')

    if fresh_model:
      self.model, self.optimizer = self.create_new_model(learning_rate)
    else:
      self.model, self.optimizer = self.create_model_from_saved_weight(learning_rate)

    if enable_finetune:
      for param in self.model.parameters():
        param.requires_grad = True

    self.model.to(device)
    # self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)

  def create_model_from_saved_weight(self, learning_rate):
    '''
    It will create a MVIT model and load pretrained weight file in the location self.model_param_path
    '''
    mvit_model = torchvision.models.video.mvit_v1_b()
    for param in mvit_model.parameters():
      param.requires_grad = False
    mvit_model.head[1] = torch.nn.Linear(in_features=768, out_features=7, bias=True)
    mvit_model.load_state_dict(torch.load(self.model_param_path))
    optimizer = torch.optim.Adam(mvit_model.head[1].parameters(), lr=0.01)
    return mvit_model, optimizer


  def create_new_model(self, learning_rate):
    '''
    A function to create transfer learing model of MVIT. For this model the parameter weight of torchvision.model package is utilized.
    '''
    if os.path.exists(self.model_param_path):
      raise Exception('Trained model is already exist. If you need create a fresh model please remove {} file'.format(self.model_param_path))
    mvit_model = torchvision.models.video.mvit_v1_b(weights=torchvision.models.video.MViT_V1_B_Weights.DEFAULT)
    for param in mvit_model.parameters():
      param.requires_grad = False
    mvit_model.head[1] = torch.nn.Linear(in_features=768, out_features=7, bias=True)
    optimizer = torch.optim.Adam(mvit_model.head[1].parameters(), lr=0.01)
    return mvit_model, optimizer