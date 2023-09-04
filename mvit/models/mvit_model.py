import os 
import torchvision
import torch
from mvit.model_utils.trainer import Trainer


class MvitTrainer(Trainer):
  '''
  Subclass of Trainer class. It is used for training Mvit deeplearning model
  The __init__ model should set the data self.model and self.model_param_path
  '''
  def __init__(self, cholec80_dataset_manager, device, save_dir='./', fresh_model=False,
               enable_finetune=False, learning_rate=0.01, optimizer_name='adam', num_head_layers=1):
    super(MvitTrainer, self).__init__(cholec80_dataset_manager, device)

    self.model_param_path = os.path.join(save_dir, 'best_model_params.pt')
    self.model, self.optimizer = self.create_model(learning_rate=learning_rate, optimizer_name=optimizer_name,
                              num_head_layers=num_head_layers, fresh_model=fresh_model, enable_finetune=enable_finetune)
    self.model.to(device)

  def create_head_layer(self, num_head_layers):
    head1 = [
        torch.nn.Dropout(p=0.5, inplace=True),
        torch.nn.Linear(in_features=768, out_features=7, bias=True)]

    head2 = [
        torch.nn.Dropout(p=0.5, inplace=True),
        torch.nn.Linear(in_features=768, out_features=100, bias=True),
        torch.nn.Linear(in_features=100, out_features=7, bias=True)]

    head3 = [
        torch.nn.Dropout(p=0.5, inplace=True),
        torch.nn.Linear(in_features=768, out_features=100, bias=True),
        torch.nn.Linear(in_features=100, out_features=100, bias=True),
        torch.nn.Linear(in_features=100, out_features=7, bias=True)

        ]

    if num_head_layers == 1:
      head = head1
    elif num_head_layers == 2:
      head = head2
    elif num_head_layers == 3:
      head = head3
      raise Exception('Please select num_head to 1, 2 or 3')

    head_layer = torch.nn.Sequential(*head)
    return head_layer


  def create_model(self, learning_rate:float, optimizer_name:str,
                    num_head_layers:int, fresh_model:bool, 
                    enable_finetune:bool, delete_existing_model=False):
    '''
    A function to create transfer learing model of MVIT. For this model the parameter weight of torchvision.model package is utilized.
    '''
    ### For preventing accedental overwrite of saved model
    if os.path.exists(self.model_param_path) and fresh_model and not delete_existing_model:
      raise Exception('Trained model is already exist. If you need create a fresh model please remove {} file'.format(self.model_param_path))

    ### For loading already existing model
    if fresh_model:
      model = torchvision.models.video.mvit_v1_b(weights=torchvision.models.video.MViT_V1_B_Weights.DEFAULT) # Loading pretrained weights
    else:
      model = torchvision.models.video.mvit_v1_b() # No need to load pretrained weights. Weights are already exist in the disk

    ### Model modification for transfer learning
    model.head = self.create_head_layer(num_head_layers)
    for param in model.parameters():
      param.requires_grad = False
    for param in model.head.parameters():
      param.requires_grad = True

    ### Optimizer Selection
    if optimizer_name == 'adam':
      optimizer = torch.optim.Adam(model.head.parameters(), lr=learning_rate)
    if optimizer_name == 'sgd':
      optimizer = torch.optim.SGD(model.head.parameters(), lr=learning_rate, momentum=0.9)
      # self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)

    ### For fine turning
    if enable_finetune:
      for param in model.parameters():
        param.requires_grad = True
    
    return model, optimizer