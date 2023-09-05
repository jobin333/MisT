import torch
import torchvision
import os
from mvit.model_utils.trainer import Trainer


def get_video_resnet(model_head, download_pretrained_weights):
  if download_pretrained_weights:
    model = torchvision.models.video.r3d_18(torchvision.models.video.R3D_18_Weights.DEFAULT)
  else:
    model = torchvision.models.video.r3d_18()
  for param in model.parameters():
    param.requires_grad = False
  if model_head == 1:
    model.avgpool = torch.nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
    model.fc = torch.nn.Linear(in_features=512, out_features=7)
  elif model_head == 2:
    model.avgpool = torch.nn.Flatten()
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(in_features=200704, out_features=7)
    ) ## in_features for 16x224x224
  else:
    raise Exception('Undefined model model_head')

  return model


def get_mvit(model_head, download_pretrained_weights):
  if download_pretrained_weights:
    model = torchvision.models.video.mvit_v1_b(weights=torchvision.models.video.MViT_V1_B_Weights.DEFAULT)
  else:
    model = torchvision.models.video.mvit_v1_b()
  for param in model.parameters():
    param.requires_grad = False
  if model_head == 1:
    model.head = torch.nn.Sequential(
        torch.nn.Dropout(p=0.5, inplace=True),
        torch.nn.Linear(in_features=768, out_features=7)
        )
  else:
    raise Exception('Undefined model model_head')

  return model

def get_s3d_model(model_head, download_pretrained_weights):
  if download_pretrained_weights:
    model = torchvision.models.video.s3d(weights=torchvision.models.video.S3D_Weights.DEFAULT)
  else:
    model = torchvision.models.video.s3d()
  for param in model.parameters():
    param.requires_grad = False
  if model_head == 1:
    model.classifier = torch.nn.Sequential(
      torch.nn.Dropout(p=0.2, inplace=False),
      torch.nn.Conv3d(1024, 7, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    )
  # elif model_head == 2:
  #   model.avgpool = torch.nn.Flatten()
  #   model.classifier = torch.nn.Sequential(
  #       torch.nn.Dropout(p=0.5),
  #       torch.nn.Linear(in_features=100352, out_features=7)
  #   )
  else:
    raise Exception('Undefined model head')

  return model


def get_swin3d(model_head, download_pretrained_weights):
  if download_pretrained_weights:
    model = torchvision.models.video.swin3d_s(torchvision.models.video.Swin3D_S_Weights.DEFAULT)
  else:
    model = torchvision.models.video.swin3d_s()
  for param in model.parameters():
    param.requires_grad = False
  if model_head == 1:
    model.head = torch.nn.Linear(in_features=768, out_features=7, bias=True)
  elif model_head == 2:
    model.avgpool = torch.nn.Flatten()
    model.head = torch.nn.Sequential(
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(in_features=301056, out_features=7, bias=True)
    )
  else:
    raise Exception('Undefined model head')

  return model


model_functions = {
    'video_resnet': get_video_resnet,
    'swin3d': get_swin3d,
    's3d': get_s3d_model,
    'mvit': get_mvit
    }


class VideoClassifierTrainer(Trainer):
  '''
  Subclass of Trainer class. It is used for training multiple deeplearning model
  The __init__ model should set the data self.model and self.model_param_path
  '''
  def __init__(self, cholec80_dataset_manager, device, model_name,
                model_head, save_dir='./', fresh_model=False,enable_finetune=False,
                  learning_rate=0.01, optimizer_name='adam', delete_existing_model=False):
    
    super(VideoClassifierTrainer, self).__init__(cholec80_dataset_manager, device)
    complete_model_name = 'save_{}_{}.pt'.format(model_name, model_head)
    self.model_param_path = os.path.join(save_dir, complete_model_name)
    if os.path.exists(self.model_param_path) and fresh_model and not delete_existing_model:
        raise Exception('Trained model is already exist. If you need create a fresh model \
                        please remove {} file'.format(self.model_param_path))
    if not os.path.exists(self.model_param_path) and not fresh_model:
        raise Exception('Trained model {} does not exist'.format(self.model_param_path))

    if fresh_model:
        self.model = model_functions[model_name](model_head=model_head,
                download_pretrained_weights=True)
    else:
        self.model = model_functions[model_name](model_head=model_head,
                download_pretrained_weights=False)
        self.model.load_state_dict(torch.load(self.model_param_path))

    if enable_finetune:
      for param in self.model.parameters:
        param.requires_grad = True

    trainable_params = [param for param in self.model.parameters() if param.requires_grad ]

    if optimizer_name == 'adam':
      self.optimizer = torch.optim.Adam(trainable_params, learning_rate)
    elif optimizer_name == 'sgd':
      self.optimizer = torch.optim.SGD(trainable_params, learning_rate, momentum=0.9)

    self.model.to(device)
    self.device = device

    