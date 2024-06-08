import torchvision
import torch
import os 


from mvit.data_utils.global_dataset_manager import VideoDatasetManager


from mvit.logging_utils.logger import logger
module_logger = logger.getChild(__name__)

class FeatureGenerator(torch.nn.Module):
  def __init__(self, model_name, device, features_save_location, 
               dataset_location, batch_size=4):
    self.device = device
    self.model_name = model_name
    self.dataset_location = dataset_location
    self.features_save_location = features_save_location
    self.batch_size = batch_size
    self.model, self.out_features = self.get_model() 
    self.save_features()
    
  def get_model(self):
    if self.model_name == 'Swin3D_B':
      model = torchvision.models.video.swin3d_b(weights=torchvision.models.video.Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1)
      out_features = model.head.in_features  ###  1024
    elif self.model_name == 'Swin3D_S':
      model = torchvision.models.video.swin3d_s(weights='KINETICS400_V1') 
      out_features = model.head.in_features  ###  768
    elif self.model_name == 'Swin3D_T':
      model = torchvision.models.video.swin3d_t(weights='KINETICS400_V1') 
      out_features = model.head.in_features  ###  768
    elif self.model_name == 'MViT_V2_S':
      model = torchvision.models.video.mvit_v2_s(weights='KINETICS400_V1')
      out_features = model.head[1].in_features  ###  768

    model.head = torch.nn.Identity()
    model = model.to(self.device)
    for param in model.parameters():
        param.requires_grad = False

    return model, out_features
  

  def create_folders(self):
    if not os.path.exists(self.features_save_location):
      raise AttributeError(f'feature save location path does not exist {self.features_save_location}')
    
    for dataset_name in ['cholec80', 'm2cai16', 'autolaparo']:
      dataset_feature_path = os.path.join(self.features_save_location, dataset_name)
      if not os.path.exists(dataset_feature_path):
        os.mkdir(dataset_feature_path)

  def save_features(self):
    dataset_manager = VideoDatasetManager(self.dataset_name, self.dataset_location, 
                                          self.batch_size, shuffle=False)
    
    self.create_folders()

      

  