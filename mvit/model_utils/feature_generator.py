import torchvision
import torch
import os 


from mvit.data_utils.global_dataset_manager import VideoDatasetManager


from mvit.logging_utils.logger import logger
module_logger = logger.getChild(__name__)

class FeatureGenerator(torch.nn.Module):
  '''
  Expecting dataset_location contain folders of "cholec80", "autolaparo", "m2cai16" dataset
  The generated features will save on feature_save_location 
  the debugging parameter used only for debugging purpose.
  '''
  def __init__(self, device, features_save_location, 
               dataset_location, batch_size=4, debugging=False, tublet_size=25):
    self.tubelet_size = tublet_size
    self.debugging = debugging 
    self.device = device
    self.dataset_location = dataset_location
    self.features_save_location = features_save_location
    self.batch_size = batch_size
    
  def get_model(self, model_name):

    if model_name == 'Swin3D_B':
      weights=torchvision.models.video.Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1
    else:
      weights='KINETICS400_V1'

    if self.debugging:  ## While debugging, it is not loading the pretrained weights
      weights = None


    if model_name == 'Swin3D_B':
      model = torchvision.models.video.swin3d_b(weights=weights)
      out_features = model.head.in_features  ###  1024
    elif model_name == 'Swin3D_S':
      model = torchvision.models.video.swin3d_s(weights=weights) 
      out_features = model.head.in_features  ###  768
    elif model_name == 'Swin3D_T':
      model = torchvision.models.video.swin3d_t(weights=weights) 
      out_features = model.head.in_features  ###  768
    elif model_name == 'MViT_V2_S':
      model = torchvision.models.video.mvit_v2_s(weights=weights)
      out_features = model.head[1].in_features  ###  768

    model.head = torch.nn.Identity()
    model = model.to(self.device)
    for param in model.parameters():
        param.requires_grad = False
    
    return model, out_features
  

  def create_folders(self, dataset_names, model_names):
    if not os.path.exists(self.features_save_location):
      raise AttributeError(f'feature save location path does not exist {self.features_save_location}')
    
    for dataset_name in dataset_names:
      dataset_feature_path = os.path.join(self.features_save_location, dataset_name)
      if not os.path.exists(dataset_feature_path):
        os.mkdir(dataset_feature_path)
      for model_name in model_names:
        model_feature_path = os.path.join(dataset_feature_path, model_name)
        if not os.path.exists(model_feature_path):
          os.mkdir(model_feature_path)

  def save_features(self, dataset_names = ['cholec80', 'm2cai16', 'autolaparo'], 
                    model_names=['Swin3D_B', 'Swin3D_S', 'Swin3D_T', 'MViT_V2_S'], overwrite=False):

    self.create_folders(dataset_names, model_names)

    for dataset_name in dataset_names:
      for model_name in model_names:
        module_logger.info(f'Generating features of dataset {dataset_name} using model {model_name}')
        self.save_features_single_dataset_model(dataset_name, model_name, overwrite=overwrite)


  def save_features_single_dataset_model(self, dataset_name, model_name, overwrite=False):
    model, _ =  self.get_model(model_name)
    dataset_path = os.path.join(self.dataset_location, dataset_name)
    dataset_manager = VideoDatasetManager(dataset_name, dataset_path, self.batch_size, shuffle=False, 
                                          tubelet_size=self.tubelet_size)
    
    video_count = len(dataset_manager)
    for i in range(1, video_count+1):
      save_file_path = os.path.join(self.features_save_location, dataset_name, model_name, str(i)+'.pt')
      if os.path.exists(save_file_path) and not overwrite :
        logger.info(f'Feature file {save_file_path} already exist; Skipping feature generation')
        continue
      data = []
      data_loader = dataset_manager.get_dataloader(i)
      for x,y in data_loader:
        x = x.to(self.device)
        feature_x = model(x)
        for item in zip(feature_x, y):
            data.append(item)
        if self.debugging:  ### For debugging of feature generator code
          break

      torch.save(data, save_file_path)
      

  
if __name__ == '__main__':
  device = torch.device('cpu')
  dataset_location = '/home/jobin/PhD/Datasets/Scaled/'
  feature_save_location = '/home/jobin/test'

  feature_generator = FeatureGenerator(device, feature_save_location, dataset_location, debugging = True)
  feature_generator.save_features()
