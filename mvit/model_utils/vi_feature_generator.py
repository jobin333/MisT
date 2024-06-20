import torchvision
import torch
import os 


from mvit.data_utils.global_dataset_manager import VideoDatasetManager
from mvit.metrics.metrics import APRFSJC


from mvit.logging_utils.logger import logger
module_logger = logger.getChild(__name__)


class FeatureConcatModel(torch.nn.Module):
  def __init__(self, video_feature_model, image_feature_model, tubelet_size):
    super().__init__()
    self.video_feature_model = video_feature_model
    self.image_feature_model = image_feature_model
    self.image_offest = tubelet_size // 2

  def forward(self, x):
    xv = self.video_feature_model(x)
    xi = self.image_feature_model(x[:,:,self.image_offest,:,:])
    x = torch.cat( (xv, xi), dim=-1 )
    return x

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
      video_weights=torchvision.models.video.Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1
      image_weights = 'IMAGENET1K_V1'
    else:
      video_weights='KINETICS400_V1'
      image_weights = 'IMAGENET1K_V1'


    if self.debugging:  ## While debugging, it is not loading the pretrained weights
      video_weights = None
      image_weights = None



    if model_name == 'Swin3D_B':
      video_model = torchvision.models.video.swin3d_b(weights=video_weights)
      image_model = torchvision.models.swin_v2_b(weights=image_weights)
      out_features = video_model.head.in_features +  image_model.head.in_features ###  1024 + 1024
    elif model_name == 'Swin3D_S':
      video_model = torchvision.models.video.swin3d_s(weights=video_weights) 
      image_model = torchvision.models.swin_v2_s(weights=image_weights)
      out_features = video_model.head.in_features +  image_model.head.in_features ###  768 + 768
    elif model_name == 'Swin3D_T':
      video_model = torchvision.models.video.swin3d_t(weights=video_weights) 
      image_model = torchvision.models.swin_v2_t(weights=image_weights)
      out_features = video_model.head.in_features +  image_model.head.in_features  ###  768 + 768
    else:
      raise NotImplementedError(f'model name {model_name} is not implemented')

    video_model.head = torch.nn.Identity()
    image_model.head = torch.nn.Identity()
    model = FeatureConcatModel(video_model, image_model, self.tubelet_size)
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
                    model_names=['Swin3D_B', 'Swin3D_S', 'Swin3D_T'], overwrite=False):

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
    self.current_dataset_manager = dataset_manager
    
    def save_step(training_set=True):
      video_count = len(dataset_manager)
      for i in range(1, video_count+1):
        file_name = str(i) if training_set else 'test_' + str(i)
        file_name = file_name + '.pt'
        save_file_path = os.path.join(self.features_save_location, dataset_name, model_name, file_name)
        if os.path.exists(save_file_path) and not overwrite :
          logger.info(f'{save_file_path} already exist; Skipping')
          continue
        data = []
        data_loader = dataset_manager.get_dataloader(i)
        self.current_data_loader = data_loader
        for x,y in data_loader:
          x = x.to(self.device)
          feature_x = model(x)
          for item in zip(feature_x, y):
              data.append(item)
          if self.debugging:  ### For debugging of feature generator code
            break
        torch.save(data, save_file_path)
        print('.', end='')
      print('')
      
    save_step(True)
    if dataset_name == 'm2cai16':
      '''
      For m2cai16, it is given a seperate dataset for testing purpose.
      '''
      dataset_manager.training_set = False
      save_step(False)



  
if __name__ == '__main__':
  device = torch.device('cpu')
  dataset_location = '/home/jobin/PhD/Datasets/Scaled/'
  feature_save_location = '/home/jobin/test'

  feature_generator = FeatureGenerator(device, feature_save_location, dataset_location, debugging = True, batch_size=1)
  feature_generator.save_features()
