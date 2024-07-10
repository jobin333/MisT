import os
import torch
import torchvision
import glob
import gc

from mvit.models.linear_models import SimpleLinearModel
from mvit.model_utils.global_trainer2 import Trainer 
from mvit.data_utils.global_dataset_manager import ModelOuptutDatasetManager
from mvit.train_utils.config_generator import TrainerConfigurationGenerator
from mvit.models.memory_models import MultiLevelMemoryModel

from mvit.logging_utils.logger import logger
module_logger = logger.getChild(__name__)

class SimpleModelOutDatasetManager():
    def __init__(self, data_file):
        self.data = torch.load(data_file)
    def get_dataloaders(self, training=True):
        return self.data[training]
    

class TrainingManager():
    
  def __init__(self, config_folder, metrics, device, 
               flm_model_class=SimpleLinearModel, 
               slm_model_class=MultiLevelMemoryModel, retrain=False,
               flm_loss_fn=torch.nn.CrossEntropyLoss(), slm_loss_fn=torch.nn.CrossEntropyLoss()):
    self.flm_loss_fn = flm_loss_fn
    self.slm_loss_fn = slm_loss_fn
    self.config_files = self.get_config_files(config_folder)
    self.device = device
    self.metrics = metrics
    self.retrain = retrain
    self.flm_model_class = flm_model_class
    self.slm_model_class = slm_model_class


  def initialize_training_data(self, config_file, for_flm=True):
    self.config_file = config_file
    self.cfg = TrainerConfigurationGenerator(self.config_file)

    if for_flm:
      self.flm = self.flm_model_class(in_features=self.cfg.in_features, out_features=self.cfg.out_features, seq_length=self.cfg.flm_seq_length)
      self.flm = self.flm.to(self.device)  
      self.dataset_manager = ModelOuptutDatasetManager(self.cfg.feature_folder, self.cfg.feature_model_name, self.cfg.dataset_name,
                                                self.cfg.train_file_indices, self.cfg.test_file_indices,  seq_length=self.cfg.flm_seq_length, 
                                                device=self.device, in_test_set=self.cfg.contain_test_set)
  
      
    else:
      self.flm = None
      self.slm = self.slm_model_class(predictor_model=self.flm, stack_length=self.cfg.slm_stack_length,
                                  dropout=self.cfg.slm_dropout, 
                                num_surg_phase=self.cfg.out_features, rolls=self.cfg.slm_rolls)
      self.slm = self.slm.to(self.device)
      self.dataset_manager = SimpleModelOutDatasetManager(self.cfg.flm_save_model_out_file)


  def save_flm_out_n_clear_memory(self, in_cpu=True):
      gc.collect()
      if in_cpu:
        device = torch.device('cpu')
      else:
        device = self.device

      self.flm = self.flm.to(device)
      self.flm.eval()
      flm_out = {True: [], False:[]} 

      for training in [True, False]:
        for dataloader in  self.dataset_manager.get_dataloaders(training):
            dataloader_out = []
            for x, y in dataloader:
                x = x.to(device)
                y = y.to(device)
                z = self.flm(x)
                dataloader_out.append((z.to(device),y.to(device)))  
            flm_out[training].append(dataloader_out)

      torch.save(flm_out, self.cfg.flm_save_model_out_file)

      del(self.dataset_manager)




  def get_config_files(self, config_folder):
    config_path_re = os.path.join(config_folder, '*pt.config')
    config_files = glob.glob(config_path_re)
    return config_files

  def train_flm(self):
    if not self.retrain and self.cfg.flm_training_completed:
      print('Training already completed exiting')
      return
    
    trainer = Trainer(self.dataset_manager, self.device, self.metrics, self.flm,
                   save_model_param_path=self.cfg.flm_save_param_path,
                   loss_fn=self.flm_loss_fn, optimizer_fn=torch.optim.Adam, 
                   optimizer_params={'lr':self.cfg.flm_lr}, save_during_training=False,
                   stop_epoch_count=self.cfg.flm_stop_epoch_count, model_name = self.cfg.flm_model_name)
    
    trainer.train(self.cfg.flm_max_epoch)
    
    for metric in  self.metrics:
      if hasattr(metric, 'details'):
          config_name = 'flm_' + metric.name + '_details'
          config_value = metric.details()
          self.cfg.__setattr__(config_name, config_value)
      config_name = 'flm_' + metric.name 
      config_value = metric.value()
      self.cfg.__setattr__(config_name, config_value)
    self.cfg.__setattr__('flm_training_completed', True)
    trainer.save_model()
    self.save_flm_out_n_clear_memory()
    self.cfg.save()
        

    
  def train_slm(self):
    trainer = Trainer(self.dataset_manager, self.device, self.metrics, self.slm,
                  save_model_param_path=self.cfg.slm_save_param_path,
                  loss_fn=self.slm_loss_fn, optimizer_fn=torch.optim.Adam, 
                  optimizer_params={'lr':self.cfg.slm_lr}, save_during_training=False,
                  stop_epoch_count=self.cfg.slm_stop_epoch_count, model_name = self.cfg.slm_model_name)
    
    trainer.train(self.cfg.slm_max_epoch)

    
    for metric in  self.metrics:
        if hasattr(metric, 'details'):
          config_name = 'slm_' + metric.name + '_details'
          config_value = metric.details()
          self.cfg.__setattr__(config_name, config_value)
        config_name = 'slm_' + metric.name + '_value'
        config_value = metric.value()
        self.cfg.__setattr__(config_name, config_value)
    
    self.cfg.__setattr__('slm_training_completed', True)
    trainer.save_model()
    self.cfg.save()

    # Code to return last metric details
    return metric.details() if hasattr(metric, 'details') else None
    
  def train(self, enable_flm_train=True, enable_slm_train=True, enable_low_memory=True):
    for config_file in self.config_files:
      print(f'Training using config file {config_file}')
      if enable_flm_train:
        self.initialize_training_data(config_file, for_flm=True)
        self.train_flm()
        
      if enable_slm_train:
        self.initialize_training_data(config_file, for_flm=False)
        metric_details = self.train_slm()
        return metric_details