import os
import torch
import torchvision
import glob

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

  def get_config_files(self, config_folder):
    config_path_re = os.path.join(config_folder, '*pt.config')
    config_files = glob.glob(config_path_re)
    return config_files

  def train_flm(self, config_file):
    cfg = TrainerConfigurationGenerator(config_file)

    if not self.retrain and cfg.flm_training_completed:
      print('Training already completed exiting')
      return
    
    dataset_manager = ModelOuptutDatasetManager(cfg.feature_folder, cfg.feature_model_name, cfg.dataset_name,
                                                cfg.train_file_indices, cfg.test_file_indices, seq_length=cfg.flm_seq_length, 
                                                device=self.device, in_test_set=cfg.contain_test_set)
    
    flm = self.flm_model_class(in_features=cfg.in_features, out_features=cfg.out_features, seq_length=cfg.flm_seq_length)
    
    trainer = Trainer(dataset_manager, self.device, self.metrics, flm,
                   save_model_param_path=cfg.flm_save_param_path,
                   loss_fn=self.flm_loss_fn, optimizer_fn=torch.optim.Adam, 
                   optimizer_params={'lr':cfg.flm_lr}, save_during_training=False,
                   stop_epoch_count=cfg.flm_stop_epoch_count, model_name = cfg.flm_model_name)
    
    trainer.train(cfg.flm_max_epoch)
    
    for metric in  self.metrics:
      if hasattr(metric, 'details'):
          config_name = 'flm_' + metric.name + '_details'
          config_value = metric.details()
          cfg.__setattr__(config_name, config_value)
      config_name = 'flm_' + metric.name 
      config_value = metric.value()
      cfg.__setattr__(config_name, config_value)
    cfg.__setattr__('flm_training_completed', True)
    trainer.save_model()
    cfg.save()
        

  def save_model_output(self, config_file):
    cfg = TrainerConfigurationGenerator(config_file)
    dataset_manager = ModelOuptutDatasetManager(cfg.feature_folder, cfg.feature_model_name, cfg.dataset_name,
                                                cfg.train_file_indices, cfg.test_file_indices, seq_length=cfg.flm_seq_length, 
                                                device=self.device, in_test_set=cfg.contain_test_set)
    
    flm = self.flm_model_class(in_features=cfg.in_features, out_features=cfg.out_features, seq_length=cfg.flm_seq_length)
    flm.eval()
    flm = flm.to(self.device)
    flm_out = {True: [], False:[]} 

    training = True
    for dataloader in  dataset_manager.get_dataloaders(training):
        dataloader_out = []
        for x, y in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)
            z = flm(x)
            dataloader_out.append((z,y))  
        flm_out[training].append(dataloader_out)
        
    training = False
    for dataloader in  dataset_manager.get_dataloaders(training):
        dataloader_out = []
        for x, y in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)
            z = flm(x)
            dataloader_out.append((z,y))  
        flm_out[training].append(dataloader_out)

    torch.save(flm_out, cfg.flm_save_model_out_file)


    
  def train_slm(self, config_file, enable_low_memory):
    cfg = TrainerConfigurationGenerator(config_file)
    if not self.retrain and cfg.slm_training_completed:
      print('Training already completed exiting')
      return
    
    if enable_low_memory:
      dataset_manager = SimpleModelOutDatasetManager(cfg.flm_save_model_out_file)
    else:
      dataset_manager = ModelOuptutDatasetManager(cfg.feature_folder, cfg.feature_model_name, cfg.dataset_name,
                                                cfg.train_file_indices, cfg.test_file_indices,  seq_length=cfg.flm_seq_length, 
                                                device=self.device, in_test_set=cfg.contain_test_set)

    if not enable_low_memory:
      flm = self.flm_model_class(in_features=cfg.in_features, out_features=cfg.out_features, seq_length=cfg.flm_seq_length)
      flm.load_state_dict( torch.load(cfg.flm_save_param_path) )
    else:
      flm = torch.zeros(1)
    
    # if self.slm_model_class.__name__ == MultiLevelMemoryModel:
    #   slm = self.slm_model_class(predictor_model=flm, stack_length=cfg.slm_stack_length,
    #                               roll_count=cfg.slm_roll_count, number_path=cfg.slm_number_path,
    #                             path_multiplier=cfg.slm_path_multiplier, dropout=cfg.slm_dropout, 
    #                             num_surg_phase=cfg.out_features, rolls=cfg.slm_rolls, roll_start_with_one=cfg.slm_roll_start_with_one)
    # else:
    #   slm = self.slm_model_class(predictor_model=flm, stack_length=cfg.slm_stack_length,
    #                              rolls=cfg.slm_rolls, dropout=cfg.slm_dropout,
    #                                nhead=cfg.slm_nhead, dim_feedforward=cfg.slm_dim_feedforward,
    #                                  num_layers=cfg.slm_num_layers,  num_surg_phase=cfg.out_features,
    #                                   dmodel=cfg.slm_dmodel)

    slm = self.slm_model_class(predictor_model=flm, stack_length=cfg.slm_stack_length,
                                  dropout=cfg.slm_dropout, 
                                num_surg_phase=cfg.out_features, rolls=cfg.slm_rolls)
   

    trainer = Trainer(dataset_manager, self.device, self.metrics, slm,
                  save_model_param_path=cfg.slm_save_param_path,
                  loss_fn=self.slm_loss_fn, optimizer_fn=torch.optim.Adam, 
                  optimizer_params={'lr':cfg.slm_lr}, save_during_training=False,
                  stop_epoch_count=cfg.slm_stop_epoch_count, model_name = cfg.slm_model_name)
    
    trainer.train(cfg.slm_max_epoch)

    
    for metric in  self.metrics:
        if hasattr(metric, 'details'):
          config_name = 'slm_' + metric.name + '_details'
          config_value = metric.details()
          cfg.__setattr__(config_name, config_value)
        config_name = 'slm_' + metric.name + '_value'
        config_value = metric.value()
        cfg.__setattr__(config_name, config_value)
    
    cfg.__setattr__('slm_training_completed', True)
    trainer.save_model()
    cfg.save()

    # Code to return last metric details
    return metric.details() if hasattr(metric, 'details') else None
    
  def train(self, enable_flm_train=True, enable_slm_train=True, enable_low_memory=True):
    for config_file in self.config_files:
      print(f'Training using config file {config_file}')
      if enable_flm_train:
        self.train_flm(config_file)
      if enable_low_memory:
         self.save_model_output(config_file)
      if enable_slm_train:
        metric_details = self.train_slm(config_file, enable_low_memory)
        # Return last metric details
        return metric_details