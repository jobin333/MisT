import os
import torch
import torchvision
import random
import glob
import pandas as pd


from mvit.models.linear_models import SimpleLinearModel
from mvit.model_utils.global_trainer2 import Trainer 
from mvit.data_utils.global_dataset_manager import ModelOuptutDatasetManager
from mvit.metrics.metrics import Accuracy
from mvit.train_utils.config_generator import TrainerConfigurationGenerator
from mvit.models.memory_models import MultiLevelMemoryModel
from mvit.train_utils.training_manager import TrainingManager


dataset_details = {
                    'cholec80':{
                        'video_count':80, 
                        'num_classes':7, 
                        'test_file_count': 11,
                    },
    
                    'm2cai16':{
                        'video_count':[27, 14], 
                        'num_classes':8, 
               
                    },
    
                    'autolaparo':{
                        'video_count':21, 
                        'num_classes':7, 
                        'test_file_count': 3,
                    
                    }
                  }

feature_model_details = { 'Swin3D_B':1024, 'Swin3D_S':768, 'Swin3D_T':768 }


configs = {
    'dataset_details': dataset_details,
    'feature_model_details': feature_model_details,
    'feature_folder' : '/workspace/Features',
    'model_save_folder' : '/workspace/Models',
    'random_train_test' : False,
    
    'flm_seq_length' : 30,
    'flm_lr' : 0.0001,
    'flm_max_epoch' : 50,
    'flm_stop_epoch_count': 5,
    'flm_model_name': 'Simple_Linear',
    
    'slm_lr' : 0.0001,
    'slm_max_epoch' : 50,
    'slm_stack_length' : 60,
    'slm_roll_count' : 20,
    'slm_number_path' : 6,
    'slm_path_multiplier' : 2,
    'slm_dropout' : 0.0,
    'slm_stop_epoch_count': 5,
    'slm_model_name': 'Multilevel_Linear',

}

#############################################################################################
#### Training using TrainingManager
device = torch.device('cuda:3')
for dataset_name in dataset_details:
    for feature_model_name in feature_model_details:    
        config  = TrainerConfigurationGenerator(dataset_name, feature_model_name, configs)
        config.save()
        
config_folder = configs['model_save_folder']
metrics = [Accuracy()]
training_manager = TrainingManager(config_folder, metrics, device)
training_manager.train()



#############################################################################################
#### Saving the result to a CSV file
config_files = glob.glob('/workspace/Models/*.config')
data = []

for file in config_files:
    cfg = TrainerConfigurationGenerator(file)
    video_count = cfg.dataset_details[cfg.dataset_name]['video_count']
    data.append({
        'Speculative Model Accuracy': round(cfg.flm_accuracy, 3),
        'Multilevel Model Accuracy': round(cfg.slm_accuracy, 3),
        'Dataset': cfg.dataset_name,
        'Feature Model': cfg.feature_model_name,
        'Dataset Size': sum(video_count) if type(video_count) == list else video_count,
        'Num Surgical Phases': cfg.out_features
        
    })
    
df = pd.DataFrame(data)
df.to_csv('evalution.csv')