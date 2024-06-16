import os
import torch
import torchvision
import random

from mvit.models.linear_models import SimpleLinearModel
from mvit.model_utils.global_trainer2 import Trainer 
from mvit.data_utils.global_dataset_manager import ModelOuptutDatasetManager
from mvit.metrics.metrics import Accuracy



class TrainerConfigurationGenerator():
    
    def __init__(self, *args):
        if len(args) == 1: 
            self.__dict__['_config'] = self.load(args[0])
        else:
            self.__dict__['_config']  = self.generate_config(*args)
            
            
    def get_file_indices(self, dataset_name, randomize):
        dataset_spec = self.dataset_details[dataset_name]
        video_count = dataset_spec['video_count']
        contain_test_set = False if type(video_count)==int else True
        if contain_test_set:
            train_file_count, test_file_count = video_count
            train_file_indices = list( range(1, train_file_count + 1) )
            test_file_indices = list( range(1, test_file_count+1) )

        else:
            file_indices = list( range(1, video_count+1) )
            if randomize:
                random.shuffle(file_indices)

            test_file_count = dataset_spec['test_file_count']
            test_file_indices = file_indices[:test_file_count]
            train_file_indices = file_indices[test_file_count:]

        return train_file_indices, test_file_indices, contain_test_set
    

    def get_file_paths(self, dataset_name, feature_model_name, model_save_folder, first_level_model_name='fl', 
                                   second_level_model_name='sl', max_files=100):

        for idx in range(max_files):
            flm_save_file_name = f'{dataset_name}_{feature_model_name}_{first_level_model_name}_{idx}.pt'
            flm_save_param_path = os.path.join(model_save_folder, flm_save_file_name)

            slm_save_file_name = f'{dataset_name}_{feature_model_name}_{second_level_model_name}_{idx}.pt'
            slm_save_param_path = os.path.join(model_save_folder, slm_save_file_name)

            config_file_name = f'{dataset_name}_{feature_model_name}_{idx}_pt.config'
            config_file_path = os.path.join(model_save_folder, config_file_name)

            if not os.path.exists(config_file_path):
                break

        return flm_save_param_path, slm_save_param_path, config_file_path        
        
    

    def generate_config(self, dataset_name, feature_model_name, global_config):
        model_save_folder = global_config ['model_save_folder']
        random_train_test = global_config['random_train_test']
        in_features = self.feature_model_details[feature_model_name]
        out_features = self.dataset_details[dataset_name]['num_classes']
        
        flm_save_param_path, slm_save_param_path, config_file_path =  self.get_file_paths(dataset_name, feature_model_name,
                                                                                    model_save_folder)

        train_file_indices, test_file_indices, contain_test_set = self.get_file_indices(dataset_name, random_train_test)


        config = global_config.copy()

        special_config = {
            'train_file_indices': train_file_indices,
            'test_file_indices': test_file_indices,
            'contain_test_set': contain_test_set,
            'dataset_name': dataset_name,
            'feature_model_name': feature_model_name,
            'in_features': in_features,
            'out_features': out_features,
            'flm_save_param_path': flm_save_param_path,
            'slm_save_param_path': slm_save_param_path,
            'config_file_path' : config_file_path

        }
        
        for key, value in special_config.items():
            config[key] = value
            
        return config
    
    
    def __getattr__(self, name):
        return self._config[name]
    
    def __setattr__(self, name, value):            
        self._config[name] = value
    
    def load(self, filename):
        return torch.load(filename)
        
    def save(self):
        torch.save(self._config, self.config_file_path)
        
        
# if __name__ == '__main__':
#   dataset_name = 'cholec80'
#   feature_model_name = 'Swin3D_S'
#   config  = TrainerConfigurationGenerator(dataset_name, feature_model_name, config)
#   config.save()
