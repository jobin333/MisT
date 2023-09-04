hyper_params = {
 'num_head_layers': [1,2,3],
 'optimizer_name': ['adam', 'sgd'],
 'learning_rate': [0.001],
 'frame_skips': [0,2,4,8,16],
 'fine_tune': [False]

}

from types import FrameType
import torch
from mvit.data_utils.dataset_manager import Cholec80DatasetManager
from mvit.models.mvit_model import MvitTrainer

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
data_path = "/content/drive/MyDrive/Datasets/Cholec80/videos_224x224/videos/"
checkpoint_save_dir = '/content/drive/MyDrive/colab_storage'
tubelet_size = 16
batch_size = 1

for num_head_layers in hyper_params['num_head_layers']:
  for optimizer_name in hyper_params['optimizer_name']:
    for learning_rate in hyper_params['learning_rate']:
      for frame_skips in hyper_params['frame_skips']:
        for fine_tune in hyper_params['fine_tune']:
          print('***************************************************************************************************************************')
          print('num_head_layers:{}; optimizer_name:{}; learning_rate:{}; frame_skips:{}; fine_tune:{}'.format(num_head_layers, optimizer_name, learning_rate, frame_skips, fine_tune))
          print('***************************************************************************************************************************')
          dataset_manager = Cholec80DatasetManager(cholec80_dataset_location=data_path,
                                         tubelet_size=tubelet_size, batch_size=batch_size, frame_skips=frame_skips, debugging=True)
          mvit_trainer = MvitTrainer(dataset_manager, fresh_model=True, device=device, save_dir=checkpoint_save_dir,
                                    learning_rate=learning_rate, optimizer_name=optimizer_name, num_head_layers=num_head_layers,
                                     enable_finetune=fine_tune, delete_existing_model=True)
          mvit_trainer.train_model(1)
          mvit_trainer.evaluate_model()

