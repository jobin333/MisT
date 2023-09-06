import torch
import torchvision

from mvit.data_utils.dataset_manager import Cholec80DatasetManager
from mvit.models.video_classifier_trainer import VideoClassifierTrainer

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
data_path = "/content/drive/MyDrive/Datasets/Cholec80/video_224x224_5fps/fps-5"
checkpoint_save_dir = './'
tubelet_size = 16
batch_size = 1
frame_skips = 0

model_names = ['video_resnet', 'swin3d', 's3d', 'mvit']

dataset_manager = Cholec80DatasetManager(cholec80_dataset_location=data_path,
                                tubelet_size=tubelet_size, batch_size=batch_size, 
                                frame_skips=frame_skips, debugging=True)

trainer = VideoClassifierTrainer(dataset_manager, fresh_model=True, device=device, 
                            save_dir=checkpoint_save_dir, learning_rate=0.001, 
                            optimizer_name='adam', model_head=1, model_name='video_resnet',
                            enable_finetune=False, delete_existing_model=True)
trainer.train_model(1)
trainer.evaluate_model()