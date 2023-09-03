import torch 
from mvit.data_utils.dataset_manager import Cholec80DatasetManager
from mvit.models.mvit_model import MvitTrainer

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
data_path = "/content/drive/MyDrive/Datasets/Cholec80/videos_224x224/videos/"
sampling_factor = 1
tubelet_size = 16
batch_size = 1
dataset_manager = Cholec80DatasetManager(cholec80_dataset_location=data_path, sampling_factor=sampling_factor,
                                         tubelet_size=tubelet_size, batch_size=batch_size)

mvit_trainer = MvitTrainer(dataset_manager, fresh_model=True)
mvit_trainer.train_model(1)
mvit_trainer.evaluate_model()