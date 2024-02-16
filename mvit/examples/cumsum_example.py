import torch
import torchvision
import copy
import os 
from mvit.data_utils.dataset_manager import Cholec80DatasetManager
from mvit.model_utils.trainer import Trainer
from mvit.model_utils.loss_functions import SequentialCrossEntropyLoss
from mvit.data_utils.dataset_manager import ModelOutputDatasetManager
from mvit.models.linear_models import SimpleLinearModel
from mvit.misc.phase_transition_sequence import get_phase_transition_sequence
from mvit.models.bayesian import generate_bayesian_prediction
from mvit.models.cumsum import CumsumModel, CumsumModel2

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
device_cpu = torch.device('cpu')

feature_model = torchvision.models.video.swin3d_s(weights='KINETICS400_V1') # Loading pretrained weights
feature_model.head = torch.nn.Identity()
feature_model = feature_model.to(device)
for param in feature_model.parameters():
  param.requires_grad = False

cholec80_dataset_location = './Dataset/Cholec80/224x224/1sk'
features_save_location = './model_outs/features'
save_model_param_path = './save_params/flow_aggregator.pt'
flow_aggregator_outs_save_location = './model_outs/linear'
cumsum_model_1_param_path = './save_params/cumsum1.pt'

### Video Feature Extraction

video_dataset_manager = Cholec80DatasetManager(cholec80_dataset_location, 
                                         tubelet_size=24,
                                         batch_size=4, 
                                         frame_skips=0, 
                                         debugging=False, shuffle=False,
                                         aproximate_keyframe_interval = 1)

feature_trainer = Trainer(video_dataset_manager, device=device, model=feature_model,
                          model_outs_save_location=features_save_location, enable_training=False)
# feature_trainer.save_model_outs_of_dataset_manager()



#### Transferlearning of Linear Model
seq_length = 30
shuffle=False ## Make it False while saving the output 

flow_aggregator_dataset_manager = ModelOutputDatasetManager(features_save_location, shuffle=shuffle, seq_length=seq_length, batch_size=4096,)
flow_model = SimpleLinearModel(in_features=768, out_features=7, seq_length=30)

flow_trainer = Trainer(flow_aggregator_dataset_manager, device=device, model=flow_model,
                          model_outs_save_location=flow_aggregator_outs_save_location, 
                       loss_fn = SequentialCrossEntropyLoss(), enable_training=True,
                       save_model_param_path=save_model_param_path,
                      optimizer_params = {'lr':0.0001})
flow_trainer.load_model()
# flow_trainer.train_model(20, run_evaluation=True)
# flow_trainer.save_model()
# flow_trainer.save_model_outs_of_dataset_manager()

print()


'''
Added information about the surgical phases encountered.
'''
    
dim = 7
cumsum_model = CumsumModel(dim=dim).to(device)
cumsum_dataset_manager = ModelOutputDatasetManager(flow_aggregator_outs_save_location, shuffle=False,
                                                      batch_size=4096, seq_length=None)


cumsum_trainer = Trainer(cumsum_dataset_manager, device=device, model=cumsum_model,
                       loss_fn = SequentialCrossEntropyLoss(), enable_training=True,
                       save_model_param_path=cumsum_model_1_param_path,
                      optimizer_params = {'lr':0.01})
# cumsum_trainer.save_model()
cumsum_trainer.load_model()
cumsum_trainer.train_model(1,  run_evaluation=True)  



## For printing prediction datapoints

# ds = cumsum_dataset_manager.get_dataloader(47)
# for x,y in ds:
#     break
# x = cumsum_trainer.model(x) 
# def print_prediction_datapoints(labels, predicted):
#     data_points = ' '.join ('[{}{}]'.format(i,j) for i,j in  zip(labels, predicted))
#     print(data_points)
    
# yr = y.detach().cpu().numpy()
# yp = x.argmax(-1).detach().cpu().numpy()  
# print_prediction_datapoints(yr, yp)