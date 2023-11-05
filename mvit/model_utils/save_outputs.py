import os
import torch 

def save_model_outs_of_dataloader(model, dataloader, filename):
  data = []
  model = model.eval()
  for i, (x,y) in enumerate(dataloader):
    feature_x = model(x)
    for item in zip(feature_x, y):
        data.append(item)
  torch.save(data, filename)

def save_model_outs_of_dataset_manager(dataset_manager, model, model_outs_location):
    if not os.path.exists(model_outs_location):
        os.makedirs(model_outs_location)

    for i in range(1,81):
        model_out_path = os.path.join(model_outs_location, 'tensors_{}.pt'.format(i))
        data_loader = dataset_manager.get_dataloader(i)
        print('Creating {}'.format(model_out_path))
        save_model_outs_of_dataloader(model, data_loader, model_out_path)