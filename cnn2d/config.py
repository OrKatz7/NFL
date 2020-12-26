
import os
import json 
  
# Opening JSON file 
with open('../settings.json') as json_file: 
    settings = json.load(json_file) 

class data_config:
    train_dir = "../../"+settings['image_root']
    train_csv_path = settings['train_csv_path']
    img_size_crop = 256
 
class efficientnetb5:
    model_name="efficientnet-b5"
    batch_size = 30*3*3
    WORKERS = 30
    classes =1
    resume = True
    gpu = "0,1,2,3"
    epochs = 5
    optimizer = "torch.optim.AdamW"
    optimizer_parm = {'lr':1e-3,'weight_decay':0.00001}
    scheduler = "torch.optim.lr_scheduler.CosineAnnealingLR"
    scheduler_parm = {'T_max':1000, 'eta_min':1e-6}
    loss_fn = 'torch.nn.BCEWithLogitsLoss'
#     MODEL_PATH = "../"+settings['MODEL_PATH2D']
    MODEL_PATH = "log/"
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
