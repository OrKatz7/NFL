
import os
import json 
  
# Opening JSON file 
with open('../settings.json') as json_file: 
    settings = json.load(json_file) 

class data_config:
    train_dir = settings['train']
    train_csv_path = settings['train_csv_path']
    jpeg_dir = settings['jpeg_dir']
    img_size_crop = 256
 
class efficientnetb3:
    model_name="efficientnet-b3"
    batch_size = 30*3
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
    MODEL_PATH = "../"+settings['MODEL_PATH2D']
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
