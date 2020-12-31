# %% [code]
import os
from matplotlib import pyplot as plt
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from torch.utils.data import TensorDataset, DataLoader,Dataset
import albumentations as albu
from skimage.color import gray2rgb
import functools
from torch import nn
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

# %% [code]
TRAIN_CSV = "/dev/cropped_images/meta_data.csv"
FOLD = 0
FEAT_DIRS = '/dev/embeddings/'
SEQ_LEN = 40
EMBED_SIZE = 2048
LSTM_UNITS = 512
DROPOUT = 0.2
POSITIVE_SAMPLES = [-3, -2, -1, 0, 1, 2, 3]

class BinaryFocalLoss(torch.nn.Module):

    def __init__(self, alpha=3, gamma=2, ignore_index=None, reduction='mean',**kwargs):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = 1e-6 # set '1e-4' when train with FP16
        self.ignore_index = ignore_index
        self.reduction = reduction

        assert self.reduction in ['none', 'mean', 'sum']

    def forward(self, output, target):
        prob = torch.sigmoid(output)
        prob = torch.clamp(prob, self.smooth, 1.0 - self.smooth)

        valid_mask = None
        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index).float()

        pos_mask = (target == 1).float()
        neg_mask = (target == 0).float()
        if valid_mask is not None:
            pos_mask = pos_mask * valid_mask
            neg_mask = neg_mask * valid_mask

        pos_weight = (pos_mask * torch.pow(1 - prob, self.gamma)).detach()
        pos_loss = -torch.sum(pos_weight * torch.log(prob)) / (torch.sum(pos_weight) + 1e-4)
        
        
        neg_weight = (neg_mask * torch.pow(prob, self.gamma)).detach()
        neg_loss = -self.alpha * torch.sum(neg_weight * F.logsigmoid(-output)) / (torch.sum(neg_weight) + 1e-4)
        loss = pos_loss + neg_loss

        return loss

class lstm_config:
    model_name="lstm" 
    batch_size = 128
    WORKERS = 30
    epochs = 20
    optimizer = torch.optim.AdamW
    optimizer_parm = {'lr':5e-4,'weight_decay':0.00001}
    scheduler = torch.optim.lr_scheduler.StepLR
    scheduler_parm = {'step_size':1, 'gamma':0.8}
    #loss_fn = torch.nn.BCEWithLogitsLoss
    #loss_parm = {}
    loss_fn = BinaryFocalLoss
    loss_parm = {'alpha':1.0, 'gamma':2}
    MODEL_PATH = './'
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

# %% [code]
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# %% [markdown]
# # Swish Fn

# %% [code]
class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class Swish_Module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)

# %% [markdown]
# # LSTM Architecture

# %% [code]
class NeuralNet(nn.Module):
    def __init__(self, embed_size=2048, LSTM_UNITS=512, DO = 0.2):
        super(NeuralNet, self).__init__()
        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True,dropout=0.0)
        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True,dropout=0.0)
        
        self.linear1 = nn.Linear(LSTM_UNITS*2, LSTM_UNITS*2)
        self.linear2 = nn.Linear(LSTM_UNITS*2, LSTM_UNITS*2)
        self.linear_pe = nn.Linear(LSTM_UNITS*2, 1)
        
        self.dropuot1 = nn.Dropout(DO)
        self.dropuot2 = nn.Dropout(DO)
        self.dropuot3 = nn.Dropout(DO)
        self.dropuot4 = nn.Dropout(DO)
        
        self.s3d = Swish_Module()
        self.s1d = Swish_Module()
        self.s2d = Swish_Module()
        
        
    def forward(self, embedding):
        
        embedding = self.dropuot1(embedding)
        self.lstm1.flatten_parameters()
        h_lstm1, _ = self.lstm1(embedding)
        h_lstm1 = self.dropuot2(h_lstm1)
        self.lstm2.flatten_parameters()
        h_lstm2, _ = self.lstm2(self.s3d(h_lstm1))
        h_lstm2 = self.dropuot3(h_lstm2)
        
        h_conc_linear1  = self.s1d(self.linear1(h_lstm1))
        h_conc_linear2  = self.s2d(self.linear2(h_lstm2))
        
        hidden = h_lstm1 + h_lstm2 + h_conc_linear1 + h_conc_linear2
        hidden = self.dropuot4(hidden)
        
        output = self.linear_pe(hidden)
        
        return output

# %% [markdown]
# # Trainer

# %% [code]
class trainer:
    def __init__(self,loss_fn,model,optimizer,scheduler,config):
        self.loss_fn = loss_fn
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.metric = loss_fn
        self.CE = torch.nn.CrossEntropyLoss()
        
    def batch_train(self, batch_embd, batch_label, batch_idx):
        batch_embd = batch_embd.cuda().float()
        batch_label = batch_label.cuda().float()
        predicted = self.model(batch_embd)
        batch_label=batch_label.view(predicted.size())
        loss = self.loss_fn(predicted,batch_label)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.detach().cpu().numpy()

    def train_epoch(self, loader):
        self.model.train()
        tqdm_loader = tqdm(loader)
        current_loss_mean = 0
        for batch_idx, (batch_embd, batch_label) in enumerate(tqdm_loader):
            loss = self.batch_train(batch_embd, batch_label, batch_idx)
            current_loss_mean = (current_loss_mean * batch_idx + loss) / (batch_idx + 1)
            tqdm_loader.set_description('loss: {:.4} lr:{:.4}'.format(
                    current_loss_mean, self.optimizer.param_groups[0]['lr']))
        return current_loss_mean
    
    def valid_epoch(self, loader,name="valid"):
        self.model.eval()
        tqdm_loader = tqdm(loader)
        current_loss_mean = 0
        correct = 0
        loss_pre_class =[]
        rsna_all = []
        true_values = []
        pred_values = []
        for batch_idx, (batch_embd, batch_label) in enumerate(tqdm_loader):
            with torch.no_grad():
                batch_embd = batch_embd.cuda().float()
                batch_label = batch_label.cuda().float()
                true_values += list(batch_label.view(-1).detach().cpu().numpy())
                predicted = self.model(batch_embd)
                pred_values += list(predicted.view(-1).detach().cpu().numpy())
                batch_label=batch_label.view(predicted.size())
                
                loss = self.loss_fn(predicted,batch_label)
                loss = loss.detach().cpu().numpy()
                current_loss_mean = (current_loss_mean * batch_idx + loss) / (batch_idx + 1)
                tqdm_loader.set_description(f"loss : {current_loss_mean:.4}")
        
        auc = roc_auc_score(true_values, pred_values)
        score = 1-current_loss_mean
        print('metric {}'.format(score))
        print('AUC {}'.format(auc))
        return auc
    
    def run(self,train_loder,val_loder):
        best_score = -100000
        for e in range(self.config.epochs):
            print("----------Epoch {}-----------".format(e))
            current_loss_mean = self.train_epoch(train_loder)
            score = self.valid_epoch(val_loder)
            self.scheduler.step()
            if best_score < score:
                best_score = score
                torch.save(self.model.state_dict(),self.config.MODEL_PATH+"/{}_best.pth".format(self.config.model_name))
                print(self.config.MODEL_PATH+"/{}_best.pth".format(self.config.model_name))

            
    def load_best_model(self):
        if os.path.exists(self.config.MODEL_PATH+"/{}_best.pth".format(self.config.model_name)):
            self.model.load_state_dict(torch.load(self.config.MODEL_PATH+"/{}_best.pth".format(self.config.model_name)))
            print("load best model")

# %% [markdown]
# # Data Loader

# %% [code]
def get_sequences(df, seq_len):
    seq = []
    v = df.video.values
    p = df.player.values
    c = 0
    a = 0
    seq_len = seq_len//2
    for i in tqdm(range(len(v))):
        if i==0:
            seq.append(c)
        else:
            if (v[i]==v[i-1]) and (p[i]==p[i-1]) and a<seq_len:
                seq.append(c)
            else:
                c+=1
                seq.append(c)
                a=0
        a+=1
    
    return seq

def get_idx(df, is_train):
    s = df.groupby(['video', 'player','seq']).impact.sum()
    df = df[['video', 'player','seq']].drop_duplicates().sort_values(by=['video', 'player','seq']).reset_index(drop=True)
    df = df.merge(s, on=['video', 'player','seq'])
    df['seq_1'] = df.groupby(['video', 'player']).seq.shift(1).reset_index().seq
    df = df.dropna().reset_index(drop=True)
    return dict(zip(range(len(df)),zip(df.seq.values, df.seq_1.values)))
    

class NFLDatasetLstm(Dataset):
    def __init__(self,df,fet_dirs,seq_len=20,is_train=False):
        self.df = df.sort_values(by=['video', 'player', 'frame']).reset_index(drop=True)
        self.fet_dirs = fet_dirs
        self.seq_len = seq_len
        self.is_train = is_train
        self.df['seq']=get_sequences(self.df, self.seq_len)
        self.idx = get_idx(self.df, self.is_train)

    def __getitem__(self, idx):
        dd = self.df[self.df.seq.isin(self.idx[idx])].sort_values(by='frame').reset_index(drop=True)
        files = dd.image_name.to_list()
        labels = dd.impact.to_numpy().reshape(-1)
        seq = np.zeros((self.seq_len,EMBED_SIZE) )
        label_seq = np.zeros((self.seq_len) )
        for i,f in enumerate(files):
            seq[i]=np.load(os.path.join(self.fet_dirs, f[:-3]+'npy'))
            label_seq[i] = labels[i]
        return torch.tensor(seq), torch.tensor(label_seq)

    def __len__(self):
        return len(self.idx)

val_videos = ['57596_002686_Endzone',
               '57596_002686_Sideline',
               '57684_001985_Endzone',
               '57684_001985_Sideline',
               '57686_002546_Endzone',
               '57686_002546_Sideline',
               '57782_000600_Endzone',
               '57782_000600_Sideline',
               '57787_003413_Endzone',
               '57787_003413_Sideline',
               '57904_001367_Endzone',
               '57904_001367_Sideline',
               '57911_000147_Endzone',
               '57911_000147_Sideline',
               '57911_002492_Endzone',
               '57911_002492_Sideline',
               '57913_000218_Endzone',
               '57913_000218_Sideline',
               '58093_001923_Endzone',
               '58093_001923_Sideline',
               '58107_004362_Endzone',
               '58107_004362_Sideline']
               
val_videos = set([i+'.mp4' for i in val_videos])


# %% [markdown]
# # Train

# %% [code]
model_config=lstm_config()

df = pd.read_csv(TRAIN_CSV).sort_values(by=['video', 'player', 'frame']).reset_index(drop=True)
df['p_impact'] = 0

for i in POSITIVE_SAMPLES:
    df['p_impact'] += df.groupby(['video', 'player']).impact.shift(i).fillna(0)
df['p_impact'] = (df.p_impact>0)*1.0

df['impact'] = df.p_impact

#t_df = df[~df.fold.isin([2*FOLD, 2*FOLD+1])].reset_index(drop=True)
t_df = df[~df.video.isin(val_videos)]
#v_df = df[df.fold.isin([2*FOLD, 2*FOLD+1])].reset_index(drop=True)
v_df = df[df.video.isin(val_videos)]

train_dataset = NFLDatasetLstm(df=t_df,fet_dirs=FEAT_DIRS, seq_len=SEQ_LEN, is_train=True)

val_dataset = NFLDatasetLstm(df=v_df,fet_dirs=FEAT_DIRS, seq_len=SEQ_LEN)

train = DataLoader(train_dataset, batch_size=model_config.batch_size, shuffle=True, num_workers=model_config.WORKERS, pin_memory=True)
val = DataLoader(val_dataset, batch_size=model_config.batch_size*1, shuffle=False, num_workers=model_config.WORKERS, pin_memory=True)

model = NeuralNet(embed_size=EMBED_SIZE, LSTM_UNITS=LSTM_UNITS, DO = DROPOUT).cuda()
model = torch.nn.DataParallel(model)

optimizer = model_config.optimizer(model.parameters(),**model_config.optimizer_parm)
scheduler = model_config.scheduler(optimizer,**model_config.scheduler_parm)
loss_fn = model_config.loss_fn(**model_config.loss_parm)

Trainer = trainer(loss_fn,model,optimizer,scheduler,config=model_config)
Trainer.run(train,val)