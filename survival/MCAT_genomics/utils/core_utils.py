from argparse import Namespace
from collections import OrderedDict
import os
import pickle 
from tqdm import tqdm
from lifelines.utils import concordance_index
import numpy as np
from sksurv.metrics import concordance_index_censored

import torch

from models.model_genomic import SNN
from models.model_set_mil import MIL_Sum_FC_surv, MIL_Attention_FC_surv, MIL_Cluster_FC_surv
from models.model_coattn import MCAT_Surv
from utils import *

# from utils.coattn_train_utils import *
# from utils.cluster_train_utils import *

# class EarlyStopping:
#     """Early stops the training if validation loss doesn't improve after a given patience."""
#     def __init__(self, warmup=5, patience=15, stop_epoch=20, verbose=False):
#         """
#         Args:
#             patience (int): How long to wait after last time validation loss improved.
#                             Default: 20
#             stop_epoch (int): Earliest epoch possible for stopping
#             verbose (bool): If True, prints a message for each validation loss improvement. 
#                             Default: False
#         """
#         self.warmup = warmup
#         self.patience = patience
#         self.stop_epoch = stop_epoch
#         self.verbose = verbose
#         self.counter = 0
#         self.best_score = None
#         self.early_stop = False
#         self.val_loss_min = np.Inf

#     def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

#         score = -val_loss

#         if epoch < self.warmup:
#             pass
#         elif self.best_score is None:
#             self.best_score = score
#             self.save_checkpoint(val_loss, model, ckpt_name)
#         elif score < self.best_score:
#             self.counter += 1
#             print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
#             if self.counter >= self.patience and epoch > self.stop_epoch:
#                 self.early_stop = True
#                 self.counter = 0

#         else:
#             print("sono qui")
#             self.best_score = score
#             self.save_checkpoint(val_loss, model, ckpt_name)
#             # self.counter = 0

#     def save_checkpoint(self, val_loss, model, ckpt_name):
#         '''Saves model when validation loss decrease.'''
#         if self.verbose:
#             print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
#         torch.save(model.state_dict(), ckpt_name)
#         self.val_loss_min = val_loss


class Monitor_CIndex:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.best_score = None

    def __call__(self, val_cindex, model, ckpt_name:str='checkpoint.pt'):

        score = val_cindex

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        elif score > self.best_score:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        else:
            pass

    def save_checkpoint(self, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), ckpt_name)



def single_train_loop_survival_coattn(epoch, model, loader, optimizer, n_classes, loss_fn=None, reg_fn=None, lambda_reg=0., gc=16, brca_label="BRCA1"):   
    
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    train_loss_surv, train_loss = 0., 0.

    print('\n')
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))
    
    # Getting the label type 
    if brca_label == "BRCA1":
        brca_col = 0
    else:
        brca_col = 1
    
    for idx, batch in tqdm(enumerate(loader)):

        data_WSI = batch['input']["patch_features"].to(device)
        omic = batch['input']["genomics"][:, brca_col].unsqueeze(1).type(torch.FloatTensor).to(device)
        label = batch['label'].type(torch.LongTensor).to(device)
        censorship = batch['censorship'].type(torch.FloatTensor).to(device)

        #Model output
        hazards, S, Y_hat, A  = model(x_path=data_WSI, x_omic=omic)
        # print(S[0], hazards)
        #Loss
        loss = loss_fn(hazards=hazards, S=S, Y=label, c=censorship)
        
        loss_value = loss.item()
        
        if reg_fn is None:
            loss_reg = 0 #Da capire
        else:
            loss_reg = reg_fn(model) * lambda_reg

        # Risk evaluation
        risk = -torch.sum(S, dim=1).detach().cpu().numpy()
        all_risk_scores[idx] = risk
        all_censorships[idx] = censorship.item()
        all_event_times[idx] = batch["original_event_time"]

        train_loss_surv += loss_value
        train_loss += loss_value + loss_reg

        if (idx + 1) % 100 == 0:
            print('batch {}, loss: {:.4f}, label: {}, event_time: {:.4f}, risk: {:.4f}, bag_size:'.format(idx, loss_value + loss_reg, label.item(), float(batch["original_event_time"]), float(risk)))
        loss = loss / gc + loss_reg
        loss.backward()

        if (idx + 1) % gc == 0: 
            optimizer.step()
            optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss_surv /= len(loader)
    train_loss /= len(loader)
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    print('Epoch: {}, train_loss_surv: {:.4f}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, train_loss_surv, train_loss, c_index))

def single_validate_survival_coattn(cur, epoch, model, loader, n_classes, early_stopping=None, monitor_cindex=None, loss_fn=None, reg_fn=None, lambda_reg=0., results_dir=None,brca_label="BRCA1"):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    val_loss_surv, val_loss = 0., 0.
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))
    
    # Getting the label type 
    if brca_label == "BRCA1":
        brca_col = 0
    else:
        brca_col = 1
    
    for idx, batch in tqdm(enumerate(loader)):

        data_WSI = batch['input']["patch_features"].to(device)
        omic = batch['input']["genomics"][:, brca_col].unsqueeze(1).type(torch.FloatTensor).to(device)
        label = batch['label'].type(torch.LongTensor).to(device)
        censorship = batch['censorship'].type(torch.FloatTensor).to(device)
        
        # Model
        with torch.no_grad():
            hazards, S, Y_hat, A =  model(x_path=data_WSI, x_omic=omic) # return hazards, S, Y_hat, A_raw, results_dict

        # Loss
        loss = loss_fn(hazards=hazards, S=S, Y=label, c=censorship, alpha=0)
        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        # Risk eval
        risk = -torch.sum(S, dim=1).cpu().numpy()
        all_risk_scores[idx] = risk
        all_censorships[idx] = censorship.cpu().numpy()
        all_event_times[idx] = batch["original_event_time"]

        val_loss_surv += loss_value
        val_loss += loss_value + loss_reg


    val_loss_surv /= len(loader)
    val_loss /= len(loader)
    #c_index eval
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]


    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss_surv, model, ckpt_name=os.path.join(results_dir, "s_{}_minloss_checkpoint.pt".format(cur)))
            
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False
    

def train_loop_survival_coattn(epoch, model, loader, optimizer, n_classes, loss_fn=None, reg_fn=None, lambda_reg=0., gc=16):   
    # device= torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    device =  "cpu" 
    model.train()
    train_loss_surv, train_loss = 0., 0.
    
    print('\n')
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))
    
    for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time, c) in enumerate(loader):
        # print(data_WSI)
        data_WSI = data_WSI.to(device)
        data_omic1 = data_omic1.type(torch.FloatTensor).to(device)
        data_omic2 = data_omic2.type(torch.FloatTensor).to(device)
        data_omic3 = data_omic3.type(torch.FloatTensor).to(device)
        data_omic4 = data_omic4.type(torch.FloatTensor).to(device)
        data_omic5 = data_omic5.type(torch.FloatTensor).to(device)
        data_omic6 = data_omic6.type(torch.FloatTensor).to(device)
        label = label.type(torch.LongTensor).to(device)
        c = c.type(torch.FloatTensor).to(device)
        
        # print(f"Il modello è su: {next(model.parameters()).device}")
        hazards, S, Y_hat, A  = model(x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)
        loss = loss_fn(hazards=hazards, S=S, Y=label, c=c)
        loss_value = loss.item()
        
        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        risk = -torch.sum(S, dim=1).detach().cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.item()
        all_event_times[batch_idx] = event_time

        train_loss_surv += loss_value
        train_loss += loss_value + loss_reg

        if (batch_idx + 1) % 100 == 0:
            print('batch {}, loss: {:.4f}, label: {}, event_time: {:.4f}, risk: {:.4f}, bag_size:'.format(batch_idx, loss_value + loss_reg, label.item(), float(event_time), float(risk)))
        loss = loss / gc + loss_reg
        loss.backward()

        if (batch_idx + 1) % gc == 0: 
            optimizer.step()
            optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss_surv /= len(loader)
    train_loss /= len(loader)
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    print('Epoch: {}, train_loss_surv: {:.4f}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, train_loss_surv, train_loss, c_index))

   


def validate_survival_coattn(cur, epoch, model, loader, n_classes, early_stopping=None, monitor_cindex=None, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., results_dir=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss_surv, val_loss = 0., 0.
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time, c) in enumerate(loader):

        data_WSI = data_WSI.to(device)
        data_omic1 = data_omic1.type(torch.FloatTensor).to(device)
        data_omic2 = data_omic2.type(torch.FloatTensor).to(device)
        data_omic3 = data_omic3.type(torch.FloatTensor).to(device)
        data_omic4 = data_omic4.type(torch.FloatTensor).to(device)
        data_omic5 = data_omic5.type(torch.FloatTensor).to(device)
        data_omic6 = data_omic6.type(torch.FloatTensor).to(device)
        label = label.type(torch.LongTensor).to(device)
        c = c.type(torch.FloatTensor).to(device)

        with torch.no_grad():
            hazards, S, Y_hat, A = model(x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6) # return hazards, S, Y_hat, A_raw, results_dict

        loss = loss_fn(hazards=hazards, S=S, Y=label, c=c, alpha=0)
        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        risk = -torch.sum(S, dim=1).cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.cpu().numpy()
        all_event_times[batch_idx] = event_time

        val_loss_surv += loss_value
        val_loss += loss_value + loss_reg


    val_loss_surv /= len(loader)
    val_loss /= len(loader)
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    if writer:
        writer.add_scalar('val/loss_surv', val_loss_surv, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/c-index', c_index, epoch)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss_surv, model, ckpt_name=os.path.join(results_dir, "s_{}_minloss_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def summary_survival_coattn(model, loader, n_classes):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0.

    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time, c) in enumerate(loader):
        
        data_WSI = data_WSI.to(device)
        data_omic1 = data_omic1.type(torch.FloatTensor).to(device)
        data_omic2 = data_omic2.type(torch.FloatTensor).to(device)
        data_omic3 = data_omic3.type(torch.FloatTensor).to(device)
        data_omic4 = data_omic4.type(torch.FloatTensor).to(device)
        data_omic5 = data_omic5.type(torch.FloatTensor).to(device)
        data_omic6 = data_omic6.type(torch.FloatTensor).to(device)
        label = label.type(torch.LongTensor).to(device)
        c = c.type(torch.FloatTensor).to(device)
        slide_id = slide_ids.iloc[batch_idx]

        with torch.no_grad():
            hazards, survival, Y_hat, A  = model(x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6) # return hazards, S, Y_hat, A_raw, results_dict

        risk = np.asscalar(-torch.sum(survival, dim=1).cpu().numpy())
        event_time = np.asscalar(event_time)
        c = np.asscalar(c)
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c
        all_event_times[batch_idx] = event_time
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'risk': risk, 'disc_label': label.item(), 'survival': event_time, 'censorship': c}})

    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    return patient_results, c_index

def train_loop_survival(epoch, model, loader, optimizer, n_classes, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., gc=16):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    train_loss_surv, train_loss = 0., 0.

    print('\n')
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    for batch_idx, (data_WSI, data_omic, label, event_time, c) in enumerate(loader):
        
        data_WSI, data_omic = data_WSI.to(device), data_omic.to(device)
        label = label.to(device)
        c = c.to(device)
        # print(data_omic)
        hazards, S, Y_hat, _, _ = model(x_path=data_WSI, x_omic=data_omic) # return hazards, S, Y_hat, A_raw, results_dict
        
        loss = loss_fn(hazards=hazards, S=S, Y=label, c=c)
        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        risk = -torch.sum(S, dim=1).detach().cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.item()
        all_event_times[batch_idx] = event_time

        train_loss_surv += loss_value
        train_loss += loss_value + loss_reg

        if (batch_idx + 1) % 100 == 0:
            print('batch {}, loss: {:.4f}, label: {}, event_time: {:.4f}, risk: {:.4f}, bag_size: {}'.format(batch_idx, loss_value + loss_reg, label.item(), float(event_time), float(risk), data_WSI.size(0)))
        # backward pass
        loss = loss / gc + loss_reg
        loss.backward()

        if (batch_idx + 1) % gc == 0: 
            optimizer.step()
            optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss_surv /= len(loader)
    train_loss /= len(loader)

    # c_index = concordance_index(all_event_times, all_risk_scores, event_observed=1-all_censorships) 
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    print('Epoch: {}, train_loss_surv: {:.4f}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, train_loss_surv, train_loss, c_index))

    if writer:
        writer.add_scalar('train/loss_surv', train_loss_surv, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/c_index', c_index, epoch)


def validate_survival(cur, epoch, model, loader, n_classes, early_stopping=None, monitor_cindex=None, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., results_dir=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss_surv, val_loss = 0., 0.
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    for batch_idx, (data_WSI, data_omic, label, event_time, c) in enumerate(loader):
        data_WSI, data_omic = data_WSI.to(device), data_omic.to(device)
        label = label.to(device)
        c = c.to(device)

        with torch.no_grad():
            hazards, S, Y_hat, _, _ = model(x_path=data_WSI, x_omic=data_omic) # return hazards, S, Y_hat, A_raw, results_dict

        loss = loss_fn(hazards=hazards, S=S, Y=label, c=c, alpha=0)
        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        risk = -torch.sum(S, dim=1).cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.cpu().numpy()
        all_event_times[batch_idx] = event_time

        val_loss_surv += loss_value
        val_loss += loss_value + loss_reg

    val_loss_surv /= len(loader)
    val_loss /= len(loader)
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    if writer:
        writer.add_scalar('val/loss_surv', val_loss_surv, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/c-index', c_index, epoch)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss_surv, model, ckpt_name=os.path.join(results_dir, "s_{}_minloss_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def test_survival(model, loader, n_classes, brca_label="BRCA1"):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss_surv, val_loss = 0., 0.

    # test_loss = 0.

    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))
    
    # Getting the label type 
    if brca_label == "BRCA1":
        brca_col = 0
    else:
        brca_col = 1
    patient_results = {}
    for idx, batch in tqdm(enumerate(loader)):

        data_WSI = batch['input']["patch_features"].to(device)
        omic = batch['input']["genomics"][:, brca_col].unsqueeze(1).type(torch.FloatTensor).to(device)
        label = batch['label'].type(torch.LongTensor).to(device)
        censorship = batch['censorship'].type(torch.FloatTensor).to(device)
        
        # Model
        with torch.no_grad():
            hazards, S, Y_hat, A =  model(x_path=data_WSI, x_omic=omic) # return hazards, S, Y_hat, A_raw, results_dict

        # Risk eval
        risk = -torch.sum(S, dim=1).cpu().numpy()
        all_risk_scores[idx] = risk
        all_censorships[idx] = censorship.cpu().numpy()
        all_event_times[idx] = batch["original_event_time"]

    
    #c_index eval
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    return patient_results, c_index

def summary_survival(model, loader, n_classes, brca_label="BRCA1"):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0.

    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))
    slide_ids = []
    # print(len(loader.dataset))
    for ids in range(0,len(loader.dataset)):
        # print(loader.dataset[ids]['slide_id'])
        slide_ids.append(loader.dataset[ids]['slide_id'])
    # print(slide_ids)

    # Getting the label type 
    if brca_label == "BRCA1":
        brca_col = 0
    else:
        brca_col = 1
    
    patient_results = {}
    for idx, batch in tqdm(enumerate(loader)):

        data_WSI = batch['input']["patch_features"].to(device)
        omic = batch['input']["genomics"][:, brca_col].unsqueeze(1).type(torch.FloatTensor).to(device)
        label = batch['label'].type(torch.LongTensor).to(device)
        censorship = batch['censorship'].type(torch.FloatTensor).to(device)

        # data_WSI, data_omic = data_WSI.to(device), data_omic.to(device)
        # label = label.to(device)
        
        slide_id = slide_ids[idx]

        with torch.no_grad():
            hazards, survival, Y_hat, _ = model(x_path=data_WSI, x_omic=omic)

        risk = np.isscalar(-torch.sum(survival, dim=1).cpu().numpy())
        event_time = np.isscalar(batch["original_event_time"])
        censorship = np.isscalar(censorship)
        all_risk_scores[idx] = risk
        all_censorships[idx] = censorship
        all_event_times[idx] = batch["original_event_time"]

        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'risk': risk, 'disc_label': label.item(), 'survival': event_time, 'censorship': censorship}})

    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    return patient_results, c_index