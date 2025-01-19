import os
import torch
import torch.optim as optim
# from pytorch_model_summary import summary
import torch.nn.functional as F
import numpy as np
from munch import Munch
from sksurv.metrics import concordance_index_censored
from copy import deepcopy
from lifelines import KaplanMeierFitter
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
import random
from lifelines.statistics import logrank_test
from MIL import *
from dataset_creation import *
# from utility import train,evaluate


def setup(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # If using CUDA.
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
### Fix random seed for reproducibility (Try different random seeds)
SEED = 42
setup(SEED)


# Models
device = "cpu"
# device = "cuda"

models = {
    'ABMIL_Multimodal': ABMIL_Multimodal,
}
model_configs = {
    'ABMIL_Multimodal': {
         "input_dim":1024,
         "genomics_input_dim" : 19962,
         "inner_dim":64, 
         "output_dim":4, 
         "use_layernorm":False, 
         "input_modalities" : ["WSI", "Genomics"], # ["WSI", "Genomics"] # ["WSI"] # ["Genomics"]
         "genomics_dropout": 0.5,
         "dropout":0.0,
    },
}
# Istantiate Model
selected_model = "ABMIL_Multimodal" 
net = models[selected_model](**model_configs[selected_model])
net = net.to(device)

# Loss
reduction = "mean" # sum or mean
loss_function = NLLSurvLoss(alpha=0.0, eps=1e-7, reduction=reduction)

# Training Parameters
MACHINE_BATCH_SIZE = 1
TARGET_BATCH_SIZE = 8
NUM_ACCUMULATION_STEPS = TARGET_BATCH_SIZE//MACHINE_BATCH_SIZE
EPOCHS = 10
PATIENCE = 7
DEBUG_BATCHES = 4

# Optimizer
LR = 0.001
WEIGHT_DECAY = 0.0001
optimizer = optim.RAdam(
                          net.parameters(),
                          lr=LR,
                          weight_decay=WEIGHT_DECAY,
            )
# Scheduler
scheduler_parameters = {
  "milestones": [],
  "gamma": 0.1
}
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_parameters) 


data_loader_config = {
  "datasets_configs": [TCGA_BRCA_dataset_config],
  "task_type": "Survival",
  "max_patches": 4096,
  "batch_size": MACHINE_BATCH_SIZE,
  "real_batch_size": TARGET_BATCH_SIZE,
  "n_bins": 4,
  "sample": True,        # sample patches during train
  "test_sample": False,   # use all available patches during testing
  "load_slides_in_RAM": False,  # load in RAM patches for increasing data loading speed
  "label_name": "FUT",
  "censorships_name": "Survival",
  "eps": 0.000001,
  "num_workers": 1,
  "train_size": 0.7,
  "val_size": 0.15,
  "test_size": 0.15,
  "random_state": 42,
}

data_loader_config = Munch.fromDict(data_loader_config)

dataset = Multimodal_WSI_Genomic_Dataset(    
                        datasets_configs=data_loader_config.datasets_configs, 
                        task_type=data_loader_config.task_type,                           
                        max_patches=data_loader_config.max_patches,
                        n_bins=data_loader_config.n_bins,
                        eps=data_loader_config.eps,
                        sample=data_loader_config.sample,
                        load_slides_in_RAM=data_loader_config.load_slides_in_RAM,
                    )
# GET INDICES FOR TRAIN, VALIDATION, AND TEST SETS
train_patients, val_patients, test_patients = dataset.get_train_test_val_splits(
                                                                                train_size=data_loader_config.train_size, 
                                                                                val_size=data_loader_config.val_size, 
                                                                                test_size=data_loader_config.test_size, 
                                                                                random_state=data_loader_config.random_state
                                                                            )
train_dataloader, val_dataloader, test_dataloader = get_dataloaders(            
                                                                    dataset=dataset,
                                                                    train_patients=train_patients, 
                                                                    val_patients=val_patients, 
                                                                    test_patients=test_patients,
                                                                    data_loader_config=data_loader_config
                                                                    )

def __reload_net__(path, device='cuda'):
    if device == 'cuda':
        print(f'\nRestoring model weigths from: {path}')
        return torch.load(path)
    else:
        print(f'\nRestoring model weigths from: {path}')
        return torch.load(path, map_location=torch.device('cpu'))

def calculate_risk(h):
    r"""
    Take the logits of the model and calculate the risk for the patient 
    
    Args: 
        - h : torch.Tensor 
    
    Returns:
        - risk : torch.Tensor 
    
    """
    hazards = torch.sigmoid(h)
    survival = torch.cumprod(1 - hazards, dim=1)
    risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
    return risk, survival.detach().cpu().numpy()

def initialize_metrics_dict(task_type="Survival"):
    log_dict = {}
    if task_type == "Survival":
        log_dict["all_risk_scores"] = []
        log_dict["all_censorships"] = []
        log_dict["all_event_times"] = []
        log_dict["all_original_event_times"] = []
        log_dict["survival_predictions"] = []
    elif task_type == "Treatment_Response":
        log_dict["all_labels"] = []
        log_dict["treatment_response_predictions"] = []
        log_dict["treatment_response_logits"] = []
    else:
        raise Exception(f"{task_type} is not supported!")
    log_dict["patient_ids"] = []
    log_dict["dataset_name"] = []
    return log_dict

def compute_metrics_dict(log_dict):
    metrics_dict = {}
    all_risk_scores = np.array(log_dict["all_risk_scores"])
    all_censorships = np.array(log_dict["all_censorships"])
    all_event_times = np.array(log_dict["all_event_times"])
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    metrics_dict["c-index"] = c_index
    return metrics_dict    

def compute_metrics(log_df, task_type="Survival", device="cuda"):
    if task_type == "Survival":
        all_risk_scores = log_df["all_risk_scores"].values
        all_censorships = log_df["all_censorships"].values
        all_event_times = log_df["all_event_times"].values
        outputs = log_df["survival_predictions"].values
        c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
        loss = loss_function(torch.tensor(outputs.tolist()), torch.tensor(all_event_times).unsqueeze(-1), None, torch.tensor(all_censorships).unsqueeze(-1))
        metrics_dict = {"c-index": c_index, "Loss": loss}
    elif task_type == "Treatment_Response":
        all_labels = log_df["all_labels"].values
        all_predictions = log_df["treatment_response_predictions"].values
        all_logits = torch.tensor(log_df["treatment_response_logits"].tolist())
        # Calculate AUC
        logits_for_auc = torch.softmax(all_logits, dim=1).numpy()[:, 1]
        auc = roc_auc_score(all_labels, logits_for_auc)    
        f1 = f1_score(all_labels, all_predictions, average='macro')
        accuracy = np.mean(all_labels == all_predictions)        
        # Calculate loss
        all_logits = all_logits.to(device)
        all_labels = torch.tensor(all_labels).long().to(device)
        loss = loss_function(all_logits, all_labels)
        metrics_dict = {"AUC": auc, "Loss": loss, "Accuracy": accuracy, "F1-Score": f1}
    else:
        raise Exception(f"{task_type} is not supported!")
    return metrics_dict

def compute_metrics_df(log_df, task_type="Survival"):
    metrics_dict = {}        
    curr_metrics_dict = compute_metrics(log_df, task_type)
    metrics_dict.update(curr_metrics_dict)

    dataset_names = log_df["dataset_name"].unique()
    for dataset in dataset_names:
        dataset_df = log_df[log_df["dataset_name"]==dataset]
        curr_metrics_dict = compute_metrics(dataset_df, task_type)
        for key, value in curr_metrics_dict.items():
            metrics_dict[f"{dataset}_{key}"] = value
            metrics_dict[f"{dataset}_{key}"] = value
                    
    metrics_dict = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in metrics_dict.items()}
    return metrics_dict    

def step(net, batch, log_dict, task_type="Survival", device="cuda"):
    batch_data = batch['input']
    labels = batch['label'] #  check this casting        
    batch_data = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch_data.items()} 
    labels = labels.to(device)   
             
    if len(labels.shape) == 1:
        labels = labels.reshape(-1,1)    
    # print(batch_data)
    
    outputs, brca1, brca2 = net(batch_data)

    braca1_predictions = torch.sigmoid(brca1).detach().cpu()
    braca2_predictions = torch.sigmoid(brca2).detach().cpu()
    if task_type == "Survival":
        censorships = batch['censorship'] #  check this casting
        censorships = censorships.to(device)     
        if len(censorships.shape) == 1:
            censorships = censorships.reshape(-1,1) 
        risk, survival = calculate_risk(outputs) # output.detach()?
        log_dict["all_risk_scores"]+=(risk.flatten().tolist())
        log_dict["all_censorships"]+=(censorships.detach().view(-1).tolist())
        log_dict["all_event_times"]+=(labels.detach().view(-1).tolist())
        log_dict["all_original_event_times"]+=(batch["original_event_time"].detach().view(-1).tolist())
        log_dict["survival_predictions"] += outputs.detach().tolist()
        #brca_1 brca_2 pred
        log_dict["braca1_predictions"] += braca1_predictions.numpy().flatten().tolist()
        log_dict["braca1_labels"] += batch['BRCA1'].detach().cpu().numpy().tolist()
        log_dict["braca2_predictions"] += braca2_predictions.numpy().flatten().tolist()
        log_dict["braca2_labels"] += batch['BRCA2'].detach().cpu().numpy().tolist()
        if len(risk.shape) == 1:
            risk = risk.reshape(-1,1)    
    elif task_type == "Treatment_Response":
        censorships = None   
        treatment_response_predictions = torch.argmax(outputs.detach().cpu(), dim=1).float()
        log_dict["treatment_response_predictions"] += treatment_response_predictions.numpy().tolist()
        log_dict["treatment_response_logits"] += outputs.detach().cpu().numpy().tolist()
        log_dict["all_labels"] += labels.detach().cpu().numpy().flatten().tolist()
    else:
        raise Exception(f"{task_type} is not supported!")
                
    log_dict["patient_ids"]+=(batch['patient_id'])
    log_dict["dataset_name"]+=(batch['dataset_name'])
    return outputs, labels, censorships, log_dict, brca1, brca2

def KaplanMeier(input_df): #Grafico che fa compare
    df = pd.DataFrame({
        "time": input_df['all_original_event_times']/365.0,
        "event": input_df['all_censorships'], 
        "risk_score": input_df['all_risk_scores']
    })

    # Step 2: Invert the event indicator if needed
    # If currently event=1 means censored and event=0 means event occurred:
    # KaplanMeierFitter expects 1 for event_occurred and 0 for censored.
    df['event_observed'] = 1 - df['event']

    # Step 3: Categorize the risk_score into two groups
    df['risk_group'] = pd.qcut(df['risk_score'], 2, labels=["Low", "High"])

    # Step 4: Fit the Kaplan-Meier curves for each group
    kmf = KaplanMeierFitter()

    plt.figure(figsize=(10, 6))
    for group in df['risk_group'].unique():
        mask = (df['risk_group'] == group)
        kmf.fit(
            durations=df.loc[mask, 'time'],
            event_observed=df.loc[mask, 'event_observed'],
            label=f"Risk: {group}"
        )
        kmf.plot_survival_function(ci_show=True)

    # Step 5: Perform a log-rank test between the two groups (Low vs High)
    low_mask = (df['risk_group'] == "Low")
    high_mask = (df['risk_group'] == "High")

    results = logrank_test(
        durations_A=df.loc[low_mask, 'time'],
        durations_B=df.loc[high_mask, 'time'],
        event_observed_A=df.loc[low_mask, 'event_observed'],
        event_observed_B=df.loc[high_mask, 'event_observed']
    )

    p_value = results.p_value
    print("Log-rank test p-value:", p_value)

    # Step 6: Customize the plot and include p-value in the title
    plt.title(f"Kaplan-Meier Survival Curves by Risk Group\nLog-rank p-value: {p_value:.4f}")
    plt.xlabel("Time (years)")
    plt.ylabel("Survival Probability")
    plt.legend(title='Risk Group')
    plt.grid(True)
    plt.show()

def train(
            net, 
            train_dataloader, 
            eval_dataloader=None, 
            test_dataloader=None, 
            task_type="Survival", 
            checkpoint="nn_model.pt", 
            device="cuda", 
            debug=False, 
            path=".", 
            kfold="", 
            best_model_criterion="highest" # se ho un eval dataloader mi trova la migliroe metrica tipo il C-Index per MCAT. Se è lowest è quello con la loss più bassa, quale devo prendere?
):
    cudnn.benchmark = False
    trainLoss = []
    validationLoss = []
    testLoss = []
    lowest_val_loss = np.inf
    highest_val_metric = 0
    STOP = False
    df_fold_suffix = f"_{kfold}"
    log_fold_string = f"/{kfold}"

    for epoch in range(EPOCHS):
        if STOP:
            print(f'\nSTOPPED at epoch {epoch}')
            break
        if kfold != "":
            print(f'\nStarting training for {kfold}')
        print('\nStarting epoch {}/{}, LR = {}'.format(epoch + 1, EPOCHS, scheduler.get_last_lr()))
        tloss = []
        bce_loss_fn = nn.BCEWithLogitsLoss() 
        batch_numb = 0
        log_dict = {}
        net.train()
        train_dataloader.dataset.dataset.set_sample(data_loader_config.sample)
        for idx, batch in tqdm(enumerate(train_dataloader)):
            if debug and batch_numb == DEBUG_BATCHES:
                print("DEBUG_BATCHES value reached")
                break
            if idx == 0:
                log_dict = initialize_metrics_dict(task_type)

            outputs, labels, censorships, log_dict, brca1 ,brca2 = step(net, batch, log_dict, task_type, device)
            
            if MACHINE_BATCH_SIZE <= TARGET_BATCH_SIZE:
                if task_type == "Survival":
                    brca1_labels = batch["brca1"].to(device).float().view(-1, 1)
                    brca2_labels = batch["brca2"].to(device).float().view(-1, 1)
                    loss = loss_function(outputs, labels, None, censorships)
                    loss_brca1 = bce_loss_fn(brca1, brca1_labels)
                    loss_brca2 = bce_loss_fn(brca2, brca2_labels)
                    loss = loss + loss_brca1 + loss_brca2 #differsent weights?
                elif task_type == "Treatment_Response":
                    loss = loss_function(outputs, labels.squeeze(1))
                else:
                    raise Exception(f"{task_type} is not supported!")

                loss = loss / NUM_ACCUMULATION_STEPS
                tloss.append(loss.item()*NUM_ACCUMULATION_STEPS)
                loss.backward()
                if ((idx + 1) % NUM_ACCUMULATION_STEPS == 0):
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1)
                    optimizer.step()    
                    optimizer.zero_grad()
            else:
                    raise Exception("MACHINE_BATCH_SIZE MUST BE EQUAL OR SMALLER THAN TARGET_BATCH_SIZE!!!")            
            batch_numb += 1
        
        scheduler.step()
        tloss = np.array(tloss)
        tloss = np.mean(tloss) 
        trainLoss.append(tloss)
        # print(log_dict)
        train_df = pd.DataFrame(log_dict)   
        print(train_df)
        train_df.to_hdf(f"{path}/train_df{df_fold_suffix}.h5", key="df", mode="w")
        train_metrics_dict = compute_metrics_df(train_df, task_type)
        train_metrics_df = pd.DataFrame(train_metrics_dict, index=[0])
        train_metrics_df.to_csv(f"{path}/train_metrics{df_fold_suffix}.csv")
        # KaplanMeier_plot(log_dict, train_dataloader.dataset.dataset.bins.astype(int))
        # predTime_vs_actualTime_confusionMatrix_plot(log_dict)
        to_log = {
                f'Epoch': epoch + 1,
                f'LR': optimizer.param_groups[0]['lr'],
                # f'Train/Loss': tloss,
                # f'Train/c-index': train_metrics_dict["c-index"],
                # f'Valid/Loss': vloss,
                # f'Valid/c-index': val_metrics_dict["c-index"],
                }
        for key, value in train_metrics_dict.items():
            to_log[f'Train{log_fold_string}/{key}'] = value
            
        train_dataloader.dataset.dataset.set_sample(data_loader_config.test_sample)
        if eval_dataloader is not None:
            net.eval()
            vloss = []
            vlossWeights = []
            batch_numb = 0
            with torch.inference_mode():
                for idx, batch in tqdm(enumerate(eval_dataloader)):
                    if debug and batch_numb == DEBUG_BATCHES:
                        break
                    if idx == 0:
                        log_dict = initialize_metrics_dict(task_type)
                    outputs, labels, censorships, log_dict = step(net, batch, log_dict, task_type, device)
                    if task_type == "Survival":
                        loss = loss_function(outputs, labels, None, censorships)
                    elif task_type == "Treatment_Response":
                        loss = loss_function(outputs, labels.squeeze(1))
                    else:
                        raise Exception(f"{task_type} is not supported!")
                    if MACHINE_BATCH_SIZE <= TARGET_BATCH_SIZE:
                        vloss.append(loss.detach().item())
                    vlossWeights.append(batch["label"].size(dim=0))
                    batch_numb += 1
            vloss = np.array(vloss)
            vloss = np.average(vloss, weights=vlossWeights)
            # vloss = np.sum(vloss)
            validationLoss.append(vloss)
            val_df = pd.DataFrame(log_dict)                
            val_df.to_hdf(f"{path}/val_df{df_fold_suffix}.h5", key="df", mode="w")
            val_metrics_dict = compute_metrics_df(val_df, task_type)
            val_metrics_df = pd.DataFrame(val_metrics_dict, index=[0])
            val_metrics_df.to_csv(f"{path}/val_metrics{df_fold_suffix}.csv")
            for key, value in val_metrics_dict.items():
                to_log[f'Valid{log_fold_string}/{key}'] = value
            vmetric = val_metrics_df["c-index"].values[0]


        if test_dataloader is not None:
            net.eval()
            ttloss = []
            ttlossWeights = []
            batch_numb = 0
            with torch.inference_mode():
                for idx, batch in tqdm(enumerate(test_dataloader)):
                    if debug and batch_numb == DEBUG_BATCHES:
                        break
                    if idx == 0:
                        log_dict = initialize_metrics_dict(task_type)
                    outputs, labels, censorships, log_dict = step(net, batch, log_dict, task_type, device)
                    if task_type == "Survival":
                        loss = loss_function(outputs, labels, None, censorships)
                    elif task_type == "Treatment_Response":
                        loss = loss_function(outputs, labels.squeeze(1))
                    else:
                        raise Exception(f"{task_type} is not supported!")
                    if MACHINE_BATCH_SIZE <= TARGET_BATCH_SIZE:
                        ttloss.append(loss.item())
                    else:
                        ttloss.append(loss.detach().mean().item())
                    ttlossWeights.append(batch["label"].size(dim=0))
                    batch_numb += 1
            ttloss = np.array(ttloss)
            ttloss = np.average(ttloss, weights=ttlossWeights)
            # ttloss = np.sum(ttloss)
            testLoss.append(ttloss)
            test_df = pd.DataFrame(log_dict)                
            test_df.to_hdf(f"{path}/test_df{df_fold_suffix}.h5", key="df", mode="w")
            test_metrics_dict = compute_metrics_df(test_df, task_type)
            test_metrics_df = pd.DataFrame(test_metrics_dict, index=[0])
            test_metrics_df.to_csv(f"{path}/test_metrics{df_fold_suffix}.csv")

            test_log = {
                # f'Test/Loss': ttloss,
                # f'Test/c-index': test_metrics_dict["c-index"],    
                } 
            for key, value in test_metrics_dict.items():
                test_log[f'Test{log_fold_string}/{key}'] = value
            
            to_log.update(test_log) 
            # if task_type == "Treatment_Response":
            #     plot_to_log = {
            #         f"Train{log_fold_string}/Confusion_Matrix": wandb.Image(train_confusion_matrix),
            #     }
            #     if eval_dataloader is not None:
            #         plot_to_log[f"Valid{log_fold_string}/Confusion_Matrix"] = wandb.Image(val_confusion_matrix)
            #     if test_dataloader is not None:
            #         plot_to_log[f"Test{log_fold_string}/Confusion_Matrix"] = wandb.Image(test_confusion_matrix)
            #     to_log.update(plot_to_log) 
                                 
               
        # wandb.log(to_log)    
        print(f"EPOCH {epoch+1}\n")    
        for k, v in to_log.items():
            pad = ' '.join(['' for _ in range(25-len(k))])
            print(f"{k}:{pad} {v}", flush=True)
  

        # Early stopping
        if eval_dataloader is not None:
            if best_model_criterion == "lowest":
                if vloss < lowest_val_loss:
                    lowest_val_loss = vloss
                    patience_counter = 0
                    lowest_val_loss_epoch = epoch + 1
                    print(
                        f"############################################ New lowest_val_loss reached: {lowest_val_loss} #########################")
                    if kfold != "":
                        checkpoint_splitted = checkpoint.split(".")
                        checkpoint = f"{checkpoint_splitted[0]}{df_fold_suffix}.pt"
                    torch.save(net, checkpoint)
                    # wandb.run.summary["Lowest_Validation_Loss/Epoch"] = lowest_val_loss_epoch
                    # wandb.run.summary["Lowest_Validation_Loss/Validation_Loss"] = lowest_val_loss
                    # wandb.run.summary["Lowest_Validation_Loss/Validation_c-index"] = val_metrics_dict["c-index"],
                    for key, value in val_metrics_dict.items():
                        # wandb.run.summary[f"Lowest_Validation_Loss/Valid{log_fold_string}/{key}"] = value
                        print(f"Lowest_Validation_Loss/Valid{log_fold_string}/{key}: {value}")
                    # # wandb.run.summary["Lowest_Validation_Loss/Validation_KM"] = val_metrics_dict["KM"],
                    train_df.to_hdf(f"{path}/best_train_df{df_fold_suffix}.h5", key="df", mode="w")
                    train_metrics_df.to_csv(f"{path}/best_train_metrics{df_fold_suffix}.csv")
                    val_df.to_hdf(f"{path}/best_val_df{df_fold_suffix}.h5", key="df", mode="w")
                    val_metrics_df.to_csv(f"{path}/best_val_metrics{df_fold_suffix}.csv")    
            elif best_model_criterion == "highest":
                if vmetric > highest_val_metric:
                    highest_val_metric = vmetric
                    patience_counter = 0
                    lowest_val_loss_epoch = epoch + 1
                    print(
                        f"############################################ New highest_val_metric reached: {highest_val_metric} #########################")
                    if kfold != "":
                        checkpoint_splitted = checkpoint.split(".")
                        checkpoint = f"{checkpoint_splitted[0]}{df_fold_suffix}.pt"
                    torch.save(net, checkpoint)
                    # wandb.run.summary["Lowest_Validation_Loss/Epoch"] = lowest_val_loss_epoch
                    # wandb.run.summary["Lowest_Validation_Loss/Validation_Loss"] = lowest_val_loss
                    # wandb.run.summary["Lowest_Validation_Loss/Validation_c-index"] = val_metrics_dict["c-index"],
                    for key, value in val_metrics_dict.items():
                        # wandb.run.summary[f"Lowest_Validation_Loss/Valid{log_fold_string}/{key}"] = value
                        print(f"Lowest_Validation_Loss/Valid{log_fold_string}/{key}: {value}")
                    # # wandb.run.summary["Lowest_Validation_Loss/Validation_KM"] = val_metrics_dict["KM"],
                    train_df.to_hdf(f"{path}/best_train_df{df_fold_suffix}.h5", key="df", mode="w")
                    train_metrics_df.to_csv(f"{path}/best_train_metrics{df_fold_suffix}.csv")
                    val_df.to_hdf(f"{path}/best_val_df{df_fold_suffix}.h5", key="df", mode="w")
                    val_metrics_df.to_csv(f"{path}/best_val_metrics{df_fold_suffix}.csv")  
            else:
                raise Exception(f"{best_model_criterion} is not supported!")

            if patience_counter == PATIENCE:
                print(f"End of training phase - Patience threshold reached\nWeights Restored from Lowest val_loss epoch: {lowest_val_loss_epoch}\nlowest_val_loss: {lowest_val_loss}")
                STOP = True
            else:
                patience_counter += 1
def evaluate(net, test_dataloader, task_type="Survival", checkpoint=None, device="cuda", best=False, path=".", kfold=""):
    cudnn.benchmark = False
    print("test")   
    df_fold_suffix = f"_{kfold}"
    log_fold_string = f"/{kfold}"     
    if best:
        if kfold != "":
            checkpoint_splitted = checkpoint.split(".")
            checkpoint = f"{checkpoint_splitted[0]}{df_fold_suffix}.pt"
        net = torch.load(checkpoint)
        print("\n Evalate best model")
    else:
        net = net
        print("\n Evalate last model")

    net = net.to(device)
    net.eval()
    tloss = []
    tlossWeights = []
    with torch.inference_mode():
        for idx, batch  in enumerate(test_dataloader):
            if idx == 0:
                    log_dict = initialize_metrics_dict(task_type)
            # batch_data = torch.squeeze(batch_data, 0)
            outputs, labels, censorships, log_dict = step(net, batch, log_dict, task_type, device)
            if task_type == "Survival":
                loss = loss_function(outputs, labels, None, censorships)
            elif task_type == "Treatment_Response":
                loss = loss_function(outputs, labels.squeeze(1))
            else:
                    raise Exception(f"{task_type} is not supported!")

            if MACHINE_BATCH_SIZE <= TARGET_BATCH_SIZE:
                tloss.append(loss.item())
            else:
                tloss.append(loss.detach().mean().item())
            tlossWeights.append(batch["label"].size(dim=0))

    tloss = np.array(tloss)
    tloss = np.average(tloss, weights=tlossWeights)
    test_df = pd.DataFrame(log_dict)
    test_metrics_dict = compute_metrics_df(test_df, task_type)

    if best:
        # wandb.run.summary["Lowest_Validation_Loss/Test_Loss"] = tloss
        # wandb.run.summary["Lowest_Validation_Loss/Test_c-index"] = test_metrics_dict["c-index"]
        for key, value in test_metrics_dict.items():
                # wandb.run.summary[f"Lowest_Validation_Loss/Test{log_fold_string}/{key}"] = value
                print(f"Lowest_Validation_Loss/Test{log_fold_string}/{key}: {value}")
        test_df.to_hdf(f"{path}/best_test_df{df_fold_suffix}.h5", key="df", mode="w")
        KaplanMeier(test_df)
        test_metrics_df = pd.DataFrame(test_metrics_dict, index=[0])
        test_metrics_df.to_csv(f"{path}/best_test_metrics{df_fold_suffix}.csv")
        # if task_type == "Treatment_Response":
        #     test_confusion_matrix = accuracy_confusionMatrix_plot(log_dict, test_metrics_df)
    else:
        for key, value in test_metrics_dict.items():
                # wandb.run.summary[f"Last_Epoch_Model/Test{log_fold_string}/{key}"] = value
                print(f"Last_Epoch_Model/Test{log_fold_string}/{key}: {value}")
        test_df.to_hdf(f"{path}/last_epoch_test_df{df_fold_suffix}.h5", key="df", mode="w")
        KaplanMeier(test_df)
        test_metrics_df = pd.DataFrame(test_metrics_dict, index=[0])
        test_metrics_df.to_csv(f"{path}/last_epoch_test_metrics{df_fold_suffix}.csv")
        # if task_type == "Treatment_Response":
        #     test_confusion_matrix = accuracy_confusionMatrix_plot(log_dict, test_metrics_df)                                                        
print("selected_model: ", selected_model)
train(
      net,
      train_dataloader, 
      val_dataloader, 
      test_dataloader,
      checkpoint="best_model.pt", 
      device=device, 
      debug=False,
      best_model_criterion="highest", # highest for highest validation metric, lowest for lowest validation loss
     )

evaluate(net, test_dataloader, task_type="Survival", checkpoint=None, device=device, best=False, path=".", kfold="")