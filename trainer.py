import time
import torch
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
import copy
import wandb
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from loss_functions import custom_loss

class BertTrainer_pt:
    def __init__(self, model, max_length, train_set, val_set, epochs, batch_size, lr, device, stop_patience, wandb_mode, project_name, wandb_name):
        
        random_seed = 42
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True

        self.model = model
        self.max_length = max_length
        self.train_set = train_set
        self.train_size = len(train_set)
        self.val_size = len(val_set)

        self.val_set = val_set
        self.epochs = epochs    
        self.batch_size = batch_size
        self.num_batches_train = self.train_size // self.batch_size
        self.num_batches_val = self.val_size // self.batch_size
        self.lr = lr
        self.weight_decay = 0.01
        self.current_epoch  = 0
        self.early_stopping_counter = 0	
        self.patience = stop_patience

        self.wandb_mode = wandb_mode
        self.project_name = project_name
        self.wandb_name = wandb_name
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.criterion = nn.CrossEntropyLoss(ignore_index = -1).to(self.device)



    def __call__(self):
        if self.wandb_mode:
            self._init_wandb()       
        self.val_set.prepare_dataset() 
        self.val_loader = DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)
        start_time = time.time()
        self.best_val_loss = float('inf')
        self._init_result_lists()
        for self.current_epoch in range(self.current_epoch, self.epochs):
            #Training
            self.model.train()
            self.train_set.prepare_dataset()
            self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
            epoch_start_time = time.time()
            avg_epoch_loss = self.train(self.current_epoch)
            self.train_losses.append(avg_epoch_loss) 
            print(f"Epoch completed in {(time.time() - epoch_start_time)/60:.1f} min")
            
            #Validation
            print("Evaluating on validation set...")
            val_results = self.evaluate(self.val_loader)
            print(f"Elapsed time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")
            self.val_losses.append(val_results[0])  
            self.val_accs.append(val_results[1])
            if self.wandb_mode:
                self._report_epoch_results()

            criterion = self.stop_early()
            if criterion:
                print(f"Training interrupted at epoch: {self.current_epoch+1}")
                wandb.finish()
                break

        print(f"-=Training completed=-")
        wandb.finish()

        results = {
            "best_epoch": self.current_epoch,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_accs": self.val_accs
        }
        return results

    def _init_result_lists(self):
        self.train_losses = []
        self.val_losses = []
        self.val_accs = []
    
    def stop_early(self):
        if self.val_losses[-1] < self.best_val_loss:
            self.best_val_loss = self.val_losses[-1]
            self.best_epoch = self.current_epoch
            self.best_model_state = self.model.state_dict()
            self.early_stopping_counter = 0
            return False
        else:
            self.early_stopping_counter += 1
            return True if self.early_stopping_counter >= self.patience else False

    def train(self, epoch: int):
        print(f"Epoch {epoch+1}/{self.epochs}")
        time_ref = time.time()
        
        epoch_loss = 0
        reporting_loss = 0
        printing_loss = 0
        for i, batch in enumerate(self.train_loader):
            input, token_target, attn_mask = batch
            
            self.optimizer.zero_grad() 

            token_predictions, resistance_predictions = self.model(input, attn_mask) 
            loss = self.criterion(token_predictions.transpose(-1, -2), token_target) 
            epoch_loss += loss.item() 
            reporting_loss += loss.item()
            printing_loss += loss.item()
            
            loss.backward() 
            self.optimizer.step() 
            
        avg_epoch_loss = epoch_loss / self.num_batches_train
        return avg_epoch_loss 
    
    def evaluate(self, loader):
        self.model.eval()
        epoch_loss = 0
        total_correct = 0
        total_tokens = 0

        with torch.no_grad():
            for i, batch in enumerate(loader):
                input, token_target, attn_mask = batch
                token_predictions, resistance_predictions = self.model(input, attn_mask)

                loss = self.criterion(token_predictions.transpose(-1, -2), token_target)
                epoch_loss += loss.item()
                
                token_mask =  token_target != -1
                predicted_tokens = token_predictions.argmax(dim=-1)
                token_target = torch.masked_select(token_target, token_mask)
                predicted_tokens = torch.masked_select(predicted_tokens, token_mask)
                
                correct = (predicted_tokens == token_target).sum().item()
                total_correct += correct
                total_tokens += token_target.numel() 
        
        avg_epoch_loss = epoch_loss / self.num_batches_val
        accuracy = total_correct / total_tokens

        return avg_epoch_loss, accuracy
    
    def _save_model(self, savepath: Path):
        torch.save(self.best_model_state, savepath)
        print(f"Model saved to {savepath}")
        
        
    def _load_model(self, savepath: Path):
        print(f"Loading model from {savepath}")
        self.model.load_state_dict(torch.load(savepath))
        print("Model loaded")
    
    def _init_wandb(self):
        self.wandb_run = wandb.init(
            project=self.project_name, # name of the project
            name=self.wandb_name, # name of the run
            
            config={
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "num_heads": self.model.attention_heads,
                "num_encoders": self.model.num_encoders,
                "emb_dim": self.model.dim_embedding,
                'ff_dim': self.model.dim_embedding,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "vocab_size": len(self.train_set.vocab_geno),
                "num_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            }
        )
    def _report_epoch_results(self):
        wandb_dict = {
            "epoch": self.current_epoch+1,
        
            "Losses/train_loss": self.train_losses[-1],
            "Losses/val_loss": self.val_losses[-1],
            
            "Accuracies/val_acc": self.val_accs[-1],
        }
        self.wandb_run.log(wandb_dict)

##----------------------------------------------------------------------------------------------------------------------------
class BertTrainer_ft:
    def __init__(self, model, max_length, train_set, val_set, epochs, batch_size, lr, device, stop_patience, wandb_mode, project_name, wandb_name):
        
        random_seed = 42
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True

        self.model = model
        self.max_length = max_length    
        self.train_set = train_set
        self.train_size = len(train_set)
        self.val_size = len(val_set)
        self.val_set = val_set
        self.epochs = epochs    
        self.batch_size = batch_size
        self.num_batches_train = self.train_size // self.batch_size
        self.num_batches_val = self.val_size // self.batch_size

        self.lr = lr
        self.weight_decay = 0.1
        self.current_epoch  = 0
        self.early_stopping_counter = 0	
        self.patience = stop_patience
        self.device = device
        self.wandb_mode = wandb_mode
        self.project_name = project_name
        self.wandb_name = wandb_name
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.token_criterion = nn.CrossEntropyLoss(ignore_index = -1).to(self.device)
        self.ab_criterion = nn.BCEWithLogitsLoss().to(self.device)


    def __call__(self):   
        if self.wandb_mode:
            self._init_wandb()   
        self.val_set.prepare_dataset() 
        self.val_loader = DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)
        start_time = time.time()
        self.best_val_loss = float('inf')
        self._init_result_lists()
        for self.current_epoch in range(self.current_epoch, self.epochs):
            #Training
            self.model.train()
            self.train_set.prepare_dataset()
            self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
            epoch_start_time = time.time()
            avg_epoch_loss_geno, avg_epoch_loss_pheno = self.train(self.current_epoch)
            self.train_losses_geno.append(avg_epoch_loss_geno) 
            self.train_losses_ab.append(avg_epoch_loss_pheno)  
            print(f"Epoch completed in {(time.time() - epoch_start_time)/60:.1f} min")
            
            #Validation
            print("Evaluating on validation set...")
            val_results = self.evaluate(self.val_loader)
            print(f"Elapsed time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")
            self.val_losses_geno.append(val_results[0])
            self.val_losses_ab.append(val_results[1])
            self.val_accs.append(val_results[2])
            self.sensitivity = val_results[3]
            self.specificity = val_results[4]
            if self.wandb_mode:
                self._report_epoch_results()
            criterion = self.stop_early()
            if criterion:
                print(f"Training interrupted at epoch: {self.current_epoch+1}")
                wandb.finish()
                break
        print(f"-=Training completed=-")
        wandb.finish()
        results = {
            "best_epoch": self.best_epoch,
            "geno_train_losses": self.train_losses_geno,
            "ab_train_losses": self.train_losses_ab,
            "geno_val_losses": self.val_losses_geno,
            "ab_val_losses": self.val_losses_ab,
            "val_accs": self.val_accs,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity
        }
        return results

    def _init_result_lists(self):

        self.train_losses_geno = []
        self.train_losses_ab = []

        self.val_losses_geno = []
        self.val_losses_ab = []

        self.sensitivity = []
        self.specificity = []

        self.val_accs = []
    
    def stop_early(self):
        if self.val_losses_ab[-1] < self.best_val_loss:
            self.best_val_loss = self.val_losses_ab[-1]
            self.best_epoch = self.current_epoch
            self.best_model_state = self.model.state_dict()
            self.early_stopping_counter = 0
            return False
        else:
            self.early_stopping_counter += 1
            return True if self.early_stopping_counter >= self.patience else False

    def train(self, epoch: int):
        print(f"Epoch {epoch+1}/{self.epochs}")
        time_ref = time.time()
        
        epoch_loss_geno = 0
        epoch_loss_pheno = 0

        for i, batch in enumerate(self.train_loader):
            input, token_target, attn_mask, AB_idx, SR_class = batch

            ABinclusion = torch.unique(AB_idx)
            ABinclusion = ABinclusion[ABinclusion != -1]
            ABinclusion = ABinclusion.tolist()
            self.model.exclude_networks(ABinclusion)

            self.optimizer.zero_grad() 

            token_predictions, resistance_predictions = self.model(input, attn_mask) 
            geno_loss = self.token_criterion(token_predictions.transpose(-1, -2), token_target) 
            
            result_list = []
            for j in range(len(AB_idx)):
                result_tensor = torch.full((self.max_length[1],), -1) 
                for idx, value in enumerate(AB_idx[j]):
                    if value != -1:
                        result_tensor[value.item()] = SR_class[j][idx]
                result_list.append(result_tensor)
            ab_loss = 0
            pheno_loss = 0
            for i, row in enumerate(resistance_predictions):
                prediction = row
                target = result_list[i]
                ab_loss = custom_loss(prediction, target.float()) 
                pheno_loss += ab_loss
            pheno_loss.backward() 
            epoch_loss_geno += geno_loss.item()
            epoch_loss_pheno += pheno_loss.item()
            self.optimizer.step()
            self.model.reset_exclusion()   
              
        avg_epoch_loss_geno = epoch_loss_geno / self.num_batches_train
        avg_epoch_loss_pheno = epoch_loss_pheno / self.num_batches_train
        return avg_epoch_loss_geno, avg_epoch_loss_pheno
    
    def evaluate(self, loader):
        self.model.eval()
        epoch_loss_geno = 0
        epoch_loss_ab = 0
        total_correct = 0
        total_sum = 0
        TP = [0 for _ in range(self.max_length[1])]
        FP = [0 for _ in range(self.max_length[1])]
        TN = [0 for _ in range(self.max_length[1])]
        FN = [0 for _ in range(self.max_length[1])]
  
        with torch.no_grad():
            for i, batch in enumerate(loader):
                input, token_target, attn_mask, AB_idx, SR_class = batch

                token_predictions, resistance_predictions = self.model(input, attn_mask) 
                geno_loss = self.token_criterion(token_predictions.transpose(-1, -2), token_target) 
                
                result_list = []
                for j in range(len(AB_idx)):
                    result_tensor = torch.full((self.max_length[1],), -1, device=self.device)  # Create tensor filled with -1 values
                    for idx, value in enumerate(AB_idx[j]):
                        if value != -1:
                            result_tensor[value.item()] = SR_class[j][idx]
                    result_list.append(result_tensor)
                ab_loss = 0
                pheno_loss = 0
                for i, row in enumerate(resistance_predictions):
                    prediction = row
                    target = result_list[i]
                    ab_loss = custom_loss(prediction, target.float()) 
                    pheno_loss += ab_loss
                epoch_loss_geno += geno_loss.item()
                epoch_loss_ab += pheno_loss.item() 
                
                list_AB_predictions = []
                pred_res = torch.where(resistance_predictions > 0, torch.ones_like(resistance_predictions), torch.zeros_like(resistance_predictions))

                for i, row in enumerate(pred_res):
                    AB_list = 0
                    AB_list = [elem.item() for elem in AB_idx[i] if elem.item() != -1]
                    current_abs = []
                    for ab in AB_list:
                        current_abs.append(row[ab].item())
                    current_abs = torch.tensor(current_abs)
                    current_abs = current_abs.type(torch.int16)
                    list_AB_predictions.append(current_abs)
                
                    processed_tensor = [row[row != -1] for row in SR_class]
                for i, row in enumerate(processed_tensor):
                    row = row.to(self.device)  # Move row tensor to the same device
                    list_AB_predictions[i] = list_AB_predictions[i].to(self.device)
                    total_correct += (row == list_AB_predictions[i]).sum().item()
                    total_sum += len(row)
                    for j in range(len(row)):
                        if row[j] == list_AB_predictions[i][j]:
                            if  list_AB_predictions[i][j] == 1:
                                Ab = AB_idx[i][j]
                                TP[Ab] += 1
                            else:
                                Ab = AB_idx[i][j]

                                TN[Ab] += 1
                        else:
                            if  list_AB_predictions[i][j] == 1:
                                Ab = AB_idx[i][j]
                                FP[Ab] += 1
                            else:
                                Ab = AB_idx[i][j]
                                FN[Ab] += 1
                specificity = []
                sensitivity = []
                for i in range(len(TP)):
                    TP_i = TP[i]
                    FP_i = FP[i]
                    TN_i = TN[i]
                    FN_i = FN[i]
                    
                    specificity_i = TN_i / (TN_i + FP_i) if (TN_i + FP_i) != 0 else 0
                    sensitivity_i = TP_i / (TP_i + FN_i) if (TP_i + FN_i) != 0 else 0
                    
                    specificity.append(specificity_i)
                    sensitivity.append(sensitivity_i)

        avg_epoch_loss_geno = epoch_loss_geno / self.num_batches_val
        avg_epoch_loss_ab = epoch_loss_ab / self.num_batches_val

        accuracy = total_correct / total_sum

        return avg_epoch_loss_geno, avg_epoch_loss_ab, accuracy, sensitivity, specificity
    
    def _save_model(self, savepath: Path):
        torch.save(self.best_model_state, savepath)
        print(f"Model saved to {savepath}")
        
        
    def _load_model(self, savepath: Path):
        print(f"Loading model from {savepath}")
        self.model.load_state_dict(torch.load(savepath))
        print("Model loaded")

    def _init_wandb(self):
        self.wandb_run = wandb.init(
            project=self.project_name, # name of the project
            name=self.wandb_name, # name of the run
            
            config={
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "num_heads": self.model.attention_heads,
                "num_encoders": self.model.num_encoders,
                "emb_dim": self.model.dim_embedding,
                'ff_dim': self.model.dim_embedding,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "vocab_size": len(self.train_set.vocab_geno),
                "num_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            }
        )
        self.wandb_run.watch(self.model) # watch the model for gradients and parameters
        self.wandb_run.define_metric("epoch", hidden=True)
        self.wandb_run.define_metric("batch", hidden=True)

        self.wandb_run.define_metric("GenoLosses/geno_train_loss", summary="min", step_metric="epoch")
        self.wandb_run.define_metric("GenoLosses/geno_val_loss", summary="min", step_metric="epoch")

        self.wandb_run.define_metric("AB_Losses/ab_train_loss", summary="min", step_metric="epoch")
        self.wandb_run.define_metric("AB_Losses/ab_val_loss", summary="min", step_metric="epoch")

        self.wandb_run.define_metric("Accuracies/val_acc", summary="min", step_metric="epoch")
        
        self.wandb_run.define_metric("Losses/final_val_loss")
        self.wandb_run.define_metric("Accuracies/final_val_acc")
        self.wandb_run.define_metric("final_epoch")

        return self.wandb_run
    
    def _report_epoch_results(self):
        wandb_dict = {
            "epoch": self.current_epoch+1,
            
            "GenoLosses/geno_train_loss": self.train_losses_geno[-1],
            "ABLosses/ab_train_loss": self.train_losses_ab[-1],

            "GenoLosses/geno_val_loss": self.val_losses_geno[-1],
            "ABLosses/ab_val_loss": self.val_losses_ab[-1],
            
            "Accuracies/val_acc": self.val_accs[-1],
        }
        self.wandb_run.log(wandb_dict)