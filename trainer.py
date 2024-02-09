import time
import torch
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader


class BertTrainer:
    def __init__(self, model, train_set, val_set, epochs, batch_size, lr, device, stop_patience, results_dir):
        
        self.model = model
        self.train_set = train_set
        self.train_size = len(train_set)
        self.val_set = val_set
        self.epochs = epochs    
        self.batch_size = batch_size
        self.num_batches = self.train_size // self.batch_size
        self.lr = lr
        self.weight_decay = 0.01
        self.current_epoch  = 0
        self.early_stopping_counter = 0	
        self.patience = stop_patience
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.criterion = nn.CrossEntropyLoss(ignore_index = -1).to(device)

        self.device = device
        self.results_dir = results_dir


    def __call__(self):      
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

            criterion = self.stop_early()
            if criterion:
                print(f"Training interrupted at epoch: {self.current_epoch+1}")
                break

        print(f"-=Training completed=-")
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

            tokens = self.model(input, attn_mask) 
            loss = self.criterion(tokens.transpose(-1, -2), token_target) 
            
            epoch_loss += loss.item() 
            reporting_loss += loss.item()
            printing_loss += loss.item()
            
            loss.backward() 
            self.optimizer.step()         
        avg_epoch_loss = epoch_loss / self.num_batches
        return avg_epoch_loss 
    
    def evaluate(self, loader):
        self.model.eval()
        epoch_loss = 0
        total_correct = 0
        total_tokens = 0

        with torch.no_grad():
            for i, batch in enumerate(loader):
                input, token_target, attn_mask = batch
                tokens = self.model(input, attn_mask)
                loss = self.criterion(tokens.transpose(-1, -2), token_target)
                epoch_loss += loss.item()
                
                token_mask =  token_target != -1
                predicted_tokens = tokens.argmax(dim=-1)
                token_target = torch.masked_select(token_target, token_mask)
                predicted_tokens = torch.masked_select(predicted_tokens, token_mask)
                
                correct = (predicted_tokens == token_target).sum().item()
                total_correct += correct
                total_tokens += token_target.numel() 
        
        avg_epoch_loss = epoch_loss / len(loader)
        accuracy = total_correct / total_tokens

        return avg_epoch_loss, accuracy
    
    def _save_model(self, savepath: Path):
        torch.save(self.best_model_state, savepath)
        print(f"Model saved to {savepath}")
        
        
    def _load_model(self, savepath: Path):
        print(f"Loading model from {savepath}")
        self.model.load_state_dict(torch.load(savepath))
        print("Model loaded")

    