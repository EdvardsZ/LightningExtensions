from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from .data_module import construct_kfold_datamodule
import lightning as L
import wandb
import os
import torch


class ExtendedTrainer(L.Trainer):
    def __init__(self, project_name: str, model_name: str, max_epochs: int, devices = [5], monitor = "val_loss", refresh_rate: int = 1, checkpoint_path = "checkpoints/", **kwargs ):
        self.model_name  = model_name
        self.project_name = project_name

        self._epochs = max_epochs

        logger = TensorBoardLogger(save_dir='lightning_logs/', name=self.model_name)
        self.wandb = WandbLogger(project = project_name, name=self.model_name, log_model=False)
        
        self.checkpoint_path = checkpoint_path
        checkpoint_callback = ModelCheckpoint(
            monitor=monitor,
            dirpath=checkpoint_path,
            filename= self.model_name + '_{epoch:02d}-{val_loss:.2f}',
            save_top_k=1,
            mode='min',
        )
        
        progress_bar = TQDMProgressBar(refresh_rate=refresh_rate)
        
        #self.checkpoint_callback = checkpoint_callback
        super().__init__(accelerator='gpu', devices=devices, max_epochs = max_epochs, enable_progress_bar=True, callbacks=[checkpoint_callback, progress_bar], logger=[logger, self.wandb], **kwargs)

    def fit(self, model, train_dataloader, val_dataloader, **kwargs):
        super().fit(model, train_dataloader, val_dataloader, **kwargs)
        self.finish_logging()

    def save_model_checkpoint(self):
        super().save_checkpoint(self.checkpoint_path + self.model_name + '.ckpt')

    def finish_logging(self):
        self.wandb.finalize("success")
        wandb.finish(quiet=True)

    def cross_validate(self, model, train_dataloader, val_dataloader, k = 5):
        
        # Initialize parameters in the model.
        batch = next(iter(train_dataloader))
        model.train()
        x, x_cond, y = batch
        model.forward(x, x_cond, y)
        
        
        # find in the model name the dataset name ( 'dataset=DATASET_NAME&)
        dataset_name = self.model_name.split("dataset=")[1].split("&")[0]
        path = f"assets/results/raw/{self.project_name}/{dataset_name}/{self.model_name}_crossval_results.pt"
        
        # Check if the results exist. If they do, return them
        
        if os.path.exists(path):
            print("Results already exist. Loading them")
            print("Training skippped")
            return torch.load(path)
        
        
    
        print("Starting crossvalidation")

        data_module = construct_kfold_datamodule(train_dataloader, val_dataloader, k) 

        # checkpoint to restore from
        # this is a bit hacky because the model needs to be saved before the fit method
        self.strategy._lightning_module = model
        path = self.checkpoint_path + f"/k_initial_weights_{self.model_name}.ckpt"
        self.save_checkpoint(path)
        self.strategy._lightning_module = None
        
        print("test")

        results = []

        for fold in range(k):
            print("Starting fold: " + str(fold))
            data_module.fold_index = fold

            self.logger = WandbLogger(project = self.project_name, name=self.get_fold_model_name(fold), log_model=False, group = self.model_name[:127])
            
            super().fit(model, data_module, ckpt_path=path)
            
            res = self.test(model=model, datamodule=data_module, ckpt_path=self.checkpoint_callback.best_model_path)
            results.append(res)

            self.finish_logging()
            
            # reset the checkpoint callback
            self.checkpoint_callback.best_model_path = None
            
        # remove k_initial
        os.remove(path)
            
        # find in the model name the dataset name ( 'dataset=DATASET_NAME&)
        dataset_name = self.model_name.split("dataset=")[1].split("&")[0]
        path = f"assets/results/raw/{self.project_name}/{dataset_name}/{self.model_name}_crossval_results.pt"
        # 1. ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # 2. save the results
        torch.save(results, path)
        
        return results
            

    def get_fold_model_name(self, fold):
        return self.model_name + "_fold_" + str(fold)




        
    