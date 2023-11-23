from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from data_module import construct_kfold_datamodule
import lightning as L
import wandb


class ExtendedTrainer(L.Trainer):
    def __init__(self, project_name: str, model_name: str, max_epochs: int, devices = [5], monitor = "val_loss", **kwargs ):
        self.model_name  = model_name
        self.project_name = project_name

        self._epochs = max_epochs

        logger = TensorBoardLogger(save_dir='lightning_logs/', name=self.model_name)
        self.wandb = WandbLogger(project = project_name, name=self.model_name, log_model="all")

        checkpoint_callback = ModelCheckpoint(
            monitor=monitor,
            dirpath='checkpoints/',
            filename= self.model_name + '_{epoch:02d}-{val_loss:.2f}',
            save_top_k=1,
            mode='min',
        )
        super().__init__(accelerator='gpu', devices=devices, max_epochs = max_epochs, enable_progress_bar=True, callbacks=[checkpoint_callback], logger=[logger, self.wandb], **kwargs)

    def fit(self, model, train_dataloader, val_dataloader, **kwargs):
        super().fit(model, train_dataloader, val_dataloader, **kwargs)
        self.finish_logging()

    def save_model_checkpoint(self):
        super().save_checkpoint('checkpoints/' + self.model_name + '.ckpt')

    def finish_logging(self):
        self.wandb.finalize("success")
        wandb.finish(quiet=True)

    def cross_validate(self, model, train_dataloader, val_dataloader, k = 5):
        print("Starting crossvalidation")

        data_module = construct_kfold_datamodule(train_dataloader, val_dataloader, k) 
        ## TODO: INITIALLY NOT SHUFFLED

        # checkpoint to restore from
        # this is a bit hacky because the model needs to be saved before the fit method
        self.strategy._lightning_module = model
        path = "checkpoints/k_initial_weights.ckpt"
        self.save_checkpoint(path)
        self.strategy._lightning_module = None

        results = []

        for fold in range(k):
            print("Starting fold: " + str(fold))
            data_module.fold_index = fold

            self.logger = WandbLogger(project = self.project_name, name=self.get_fold_model_name(fold), log_model="all", group = self.model_name)
            
            super().fit(model, data_module, ckpt_path=path)
            self.save_model_checkpoint()

            res = self.test(model=model, datamodule=data_module)
            results.append(res)

            self.finish_logging()
            

    def get_fold_model_name(self, fold):
        return self.model_name + "_fold_" + str(fold)




        
    