from pl_crossvalidate import KFoldDataModule
from torch.utils.data import DataLoader



def construct_kfold_datamodule(
        train_dataloader: DataLoader,
        val_dataloaders: DataLoader,
        num_folds: int = 5,
        shuffle: bool = False,
        stratified: bool = False,
) -> KFoldDataModule:
        return KFoldDataModule(
            num_folds,
            shuffle,
            stratified,
            train_dataloader=train_dataloader,
            val_dataloaders=val_dataloaders,
            datamodule=None,
        )