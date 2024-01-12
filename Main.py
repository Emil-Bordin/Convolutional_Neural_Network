import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping
from EmilBordin969367_CNN import CNNClassification
from EmilBordin969367_Data import DataLoading


#Exercise 6: Training and Evaluation
if __name__ == "__main__":

    data_loading = DataLoading()
    train_loader, val_loader, test_loader = data_loading.load_data()

    model = CNNClassification()

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/")

    trainer = pl.Trainer(devices=1, accelerator="auto", max_epochs=100, logger=tb_logger, callbacks=EarlyStopping(monitor="val_loss", min_delta=0.01, patience=5, verbose=True))

    trainer.fit(model, train_dataloaders= train_loader, val_dataloaders=val_loader)
    trainer.test(ckpt_path="best", dataloaders=test_loader)

#Exercise 7: Results
#Training accuracy: 0.9988
#Test accuracy: 1.0

