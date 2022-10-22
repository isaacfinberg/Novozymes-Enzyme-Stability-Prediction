if __name__ == "__main__":
    import pytorch_lightning as pl
    from src.datamodule import AminoAcidSequenceDataModule
    from src.model import LitProtBert
    from src.train import train

    model = LitProtBert()
    data_module = AminoAcidSequenceDataModule(
        train_dir="data/train.csv", prediction_dir="data/test.csv", batch_size=2
    )
    data_module.prepare_data()
    data_module.setup()
    trained_model = train(model, data_module)
