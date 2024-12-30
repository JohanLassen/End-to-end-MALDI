# Import modules
from maldi_learn.model import MALDI_1D_CNN, backboneModel
from maldi_learn.analyzer import DataAnalyzer
from maldi_learn.utils import load_all_data
import pytorch_lightning as pl
import json
import sys

def main(args):

    # Set seed
    pl.seed_everything(1)
    
    # Get parameters and prefixes
    parameters = json.loads(args[1])

    #parameters = json.loads(args[1])

    prefixes = [
        "/faststorage/project/amr_driams/data/DRIAMS-B/", # Unzipped data folders from the DRIAMS database
        "/faststorage/project/amr_driams/data/DRIAMS-C/", 
        "/faststorage/project/amr_driams/data/DRIAMS-D/", 
        "/faststorage/project/amr_driams/data/DRIAMS-A/"]

    # Load data
    metadata, X = load_all_data(prefixes)
    analysis = DataAnalyzer(X, metadata, parameters)
    analysis.setup()
    # Instantialize pytorch lightning model and callbacks
    model = MALDI_1D_CNN(
        backbone = backboneModel, 
        classes  = analysis.params["num_classes"], 
        params   = analysis.params, 
        learning_rate = analysis.params["lr"],
        cosine_t_max  = analysis.params["epochs"]*len(analysis.train_dataloader())
    )
    
    callbacks = [
        pl.callbacks.EarlyStopping(monitor="val_loss", patience=50, mode="min"),
        pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=3),
        pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
    ]
    

    # Train the model
    analysis.train(model, callbacks)
    
    # Get the predictions
    analysis.evaluate()

    if parameters["sampling"] != "kfold":
        # Get the attributions
        analysis.get_attributions(n_baselines=500, targets=[9, 7, 4])

    # Make the UMAP
    #analysis.make_umap()

if __name__ == "__main__":
    main(sys.argv)
