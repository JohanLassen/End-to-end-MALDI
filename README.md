# End-to-end-MALDI


This readme demonstrates how to run the analysis. All results are saved in the parameters output directory. When running the analysis, we used a job scheduler and json files to set the parameters.


#### Import modules
```
from maldi_learn.model import MALDI_1D_CNN, backboneModel
from maldi_learn.analyzer import DataAnalyzer
from maldi_learn.utils import load_all_data
import pytorch_lightning as pl
import sys
```

#### Set parameters
```
# Set seed
pl.seed_everything(1)

# Get parameters and prefixes
parameters = { 
    "output_directory":"./results_random_end_to_end/",
    "task":"multilabel", # multiclass, multilabel or binary 
    "num_classes":28, 
    "num_workers":4, 
    "batch_size":128, 
    "lr":0.005, 
    "optimizer":"Adam", 
    "epochs":100,
    "problem":"antibiotic",
    "cnn1_kernel":60, 
    "fc1_nodes":2656, 
    "max_pool_kernel":30, 
    "max_pool_stride":30, 
    "dropout":0.3,
    "sampling":"random" #random, DRIAMS-A, or DRIAMS-B
    }



prefixes = [
    "/faststorage/project/amr_driams/data/DRIAMS-B/", # Unzipped data folders from the DRIAMS database
    "/faststorage/project/amr_driams/data/DRIAMS-C/", 
    "/faststorage/project/amr_driams/data/DRIAMS-D/", 
    "/faststorage/project/amr_driams/data/DRIAMS-A/"]
```

#### Load the data
```
metadata, X = load_all_data(prefixes)
analysis = DataAnalyzer(X, metadata, parameters)
analysis.setup()
```

#### Instantialize pytorch lightning model and callbacks
```
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
```

#### Train the model
```
analysis.train(model, callbacks)
```

#### Get the predictions (saved in the output directory)
```
analysis.evaluate()
```

#### Get the attributions
```
if parameters["sampling"] != "kfold":
    # Get the attributions
    analysis.get_attributions(n_baselines=500, targets=[9, 7, 4])
```