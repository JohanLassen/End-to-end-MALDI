# Imports
import os
import re
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Helper functions for loading data
def get_preprocessed_path(paths):

    total_output = []

    for data_path in paths:
        data_path = data_path + "raw/"

        # List all files in data_path recursively
        for root, _, files in os.walk(data_path):
            for file in files:
                input_path = os.path.join(root, file)
                output_path = input_path#.replace("raw", "preprocessed_raw")
                total_output.append(output_path)
    return total_output


def get_metadata(paths):

    metadata_files = []
    for prefix in paths:
        metadata_path = prefix+"id/"
        for root, _, files in os.walk(metadata_path):
            if len(files)==0:
                continue
            for file in files:
                if not "_clean" in file:
                    continue
                file = root+"/"+file
                metadata_files.append(pd.read_csv(file))

    return pd.concat(metadata_files)

def load_preprocessed(output_paths):
    X = []
    successfully_loaded = []

    for path in tqdm(list(output_paths)):
        try:
            if "raw" in path and not "raw_end_to_end" in path:
                file = np.loadtxt(path, skiprows=3, delimiter = " ")
                file = file[file[:,0]>2000,]
                intensity = file[:,1]**0.25
                x = (-1 + 2*((intensity-intensity.min())/(intensity.max()-intensity.min())))[:20000].reshape(1, -1)
            if x.shape[1] == 20000:#36001:
                X.append(x)
                successfully_loaded.append(True)
            else:
                raise Exception("X is not the supposed length. Skipping to next file")
        except:
            successfully_loaded.append(False)
            continue
    X = np.concatenate(X, axis=0)
    return X, successfully_loaded

def serialize_data(metadata, name):

    if not os.path.exists(name):

        # X of shape (sum(success), n_features) and success of shape (n_files,)
        X, successfully_loaded = load_preprocessed(metadata.full_path.to_list())
        metadata   = metadata[successfully_loaded]

        my_object = (metadata, X)
        # Open a file handle in write mode
        with open(name, 'wb') as file:
            # Serialize the object and write it to the file
            pickle.dump(my_object, file)
    else:
        # Open the file handle in read mode
        with open(name, 'rb') as file:
            # Load the serialized object from the file
            metadata, X = pickle.load(file)

    return metadata, X

# Parse all data across different years
def load_all_data(prefixes):

    # get all preprocessed paths and metadata
    outputs = get_preprocessed_path(prefixes)
    metadata   = get_metadata(prefixes)
    path_dict = {}
    for path in outputs:
        filename = os.path.basename(path)
        path_dict[filename] = path

    # Do simple filtering - mininum 100 samples per species and no MIX samples
    metadata["filename"] = metadata["code"].apply(lambda x: x + ".txt")

    # keep only file names and remove suffix from outputs

    out = np.array([re.sub(".*/|.txt", "", x) for x in outputs])
    metadata = metadata[metadata["code"].isin(out)]

    metadata = metadata[~metadata['species'].str.contains("MIX")]
    species, count = np.unique(metadata.species, return_counts=True)
    accepted_species = species[count > 100]
    metadata = metadata[metadata["species"].isin(accepted_species)]
    metadata['full_path'] = metadata['filename'].map(path_dict)

    # Serialize data
    name = "./datapackage_"+str(metadata.shape[0])+".pickle"
    metadata, X = serialize_data(metadata, name)

    # Add extra metadata to the full dataset
    metadata["sampling_site"] = [re.sub(".*/data/|/raw.*", "", filename) for filename in metadata.full_path.to_list()]
    metadata["year"] = [re.sub(".*/raw/", "", os.path.dirname(filename)) for filename in metadata.full_path.to_list()]

    X = X.reshape(X.shape[0], 1, X.shape[1]) # Shape (n_samples, n_channels, n_features)
    return(metadata, X)



# Generate train test splits for dataloaders
def MaldiDataset(X, metadata, split, params):

    # Not included in the study
    if params["problem"] == "species":
        y = metadata.species.to_numpy()
        metadata = y
        y = (pd.get_dummies(y)*1).to_numpy()
        y = torch.from_numpy(y).float()
        X_train, X_test, y_train, y_test   = train_test_split(X, y, stratify=y, random_state=42)
        X_train, X_val, y_train, y_val     = train_test_split(X_train, y_train, stratify=y_train, random_state=42)
        metadata_train, metadata_test      = train_test_split(metadata, random_state=42)
        metadata_train, metadata_val       = train_test_split(metadata_train, random_state=42)

    if params["problem"] == "antibiotic":

        minimum_occurences = 1000
        metadata_orig = metadata
        species = metadata.species.to_numpy()


        # Remove duplicated columns + meta info
        antibiotic = metadata.drop(columns =
              ["code", 'Unnamed: 0.1', "Unnamed: 0", "species",
               "combined_code", 'filename', "full_path", 'genus',
               'sampling_site', 'year', 'laboratory_species'])

        columns = antibiotic.columns.to_numpy()
        columns = np.array([re.sub("_.*|[.].*", "", x) for x in columns])
        unique  = np.unique(columns)
        y       = ((antibiotic=="R").to_numpy()*1 + (antibiotic=="I").to_numpy()*1)

        agents = list()
        for i in unique:
            index = np.array([i == x for x in columns])
            agents.append(y[...,index].max(axis = 1, keepdims=True))

        # Redefine cleaned data structures
        y = np.concatenate(agents, axis=1)
        antibiotic = pd.DataFrame(y, columns = unique)

        y_keep = y[:,y.sum(axis=0)>minimum_occurences]
        metadata = antibiotic.iloc[:,y.sum(axis=0)>minimum_occurences]
        y = torch.from_numpy(y_keep).float()

        if params["sampling"] == "kfold":

            # Perform a random train, val, test split using random_split on the data and use partitions as TensorDatasets
            dataset = TensorDataset(torch.Tensor(X), torch.Tensor(y))
            Five_folds = torch.utils.data.random_split(dataset, [1/params["kfold"]]*params["kfold"])
            test_dataset = Five_folds[params["fold"]]
            train_dataset = torch.utils.data.ConcatDataset([Five_folds[i] for i in range(5) if i != params["fold"]])
            train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.9, 0.1])

            # Use train val test indices to save metadata, species, and metadata_orig partitions
            train_indices, val_indices, test_indices = train_dataset.indices, val_dataset.indices, test_dataset.indices
            metadata_train, metadata_val, metadata_test = metadata.iloc[train_indices], metadata.iloc[val_indices], metadata.iloc[test_indices]
            species_train, species_val, species_test = species[train_indices], species[val_indices], species[test_indices]
            metadata_orig_train, metadata_orig_val, metadata_orig_test = metadata_orig.iloc[train_indices], metadata_orig.iloc[val_indices], metadata_orig.iloc[test_indices]


        if params["sampling"] == "random":

            # Perform a random train, val, test split using random_split on the data and use partitions as TensorDatasets
            dataset = TensorDataset(torch.Tensor(X), torch.Tensor(y))
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1])

            # Use train val test indices to save metadata, species, and metadata_orig partitions
            train_indices, val_indices, test_indices = train_dataset.indices, val_dataset.indices, test_dataset.indices
            metadata_train, metadata_val, metadata_test = metadata.iloc[train_indices], metadata.iloc[val_indices], metadata.iloc[test_indices]
            species_train, species_val, species_test = species[train_indices], species[val_indices], species[test_indices]
            metadata_orig_train, metadata_orig_val, metadata_orig_test = metadata_orig.iloc[train_indices], metadata_orig.iloc[val_indices], metadata_orig.iloc[test_indices]

        if params["sampling"] == "DRIAMS-A":
            train = ((metadata_orig.sampling_site == "DRIAMS-A") & (metadata_orig.year.isin(["2015", "2016", "2017"]))).to_numpy()
            test = ((metadata_orig.sampling_site == "DRIAMS-A") & (metadata_orig.year.isin(["2018"]))).to_numpy()

            X_train, X_test = X[train], X[test]
            y_train, y_test = y[train], y[test]

            # Perform a random train, val split using random_split on the data and use partitions as TensorDatasets
            train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
            train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.9, 0.1])

            # Use train val test indices to save metadata, species, and metadata_orig partitions
            train_indices, val_indices = train_dataset.indices, val_dataset.indices
            metadata_train, metadata_val = metadata.iloc[train_indices], metadata.iloc[val_indices]
            species_train, species_val = species[train_indices], species[val_indices]
            metadata_orig_train, metadata_orig_val = metadata_orig.iloc[train_indices], metadata_orig.iloc[val_indices]

            # Create test dataset and test metadata
            test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
            metadata_test = metadata[test]
            species_test = species[test]
            metadata_orig_test = metadata_orig[test]


        if params["sampling"] == "DRIAMS-B": # test whether training on all other datasets can predict on a completely new dataset
            train = ((metadata_orig.sampling_site.isin(["DRIAMS-A", "DRIAMS-C", "DRIAMS-D"]))).to_numpy()
            test = ((metadata_orig.sampling_site.isin(["DRIAMS-B"]))).to_numpy()

            X_train, X_test = X[train], X[test]
            y_train, y_test = y[train], y[test]

            # Perform a random train, val split using random_split on the data and use partitions as TensorDatasets
            train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
            train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.9, 0.1])

            # Use train val test indices to save metadata, species, and metadata_orig partitions
            train_indices, val_indices = train_dataset.indices, val_dataset.indices
            metadata_train, metadata_val = metadata.iloc[train_indices], metadata.iloc[val_indices]
            species_train, species_val = species[train_indices], species[val_indices]
            metadata_orig_train, metadata_orig_val = metadata_orig.iloc[train_indices], metadata_orig.iloc[val_indices]

            # Create test dataset and test metadata
            test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
            metadata_test = metadata[test]
            species_test = species[test]
            metadata_orig_test = metadata_orig[test]

    # Split data
    if split == "train":
        return (metadata_train, species_train, metadata_orig_train), train_dataset
    elif split == "val":
        return (metadata_val, species_val, metadata_orig_val), val_dataset
    elif split == "test":
        return (metadata_test, species_test, metadata_orig_test), test_dataset
    else:
        raise ValueError("Split must be train, val or test")



# Create a pytorch lightning data module
class MaldiDataModule(pl.LightningDataModule):

    def __init__(self, X, metadata, params):
        super().__init__()
        self.params = params
        self.batch_size = params["batch_size"]
        self.num_workers = params["num_workers"]
        self.X = X
        self.metadata = metadata
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.attributions = dict()
        self.best_model_path = None

    def prepare_data(self) -> None:
        return super().prepare_data()

    def setup(self, stage=str):
        self.metadata_train, self.train_dataset = MaldiDataset(self.X, self.metadata, "train", self.params)
        self.metadata_val, self.val_dataset = MaldiDataset(self.X, self.metadata, "val", self.params)
        self.metadata_test, self.test_dataset = MaldiDataset(self.X, self.metadata, "test", self.params)

        # get class weights for train y
        if self.params["problem"] == "antibiotic":
            y = self.metadata_train[0].to_numpy()
            print(y)
            class_weights = (y == 0).sum(axis=0)/y.sum(axis=0)
            print(class_weights)
            self.params["class_weights"] = torch.tensor(class_weights).to(self.device)
        else:
            self.params["class_weights"] = None

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=self.num_workers, shuffle=False) # batch_size=1 to avoid memory issues in get_attributions

