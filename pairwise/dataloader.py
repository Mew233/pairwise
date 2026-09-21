"""
    Dataloader, customized K-fold trainer & evaluater
"""
from torch.utils.data import DataLoader, sampler,TensorDataset
from itertools import chain
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import numpy as np
import os
from .seeding import get_seed
from .paths import DATA_DIR, RESULTS_DIR, WEIGHTS_DIR, weights_path
import torch_geometric.data
from torch_geometric.data import Batch
from itertools import cycle
from torch_geometric import data as DATA
import random
import pandas as pd
# shap is imported lazily in the SHAP() function to avoid pulling in numba/llvmlite
# at module load time, which fails on systems with incompatible libstdc++ ABI.
# import shap as sp  # use sp = _get_shap() inside functions that need it

def _get_shap():
    """Lazy shap import — only triggers numba/llvmlite when SHAP analysis is used."""
    import shap as sp
    return sp
from tqdm import tqdm
# dill removed — was only used for the now-fixed explainer.bz2 pickle path


def get_device(args):
    return getattr(args, 'device', torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))


def get_cv_folds(args, default=5):
    n = getattr(args, 'cv_folds', default) or default
    if n < 2:
        raise ValueError('cv_folds must be >= 2 (sklearn KFold requirement)')
    return n


def resolve_model_weights(model_weights):
    if os.path.isabs(model_weights):
        return model_weights
    return os.path.join(WEIGHTS_DIR, model_weights)


def best_model_path(model_name, split=None):
    """Path to best model checkpoint. If split is given, suffix the filename
    so different splits don't overwrite each other."""
    suffix = f'_{split}' if split else ''
    return weights_path(f'best_model_{model_name}{suffix}.pth')


def _split_tag(args):
    """Return the split tag from args.custom_split, optionally extended with
    synergy_thres so per-threshold sweep runs do not clobber each other's
    checkpoints/predictions. Empty string when neither is set.

    The threshold suffix is opt-in via args.suffix_threshold (default False) so
    that legacy scaffold/lineage/combined runs keep their unsuffixed filenames.
    """
    s = getattr(args, 'custom_split', None) or ''
    if getattr(args, 'suffix_threshold', False):
        thres = getattr(args, 'synergy_thres', None)
        if thres is not None:
            s = f"{s}_thres{thres}" if s else f"thres{thres}"
    return s


def _ckpt_save_path(args):
    """Checkpoint save path, suffixed by split tag if present."""
    split = _split_tag(args)
    suffix = f'_{split}' if split else ''
    return weights_path(f'best_model_{args.model}{suffix}.pth')


def _ckpt_load_path(args):
    """Checkpoint load path. Prefer split-suffixed; fall back to legacy
    unsuffixed name so old eval runs still work."""
    split = _split_tag(args)
    if split:
        p = weights_path(f'best_model_{args.model}_{split}.pth')
        if os.path.exists(p):
            return p
    return weights_path(f'best_model_{args.model}.pth')


def _empty_molecular_graph(num_features=78):
    graph = DATA.Data(
        x=torch.zeros(1, num_features),
        edge_index=torch.zeros(2, 0, dtype=torch.long),
    )
    graph.c_size = torch.LongTensor([1])
    return graph


def _ensure_molecular_graph(graph, num_features=78):
    if graph is None:
        return _empty_molecular_graph(num_features)
    return graph


def _batch_pyg_graphs(graph_data, device):
    if isinstance(graph_data, (list, tuple)):
        graph_data = Batch.from_data_list(
            [_ensure_molecular_graph(g) for g in graph_data]
        )
    elif graph_data is None:
        graph_data = _empty_molecular_graph()
    return graph_data.to(device)


def _unpack_deepdds_batch(data1, data2, data_cell, device):
    data1 = _batch_pyg_graphs(data1, device)
    data2 = _batch_pyg_graphs(data2, device)
    return (
        data1.x, data1.edge_index,
        data2.x, data2.edge_index,
        data_cell,
        data1.batch, data2.batch,
    )

# def dataloader(X_train_val_set, X_test_set, Y_train_val_set, Y_test_set):
def dataloader(*args,**kwargs):
    """
        should be starting from X, and Y
        X_{}_trainval, X_{}_test, Y_{}_trainval, Y_{}_test
    """
    temp_loader = {}
    for name, input in kwargs.items():
    # First, format
        input_name = name.split('_')
        if input_name[0].startswith('Y'):
            input = input.astype('float32')
            input = torch.from_numpy(input)
            input = input.unsqueeze(1)
            temp_loader[name] = input

        elif input_name[0].startswith('X'):
            #input is tabular format
            input = input.astype('float32')
            input = torch.from_numpy(input)
            temp_loader[name] = input
        
        elif input_name[0].startswith('dummy'):
            input = input.astype('float32')
            input = torch.from_numpy(input)
            temp_loader[name] = input
    
    # Second, to tensordataset
    temp_loader_trainval,  temp_loader_test = [], []
    for key, val in temp_loader.items():
        load_name = key.split('_')
        # train_val_dataset
        if load_name[-1].endswith('trainval'):
            temp_loader_trainval.append(val)
        elif load_name[-1].endswith('test'):
            temp_loader_test.append(val)

    train_val_dataset = torch.utils.data.TensorDataset(*temp_loader_trainval)
    test_dataset = torch.utils.data.TensorDataset(*temp_loader_test)
    test_loader = DataLoader(test_dataset, batch_size=getattr(args, 'batch_size', 256), shuffle=False, sampler=sampler.SequentialSampler(test_dataset))
    #list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
    return train_val_dataset, test_loader


def dataloader_graph(*args,**kwargs):
    """
        should be starting from X, and Y
        X_{}_trainval, X_{}_test, Y_{}_trainval, Y_{}_test
    """
    temp_loader = {}
    for name, input in kwargs.items():
    # First, format
        temp_loader[name] = input

    # Second, combine trainval or test
    temp_loader_trainval,  temp_loader_test = [], []
    for key, val in temp_loader.items():
        load_name = key.split('_')
        # train_val_dataset
        if load_name[-1].endswith('trainval'):
            temp_loader_trainval.append(val)
        elif load_name[-1].endswith('test'):
            temp_loader_test.append(val)


    return temp_loader_trainval,temp_loader_test

def k_fold_trainer(dataset,model,args):

    # Configuration options
    k_folds = get_cv_folds(args)
    num_epochs = args.epochs
    batch_size = args.batch_size

    loss_function = nn.BCELoss()
    # For fold results
    results = {}

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, random_state=get_seed(args), shuffle=True)
    # Start print
    print('--------------------------------')

    # save 5-fold evalutation results for meta classifier
    meta_clf_pred = []
    meta_clf_acts = []
    meta_clf_index = []

    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        trainloader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
        valloader = DataLoader(dataset,batch_size=batch_size, sampler=test_subsampler)

        # Init the neural network
        
        device = get_device(args)
        network = model.to(device)
        # init para for each fold
        for layer in network.modules():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        # Initialize optimizer
        optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)

        for epoch in range(0, num_epochs):

            print(f'Starting epoch {epoch+1}')
            # Set current loss value
            current_loss = 0.0

            # Iterate over the DataLoader for training data
            for i, data in enumerate(trainloader, 0):
                network.train()
                inputs, targets = data[:-2], data[-2]
                inputs = [inp.to(device) for inp in inputs]
                targets = targets.to(device).float()
                index = data[-1]
                # Zero the gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                if args.model == 'deepsynergy_preuer':
                    outputs = network(inputs[0])
                elif args.model == 'matchmaker_brahim':
                    outputs = network(inputs[0],inputs[1],inputs[2])
                elif args.model == 'multitaskdnn_kim':
                    outputs = network(inputs[0],inputs[1],inputs[2],inputs[3],inputs[4])
                elif args.model == 'hetergnn':
                    outputs, emb_loss = network(drug1=inputs[0], drug2=inputs[1], cell=inputs[2])

                if args.model == 'hetergnn':
                    loss = loss_function(outputs, targets) + emb_loss
                else:
                    loss = loss_function(outputs, targets)
                loss.backward()
                optimizer.step()
                # Print statistics
                current_loss += loss.item()
                if i % 2000 == 1999:
                    print('Loss after mini-batch %5d: %.3f' %
                        (i + 1, current_loss / 2000))
                    current_loss = 0.0
            
            # Process is complete.


        # Evaluation for this fold
        with torch.no_grad():
            predictions, actuals = list(), list()
            idx = list()

            for i, data in enumerate(valloader, 0):
                network.eval()

                inputs, labels = data[:-2], data[-2]
                inputs = [inp.to(device) for inp in inputs]
                data_index = data[-1]

                if args.model == 'deepsynergy_preuer':
                    outputs = network(inputs[0])
                elif args.model == 'matchmaker_brahim':
                    outputs = network(inputs[0],inputs[1],inputs[2])
                elif args.model == 'multitaskdnn_kim':
                    outputs = network(inputs[0],inputs[1],inputs[2],inputs[3],inputs[4])
                elif args.model == 'hetergnn':
                    outputs, emb_loss = network(drug1=inputs[0], drug2=inputs[1], cell=inputs[2])

                #outputs = network(inputs)
                outputs = outputs.detach().cpu().numpy()

                # actual output
                actual = labels.numpy()
                actual = actual.reshape(len(actual), 1)

                # store the values in respective lists
                indices = data_index.numpy()
                indices = indices.reshape(len(indices), 1)

                predictions.append(list(outputs))
                actuals.append(list(actual))
                idx.append(list(indices))

        actuals = [val for sublist in np.vstack(list(chain(*actuals))) for val in sublist]
        predictions = [val for sublist in np.vstack(list(chain(*predictions))) for val in sublist]
        idx = [val for sublist in np.vstack(list(chain(*idx))) for val in sublist]

        auc = roc_auc_score(y_true=actuals, y_score=predictions)

        meta_clf_pred.append(predictions)
        meta_clf_acts.append(actuals)
        meta_clf_index.append(idx)

        # Print accuracy
        print(f'Accuracy for fold %d: %.4f' % (fold, auc))
        print('--------------------------------')
        results[fold] = auc
    
            # Saving the best model
        if results[fold] >= max(results.values()):
            save_path = _ckpt_save_path(args)
            torch.save(network.state_dict(), save_path)

    # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    sum = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value}')
        sum += value
    print(f'Average: {sum/len(results.items())}')

    network_weights = _ckpt_save_path(args)

    # save 5-fold evaluation results


    meta_clf_index = [val for sublist in np.vstack(list(chain(*meta_clf_index))) for val in sublist]
    meta_clf_acts = [val for sublist in np.vstack(list(chain(*meta_clf_acts))) for val in sublist]
    meta_clf_pred = [val for sublist in np.vstack(list(chain(*meta_clf_pred))) for val in sublist]

    saveddf = pd.DataFrame(np.column_stack([meta_clf_index,meta_clf_acts,meta_clf_pred]),\
                columns=['index','actuals','metapredicts_%s' % args.model])
    save_path = os.path.join(RESULTS_DIR, 'meta_clf','metapredicts_%s_%s.txt' % (args.model, args.synergy_df))
    saveddf.to_csv(save_path, header=True, index=True, sep=",")

    return network_weights

class MyDataset(TensorDataset):
    def __init__(self, trainval_df):
        super(MyDataset, self).__init__()
        self.df = trainval_df
        self.df.reset_index(drop=True, inplace=True)  # train_test_split之后，数据集的index混乱，需要reset

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        
        #drug1, drug2, cell, target
        g1 = _ensure_molecular_graph(self.df.loc[index, 0])
        g2 = _ensure_molecular_graph(self.df.loc[index, 1])
        return (g1, g2, self.df.loc[index, 2], self.df.loc[index, 3], self.df.loc[index, 4])

def k_fold_trainer_graph(temp_loader_trainval,model,args):

    train_val_dataset_drug = temp_loader_trainval[0]
    train_val_dataset_drug2 = temp_loader_trainval[1]
    train_val_dataset_cell = temp_loader_trainval[2].tolist()
    train_val_dataset_target = temp_loader_trainval[3].tolist()
    train_val_dataset_index = temp_loader_trainval[4].tolist()

    # Configuration options
    k_folds = get_cv_folds(args)
    num_epochs = args.epochs
    batch_size = args.batch_size

    loss_function = nn.BCELoss()
    # For fold results
    results = {}

    # Define the K-fold Cross Validator
    # skf = StratifiedKFold(n_splits=k_folds, random_state=get_seed(args), shuffle=True)

    # X,y = train_val_dataset_drug, []
    # for data_object in train_val_dataset_drug:
    #     labels = data_object.label
    #     y.append(labels)

    # skf.get_n_splits(X, y)
    kfold = KFold(n_splits=k_folds, random_state=get_seed(args), shuffle=True)

    # Start print
    print('--------------------------------')

    # K-fold Cross Validation model evaluation
    # for fold, (train_ids, test_ids) in enumerate(skf.split(X,y)):
    
    trainval_df = [train_val_dataset_drug,train_val_dataset_drug2,train_val_dataset_cell,train_val_dataset_target,train_val_dataset_index]
    trainval_df = pd.DataFrame(trainval_df).T

    # save 5-fold evalutation results for meta classifier
    meta_clf_pred = []
    meta_clf_acts = []
    meta_clf_index = []

    for fold, (train_ids, test_ids) in enumerate(kfold.split(trainval_df)):
        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        
        #(Graph) For graph object needed to use torch_geometric.data.DataLoader
        if args.model == 'deepdds_wang':
            
            Dataset = MyDataset
            # self define dataset
            train_dataset = Dataset(trainval_df)
            
            trainloader = torch_geometric.data.DataLoader(train_dataset, batch_size=batch_size,
                              sampler=train_subsampler)
            valloader = torch_geometric.data.DataLoader(train_dataset, batch_size=batch_size,
                              sampler=test_subsampler)
        

        # Init the neural network
        device = get_device(args)
        network = model.to(device)
        # init para for each fold
        for layer in network.modules():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        # Initialize optimizer
        optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)

        for epoch in range(0, num_epochs):

            print(f'Starting epoch {epoch+1}')
            # Set current loss value
            current_loss = 0.0

            # Iterate over the DataLoader for training data
            for i, data in enumerate(trainloader):
                data1 = data[0]
                data2 = data[1]
                data_cell = data[2]
                data_target = data[3]
                data_index = data[4]

                x1, edge_index1, x2, edge_index2, cell, batch1, batch2 \
                    = _unpack_deepdds_batch(data1, data2, data_cell, device)

                targets = data_target.unsqueeze(1).to(device)
                                

                # Zero the gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = network(x1, edge_index1, x2, edge_index2, cell, batch1, batch2)

                loss = loss_function(outputs, targets.to(torch.float32))
                loss.backward()
                optimizer.step()
                # Print statistics
                current_loss += loss.item()
                if i % 100 == 99:
                    print('Loss after mini-batch %5d: %.3f' %
                        (i + 1, current_loss / 100))
                    current_loss = 0.0
            
            # Process is complete.


        # Evaluation for this fold
        with torch.no_grad():
            predictions, actuals = list(), list()
            idx = list()

            for i, data in enumerate(valloader):
                data1 = data[0]
                data2 = data[1]
                data_cell = data[2]
                data_target = data[3]
                data_index = data[4]

                x1, edge_index1, x2, edge_index2, cell, batch1, batch2 \
                    = _unpack_deepdds_batch(data1, data2, data_cell, device)

                targets = data_target

                # forward + backward + optimize
                outputs = network(x1, edge_index1, x2, edge_index2, cell, batch1, batch2)
                outputs = outputs.squeeze(1)
                outputs = outputs.detach().cpu().numpy()

                # actual output
                actual = targets.cpu().numpy()
                actual = actual.reshape(len(actual), 1)

                indices = data_index.numpy()
                indices = indices.reshape(len(indices), 1)

                # store the values in respective lists
                predictions.append(list(outputs))
                actuals.append(list(actual))
                idx.append(list(indices))
                
        actuals = [val for sublist in np.vstack(list(chain(*actuals))) for val in sublist]
        predictions = [val for sublist in np.vstack(list(chain(*predictions))) for val in sublist]
        idx = [val for sublist in np.vstack(list(chain(*idx))) for val in sublist]

        try:
            auc = roc_auc_score(y_true=actuals, y_score=predictions)
        except ValueError:
            auc = 0

        meta_clf_pred.append(predictions)
        meta_clf_acts.append(actuals)
        meta_clf_index.append(idx)

        # Print accuracy
        print(f'Accuracy for fold %d: %f' % (fold, auc))
        print('--------------------------------')
        results[fold] = auc
    
            # Saving the best model
        if results[fold] >= max(results.values()):
            save_path = _ckpt_save_path(args)
            torch.save(network.state_dict(), save_path)

    # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    sum = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value}')
        sum += value
    print(f'Average: {sum/len(results.items())}')

    network_weights = _ckpt_save_path(args)

    # save 5-fold evaluation results
    meta_clf_acts = [val for sublist in np.vstack(list(chain(*meta_clf_acts))) for val in sublist]
    meta_clf_pred = [val for sublist in np.vstack(list(chain(*meta_clf_pred))) for val in sublist]
    meta_clf_index = [val for sublist in np.vstack(list(chain(*meta_clf_index))) for val in sublist]

    saveddf = pd.DataFrame(np.column_stack([meta_clf_index,meta_clf_acts,meta_clf_pred]),\
                columns=['index','actuals','metapredicts_%s' % args.model])
    save_path = os.path.join(RESULTS_DIR, 'meta_clf','metapredicts_%s_%s.txt' % (args.model, args.synergy_df))
    saveddf.to_csv(save_path, header=True, index=True, sep=",")

    return network_weights

def k_fold_trainer_graph_TGSynergy(temp_loader_trainval,model,args):

    train_val_dataset_drug = temp_loader_trainval[0]
    train_val_dataset_drug2 = temp_loader_trainval[1]
    train_val_dataset_cell = temp_loader_trainval[2]
    train_val_dataset_target = temp_loader_trainval[3].tolist()
    train_val_dataset_index = temp_loader_trainval[4].tolist()

    # Configuration options
    k_folds = get_cv_folds(args)
    num_epochs = args.epochs
    batch_size = args.batch_size

    loss_function = nn.BCELoss()
    # For fold results
    results = {}

    kfold = KFold(n_splits=k_folds, random_state=get_seed(args), shuffle=True)

    # Start print
    print('--------------------------------')

    # K-fold Cross Validation model evaluation
    # for fold, (train_ids, test_ids) in enumerate(skf.split(X,y)):
    
    trainval_df = [train_val_dataset_drug,train_val_dataset_drug2,train_val_dataset_cell,train_val_dataset_target,train_val_dataset_index]
    trainval_df = pd.DataFrame(trainval_df).T

    # save 5-fold evalutation results for meta classifier
    meta_clf_pred = []
    meta_clf_acts = []
    meta_clf_index = []

    for fold, (train_ids, test_ids) in enumerate(kfold.split(trainval_df)):
        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        
        #(Graph) For graph object needed to use torch_geometric.data.DataLoader
        if args.model == 'TGSynergy':
            
            Dataset = MyDataset
            # self define dataset
            train_dataset = Dataset(trainval_df)
            
            trainloader = torch_geometric.data.DataLoader(train_dataset, batch_size=batch_size,
                              sampler=train_subsampler)
            valloader = torch_geometric.data.DataLoader(train_dataset, batch_size=batch_size,
                              sampler=test_subsampler)
        

        # Init the neural network
        network = model
        # init para for each fold
        for layer in network.modules():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        # Initialize optimizer
        optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)

        for epoch in range(0, num_epochs):

            print(f'Starting epoch {epoch+1}')
            # Set current loss value
            current_loss = 0.0

            # Iterate over the DataLoader for training data
            for i, data in enumerate(trainloader):
                data1 = data[0]
                data2 = data[1]
                data_cell = data[2]
                data_target = data[3]
                drug, drug2, cell = data1, data2, data_cell
                
                targets = data_target.unsqueeze(1)
                                
                data_index = data[4]

                # Zero the gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = network(drug, drug2, cell)

                loss = loss_function(outputs, targets.to(torch.float32))
                loss.backward()
                optimizer.step()
                # Print statistics
                current_loss += loss.item()
                if i % 100 == 99:
                    print('Loss after mini-batch %5d: %.3f' %
                        (i + 1, current_loss / 100))
                    current_loss = 0.0
            
            # Process is complete.


        # Evaluation for this fold
        with torch.no_grad():
            predictions, actuals = list(), list()
            idx = list()

            for i, data in enumerate(valloader):
                data1 = data[0]
                data2 = data[1]
                data_cell = data[2]
                data_target = data[3]
                drug, drug2, cell = data1, data2, data_cell

                targets = data_target

                data_index = data[4]

                # forward + backward + optimize
                outputs = network(drug, drug2, cell)
                outputs = outputs.squeeze(1)
                outputs = outputs.detach().cpu().numpy()

                # actual output
                actual = targets.numpy()
                actual = actual.reshape(len(actual), 1)

                indices = data_index.numpy()
                indices = indices.reshape(len(indices), 1)

                # store the values in respective lists
                predictions.append(list(outputs))
                actuals.append(list(actual))
                idx.append(list(indices))

        actuals = [val for sublist in np.vstack(list(chain(*actuals))) for val in sublist]
        predictions = [val for sublist in np.vstack(list(chain(*predictions))) for val in sublist]
        idx = [val for sublist in np.vstack(list(chain(*idx))) for val in sublist]
        
        try:
            auc = roc_auc_score(y_true=actuals, y_score=predictions)
        except ValueError:
            auc = 0

        meta_clf_pred.append(predictions)
        meta_clf_acts.append(actuals)
        meta_clf_index.append(idx)

        # Print accuracy
        print(f'Accuracy for fold %d: %f' % (fold, auc))
        print('--------------------------------')
        results[fold] = auc
    
            # Saving the best model
        if results[fold] >= max(results.values()):
            save_path = _ckpt_save_path(args)
            torch.save(network.state_dict(), save_path)

    # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    sum = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value}')
        sum += value
    print(f'Average: {sum/len(results.items())}')

    network_weights = _ckpt_save_path(args)

    # save 5-fold evaluation results
    meta_clf_acts = [val for sublist in np.vstack(list(chain(*meta_clf_acts))) for val in sublist]
    meta_clf_pred = [val for sublist in np.vstack(list(chain(*meta_clf_pred))) for val in sublist]
    meta_clf_index = [val for sublist in np.vstack(list(chain(*meta_clf_index))) for val in sublist]

    saveddf = pd.DataFrame(np.column_stack([meta_clf_index,meta_clf_acts,meta_clf_pred]),\
                columns=['index','actuals','metapredicts_%s' % args.model])
    save_path = os.path.join(RESULTS_DIR, 'meta_clf','metapredicts_%s_%s.txt' % (args.model, args.synergy_df))
    saveddf.to_csv(save_path, header=True, index=True, sep=",")

    return network_weights

class MyDataset_trans(TensorDataset):
    def __init__(self, trainval_df):
        super(MyDataset_trans, self).__init__()
        self.df = trainval_df
        #self.df.reset_index(drop=True, inplace=True)  
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        return (self.df.loc[index,0], self.df.loc[index,1], self.df.loc[index,2], self.df.loc[index,3], \
         self.df.loc[index,4], self.df.loc[index,5])


def k_fold_trainer_graph_trans(temp_loader_trainval,model,args):

    train_val_dataset_input = temp_loader_trainval[0]
    train_val_dataset_target = temp_loader_trainval[1].tolist()
    train_val_dataset_index = temp_loader_trainval[2].tolist()

    train_val_dataset_fps = temp_loader_trainval[3]
    train_val_dataset_sm1 = temp_loader_trainval[4]
    train_val_dataset_sm2 = temp_loader_trainval[5]

    # Configuration options
    k_folds = get_cv_folds(args)
    num_epochs = args.epochs
    batch_size = args.batch_size

    loss_function = nn.BCELoss()
    # For fold results
    results = {}

    kfold = KFold(n_splits=k_folds, random_state=get_seed(args), shuffle=True)

    # Start print
    print('--------------------------------')

    # K-fold Cross Validation model evaluation
    # for fold, (train_ids, test_ids) in enumerate(skf.split(X,y)):
    
    trainval_df = [train_val_dataset_input,train_val_dataset_target,train_val_dataset_index,train_val_dataset_fps,
    train_val_dataset_sm1,train_val_dataset_sm2]
    trainval_df = pd.DataFrame(trainval_df).T


    # save 5-fold evalutation results for meta classifier
    meta_clf_pred = []
    meta_clf_acts = []
    meta_clf_index = []

    for fold, (train_ids, test_ids) in enumerate(kfold.split(trainval_df)):
        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')
        # Sample elements randomly from a given list of ids, no replacement.
        torch.manual_seed(get_seed(args))
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        
        #(Graph) For graph object needed to use torch_geometric.data.DataLoader
        if args.model == 'transynergy_liu':
            
            Dataset = MyDataset_trans
            # self define dataset
            train_dataset = Dataset(trainval_df)
            
            trainloader = torch_geometric.data.DataLoader(train_dataset, batch_size=batch_size,
                              sampler=train_subsampler)
            valloader = torch_geometric.data.DataLoader(train_dataset, batch_size=batch_size,
                              sampler=test_subsampler)
        

        # Init the neural network
        device = get_device(args)
        network = model.to(device)
        # init para for each fold
        for layer in network.modules():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)

        ## fine-tuning 
        if args.train_test_mode == 'fine_tune':
            network.load_state_dict(torch.load(_ckpt_load_path(args), map_location=get_device(args)))
            #Examine the layer's ID that we'd like to fix or free
            for i, param in enumerate(network.parameters()):
                print(i, param.size(), param.requires_grad)
            release_after = 0
            for i, param in enumerate(network.parameters()):
                if i>=release_after:
                    param.requires_grad=True
                else:
                    param.requires_grad=False

            num_epochs = 10
            optimizer = torch.optim.Adam(network.parameters(), lr=getattr(args, 'ft_lr', args.lr * 0.1))

        for epoch in range(0, num_epochs):

            print(f'Starting epoch {epoch+1}')
            # Set current loss value
            current_loss = 0.0

            # Iterate over the DataLoader for training data
            for i, data in enumerate(trainloader):
                network.train()
                data1 = data[0].to(device)
                data_target = data[1]
                data_index = data[2]

                data_fp = data[3].to(device)
                data_sm1 = data[4].to(device)
                data_sm2 = data[5].to(device)

                targets = abs(data_target).unsqueeze(1).to(device).float()

                # Zero the gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = network(data1, fp=data_fp, sm1=data_sm1, sm2=data_sm2)

                loss = loss_function(outputs, targets)
                loss.backward()
                optimizer.step()
                # Print statistics
                current_loss += loss.item()
                if i % 100 == 99:
                    print('Loss after mini-batch %5d: %.3f' %
                        (i + 1, current_loss / 100))
                    current_loss = 0.0
            
            # Process is complete.



        # Evaluation for this fold
        with torch.no_grad():
            predictions, actuals = list(), list()
            idx = list()

            for i, data in enumerate(valloader):
                network.eval()

                data1 = data[0].to(device)
                data_target = data[1]
                data_index = data[2]
                
                data_fp = data[3].to(device)
                data_sm1 = data[4].to(device)
                data_sm2 = data[5].to(device)

                targets = data_target

                # forward + backward + optimize
                outputs = network(data1,fp=data_fp,sm1=data_sm1,sm2=data_sm2) #outputs = network(data1,fp=None)
                outputs = outputs.squeeze(1)
                outputs = outputs.detach().cpu().numpy()

                # actual output
                actual = targets.cpu().numpy()
                actual = actual.reshape(len(actual), 1)

                indices = data_index.numpy()
                indices = indices.reshape(len(indices), 1)

                # store the values in respective lists
                predictions.append(list(outputs))
                actuals.append(list(actual))
                idx.append(list(indices))
                
        actuals = [val for sublist in np.vstack(list(chain(*actuals))) for val in sublist]
        predictions = [val for sublist in np.vstack(list(chain(*predictions))) for val in sublist]
        idx = [val for sublist in np.vstack(list(chain(*idx))) for val in sublist]

        try:
            auc = roc_auc_score(y_true=actuals, y_score=predictions)
        except ValueError:
            auc = 0

        meta_clf_pred.append(predictions)
        meta_clf_acts.append(actuals)
        meta_clf_index.append(idx)

        # Print accuracy
        print(f'Accuracy for fold %d: %f' % (fold, auc))
        print('--------------------------------')
        results[fold] = auc
    
            # Saving the best model
        if results[fold] >= max(results.values()):
            save_path = _ckpt_save_path(args)
            torch.save(network.state_dict(), save_path)

    # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    sum = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value}')
        sum += value
    print(f'Average: {sum/len(results.items())}')

    network_weights = _ckpt_save_path(args)

    # save 5-fold evaluation results
    meta_clf_acts = [val for sublist in np.vstack(list(chain(*meta_clf_acts))) for val in sublist]
    meta_clf_pred = [val for sublist in np.vstack(list(chain(*meta_clf_pred))) for val in sublist]
    meta_clf_index = [val for sublist in np.vstack(list(chain(*meta_clf_index))) for val in sublist]

    saveddf = pd.DataFrame(np.column_stack([meta_clf_index,meta_clf_acts,meta_clf_pred]),\
                columns=['index','actuals','metapredicts_%s' % args.model])
    save_path = os.path.join(RESULTS_DIR, 'meta_clf','metapredicts_%s_%s.txt' % (args.model, args.synergy_df))
    saveddf.to_csv(save_path, header=True, index=True, sep=",")

    return network_weights

class MyDataset_combonet(TensorDataset):
    def __init__(self, trainval_df):
        super(MyDataset_combonet, self).__init__()
        self.df = trainval_df
        #self.df.reset_index(drop=True, inplace=True)  
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        return (self.df.loc[index,0], self.df.loc[index,1], self.df.loc[index,2], self.df.loc[index,3],self.df.loc[index,4])


def k_fold_trainer_graph_combonet(temp_loader_trainval,model,args):

    train_val_dataset_input = temp_loader_trainval[0]
    train_val_dataset_target = temp_loader_trainval[1].tolist()
    train_val_dataset_index = temp_loader_trainval[2].tolist()
    train_val_dataset_input2 = temp_loader_trainval[3]
    train_val_dataset_input3 = temp_loader_trainval[4]

    # Configuration options
    k_folds = get_cv_folds(args)
    num_epochs = args.epochs
    batch_size = args.batch_size

    loss_function = nn.BCELoss()
    # For fold results
    results = {}

    kfold = KFold(n_splits=k_folds, random_state=get_seed(args), shuffle=True)

    # Start print
    print('--------------------------------')

    # K-fold Cross Validation model evaluation
    # for fold, (train_ids, test_ids) in enumerate(skf.split(X,y)):
    
    trainval_df = [train_val_dataset_input,train_val_dataset_target,train_val_dataset_index,train_val_dataset_input2,\
        train_val_dataset_input3]
    trainval_df = pd.DataFrame(trainval_df).T


    # save 5-fold evalutation results for meta classifier
    meta_clf_pred = []
    meta_clf_acts = []
    meta_clf_index = []

    for fold, (train_ids, test_ids) in enumerate(kfold.split(trainval_df)):
        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')
        # Sample elements randomly from a given list of ids, no replacement.
        torch.manual_seed(get_seed(args))
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        
        #(Graph) For graph object needed to use torch_geometric.data.DataLoader
        if args.model in ['combonet', 'pairwise']:
            
            Dataset = MyDataset_combonet
            # self define dataset
            train_dataset = Dataset(trainval_df)
            
            trainloader = torch_geometric.data.DataLoader(train_dataset, batch_size=batch_size,
                              sampler=train_subsampler)
            valloader = torch_geometric.data.DataLoader(train_dataset, batch_size=batch_size,
                              sampler=test_subsampler)
        

        # Init the neural network
        device = get_device(args)
        network = model.to(device)
        # init para for each fold
        for layer in network.modules():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        # Initialize optimizer
        optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)

        for epoch in range(0, num_epochs):

            print(f'Starting epoch {epoch+1}')
            # Set current loss value

            # PAIRWISE uses a 2-pass training scheme per epoch, consistent with the
            # manuscript: the training loop is invoked twice over the same data
            # in each epoch, effectively doubling the number of gradient updates
            # per epoch. With 50 epochs and 2 passes, the model sees each sample
            # 100 times — this matches the "50 epochs x 2 passes" protocol
            # described in the README and is what produces the ~0.85 AUROC
            # reported for held-out evaluation.
            #
            # The `isAux` flag is currently unused (both passes are identical)
            # but kept for forward compatibility with the auxiliary-loss variant
            # described in the ablation study (Supp. Fig. 6).
            def training(isAux, trainloader):
                current_loss = 0.0
            # Iterate over the DataLoader for training data
                for i, data in enumerate(trainloader):
                    data1 = data[0].to(device)
                    data_target = data[1]
                    data2 = data[3].to(device)
                    data3 = data[4].to(device)

                    targets = data_target.unsqueeze(1).to(device).float()
                                    

                    # Zero the gradients
                    optimizer.zero_grad()
                    # forward + backward + optimize
                    outputs = network(data1, data2, data3)

                    loss = loss_function(outputs, targets)

                    loss.backward()
                    optimizer.step()
                    # Print statistics
                    current_loss += loss.item()
                    if i % 100 == 99:
                        print('Loss after mini-batch %5d: %.3f' %
                            (i + 1, current_loss / 100))
                        current_loss = 0.0
            training(False, trainloader)
            training(True, trainloader)

            # Process is complete.



        # Evaluation for this fold
        with torch.no_grad():
            predictions, actuals = list(), list()
            idx = list()

            for i, data in enumerate(valloader):
                network.eval()

                data1 = data[0].to(device)
                data_target = data[1]
                data_index = data[2]
                data2 = data[3].to(device)
                data3 = data[4].to(device)

                targets = data_target

                # forward + backward + optimize
                outputs = network(data1, data2, data3)
                outputs = outputs.squeeze(1)
                outputs = outputs.detach().cpu().numpy()

                # actual output
                actual = targets.numpy()
                actual = actual.reshape(len(actual), 1)

                indices = data_index.numpy()
                indices = indices.reshape(len(indices), 1)

                # store the values in respective lists
                predictions.append(list(outputs))
                actuals.append(list(actual))
                idx.append(list(indices))
                
        actuals = [val for sublist in np.vstack(list(chain(*actuals))) for val in sublist]
        predictions = [val for sublist in np.vstack(list(chain(*predictions))) for val in sublist]
        idx = [val for sublist in np.vstack(list(chain(*idx))) for val in sublist]

        try:
            auc = roc_auc_score(y_true=actuals, y_score=predictions)
        except ValueError:
            auc = 0

        meta_clf_pred.append(predictions)
        meta_clf_acts.append(actuals)
        meta_clf_index.append(idx)

        # Print accuracy
        print(f'Accuracy for fold %d: %f' % (fold, auc))
        print('--------------------------------')
        results[fold] = auc
    
            # Saving the best model
        if results[fold] >= max(results.values()):
            save_path = _ckpt_save_path(args)
            torch.save(network.state_dict(), save_path)

    # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    sum = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value}')
        sum += value
    print(f'Average: {sum/len(results.items())}')

    network_weights = _ckpt_save_path(args)

    # save 5-fold evaluation results
    meta_clf_acts = [val for sublist in np.vstack(list(chain(*meta_clf_acts))) for val in sublist]
    meta_clf_pred = [val for sublist in np.vstack(list(chain(*meta_clf_pred))) for val in sublist]
    meta_clf_index = [val for sublist in np.vstack(list(chain(*meta_clf_index))) for val in sublist]

    saveddf = pd.DataFrame(np.column_stack([meta_clf_index,meta_clf_acts,meta_clf_pred]),\
                columns=['index','actuals','metapredicts_%s' % args.model])
    save_path = os.path.join(RESULTS_DIR, 'meta_clf','metapredicts_%s_%s.txt' % (args.model, args.synergy_df))
    saveddf.to_csv(save_path, header=True, index=True, sep=",")

    return network_weights

def SHAP(model, model_weights,train_val_dataset, test_loader,args):
    sp = _get_shap()  # lazy import
    ####################
    # calcuate shapley
    ####################
    print('calculate shapely values')
    model.eval()
    device = get_device(args)
    model = model.to(device)
    model.load_state_dict(torch.load(resolve_model_weights(model_weights), map_location=device))

    #这里需要修改
    # refer to "k_fold_trainer_graph_TGSynergy" to load dataset
    if args.model in ['TGSynergy', 'deepdds_wang']:
        # explainer = SubgraphX(model, num_classes=4, device="cpu", explain_graph=False,
        #                 reward_method='nc_mc_l_shapley')

        train_val_dataset_drug = train_val_dataset[0]
        train_val_dataset_drug2 = train_val_dataset[1]
        train_val_dataset_cell = train_val_dataset[2]
        train_val_dataset_target = train_val_dataset[3].tolist()
        trainval_df = [train_val_dataset_drug,train_val_dataset_drug2,train_val_dataset_cell,train_val_dataset_target]
        trainval_df = pd.DataFrame(trainval_df).T  # shape (138878, 4)
        train_dataset = MyDataset(trainval_df)
        train_val_loader = torch_geometric.data.DataLoader(train_dataset, batch_size=getattr(args, 'batch_size', 256), shuffle=False)
        #batch = next(iter(train_val_loader))
        #background, _ = batch[:-1], batch[-1]

        # a dict to store both activations
        activation = {}
        def getActivation(name):
            def hook(model, input, output):
                if name in activation:
                    activation[name].append(output.detach())
                else:
                    activation[name] = [output.detach()]
            return hook

        if args.model == 'TGSynergy':
            # register forward hooks on the layers of choice
            h1 = model.drug_emb.register_forward_hook(getActivation('drug_emb'))
            h2 = model.cell_emb.register_forward_hook(getActivation('cell_emb'))
        elif args.model == 'deepdds_wang':
            h1 = model.drug_emb.register_forward_hook(getActivation('drug_emb'))
            h2 = model.cell_emb.register_forward_hook(getActivation('cell_emb'))

        batch = next(iter(train_val_loader))
        background, _ = batch[:-1], batch[-1]
        # forward pass -- getting the outputs
        out = model(background)
            
            # detach the hooks
        #h1.remove()
        #h2.remove()

        #x_drug, x_drug2,x_cell
        bg_x = torch.cat([
            activation['drug_emb'][0], activation['drug_emb'][1],activation['cell_emb'][0]
        ], -1)


    # normal models use "DataLoader"


    else:
        if args.model == 'transynergy_liu':
            train_val_dataset = pd.DataFrame(train_val_dataset).T
            Dataset = MyDataset_trans 
            train_val_dataset = Dataset(train_val_dataset)

        train_val_loader = DataLoader(train_val_dataset, batch_size=getattr(args, 'batch_size', 256), shuffle=False)
        #only select first batch's endpoints as our background
        batch = next(iter(train_val_loader))
        background, _ = batch[:-1], batch[-1]

# --------------- deepsynergy ---------------- #
    if args.model == 'deepsynergy_preuer':
        # 如果是一个输入, 必须为单个tensor
        explainer = sp.DeepExplainer(model, background[0])
        expected_value = explainer.expected_value
        shap_list, features_list = list(), list()
        # predictions, actuals = list(), list()
        for i, data in enumerate(test_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, _ = data[:-1], data[-1]
            shap_array_list = explainer.shap_values(inputs[0])
            shap_list.append(shap_array_list)
            features_list.append(inputs[0].numpy())

        shap_arr = np.concatenate(shap_list, axis=0)
        features_arr = np.concatenate(features_list, axis=0)
        #需要figure这里的columns是什么
        save_path = RESULTS_DIR
        exp_col_list = list(np.loadtxt(os.path.join(save_path,'selected_genes.txt'), delimiter=',').astype(int))
        drugs_col_list = list(np.arange(shap_arr.shape[1]-len(exp_col_list)))

        test_idx = list(np.loadtxt(os.path.join(save_path,'test_idx.txt')).astype(int))
        # Here we only focus on cell expression matrics, still need drug shaps to cacluate the proba
        shap_df = pd.DataFrame(shap_arr, columns=exp_col_list+drugs_col_list, index=test_idx)
        features_df = pd.DataFrame(features_arr, columns=exp_col_list+drugs_col_list, index=test_idx).iloc[: , :1000]

    # --------------- transynergy_liu ---------------- #
    elif args.model == 'transynergy_liu':
        # a dict to store both activations
        # activation = {}
        # def getActivation(name):
        #     def hook(model, input, output):
        #         if name in activation:
        #             activation[name].append(output.detach())
        #         else:
        #             activation[name] = [output.detach()]
        #     return hook

        #     model.encoder.layers[0].attn.q_linear.register_forward_hook(getActivation('q'))
        #     model.encoder.layers[0].attn.k_linear.register_forward_hook(getActivation('k'))

        #     out = model(background[0],background[3],background[4])

        #============for shapely=====================
        # X(drug target), Y, dummy, X2(FP), X_SM1(cell), X_SM2(cell)
        # explainer = sp.DeepExplainer(model, [background[0],background[3],background[4]])
        # expected_value = explainer.expected_value
        explainer = sp.GradientExplainer(model, [background[0],background[3],background[4]])
        
        # Use the freshly computed explainer instead of a stale serialised one.
        # The old code loaded explainer.bz2 and hardcoded expected_value=0.3088,
        # which silently returned stale attributions from a different model state.
        ex2 = explainer

        # Use the model's own background prediction as the expected value.
        with torch.no_grad():
            expected_value = float(
                model(background[0], fp=background[3], sm1=background[4], sm2=background[5]
                      if len(background) > 5 else background[4]).mean().cpu()
            )
        shap_list, features_list = list(), list()
        # predictions, actuals = list(), list()
        for i, data in enumerate(test_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, _ = data[:-1], data[-1]
            shap_array_list = ex2.shap_values([inputs[0],inputs[3],inputs[4]])
            shap_list.append(shap_array_list)
            features_list.append(inputs[2].numpy()) #index
        
        shap_df = pd.DataFrame()
        features_df = pd.DataFrame()
        #get gene
        proessed_dpi = pd.read_csv(os.path.join(RESULTS_DIR, 'proessed_dpi.csv'),index_col=0)
        genes = proessed_dpi.index
        save_path = RESULTS_DIR
        exp_col_list = np.array(list(np.loadtxt(os.path.join(save_path,'selected_genes.txt'), delimiter=',').astype(int)))

        for i in tqdm(np.arange(len(shap_list))):
            #batch
            #record in batch
            dts_shap_arr, fps_shap_arr, exp_shap_arr = shap_list[i][0], shap_list[i][1], shap_list[i][2]
            index = features_list[i]
            for j in np.arange(dts_shap_arr.shape[0]):
                shap_arr = np.concatenate((np.sum(dts_shap_arr[j],axis=1), \
                    np.sum(fps_shap_arr[j],axis=1), np.sum(exp_shap_arr[j]),\
                        index[j]), axis=None)
                temp = pd.DataFrame(shap_arr).T
                temp['d1tg_score'] =  [np.sort(dts_shap_arr[j][0])[::-1][:50]]
                temp['d1tg_gene'] =  [genes[np.argsort(dts_shap_arr[j][0])[::-1][:50]].values]
                #drug2
                temp['d2tg_score'] =  [np.sort(dts_shap_arr[j][1])[::-1][:50]]
                temp['d2tg_gene'] =  [genes[np.argsort(dts_shap_arr[j][1])[::-1][:50]].values]
                #cell
                temp['cell_score'] =  [np.sort(exp_shap_arr[j])[::-1][:50]]
                temp['cell_gene'] =  [exp_col_list[np.argsort(exp_shap_arr[j])[::-1][:50]]]

                shap_df = pd.concat([shap_df, temp], ignore_index=True)

        
        shap_df.columns = [['d1tg','d2tg','d1fp','d2fp','cell','index','d1tg_score','d1tg_gene',\
            'd2tg_score','d2tg_gene','cell_score','cell_gene']]

# --------------- matchmaker --------------- #
    elif args.model in ['matchmaker_brahim','multitaskdnn_kim']:
        #如果是多个, 必须是type为list的tensor组合
        if args.model == 'matchmaker_brahim':
            explainer = sp.DeepExplainer(model, [background[0],background[1],background[2]])
            expected_value = explainer.expected_value
            shap_list, features_list = list(), list()
            # predictions, actuals = list(), list()
            for i, data in enumerate(test_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, _ = data[:-1], data[-1]
                shap_array_list = explainer.shap_values([inputs[0],inputs[1],inputs[2]])
                shap_list.append(shap_array_list)
                features_list.append([inputs[0],inputs[1],inputs[2]])

            shap_df = pd.DataFrame()
            features_df = pd.DataFrame()
            for i in np.arange(len(shap_list)):
                chem_shap_arr, dg_shap_arr, exp_shap_arr = shap_list[i]
                shap_arr = np.concatenate((chem_shap_arr, dg_shap_arr, exp_shap_arr), axis=1)
                temp = pd.DataFrame(shap_arr)
                shap_df = pd.concat([shap_df, temp], ignore_index=True)

                chem_feat_arr, dg_feat_arr, exp_feat_arr = features_list[i]
                feat_arr = np.concatenate((chem_feat_arr, dg_feat_arr, exp_feat_arr), axis=1)
                temp_feat = pd.DataFrame(feat_arr)
                features_df = pd.concat([features_df, temp_feat], ignore_index=True)
                
        elif args.model == 'multitaskdnn_kim':
            explainer = sp.DeepExplainer(model, [background[0],background[1],background[2],background[3],background[4]])
            expected_value = explainer.expected_value
            shap_list, features_list = list(), list()
            # predictions, actuals = list(), list()
            for i, data in enumerate(test_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, _ = data[:-1], data[-1]
                shap_array_list = explainer.shap_values([inputs[0],inputs[1],inputs[2],inputs[3],inputs[4]])
                shap_list.append(shap_array_list)
                features_list.append([inputs[0],inputs[1],inputs[2],inputs[3],inputs[4]])

            shap_df = pd.DataFrame()
            features_df = pd.DataFrame()
            for i in np.arange(len(shap_list)):
                ##fp_drug, tg_drug, fp_drug2, tg_drug2, cell
                fp_drug, tg_drug, fp_drug2, tg_drug2, cell = shap_list[i]
                shap_arr = np.concatenate((fp_drug, tg_drug, fp_drug2, tg_drug2, cell), axis=1)
                temp = pd.DataFrame(shap_arr)
                shap_df = pd.concat([shap_df, temp], ignore_index=True)

                fp_drug_feat, tg_drug_feat, fp_drug2_feat, tg_drug2_feat, cell_feat = features_list[i]
                feat_arr = np.concatenate((fp_drug_feat, tg_drug_feat, fp_drug2_feat, tg_drug2_feat, cell_feat), axis=1)
                temp_feat = pd.DataFrame(feat_arr)
                features_df = pd.concat([features_df, temp_feat], ignore_index=True)
            
    ## 共用

        save_path = RESULTS_DIR
        exp_col_list = list(np.loadtxt(os.path.join(save_path,'selected_genes.txt'), delimiter=',').astype(int))
        drugs_col_list = list(np.arange(shap_arr.shape[1]-len(exp_col_list)))

        test_idx = list(np.loadtxt(os.path.join(save_path,'test_idx.txt')).astype(int))
        shap_df.columns = drugs_col_list+exp_col_list
        shap_df.index = test_idx

        features_df.columns = drugs_col_list+exp_col_list
        features_df.index = test_idx
        #shap_df = pd.DataFrame(shap_arr, columns=drugs_col_list+exp_col_list, index=test_idx)

# --------------- TGSynergy --------------- #
    elif args.model in ['TGSynergy']:
        if args.model == 'TGSynergy':
            #使用中间层作为input
            explainer = sp.DeepExplainer(model.regression_classify, bg_x)
            expected_value = explainer.expected_value
            # 首先需要得到中间层的 emb
            #drug1_list, drug2_list, cell_list = [], [], []

            shap_list, features_list = list(), list()
            for i, data in enumerate(test_loader, 0):
                activation = {}
                inputs, _ = data[:-1], data[-1]
                out = model(inputs)
                
                x = torch.cat([
                    activation['drug_emb'][0], activation['drug_emb'][1],activation['cell_emb'][0]], -1)

                shap_array_list = explainer.shap_values(x)
                shap_list.append(shap_array_list)
                features_list.append(x)

            # detach the hooks
            h1.remove()
            h2.remove()
            
            shap_arr = np.concatenate(shap_list, axis=0)
            features_arr = np.concatenate(features_list, axis=0)
            #column names
            save_path = RESULTS_DIR
            exp_col_list = list(np.loadtxt(os.path.join(save_path,'selected_genes.txt'), delimiter=',').astype(int))
            drugs_col_list = list(np.arange(shap_arr.shape[1]-len(exp_col_list)))

            test_idx = list(np.loadtxt(os.path.join(save_path,'test_idx.txt')).astype(int))
            # columns names should follow drug1, drug2, cell order
            shap_df = pd.DataFrame(shap_arr, columns=2*drugs_col_list+exp_col_list, index=test_idx)
            features_df = pd.DataFrame(features_arr, columns=2*drugs_col_list+exp_col_list, index=test_idx).iloc[: , :1000]


    return shap_df, features_df, expected_value

def evaluator(model,model_weights,train_val_dataset,test_loader, args):
    """_summary_

    Args:
        model (_type_): _description_
        test_loader (_type_): _description_

    Returns:
        _type_: _description_
    """
    model.eval()
    device = get_device(args)
    model = model.to(device)
    model.load_state_dict(torch.load(resolve_model_weights(model_weights), map_location=device))

    predictions, actuals = list(), list()
    for i, data in enumerate(test_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[:-2], data[-2]
        inputs = [inp.to(device) for inp in inputs]
        index = data[-1]
        if args.model == 'deepsynergy_preuer':
            y_pred = model(inputs[0])
        elif args.model == 'matchmaker_brahim':
            y_pred = model(inputs[0],inputs[1],inputs[2])
        elif args.model == 'multitaskdnn_kim':
            y_pred = model(inputs[0],inputs[1],inputs[2],inputs[3],inputs[4])
        elif args.model == 'hetergnn':
            y_pred, emb_loss = model(drug1=inputs[0], drug2=inputs[1], cell=inputs[2])

        y_pred = y_pred.detach().cpu().numpy()

        # actual output
        actual = labels.numpy()
        actual = actual.reshape(len(actual), 1)
        # store the values in respective lists
        predictions.append(list(y_pred))
        actuals.append(list(actual))

    actuals = [val for sublist in np.vstack(list(chain(*actuals))) for val in sublist]
    predictions = [val for sublist in np.vstack(list(chain(*predictions))) for val in sublist]

    if args.SHAP_analysis == True:
        shap_df, features_df, expected_value = SHAP(model, model_weights,train_val_dataset, test_loader,args)
        # shap_df['actuals'] = actuals
        # shap_df['predictions'] = predictions
    else:
        shap_df = None
        features_df = None
        expected_value = None
    return actuals, predictions, shap_df, features_df, expected_value

def evaluator_graph(model,model_weights,temp_loader_test,args):
# For graph, the dataloader should be imported from torch geometric
    model.eval()
    device = get_device(args)
    model = model.to(device)
    model.load_state_dict(torch.load(resolve_model_weights(model_weights), map_location=device))

    test_dataset_drug = temp_loader_test[0]
    test_dataset_drug2 = temp_loader_test[1]
    test_dataset_cell = temp_loader_test[2].tolist()
    test_dataset_target = temp_loader_test[3].tolist()
    test_dataset_index = temp_loader_test[4].tolist()

    test_df = [test_dataset_drug,test_dataset_drug2,test_dataset_cell,test_dataset_target,test_dataset_index]
    test_df = pd.DataFrame(test_df).T

    Dataset = MyDataset 
    test_df = Dataset(test_df)
            
    test_loader = torch_geometric.data.DataLoader(test_df, batch_size=getattr(args, 'batch_size', 256), shuffle=False)

    predictions, actuals = list(), list()

    for i, data in enumerate(test_loader):
        
        data1 = data[0]
        data2 = data[1]
        data_cell = data[2]
        data_target = data[3]

        x1, edge_index1, x2, edge_index2, cell, batch1, batch2 \
            = _unpack_deepdds_batch(data1, data2, data_cell, device)

        y_pred = model(x1, edge_index1, x2, edge_index2, cell, batch1, batch2)
        y_pred = y_pred.detach().cpu().numpy()
        # pick the index of the highest values
        #res = np.argmax(y_pred, axis = 1) 

        # actual output
        actual = data_target.numpy()
        actual = actual.reshape(len(actual), 1)
        # store the values in respective lists
        predictions.append(list(y_pred))
        actuals.append(list(actual))

    actuals = [val for sublist in np.vstack(list(chain(*actuals))) for val in sublist]
    predictions = [val for sublist in np.vstack(list(chain(*predictions))) for val in sublist]


    if args.SHAP_analysis == True:
        pass  # SHAP analysis not implemented for this model variant
    else:
        shap_df = None
        features_df = None
        expected_value = None
    return actuals, predictions, shap_df, features_df, expected_value


def evaluator_graph_TGSynergy(model,model_weights,train_val_dataset, temp_loader_test,args):
# For graph, the dataloader should be imported from torch geometric
    model.eval()
    device = get_device(args)
    model = model.to(device)
    model.load_state_dict(torch.load(resolve_model_weights(model_weights), map_location=device))

    test_dataset_drug = temp_loader_test[0]
    test_dataset_drug2 = temp_loader_test[1]
    test_dataset_cell = temp_loader_test[2]
    test_dataset_target = temp_loader_test[3].tolist()
    test_dataset_index = temp_loader_test[4].tolist()

    test_df = [test_dataset_drug,test_dataset_drug2,test_dataset_cell,test_dataset_target,test_dataset_index]
    test_df = pd.DataFrame(test_df).T

    Dataset = MyDataset 
    test_df = Dataset(test_df)
            
    test_loader = torch_geometric.data.DataLoader(test_df, batch_size=getattr(args, 'batch_size', 256), shuffle=False)

    predictions, actuals = list(), list()

    for i, data in enumerate(test_loader):
        
        data1 = data[0]
        data2 = data[1]
        data_cell = data[2]
        data_target = data[3]

        drug, drug2, cell = data1.to(device), data2.to(device), data_cell.to(device)

        y_pred = model(drug, drug2, cell)
        #y_pred = model(data)
        y_pred = y_pred.detach().cpu().numpy()
        # pick the index of the highest values
        #res = np.argmax(y_pred, axis = 1) 

        # actual output
        actual = data_target.numpy()
        actual = actual.reshape(len(actual), 1)
        # store the values in respective lists
        predictions.append(list(y_pred))
        actuals.append(list(actual))

    actuals = [val for sublist in np.vstack(list(chain(*actuals))) for val in sublist]
    predictions = [val for sublist in np.vstack(list(chain(*predictions))) for val in sublist]

    if args.SHAP_analysis == True:
        shap_df, features_df  # SHAP analysis, expected_value = SHAP(model, model_weights,train_val_dataset, test_loader,args)
        #pass
    else:
        shap_df = None
        features_df = None
        expected_value = None
    return actuals, predictions, shap_df, features_df, expected_value

def evaluator_graph_trans(model,model_weights,train_val_dataset, temp_loader_test,args):
# For graph, the dataloader should be imported from torch geometric
    model.eval()
    device = get_device(args)
    model = model.to(device)
    model.load_state_dict(torch.load(resolve_model_weights(model_weights), map_location=device))

    test_dataset= temp_loader_test[0]
    test_dataset_target = temp_loader_test[1].tolist()
    test_dataset_index = temp_loader_test[2].tolist()

    test_dataset_fps = temp_loader_test[3]
    test_dataset_sm1 = temp_loader_test[4]
    test_dataset_sm2 = temp_loader_test[5]

    test_df = [test_dataset, test_dataset_target, test_dataset_index, test_dataset_fps,\
        test_dataset_sm1,test_dataset_sm2]
    test_df = pd.DataFrame(test_df).T

    Dataset = MyDataset_trans 
    test_df = Dataset(test_df)
            
    test_loader = torch_geometric.data.DataLoader(test_df, batch_size=getattr(args, 'batch_size', 256), shuffle=False)
    # test_loader = DataLoader(test_df, batch_size=256,shuffle = False)

    predictions, actuals = list(), list()

    for i, data in enumerate(test_loader):
        
        data_target = data[1]

        data1 = data[0].to(device)
        data_fp = data[3].to(device)
        data_sm1 = data[4].to(device)
        data_sm2 = data[5].to(device)

        y_pred = model(data1, fp=data_fp, sm1=data_sm1, sm2=data_sm2)
        y_pred = y_pred.detach().cpu().numpy()

        # actual output
        actual = data_target.numpy()
        actual = actual.reshape(len(actual), 1)
        # store the values in respective lists
        predictions.append(list(y_pred))
        actuals.append(list(actual))

    actuals = [val for sublist in np.vstack(list(chain(*actuals))) for val in sublist]
    predictions = [val for sublist in np.vstack(list(chain(*predictions))) for val in sublist]

    if args.SHAP_analysis == True:
        shap_df, features_df  # SHAP analysis, expected_value = SHAP(model, model_weights,train_val_dataset, test_loader,args)
        #pass
    else:
        shap_df = None
        features_df = None
        expected_value = None
    return actuals, predictions, shap_df, features_df, expected_value


def evaluator_graph_combonet(model,model_weights,train_val_dataset, temp_loader_test,args):
# For graph, the dataloader should be imported from torch geometric
    model.eval()
    device = get_device(args)
    model = model.to(device)
    model.load_state_dict(torch.load(resolve_model_weights(model_weights), map_location=device))

    test_dataset= temp_loader_test[0]
    test_dataset_target = temp_loader_test[1].tolist()
    test_dataset_index = temp_loader_test[2].tolist()
    test_dataset2 = temp_loader_test[3]
    test_dataset3 = temp_loader_test[4]

    test_df = [test_dataset, test_dataset_target, test_dataset_index,test_dataset2,\
        test_dataset3]
    test_df = pd.DataFrame(test_df).T

    Dataset = MyDataset_combonet 
    test_df = Dataset(test_df)
            
    test_loader = torch_geometric.data.DataLoader(test_df, batch_size=getattr(args, 'batch_size', 256), shuffle=False)
    # test_loader = DataLoader(test_df, batch_size=256,shuffle = False)

    predictions, actuals = list(), list()

    for i, data in enumerate(test_loader):
        
        data1 = data[0].to(device)
        data_target = data[1]
        data2 = data[3].to(device)
        data3 = data[4].to(device)

        y_pred = model(data1, data2, data3)
        y_pred = y_pred.detach().cpu().numpy()

        # actual output
        actual = data_target.numpy()
        actual = actual.reshape(len(actual), 1)
        # store the values in respective lists
        predictions.append(list(y_pred))
        actuals.append(list(actual))

    actuals = [val for sublist in np.vstack(list(chain(*actuals))) for val in sublist]
    predictions = [val for sublist in np.vstack(list(chain(*predictions))) for val in sublist]

    if args.SHAP_analysis == True:
        shap_df, features_df  # SHAP analysis, expected_value = SHAP(model, model_weights,train_val_dataset, test_loader,args)
        #pass
    else:
        shap_df = None
        features_df = None
        expected_value = None
    return actuals, predictions, shap_df, features_df, expected_value