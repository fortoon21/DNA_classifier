from torchvision import datasets, transforms
from torch.utils.data import Dataset
from base import BaseDataLoader
import pandas as pd
import torch
import numpy as np

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class sklearn_Dataset(Dataset):
    """gene dataset."""

    def __init__(self, data_dir, transform=None):

        self.data_dir = data_dir
        from sklearn.datasets import make_classification
        # X, labels = load_iris(return_X_y=True)
        # from sklearn.preprocessing import LabelEncoder
        # lb = LabelEncoder()
        # lb.fit(labels)
        # y=lb.transform(labels)

        # mean_data = np.mean(X, axis=0)
        # std_data = np.std(X, axis=0)
        # norm_X=np.zeros(X.shape)
        # for j in range(4):
        #     for i in range(X.shape[0]):
        #         norm_X[i, j] = (X[i, j] - mean_data[j])/std_data[j]
        
        pre_X, y = make_classification(n_samples=2000, n_classes=2, n_informative=3, n_redundant=0, 
                            n_features=3, random_state=123)
        X=pre_X + abs(np.min(pre_X)) + np.array([1,1,1])
        X=np.around(X*0.1, 4)

        self.X=torch.tensor(X, dtype=torch.float32)
        self.y=torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.X[idx], self.y[idx]

class sklearn_DataLoader(BaseDataLoader):
    """
    fake data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.2, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.data_dir = data_dir
        self.dataset = sklearn_Dataset(self.data_dir)
        
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class gene1Dataset(Dataset):
    """gene dataset."""

    def __init__(self, data_dir, transform=None):

        self.data_dir = data_dir
        rna_exp = pd.read_csv(data_dir+'/data.csv', index_col=[0])
        labels = pd.read_csv(data_dir+'/labels.csv', index_col=[0]) 
        rna_merged = pd.concat([rna_exp, labels], axis=1)
        exp = rna_merged.set_index('Class').sort_index()
        X = exp.reset_index().drop('Class', axis=1)[['gene_220',
            'gene_2136',
            'gene_4376',
            'gene_4618',
            'gene_4773',
            'gene_5453',
            'gene_5577',
            'gene_8891',
            'gene_9446',
            'gene_15301',
            'gene_15895',
            'gene_16829',
            'gene_16887',
            'gene_17595',
            'gene_18217',
            'gene_18906',
            'gene_19296',
            'gene_19313',
            'gene_19913']]
        y_ = exp.index
        from sklearn.preprocessing import LabelEncoder
        lb = LabelEncoder()
        lb.fit(y_)
        y=lb.transform(y_)

        self.X=torch.tensor(X.to_numpy(), dtype=torch.float32)
        self.y=torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.X[idx], self.y[idx]

class gene1DataLoader(BaseDataLoader):
    """
    gene data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.data_dir = data_dir
        self.dataset = gene1Dataset(self.data_dir)
        
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class gene2Dataset(Dataset):
    """gene dataset."""

    def __init__(self, data_dir, transform=None):

        self.data_dir = data_dir
        rna_exp = pd.read_csv(data_dir+'/data.csv', index_col=[0])
        labels = pd.read_csv(data_dir+'/labels.csv', index_col=[0]) 
        rna_merged = pd.concat([rna_exp, labels], axis=1)
        exp = rna_merged.set_index('Class').sort_index()
        exp_s = exp.loc[['BRCA', 'KIRC', 'LUAD']]
        dic={'BRCA':0, 'KIRC':1, 'LUAD':2} 
        X = exp_s.reset_index().drop('Class', axis=1)[['gene_220',
            'gene_2136',
            'gene_4376',
            'gene_4618',
            'gene_4773',
            'gene_5453',
            'gene_5577',
            'gene_8891',
            'gene_9446',
            'gene_15301',
            'gene_15895',
            'gene_16829',
            'gene_16887',
            'gene_17595',
            'gene_18217',
            'gene_18906',
            'gene_19296',
            'gene_19313',
            'gene_19913']]
        y_ = exp_s.index
        from sklearn.preprocessing import LabelEncoder
        lb = LabelEncoder()
        lb.fit(y_)
        y=lb.transform(y_)

        self.X=torch.tensor(X.to_numpy(), dtype=torch.float32)
        self.y=torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.X[idx], self.y[idx]

class gene2DataLoader(BaseDataLoader): 
    """
    gene data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.data_dir = data_dir
        self.dataset = gene2Dataset(self.data_dir)
        
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class gene3Dataset(Dataset):
    """gene dataset."""

    def __init__(self, data_dir, transform=None):

        self.data_dir = data_dir
        rna_exp = pd.read_csv(data_dir+'/tiny_data.tsv', sep='\t')
        labels = pd.read_csv(data_dir+'/tiny_label.tsv', sep='\t')
        X = rna_exp
        y_ = labels['cancer type']
        from sklearn.preprocessing import LabelEncoder
        lb = LabelEncoder()
        lb.fit(y_)
        y=lb.transform(y_)

        self.X=torch.tensor(X.to_numpy(), dtype=torch.float32)
        self.y=torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.X[idx], self.y[idx]

class gene3DataLoader(BaseDataLoader): 
    """
    gene data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.data_dir = data_dir
        self.dataset = gene3Dataset(self.data_dir)
        
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class gene3Dataset_10(Dataset):
    """gene dataset."""

    def __init__(self, data_dir, transform=None):

        self.data_dir = data_dir
        rna_exp = pd.read_csv(data_dir+'/tiny_data_10.tsv', sep='\t')
        labels = pd.read_csv(data_dir+'/tiny_label_10.tsv', sep='\t')
        X = rna_exp
        y_ = labels['cancer type']
        from sklearn.preprocessing import LabelEncoder
        lb = LabelEncoder()
        lb.fit(y_)
        y=lb.transform(y_)

        self.X=torch.tensor(X.to_numpy(), dtype=torch.float32)
        self.y=torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.X[idx], self.y[idx]

class gene3DataLoader_10(BaseDataLoader): 
    """
    gene data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.data_dir = data_dir
        self.dataset = gene3Dataset_10(self.data_dir)
        
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class gene3Dataset_20(Dataset):
    """gene dataset."""

    def __init__(self, data_dir, transform=None):

        self.data_dir = data_dir
        rna_exp = pd.read_csv(data_dir+'/tiny_data_20.tsv', sep='\t')
        labels = pd.read_csv(data_dir+'/tiny_label_20.tsv', sep='\t')
        X = rna_exp
        y_ = labels['cancer type']
        from sklearn.preprocessing import LabelEncoder
        lb = LabelEncoder()
        lb.fit(y_)
        y=lb.transform(y_)

        self.X=torch.tensor(X.to_numpy(), dtype=torch.float32)
        self.y=torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.X[idx], self.y[idx]

class gene3DataLoader_20(BaseDataLoader): 
    """
    gene data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.data_dir = data_dir
        self.dataset = gene3Dataset_20(self.data_dir)
        
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class gene3Dataset_30(Dataset):
    """gene dataset."""

    def __init__(self, data_dir, transform=None):

        self.data_dir = data_dir
        rna_exp = pd.read_csv(data_dir+'/tiny_data_30.tsv', sep='\t')
        labels = pd.read_csv(data_dir+'/tiny_label_30.tsv', sep='\t')
        X = rna_exp
        y_ = labels['cancer type']
        from sklearn.preprocessing import LabelEncoder
        lb = LabelEncoder()
        lb.fit(y_)
        y=lb.transform(y_)

        self.X=torch.tensor(X.to_numpy(), dtype=torch.float32)
        self.y=torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.X[idx], self.y[idx]

class gene3DataLoader_30(BaseDataLoader): 
    """
    gene data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.data_dir = data_dir
        self.dataset = gene3Dataset_30(self.data_dir)
        
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class gene3Dataset_40(Dataset):
    """gene dataset."""

    def __init__(self, data_dir, transform=None):

        self.data_dir = data_dir
        rna_exp = pd.read_csv(data_dir+'/tiny_data_40.tsv', sep='\t')
        labels = pd.read_csv(data_dir+'/tiny_label_40.tsv', sep='\t')
        X = rna_exp
        y_ = labels['cancer type']
        from sklearn.preprocessing import LabelEncoder
        lb = LabelEncoder()
        lb.fit(y_)
        y=lb.transform(y_)

        self.X=torch.tensor(X.to_numpy(), dtype=torch.float32)
        self.y=torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.X[idx], self.y[idx]

class gene3DataLoader_40(BaseDataLoader): 
    """
    gene data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.data_dir = data_dir
        self.dataset = gene3Dataset_40(self.data_dir)
        
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class gene3Dataset_50(Dataset):
    """gene dataset."""

    def __init__(self, data_dir, transform=None):

        self.data_dir = data_dir
        rna_exp = pd.read_csv(data_dir+'/tiny_data_50.tsv', sep='\t')
        labels = pd.read_csv(data_dir+'/tiny_label_50.tsv', sep='\t')
        X = rna_exp
        y_ = labels['cancer type']
        from sklearn.preprocessing import LabelEncoder
        lb = LabelEncoder()
        lb.fit(y_)
        y=lb.transform(y_)

        self.X=torch.tensor(X.to_numpy(), dtype=torch.float32)
        self.y=torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.X[idx], self.y[idx]

class gene3DataLoader_50(BaseDataLoader): 
    """
    gene data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.data_dir = data_dir
        self.dataset = gene3Dataset_50(self.data_dir)
        
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class gene3Dataset_100(Dataset):
    """gene dataset."""

    def __init__(self, data_dir, transform=None):

        self.data_dir = data_dir
        rna_exp = pd.read_csv(data_dir+'/tiny_data_100.tsv', sep='\t')
        labels = pd.read_csv(data_dir+'/tiny_label_100.tsv', sep='\t')
        X = rna_exp
        y_ = labels['cancer type']
        from sklearn.preprocessing import LabelEncoder
        lb = LabelEncoder()
        lb.fit(y_)
        y=lb.transform(y_)

        self.X=torch.tensor(X.to_numpy(), dtype=torch.float32)
        self.y=torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.X[idx], self.y[idx]

class gene3DataLoader_100(BaseDataLoader): 
    """
    gene data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.data_dir = data_dir
        self.dataset = gene3Dataset_100(self.data_dir)
        
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class gene3Dataset_150(Dataset):
    """gene dataset."""

    def __init__(self, data_dir, transform=None):

        self.data_dir = data_dir
        rna_exp = pd.read_csv(data_dir+'/tiny_data_150.tsv', sep='\t')
        labels = pd.read_csv(data_dir+'/tiny_label_150.tsv', sep='\t')
        X = rna_exp
        y_ = labels['cancer type']
        from sklearn.preprocessing import LabelEncoder
        lb = LabelEncoder()
        lb.fit(y_)
        y=lb.transform(y_)

        self.X=torch.tensor(X.to_numpy(), dtype=torch.float32)
        self.y=torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.X[idx], self.y[idx]

class gene3DataLoader_150(BaseDataLoader): 
    """
    gene data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.data_dir = data_dir
        self.dataset = gene3Dataset_150(self.data_dir)
        
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
