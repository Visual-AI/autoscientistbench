# Minimal EasyTSAD 0.3.0.2 method dependencies; scientific operations unchanged.
import numpy as np
from types import SimpleNamespace as TSData

class BaseMethodMeta(type):
    """
    Metaclass register implemented methods automaticly. This allows the usage of runtime arguments to specify the method to run experiments.

    Attributes:
        registry (dict): Registry to store the registered methods.

    """
    registry = {}

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        if name != 'BaseMethod':
            BaseMethodMeta.registry[name] = cls
     
            
class BaseMethod(metaclass=BaseMethodMeta):
    """
    BaseMethod class. Assuming that the name of your method is A, following the steps to run your method:
    1. Create "A" folder under "Method" directory; (Recommanded & Optional)

    2. Create "config.toml" to set the parameters needed for DataPreprocess and Analysis.
    
    3. Create "A.py" and create class A which should inherite abstract class BaseMethod. 
     - Override the function "train_valid_phase" to train model A; 
     - Override the function "test_phase" to test A and generate anomaly scores; 
     - Override the function "anomaly_score" which returns the anomaly score in np.ndarray format for further evaluation;
     
     Optional: 
      if your method support training in all_in_one mode or transfer mode, OVERRIDE the function "train_valid_phase_all_in_one" to train model A; 
     Optional:
      if you want to active "plot_y_hat" option in config.toml, OVERRIDE the function "get_y_hat" to save y_hat values; 
     NOTE: 
      the function "train_valid_phase_all_in_one" receive a Dict of TSDatas INSTEAD OF one TSData instanece in function "train_valid_phase" as its args.
    
    4. Fullfill "config.toml" to set the parameters needed in your class. 

    Methods:
        anomaly_score(self) -> np.ndarray:
            Calculates the anomaly scores for the time series data. Must be implemented by subclasses.

        get_y_hat(self) -> np.ndarray:
            Returns the  values for the time series data if . Can be overridden by subclasses.

        train_valid_phase(self, tsTrain: TSData):
            Performs the training and validation phase for the given training time series data. Must be implemented by subclasses.

        train_valid_phase_all_in_one(self, tsTrains: Dict[str, TSData]):
            Performs the training and validation phase, considering multiple training time series datasets. Can be overridden by subclasses.

        test_phase(self, tsData: TSData):
            Performs the testing phase on the given time series data. Must be implemented by subclasses.
            
        param_statistic(self):
            Counting the params/Flops before training. Torchinfo is recommended if the method is implemented in Pytorch.
    """
    def anomaly_score(self) -> np.ndarray:
        """
        Calculates the anomaly scores for the time series data.

        Returns:
            np.ndarray: Array of anomaly scores.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.

        """
        raise NotImplementedError()
    
    def get_y_hat(self) -> np.ndarray:
        """
        Returns the predicted values for the time series data. Recommended to override when you employ MAE/MSE loss.

        Returns:
            np.ndarray: Array of predicted values.

        """
        pass
    
    def train_valid_phase(self, tsTrain: TSData):
        """
        Performs the training and validation phase for the given training time series data.

        Args:
            tsTrain (TSData): Training time series data.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.

        """
        raise NotImplementedError()
    
    def train_valid_phase_all_in_one(self, tsTrains):
        """
        Performs the training and validation phase, considering multiple training time series datasets.

        Args:
            tsTrains (Dict[str, TSData]): Dictionary of training time series datasets.

        """
        pass
    
    def test_phase(self, tsData: TSData):
        """
        Performs the testing phase on the given time series data.

        Args:
            tsData (TSData): Time series data for testing.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.

        """
        raise NotImplementedError()
    
    def param_statistic(self, save_file):
        pass

# __all__ = ['AE', 'AnomalyTransformer', 'AR', 'Donut', 'EncDecAD', 'FCVAE', 'LSTMADalpha', 'LSTMADbeta', 'SRCNN', 'TFAD', 'TimesNet']

# from .AE.AE import AE
# from .AnomalyTransformer.AnomalyTransformer import AnomalyTransformer
# from .AR.AR import AR
# from .Donut.Donut import Donut
# from .EncDecAD.EncDecAD import EncDecAD
# from .FCVAE.FCVAE import FCVAE
# from .LSTMADalpha.LSTMADalpha import LSTMADalpha
# from .LSTMADbeta.LSTMADbeta import LSTMADbeta
# from .SRCNN.SRCNN import SRCNN
# from .TFAD.TFAD import TFAD
# from .TimesNet.TimesNet import TimesNet
import numpy as np
import torch
import os

class EarlyStoppingTorch:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path=None, patience=7, verbose=False, delta=0.0001):
        """
        Args:
            save_path : 
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if self.save_path:
            path = os.path.join(self.save_path, 'best_network.pth')
            torch.save(model.state_dict(), path)	
        self.val_loss_min = val_loss
import torch
import torch.utils.data
import numpy as np
from types import SimpleNamespace as TSData
from typing import Tuple, Sequence, Dict

class UTSOneByOneDataset(torch.utils.data.Dataset):
    '''
    The Dateset for one by one training and testing. 
    '''
    def __init__(self, tsData: TSData, phase:str, window_size: int, horizon: int = 1) -> None:
        super().__init__()
        self.window_size = window_size
        self.horizon = horizon
        
        if phase == "train":
            self.len, = tsData.train.shape
            self.sample_num = max(self.len - self.window_size - self.horizon + 1, 0)
            X = torch.zeros((self.sample_num, self.window_size))
            Y = torch.zeros((self.sample_num, 1))
            
            for i in range(self.sample_num):
                X[i, :] = torch.from_numpy(tsData.train[i : i + self.window_size])
                Y[i, :] = torch.from_numpy(np.array(tsData.train[i + self.window_size + self.horizon - 1]))
                
        elif phase == "valid":
            self.len, = tsData.valid.shape
            self.sample_num = max(self.len - self.window_size - self.horizon + 1, 0)
            X = torch.zeros((self.sample_num, self.window_size))
            Y = torch.zeros((self.sample_num, 1))
            
            for i in range(self.sample_num):
                X[i, :] = torch.from_numpy(tsData.valid[i : i + self.window_size])
                Y[i, :] = torch.from_numpy(np.array(tsData.valid[i + self.window_size + self.horizon - 1]))
                
        elif phase == "test":
            self.len, = tsData.test.shape
            self.sample_num = max(self.len - self.window_size - self.horizon + 1, 0)
            X = torch.zeros((self.sample_num, self.window_size))
            Y = torch.zeros((self.sample_num, 1))
            
            for i in range(self.sample_num):
                X[i, :] = torch.from_numpy(tsData.test[i : i + self.window_size])
                Y[i, :] = torch.from_numpy(np.array((tsData.test[i + self.window_size + self.horizon - 1])))
                
        else:
            raise ValueError('Arg "phase" in OneByOneDataset() must be one of "train", "valid", "test"')
            
        self.samples, self.targets = X, Y
    
    def __len__(self):
        return self.sample_num
    
    def __getitem__(self, index):
        return self.samples[index, :], self.targets[index, :]
    
    
class UTSAllInOneDataset(torch.utils.data.Dataset):
    '''
    The Dateset for one by one training and testing. 
    '''
    def __init__(self, datas: Dict[str, TSData], phase:str, window_size: int, horizon: int = 1) -> None:
        super().__init__()
        self.window_size = window_size
        self.horizon = horizon
        self.sample_num = 0
        
        if phase == "train":
            for _, i in datas.items():
                ll, = i.train.shape
                self.sample_num += max(ll - self.window_size - self.horizon + 1, 0)
            
            X = torch.zeros((self.sample_num, self.window_size))
            Y = torch.zeros((self.sample_num, 1))
            
            cnt = 0
            for _, i in datas.items():
                ll, = i.train.shape
                ll = max(ll - self.window_size - self.horizon + 1, 0)
                for j in range(ll):
                    X[cnt, :] = torch.from_numpy(i.train[j: j + self.window_size])
                    Y[cnt, :] = torch.from_numpy(np.array(i.train[j + self.window_size + self.horizon - 1]))
                    cnt += 1
        
        elif phase == "valid":
            for _, i in datas.items():
                ll, = i.valid.shape
                self.sample_num += max(ll - self.window_size - self.horizon + 1, 0)
            
            X = torch.zeros((self.sample_num, self.window_size))
            Y = torch.zeros((self.sample_num, 1))
            
            cnt = 0
            for _, i in datas.items():
                ll, = i.valid.shape
                ll = max(ll - self.window_size - self.horizon + 1, 0)
                for j in range(ll):
                    X[cnt, :] = torch.from_numpy(i.valid[j: j + self.window_size])
                    Y[cnt, :] = torch.from_numpy(np.array(i.valid[j + self.window_size + self.horizon - 1]))
                    cnt += 1
        
        else:
            raise ValueError('Arg "phase" in AllInOneDataset() must be one of "train", "valid"')
        
        assert cnt == self.sample_num
        
        self.samples, self.targets = X, Y
    
    
    def __len__(self):
        return self.sample_num
    
    def __getitem__(self, index):
        return self.samples[index, :], self.targets[index, :]
        