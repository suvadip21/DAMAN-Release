
import numpy as np
import json
from util.tools import ImageDataset
from networks.daman import DaManNet as Net
from networks.daman_lite import DaManLiteNet as LiteNet
from torch.utils.data import DataLoader
import torch.optim as optim

class AmoebaSegmentor:
    
    def __init__(self, paramfile='params.json'):
        self.paramfile = paramfile
        self.src_test_dataset = None
        self.tgt_test_dataset = None
        self.src_train_dataset = None
        self.tgt_train_dataset = None

    def _set_src_datasets(self):
        self.src_train_dataset  = ImageDataset(base_directory=self._data_dir, src=True, transform=True, resize=True, res_factor=self._res_factor, times_augment=10)
        self.src_test_dataset  = ImageDataset(base_directory=self._data_dir, trainset=False, src=True, transform=False, resize=True, res_factor=self._res_factor, times_augment=1)

    def _set_tgt_datasets(self):
        self.tgt_train_dataset  = ImageDataset(base_directory=self._data_dir, src=False, transform=True, resize=True, res_factor=self._res_factor, times_augment=10)
        self.tgt_test_dataset  = ImageDataset(base_directory=self._data_dir, trainset=False, src=False, transform=False, resize=True, res_factor=self._res_factor, times_augment=1)

    def build(self):
        try:
            f = open(self.paramfile)
        except OSError as e:
            print ("Parameter file not specified")
        
        with f:
            loader = json.load(f)
            
            self._data_dir = loader['data_dir']
            self._lr = float(loader['learning_rate'])
            self._num_epochs = int(loader['num_epochs'])
            self._save_intervl = int(loader['model_save_interval'])
            self._pretrained_model = loader['pretrained_model']
            self._model_savedir = loader['model_savepath']
            self._res_factor = float(loader['res_factor'])
            self._batch_sz = int(loader['batch_size'])

            self._set_src_datasets()
            self._set_tgt_datasets()

    def train(self, use_lite=True):        
        src_train_loader = DataLoader(self.src_train_dataset, batch_size=self._batch_sz)
        tar_train_loader = DataLoader(self.tgt_train_dataset, batch_size=self._batch_sz)
        if use_lite:
            net = LiteNet(savepath=self._model_savedir, save_intervl=self._save_intervl)
        else:
            net = Net(savepath=self._model_savedir, model_pretrained=None, save_intervl=self._save_intervl)
        optimizer = optim.Adam(net.parameters(), lr=self._lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        net.train_net(src_train_loader, tar_train_loader, 
                num_epochs=self._num_epochs,
                batch_sz=self._batch_sz,
                optimizer=optimizer,
                scheduler=scheduler,
                lr=self._lr,
                lambda_coral=1.)

def main():
    segmentor = AmoebaSegmentor(paramfile='params.json')
    segmentor.build()
    segmentor.train(use_lite=True)

if __name__=="__main__":
    main()


