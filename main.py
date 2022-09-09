
import numpy as np
import json
from util.tools import ImageDataset
from util.helpers_docker import StdIO as IO
from util.helpers_docker import StdIP as IP
from networks.daman import DaManNet as Net
from networks.daman_lite import DaManLiteNet as LiteNet
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from util.performance_analysis import ReconstructObjects
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

class AmoebaSegmentor:
    
    def __init__(self, paramfile='params.json'):
        self.paramfile = paramfile
        # self.src_test_dataset = None
        # self.tgt_test_dataset = None
        self.src_train_dataset = None
        self.tgt_train_dataset = None

    def set_train_datasets(self):
        self.src_train_dataset  = ImageDataset(base_directory=self._data_dir, src=True, transform=True, resize=True, res_factor=self._res_factor, times_augment=10)
        self.tgt_train_dataset  = ImageDataset(base_directory=self._data_dir, src=False, transform=True, resize=True, res_factor=self._res_factor, times_augment=10)


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
            self._test_img = loader['test_image']

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

    def inference(self, resize_to_256=True, savename='segmentation.png'):

        img = IO.imread_2d(self._test_img)
        if resize_to_256:   # The provided model was trained on 256x256 images
            img = IP.imresize(img, orig_res_mm=1., des_res_mm=2.)
        img = IP.linstretch(img)
        trained_model = Net()
        trained_model.load_state_dict(torch.load(self._pretrained_model))
        torch_img = torch.Tensor(img[np.newaxis, np.newaxis, ...]).type(torch.FloatTensor)
        fg, bg, edg, _, _ = trained_model(torch_img)
        fg = fg[0,...].data.cpu().numpy()
        bg = bg[0,...].data.cpu().numpy()
        edg = edg[0,...].data.cpu().numpy()  

        
        fg_img = IP.linstretch(fg[0])
        bg_img = IP.linstretch(bg[0])
        edg_img = IP.linstretch(edg[0])
        cells = ReconstructObjects(fg_img, bg_img, edg_img, img)

        self._writeimg(img, cells, savename=savename)



    def _writeimg(self, img, cellimg, savename='inference.png'):
        f = plt.figure(figsize=(10, 4))
        ax1, ax2= f.add_subplot(1,2,1), f.add_subplot(1,2,2)
        ax1.imshow(img, cmap='gray')
        ax2.imshow(cellimg, cmap='nipy_spectral')
        ax1.axis('off'), ax2.axis('off')
        f.savefig(savename)


def main(do_training):

    segmentor = AmoebaSegmentor(paramfile='params.json')
    segmentor.build()

    if do_training:
        segmentor.set_train_datasets()
        segmentor.train(use_lite=True)

    segmentor.inference(savename='inference.png')

if __name__=="__main__":

    main(do_training=False)


