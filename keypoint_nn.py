"""Models for facial keypoint detection"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms



class KeypointModel(pl.LightningModule):
    """Facial keypoint detection model"""
        
    #def __init__(self, hparams):
    def __init__(self, pretr_cnn, hparams):
        """
        Initialize your model from a given dict containing all your hparams
        Warning: Don't change the method declaration (i.e. by adding more
            arguments), otherwise it might not work on the submission server
        """
        #super(KeypointModel, self).__init__()
        super().__init__()
        self.hparams = hparams


        #self.trn_batches = []
        #self.vl_batches = []
        ########################################################################
        # TODO: Define all the layers of your CNN, the only requirements are:  #
        # 1. The network takes in a batch of images of shape (Nx1x96x96)       #
        # 2. It ends with a linear layer that represents the keypoints.        #
        # Thus, the output layer needs to have shape (Nx30),                   #
        # with 2 values representing each of the 15 keypoint (x, y) pairs      #
        #                                                                      #
        # Some layers you might consider including:                            #
        # maxpooling layers, multiple conv layers, fully-connected layers,     #
        # and other layers (such as dropout or batch normalization) to avoid   #
        # overfitting.                                                         #
        ########################################################################
        '''
        self.cnn = nn.Sequential(
                nn.Conv2d(1,10,(3,3),2),
                nn.ReLU(),
                nn.BatchNorm2d(10),
                nn.Conv2d(10,30,(3,3),2),
                nn.ReLU(),
                nn.BatchNorm2d(30),
                #nn.Conv2d(20,30,(3,3)),
                #nn.ReLU(),
                #nn.BatchNorm2d(30),
                #nn.Conv2d(30,20,(1,1)),
                #nn.ReLU(),
                #nn.BatchNorm2d(20),
                nn.Conv2d(30,10,(1,1)),
                nn.ReLU(),
                nn.BatchNorm2d(10),
                #nn.Conv2d(10,5,(1,1)),
                #nn.ReLU(),
                #nn.BatchNorm2d(5),
                nn.Flatten()
        )
        '''

        self.cnn = pretr_cnn

        
        #sample = torch.randn(3,3,224,224)


        sample = torch.randn(self.hparams['batch_size'],3,224,224)
        print(sample.size())
         

        #sample_out = self.cnn(sample)    
        #print(self.cnn(sample).size())
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(25088, 5290),
            nn.Linear(5290, 1000),
            nn.BatchNorm1d(1000),
            nn.Linear(1000,30),
            nn.Tanh()
        )
        
        self.model = nn.Sequential(
          self.cnn,
          self.classifier
        )

        print(self.model(sample).size())
        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        ########################################################################
        # TODO: Define the forward pass behavior of your model                 #
        # for an input image x, forward(x) should return the                   #
        # corresponding predicted keypoints                                    #
        ########################################################################
        
      y = self.model(x)
      return y.view(-1,15,2)
      pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        
    def training_step(self, batch, batch_idx):
      images, targets = batch


      #print(batch.device)
      #ADDED ADDITIONALLY
      images= images.to(torch.device('cuda'))
      targets = targets.to(torch.device('cuda'))
      #self.trn_batches.append((images_resized_train,targets))


      # forward pass
      out = self.forward(images)

      # loss
      #print(out.size())
      loss = F.mse_loss(out, targets)

      #ADDED ADDITIONALLY
      images.to(torch.device('cpu'))
      targets.to(torch.device('cpu'))

      # logs
      tensorboard_logs = {'loss': loss}

      return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
      images, targets = batch

      #ADDED ADDITIONALLY
      images = images.to(torch.device('cuda'))
      targets = targets.to(torch.device('cuda'))
      #self.vl_batches.append((images_,targets))

      # forward pass
      out = self.forward(images)

      # loss
      loss = F.mse_loss(out, targets)

      #ADDED ADDITIONALLY
      images.to(torch.device('cpu'))
      targets.to(torch.device('cpu'))

      tensorboard_logs = {'val_loss': loss}
      return {'val_loss': loss, 'log':tensorboard_logs}



    def training_epoch_end(self, outputs):
      #print(type(outputs))
      #print(outputs)
      #print(outputs['loss'])
      avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
      #avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

      tensorboard_logs = {'train_loss_epoch': avg_loss}

      return {'train_loss': avg_loss, 'log': tensorboard_logs}


    def validation_epoch_end(self, outputs):
      avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
      #avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

      tensorboard_logs = {'val_loss_epoch': avg_loss}

      return {'val_loss': avg_loss, 'log': tensorboard_logs}

    #def prepare_data(self):
      
      #self.train_dataset = self.hparams['train']
      #self.val_dataset = self.hparams['val']
      #self.test_dataset = self.hparams['test']
    
    '''
    def training_epoch_end(self, outputs):
      for img,targ in self.train_batches:
        img = img.to(torch.device('cpu'))
        targ = targ.to(torch.device('cpu'))
      self.trn_batches.clear()
      #return outputs
      #return {}


    def validation_epoch_end(self, outputs):
      for img,targ in self.val_batches:
        img = img.to(torch.device('cpu'))
        targ = targ.to(torch.device('cpu'))
      self.vl_batches.clear()
      #return outputs
      #return {}
    '''

    #@pl.data_loader
    #def train_dataloader(self):
    #    return DataLoader(self.train_dataset, shuffle=True, batch_size=self.hparams["batch_size"])

    #@pl.data_loader
    #def val_dataloader(self):
        #return DataLoader(self.val_dataset, batch_size=self.hparams["batch_size"])

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.model.parameters(), self.hparams["learning_rate"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience = 5)
        return [optim],[scheduler]



class DummyKeypointModel(pl.LightningModule):
    """Dummy model always predicting the keypoints of the first train sample"""
    def __init__(self):
        super().__init__()
        self.prediction = torch.tensor([[
            0.4685, -0.2319,
            -0.4253, -0.1953,
            0.2908, -0.2214,
            0.5992, -0.2214,
            -0.2685, -0.2109,
            -0.5873, -0.1900,
            0.1967, -0.3827,
            0.7656, -0.4295,
            -0.2035, -0.3758,
            -0.7389, -0.3573,
            0.0086, 0.2333,
            0.4163, 0.6620,
            -0.3521, 0.6985,
            0.0138, 0.6045,
            0.0190, 0.9076,
        ]])

    def forward(self, x):
        return self.prediction.repeat(x.size()[0], 1, 1, 1)
