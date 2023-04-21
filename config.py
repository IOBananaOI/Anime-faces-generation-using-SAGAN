import torch

class Config:
    def __init__(self):

        self.batch_size = 32
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')
        self.model_save_path = 'weights/'

        self.lr = 1e-4
        self.beta1 = 0.5
        
        #### Generator parameters

        # Size of latent vector (z)
        self.nz = 64

        self.generator_hdim = 64

        self.channels_number = 3

        self.image_size = 256

        #### Discriminator parameters

        self.discriminator_hdim = 64