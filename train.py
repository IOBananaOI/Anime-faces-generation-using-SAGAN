import torch

from torchvision.transforms.functional import to_pil_image
from utils import show_image, save_image, save_model
from diff_augment import DiffAugment

def train_model(g, d, d_opt, g_opt, criterion, epochs, dataloader, cfg):
    """
    Function for model training.

        :param g: Generator model
        :param d: Discriminator model
        :param g_opt: Generator optimizer
        :param d_opt: Discriminator optimizer
        :param epochs: Number epochs
        :param dataloader: Dataloader
        :param cfg: Config class instance

    """

    test_noise = torch.randn(1, cfg.nz, 1, 1, device=cfg.device)

    results = {"G_loss": [], "D_loss": []}

    g_state = g.state_dict()
    d_state = d.state_dict()

    min_g_loss = 1e5
    min_d_loss = 1e5

    for epoch in range(1, epochs+1):
        g_epoch_loss = 0.0
        d_epoch_loss = 0.0

        print(f"====== Epoch {epoch} ======")

        for batch, (X, _) in enumerate(dataloader, 0):
            ###### Discriminator training
            policy = 'color,translation,cutout'

            d.zero_grad()

            ## Loss on real batch

            real_batch = X.to(cfg.device)
#             r_noise = torch.randn((cfg.batch_size, cfg.channels_number, cfg.image_size, cfg.image_size), device=cfg.device)

#             real_batch = real_batch + r_noise

            real_label = torch.FloatTensor(cfg.batch_size).uniform_(0.8, 1.1).to(cfg.device)

            real_discriminator_pred = d(DiffAugment(real_batch, policy=policy)).view(-1)
            
            real_errD = criterion(real_discriminator_pred, real_label)
    
            
            real_errD.backward()

            ## Loss on fake batch

            z = torch.randn(cfg.batch_size, cfg.nz, 1, 1, device=cfg.device)

            fake_batch = g(z)
            # f_noise = torch.randn((cfg.batch_size, cfg.channels_number, cfg.image_size, cfg.image_size), device=cfg.device)
            # fake_batch = fake_batch + f_noise

            fake_label = torch.FloatTensor(cfg.batch_size).uniform_(0.0, 0.1).to(cfg.device)

            fake_discriminator_pred = d(DiffAugment(fake_batch.detach(), policy=policy)).view(-1)
            

            fake_errD = criterion(fake_discriminator_pred, fake_label)
            fake_errD.backward()

            errD = real_errD + fake_errD

            d_opt.step()

            ###### Generator training

            g.zero_grad()

            fake_discriminator_pred = d(DiffAugment(fake_batch, policy=policy)).view(-1)

            errG = criterion(fake_discriminator_pred, real_label)
            errG.backward()

            g_opt.step()

            results["D_loss"].append(errD.item())
            results["G_loss"].append(errG.item())

            g_epoch_loss += errG.item()
            d_epoch_loss += errD.item()

            if batch % 50 == 0:
                print(f"Loss D: {errD.item()} ||| Loss G: {errG.item()}")
                print()

        g_epoch_loss /= batch
        d_epoch_loss /= batch

        print(f"\nAverage epoch's Generator loss: {g_epoch_loss:.4f}")
        print(f"\nAverage epoch's Discriminator loss: {d_epoch_loss:.4f}")

        test_img = to_pil_image(g(test_noise).squeeze())

        if epoch % 10 == 0: 
            show_image(test_img, epoch)
            save_image(test_img, epoch, cfg)

    g_state = g.state_dict()
    d_state = d.state_dict()
    
    save_model(g_state, d_state, epochs, d_opt, g_opt, criterion)
    print(f"Generator and Discriminator were saved with minimal losses {min_g_loss:.4f} and {min_d_loss:.4f} respectively.")

    return results
            