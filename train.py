import torch

from torchvision.transforms.functional import to_pil_image
from utils import show_image, save_image, save_model

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

    test_noise = torch.randn(cfg.batch_size, cfg.nz, 1, 1, device=cfg.device)

    results = {"G_loss": [], "D_loss": []}

    g_state = g.state_dict()
    d_state = d.state_dict()

    min_g_loss = 1e5
    min_d_loss = 1e5

    for epoch in range(1, epochs+1):
        g_epoch_loss = 0.0
        d_epoch_loss = 0.0

        print(f"====== Epoch {epoch} ======")

        for batch, (X, _) in enumerate(dataloader):
            ###### Discriminator training

            d.zero_grad()

            ## Loss on real batch

            real_batch = X.to(cfg.device)
            real_label = torch.full((cfg.batch_size,), 1., dtype=torch.float, device=cfg.device)

            real_discriminator_pred = d(real_batch).view(-1)

            real_errD = criterion(real_label, real_discriminator_pred)
            real_errD.backward()

            ## Loss on fake batch

            z = torch.randn(cfg.batch_size, cfg.nz, 1, 1, device=cfg.device)

            fake_batch = g(z)
            fake_label = torch.full((cfg.batch_size,), 0., dtype=torch.float, device=cfg.device)

            fake_discriminator_pred = d(fake_batch).view(-1)

            fake_errD = criterion(fake_label, fake_discriminator_pred)
            fake_errD.backward()

            errD = real_errD + fake_errD

            d_opt.step()

            ###### Generator training

            g.zero_grad()

            fake_discriminator_pred = d(fake_batch).view(-1)

            errG = criterion(fake_discriminator_pred, real_label)
            errG.backward()

            g_opt.step()

            results["D_loss"].append(errD.item())
            results["G_loss"].append(errG.item())

            g_epoch_loss += errG.item()
            d_epoch_loss += errD.item()

            if errG.item() < min_g_loss:
                min_g_loss = errG.item()
                g_state = g.state_dict()

            if errD.item() < min_d_loss:
                min_d_loss = errD.item() 
                d_state = g.state_dict()

            if batch % 50 == 0:
                print(f"Loss D: {errD.item()} ||| Loss G: {errG.item()}")
                print()

        g_epoch_loss /= batch
        d_epoch_loss /= batch

        print(f"\nAverage epoch's Generator loss: {g_epoch_loss:.4f}")
        print(f"\nAverage epoch's Discriminator loss: {d_epoch_loss:.4f}")

        test_img = to_pil_image(g(test_noise))
        
        show_image(test_img, epoch)
        save_image(test_img, epoch)

    
    save_model(g_state, d_state, epochs, d_opt, g_opt, criterion, min_g_loss, min_d_loss, cfg.model_save_path)
    print(f"Generator and Discriminator were saved with minimal losses {min_g_loss:.4f} and {min_d_loss:.4f} respectively.")

    return results
            