import os

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.utils import save_image
from tqdm import tqdm

WEEK_DIR = os.path.dirname(os.path.abspath(__file__))      # stage-3/dcgan/, outputs go here
STAGE_DIR = os.path.dirname(WEEK_DIR)                    # stage-3/, where data/ lives
DATA_DIR = os.path.join(STAGE_DIR, "data", "animefaces")
PROGRESS_DIR = os.path.join(STAGE_DIR, "data", "dcgan-progress")   # one grid per epoch, same 64 noise vectors
GENERATOR_PATH = os.path.join(STAGE_DIR, "data", "dcgan-generator.pth")

IMAGE_SIZE = 64
NZ = 100            # length of the random vector z that G turns into a face
NGF = 64            # width of G's feature maps (channels of the last conv layer)
NDF = 64            # width of D's feature maps (channels of the first conv layer)
BATCH_SIZE = 128
EPOCHS = 25
LEARNING_RATE = 0.0002
BETA1 = 0.5         # Adam momentum; the DCGAN paper found the default 0.9 unstable for GANs
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# 把圖片讀進來、縮成 64×64、疊成 (N, 3, 64, 64) 的 tensor
def load_images():
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".jpg"))
    images = []
    for f in tqdm(files, desc="loading"):
        image = Image.open(os.path.join(DATA_DIR, f)).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        images.append(torch.from_numpy(np.array(image)).permute(2, 0, 1))   # (64, 64, 3) -> (3, 64, 64)
    return torch.stack(images)                                              # (N, 3, 64, 64) uint8

# 搬到 GPU
def to_model_range(images):
    return images.to(DEVICE).float() / 127.5 - 1    # 0~255 -> -1~1, same range as G's Tanh output


def weights_init(layer):
    # the DCGAN paper's recipe: small gaussian weights, batchnorm scale around 1
    if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(layer.weight, 0.0, 0.02)
    elif isinstance(layer, nn.BatchNorm2d):
        nn.init.normal_(layer.weight, 1.0, 0.02)
        nn.init.zeros_(layer.bias)


def build_generator():
    # G: z (100, 1, 1) -> 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 picture
    return nn.Sequential(
        nn.ConvTranspose2d(NZ, NGF * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(NGF * 8), nn.ReLU(True),  # (512, 4, 4)
        nn.ConvTranspose2d(NGF * 8, NGF * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(NGF * 4), nn.ReLU(True),  # (256, 8, 8)
        nn.ConvTranspose2d(NGF * 4, NGF * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(NGF * 2), nn.ReLU(True),  # (128, 16, 16)
        nn.ConvTranspose2d(NGF * 2, NGF, 4, 2, 1, bias=False), nn.BatchNorm2d(NGF), nn.ReLU(True),          # (64, 32, 32)
        nn.ConvTranspose2d(NGF, 3, 4, 2, 1, bias=False), nn.Tanh(),                                         # (3, 64, 64)
    )


def sample(generator, z):
    # BatchNorm must be in eval mode here, otherwise sampling would shift its running statistics
    generator.eval()
    with torch.inference_mode():
        images = (generator(z) + 1) / 2             # -1~1 -> 0~1 for saving
    generator.train()
    return images.cpu()


def main():
    images = load_images()
    print(f"{len(images)} images, {tuple(images.shape[1:])} each")

    torch.manual_seed(0)
    generator = build_generator().to(DEVICE)
    # D: 64x64 picture -> one number, the probability that it is a real photo
    discriminator = nn.Sequential(
        nn.Conv2d(3, NDF, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),                                     # (64, 32, 32)
        nn.Conv2d(NDF, NDF * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(NDF * 2), nn.LeakyReLU(0.2, True),      # (128, 16, 16)
        nn.Conv2d(NDF * 2, NDF * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(NDF * 4), nn.LeakyReLU(0.2, True),  # (256, 8, 8)
        nn.Conv2d(NDF * 4, NDF * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(NDF * 8), nn.LeakyReLU(0.2, True),  # (512, 4, 4)
        nn.Conv2d(NDF * 8, 1, 4, 1, 0, bias=False), nn.Sigmoid(), nn.Flatten(0),                             # (1,) per image
    ).to(DEVICE)
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    loss_fn = nn.BCELoss()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))

    fixed_noise = torch.randn(64, NZ, 1, 1, device=DEVICE)   # same 64 z every epoch: watch the same faces evolve
    os.makedirs(PROGRESS_DIR, exist_ok=True)
    batches = (len(images) + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"Start Training on {DEVICE}")
    for epoch in range(1, EPOCHS + 1):
        order = torch.randperm(len(images))
        for batch, start in enumerate(range(0, len(images), BATCH_SIZE), 1):
            real = to_model_range(images[order[start:start + BATCH_SIZE]])
            n = len(real)                                    # last batch is smaller than BATCH_SIZE
            ones, zeros = torch.ones(n, device=DEVICE), torch.zeros(n, device=DEVICE)

            # 1) train D: real photos should score 1, G's fakes should score 0
            discriminator.zero_grad()
            d_real = discriminator(real)
            fake = generator(torch.randn(n, NZ, 1, 1, device=DEVICE))
            d_fake = discriminator(fake.detach())             # detach: this step must not touch G
            loss_d = loss_fn(d_real, ones) + loss_fn(d_fake, zeros)
            loss_d.backward()
            optimizer_d.step()

            # 2) train G: the same fakes should fool the freshly updated D into scoring 1
            generator.zero_grad()
            d_fake_after = discriminator(fake)
            loss_g = loss_fn(d_fake_after, ones)
            loss_g.backward()
            optimizer_g.step()

            if batch % 50 == 0 or batch == batches:
                print(f"[{epoch}/{EPOCHS}][{batch}/{batches}]  Loss_D: {loss_d.item():.4f}  Loss_G: {loss_g.item():.4f}  "
                      f"D(x): {d_real.mean().item():.4f}  D(G(z)): {d_fake.mean().item():.4f} / {d_fake_after.mean().item():.4f}")

        save_image(sample(generator, fixed_noise), os.path.join(PROGRESS_DIR, f"epoch-{epoch:02d}.png"), nrow=8)

    torch.save(generator.state_dict(), GENERATOR_PATH)
    print(f"saved {EPOCHS} progress grids in {os.path.basename(PROGRESS_DIR)}/ and the generator weights; "
          f"run dcgan_generate.py to draw new faces")


if __name__ == "__main__":
    main()
