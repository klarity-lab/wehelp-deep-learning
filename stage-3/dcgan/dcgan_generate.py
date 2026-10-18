import os

import torch
from torchvision.utils import save_image

from dcgan import DEVICE, GENERATOR_PATH, NZ, WEEK_DIR, build_generator, sample

generator = build_generator().to(DEVICE)
generator.load_state_dict(torch.load(GENERATOR_PATH, map_location=DEVICE))

for i in range(1, 4):
    save_image(sample(generator, torch.randn(1, NZ, 1, 1, device=DEVICE)), os.path.join(WEEK_DIR, f"generated-{i}.png"))
print("saved generated-1.png, generated-2.png, generated-3.png")
