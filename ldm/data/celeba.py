import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CelebAHQBase(Dataset):
    def __init__(self,
                 data_root,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5
                 ):
        self.data_root = data_root
        self.image_paths = []
        for fname in sorted(os.listdir(data_root)):
            if fname.endswith('.jpg') or fname.endswith('.png'):
                self.image_paths.append(os.path.join(data_root, fname))
                
        self._length = len(self.image_paths)

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        
    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i])
        
        if not image.mode == "RGB":
            image = image.convert("RGB")

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        return example


class CelebAHQTrain(CelebAHQBase):
    def __init__(self, **kwargs):
        super().__init__(flip_p=0.5, **kwargs)


class CelebAHQValidation(CelebAHQBase):
    def __init__(self, **kwargs):
        super().__init__(flip_p=0.0, **kwargs)