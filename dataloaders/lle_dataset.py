import os
import glob
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F


class LowLightDataset(torch.utils.data.Dataset):
    """
    Dataset for Low-Light Enhancement (LLE) task.

    Loads paired low-light and normal-light images from a directory structure:
        data_root/
            low_light_subdir/   (e.g., i_n/low/)
                xxx.png
            normal_subdir/      (e.g., i_n/normal/)
                xxx.png

    Both subdirectories must contain the same number of images with matching filenames.
    Images are sorted by filename to ensure pairing.

    During training: random crop (image_size x image_size) + random horizontal flip.
    During validation/test: center crop to the smaller of (width, height), then resize
    to image_size x image_size.
    """
    def __init__(self, split=None, args=None):
        super().__init__()
        self.args = args
        self.split = split

        # Recursively scan all scene subdirectories (e.g. i_0/, i_12/, ...) for
        # low/ and normal/ image files.  The actual structure is:
        #   train_data_root/
        #     i_0/low/xxx.png    i_0/normal/xxx.png
        #     i_12/low/xxx.png   i_12/normal/xxx.png
        #     ...
        low_list = []
        normal_list = []
        # low_light_subdir may contain a redundant prefix like "i_n/" when passed via CLI.
        # Strip any leading "i_n/" or "i_n\\" to get the actual subfolder name.
        _actual_sub = args.low_light_subdir.replace("i_n/", "").replace("i_n\\", "")
        for scene_dir in sorted(os.listdir(args.train_data_root)):
            scene_path = os.path.join(args.train_data_root, scene_dir)
            if not os.path.isdir(scene_path):
                continue
            low_subdir = os.path.join(scene_path, _actual_sub)
            normal_subdir = os.path.join(scene_path, _actual_sub.replace("low", "normal"))
            if not os.path.isdir(low_subdir) or not os.path.isdir(normal_subdir):
                print(f"[DEBUG] skip {scene_dir}: low={low_subdir} exists={os.path.isdir(low_subdir)}, "
                      f"normal={normal_subdir} exists={os.path.isdir(normal_subdir)}")
                continue
            low_list += sorted(glob.glob(os.path.join(low_subdir, "*.png")))
            normal_list += sorted(glob.glob(os.path.join(normal_subdir, "*.png")))

        # Also support jpg if no png found
        if len(low_list) == 0:
            for scene_dir in sorted(os.listdir(args.train_data_root)):
                scene_path = os.path.join(args.train_data_root, scene_dir)
                if not os.path.isdir(scene_path):
                    continue
                low_subdir = os.path.join(scene_path, _actual_sub)
                normal_subdir = os.path.join(scene_path, _actual_sub.replace("low", "normal"))
                if not os.path.isdir(low_subdir) or not os.path.isdir(normal_subdir):
                    continue
                low_list += sorted(glob.glob(os.path.join(low_subdir, "*.jpg")))
                normal_list += sorted(glob.glob(os.path.join(normal_subdir, "*.jpg")))

        self.low_list = low_list
        self.normal_list = normal_list

        assert len(self.low_list) == len(self.normal_list), (
            f"Low/Normal image count mismatch: "
            f"{len(self.low_list)} low images vs {len(self.normal_list)} normal images"
        )
        assert len(self.low_list) > 0, f"No images found in {args.train_data_root}"

        print(f"[LowLightDataset] Loaded {len(self.low_list)} image pairs from {args.train_data_root}")

        if split == 'train':
            self.crop_preproc = transforms.Compose([
                transforms.RandomCrop((args.image_size, args.image_size)),
                transforms.RandomHorizontalFlip(),
            ])
        elif split == 'test':
            pass  # handled in __getitem__
        else:
            raise ValueError(f"Unknown split: {split}")

    def __len__(self):
        return len(self.low_list)

    def __getitem__(self, idx):
        low_img = Image.open(self.low_list[idx]).convert('RGB')
        normal_img = Image.open(self.normal_list[idx]).convert('RGB')

        if self.split == 'train':
            # Apply identical random crop + flip to both images to keep alignment
            low_img, normal_img = self._train_augment(low_img, normal_img)
        elif self.split == 'test':
            # Center crop to the minimum dimension to keep aspect ratio, then resize
            w, h = low_img.size
            crop_size = min(w, h)
            low_img = transforms.CenterCrop(crop_size)(low_img)
            low_img = low_img.resize((self.args.image_size, self.args.image_size), Image.LANCZOS)

            w_n, h_n = normal_img.size
            crop_size_n = min(w_n, h_n)
            normal_img = transforms.CenterCrop(crop_size_n)(normal_img)
            normal_img = normal_img.resize((self.args.image_size, self.args.image_size), Image.LANCZOS)

        # Normalize to [-1, 1]
        low_tensor = F.normalize(transforms.ToTensor()(low_img), mean=[0.5], std=[0.5])
        normal_tensor = F.normalize(transforms.ToTensor()(normal_img), mean=[0.5], std=[0.5])

        return {
            "conditioning_pixel_values": low_tensor,
            "output_pixel_values": normal_tensor,
            "neg_prompt": self.args.neg_prompt,
            "prompt": [self.args.fixed_prompt],
        }

    def _train_augment(self, low_img, normal_img):
        """
        Apply identical geometric augmentation to both low and normal images.
        Uses RandomCrop + RandomHorizontalFlip with a shared random seed per call.
        """
        # Convert to tensors for aligned transforms
        low_t = transforms.ToTensor()(low_img)
        normal_t = transforms.ToTensor()(normal_img)

        # Random horizontal flip (share flip state)
        if torch.rand(1).item() > 0.5:
            low_t = torch.flip(low_t, dims=[2])
            normal_t = torch.flip(normal_t, dims=[2])

        # Random crop (crop same region from both)
        w, h = low_img.size
        crop_size = min(self.args.image_size, min(w, h))
        if crop_size < self.args.image_size:
            # Image too small; resize up first
            scale = self.args.image_size / crop_size
            new_w, new_h = int(w * scale), int(h * scale)
            low_img = low_img.resize((new_w, new_h), Image.LANCZOS)
            normal_img = normal_img.resize((new_w, new_h), Image.LANCZOS)
            low_t = transforms.ToTensor()(low_img)
            normal_t = transforms.ToTensor()(normal_img)

        w, h = low_t.shape[2], low_t.shape[1]
        crop_size = self.args.image_size

        top = torch.randint(0, h - crop_size + 1, (1,)).item()
        left = torch.randint(0, w - crop_size + 1, (1,)).item()

        low_t = low_t[:, top:top + crop_size, left:left + crop_size]
        normal_t = normal_t[:, top:top + crop_size, left:left + crop_size]

        return transforms.ToPILImage()(low_t), transforms.ToPILImage()(normal_t)
