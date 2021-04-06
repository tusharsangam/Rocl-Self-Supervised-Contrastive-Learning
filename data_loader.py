from torchvision.datasets import CIFAR10
from torchvision import transforms
import VisionDataset
import torch
from typing import Any, Callable, Optional, Tuple

class CIFAR10_Contrastive(torch.utils.data.Dataset):
    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False):
        #super(CIFAR10_Contrastive, self).__init__(root, transform=transform, target_transform=target_transform)
        self.transform = transform
        self.target_transform = target_transform 
        self.base_dataset = CIFAR10(root=root, train=train, download=download)
    def __len__(self):
        return len(self.base_dataset)
    def __getitem__(self, index: int) -> Tuple[Any, Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (originalimage, crop1, crop2, target) where target is index of the target class.
        """
        img, target = self.base_dataset.__getitem__(index)
        ori_img = img.copy()
        toTensor = transforms.ToTensor()
        ori_img = toTensor(ori_img)
        img2 = img.copy()
        if self.transform is not None:
            img2 = self.transform(img2)
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        return ori_img, img, img2, target

def get_loader(dataset="CIFAR10",  contrastive_mode=True, batch_size=256, local_rank=0):
    if dataset == "CIFAR10":
        if contrastive_mode is True:
            color_jitter_strength = 0.5
            color_jitter = transforms.ColorJitter(0.8*color_jitter_strength, 0.8*color_jitter_strength, 0.8*color_jitter_strength, 0.2*color_jitter_strength)
            rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
            rnd_gray = transforms.RandomGrayscale(p=0.2)
            transform_train = transforms.Compose([
                rnd_color_jitter,
                rnd_gray,
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(32),
                transforms.ToTensor(),
            ])
            transform_test = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(32),
                transforms.ToTensor(),
            ])
            train_dataset = CIFAR10_Contrastive(root="../cifar10_data", train=True, transform=transform_train, download=True)
            test_dataset = CIFAR10_Contrastive(root="../cifar10_data", train=False, transform=transform_test, download=True)
            
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset,
                num_replicas=2,
                rank=local_rank,
                )
            
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,num_workers=4,
                    pin_memory=False,
                    shuffle=False,
                    sampler=train_sampler,
                )

            val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100,
                    num_workers=4,
                    pin_memory=False,
                    shuffle=False,
                )
            return train_sampler, train_loader, val_loader