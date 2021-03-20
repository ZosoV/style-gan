import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch


def load_data(dataroot,image_size, batch_size, workers):
    # We can use an image folder dataset the way we have it setup.
    # Create the dataset
    dataset = dset.CelebA(root=dataroot,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)

    return dataloader