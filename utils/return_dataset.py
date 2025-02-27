import os
import torch
from torchvision import transforms
from loaders.data_list import Imagelists_VISDA, return_classlist


class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))

def return_dataset(args):
    base_path = './data/%s' % args.dataset
    root = ''
    image_set_file_s = \
        os.path.join(base_path,
                     args.source + '.txt')
    image_set_file_t = \
        os.path.join(base_path,
                     args.target + '.txt' )
    image_set_file_unl = \
        os.path.join(base_path,
                     args.target + '.txt')
    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
    data_transforms = {
        'train': transforms.Compose([
            ResizeImage(args.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            ResizeImage(args.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val_test': transforms.Compose([
            ResizeImage(args.img_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            ResizeImage(args.img_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    source_dataset = Imagelists_VISDA(image_set_file_s, root=root,
                                      transform=data_transforms['train'],CLass_N=args.Class_N)
    target_dataset = Imagelists_VISDA(image_set_file_t, root=root,
                                      transform=data_transforms['train'],CLass_N=args.Class_N)
    target_dataset_test = Imagelists_VISDA(image_set_file_unl, root=root,
                                           transform=data_transforms['test'],CLass_N=args.Class_N)
    class_list = return_classlist(image_set_file_s)

    print("%d classes in this dataset" % len(class_list))
    bs = args.bs
    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=bs,
                                                num_workers=4, shuffle=True,pin_memory=True,
                                                drop_last=True)
    target_loader = \
        torch.utils.data.DataLoader(target_dataset,
                                    batch_size=min(bs, len(target_dataset)),
                                    num_workers=4,
                                    shuffle=True,pin_memory=True, drop_last=True)


    target_loader_test = \
        torch.utils.data.DataLoader(target_dataset_test,
                                    batch_size=bs , num_workers=4,
                                    shuffle=False,pin_memory=True, drop_last=True)
    return source_loader,target_loader,target_loader_test, class_list






def return_dataset_test(args):
    base_path = './data/txt/%s' % args.dataset
    root = './data/%s/' % args.dataset
    image_set_file_s = os.path.join(base_path, args.source + '_all' + '.txt')
    image_set_file_test = os.path.join(base_path,
                                       'unlabeled_target_images_' +
                                       args.target + '_%d.txt' % (args.num))
    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
    data_transforms = {
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    target_dataset_unl = Imagelists_VISDA(image_set_file_test, root=root,
                                          transform=data_transforms['test'],
                                          test=True)
    class_list = return_classlist(image_set_file_s)
    print("%d classes in this dataset" % len(class_list))

    bs=32
    target_loader_unl = \
        torch.utils.data.DataLoader(target_dataset_unl,
                                    batch_size=bs * 2, num_workers=8,
                                    shuffle=False,pin_memory=True, drop_last=False)
    return target_loader_unl, class_list








