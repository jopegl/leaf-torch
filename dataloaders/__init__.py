from dataloaders.datasets import leaf
from torch.utils.data import DataLoader

def make_data_loader(args, **kwargs):

    if args.dataset == 'leaf':
        train_set = leaf.LeafSegmentation(args, split = 'train')
        val_set = leaf.LeafSegmentation(args, split = 'val')
        test_set = leaf.LeafSegmentation(args, split = 'test')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        return train_loader, val_loader, test_loader, num_class
    elif args.dataset == 'leaf-cross-validation':
        train_set = leaf.LeafSegmentation(args, split = 'train', cross_val_folder='dataset/cross-validation/cv_1')
        test_set = leaf.LeafSegmentation(args, split = 'test', cross_val_folder='dataset/cross-validation/cv_1')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        return train_loader, test_loader, num_class
    else:
        raise NotImplementedError

