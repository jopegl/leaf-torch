import os
import numpy as np
from mypath import Path
from dataloaders import make_data_loader
from torch.utils.tensorboard import SummaryWriter
import argparse
import time
import torch
import torchvision

from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.metrics import Evaluator

class CrossValidator:
    def __init__(self, args):
        self.args = args
        print(f"[DEBUG] Dataset argument: '{args.dataset}'")

        # Carrega dataloaders
        self.train_loader, self.test_loader, self.num_class = make_data_loader(self.args)

        # Modelo DeepLabV3 com MobileNet backbone
        self.model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(
            num_classes=self.num_class, aux_loss=False
        )

        # Configura pesos balanceados
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset),
                                                args.dataset+'_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.num_class)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None

        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=args.lr,
                                         momentum=args.momentum,
                                         weight_decay=args.weight_decay,
                                         nesterov=args.nesterov)

        self.seg_loss = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(args.loss_type)
        self.evaluator = Evaluator(self.num_class)
        self.lr_scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                         args.epochs, len(self.train_loader))

        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, args.gpu_ids)
            self.model = self.model.cuda()

        # Resume checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError(f"=> no checkpoint found at '{args.resume}'")
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")

        # Limpa start_epoch se fine-tuning
        if args.ft:
            args.start_epoch = 0

        self.writer = SummaryWriter(log_dir=f'runs_cross_validation/{args.checkname}')

    def training_loop(self):
        best_total_loss = float('inf')
        device = 'cuda' if self.args.cuda else 'cpu'
        model = self.model
        seg_loss_fn = self.seg_loss
        optimizer = self.optimizer

        train_loader = self.train_loader
        test_loader = self.test_loader

        for epoch in range(self.args.start_epoch, self.args.epochs):
            start_time = time.time()
            print(f"\033[94m=== Epoch {epoch+1}/{self.args.epochs} ===\033[0m")
            train_seg_loss = 0.0

            model.train()
            for i, (images, masks) in enumerate(train_loader):
                if i % 100 == 0:
                    print(f"Batch {i}: images {images.shape}, masks {masks.shape}")
                images = images.to(device)
                masks = masks.to(device)

                self.lr_scheduler(optimizer, i, epoch, self.best_pred)

                optimizer.zero_grad()
                seg_pred = model(images)['out']  # DeepLab retorna dict {'out', 'aux'}
                seg_loss = seg_loss_fn(seg_pred, masks)
                seg_loss.backward()
                optimizer.step()

                train_seg_loss += seg_loss.item()
                global_step = epoch * len(train_loader) + i
                if i % 100 == 0:
                    self.writer.add_scalar("train/seg_loss_iter", seg_loss.item(), global_step)
                    self.writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], global_step)

            train_seg_loss /= len(train_loader)

            # Avaliação
            model.eval()
            test_seg_loss = 0.0
            self.evaluator.reset()
            with torch.no_grad():
                for images, masks in test_loader:
                    images = images.to(device)
                    masks = masks.to(device)
                    seg_pred = model(images)['out']
                    seg_loss = seg_loss_fn(seg_pred, masks)
                    test_seg_loss += seg_loss.item()

                    seg_pred_cpu = seg_pred.detach().cpu().numpy()
                    masks_cpu = masks.detach().cpu().numpy()
                    pred = np.argmax(seg_pred_cpu, axis=1)
                    self.evaluator.add_batch(masks_cpu, pred)

            test_seg_loss /= len(test_loader)

            # Tensorboard
            self.writer.add_scalar("epoch/train_seg_loss", train_seg_loss, epoch)
            self.writer.add_scalar("epoch/test_seg_loss", test_seg_loss, epoch)
            self.writer.add_scalar("epoch/total_loss", train_seg_loss + test_seg_loss, epoch)

            Acc = self.evaluator.Pixel_Accuracy()
            mIoU = self.evaluator.Mean_Intersection_over_Union()
            FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()

            print(f"\033[96m[Epoch Summary]\033[0m "
                  f"Train Seg Loss: \033[91m{train_seg_loss:.4f}\033[0m, "
                  f"Test Seg Loss: \033[92m{test_seg_loss:.4f}\033[0m, "
                  f"mIoU: \033[93m{mIoU:.4f}\033[0m, "
                  f"Acc: \033[93m{Acc:.4f}\033[0m")

            self.writer.add_scalar("val/Acc", Acc, epoch)
            self.writer.add_scalar("val/mIoU", mIoU, epoch)
            self.writer.add_scalar("val/fwIoU", FWIoU, epoch)

            # Checkpoint
            total_test_loss = test_seg_loss
            checkpoint = {
                'epoch': epoch+1,
                'state_dict': model.module.state_dict() if self.args.cuda else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_pred': best_total_loss
            }
            os.makedirs("all_2nd_crossv", exist_ok=True)
            torch.save(checkpoint, f"all_2nd_crossv/checkpoint_epoch_{epoch}.pth.tar")

            if total_test_loss < best_total_loss:
                best_total_loss = total_test_loss
                torch.save(checkpoint, "2nd_best_model_crossv.pth.tar")
                print("\033[92mBest model updated!\033[0m")

            print("best model saved")
            end_time = time.time()
            print('Epoch duration: ', end_time - start_time)

        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3 Training")
    parser.add_argument('--dataset', type=str, default='leaf-cross-validation',
                        choices=['pascal', 'coco', 'cityscapes', 'leaf', 'leaf-cross-validation'],
                        help='dataset name')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--test-batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.007)
    parser.add_argument('--use-balanced-weights', action='store_true', default=False)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--nesterov', action='store_true', default=False)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--gpu-ids', type=str, default='0')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--checkname', type=str, default='deeplab_mobilenet')
    parser.add_argument('--ft', action='store_true', default=False)

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        args.gpu_ids = [int(x) for x in args.gpu_ids.split(',')]

    cross_validation = CrossValidator(args)
    cross_validation.training_loop()


if __name__ == "__main__":
    main()