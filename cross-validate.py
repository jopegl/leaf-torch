import os
import numpy as np
from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from torch.utils.tensorboard import SummaryWriter
from utils.metrics import Evaluator
from area_loss_fn import masked_mse_loss
import argparse

class CrossValidator:
    def __init__(self, args):
        self.args = args
        print(f"[DEBUG] Dataset argument: '{args.dataset}'")
        kwargs = {'num_workers': 0, 'pin_memory': True} 
        self.train_loader, self.test_loader, self.num_class = make_data_loader(self.args, **kwargs)
        model = DeepLab(
            num_classes=self.num_class,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=True
        )

        self.model = model

        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]
        
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset+'_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.num_class)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None

        self.optimizer = torch.optim.SGD(train_params, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
        self.seg_loss = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(args.loss_type)
        self.area_loss = masked_mse_loss

        self.evaluator = Evaluator(self.num_class)

        self.lr_scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.train_loader))
        

        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume, weights_only=False)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

        self.writer = SummaryWriter(log_dir=f'runs_cross_validation/{args.checkname}')

    
    def training_loop(self):
        best_total_loss = float('inf')

        device = 'cuda' if self.args.cuda else 'cpu'
        model = self.model
        seg_loss_fn = self.seg_loss
        area_loss_fn = self.area_loss

        optimizer = self.optimizer

        alpha = 1.0 # weight for segmentation loss
        beta = 5.0 # weight for area loss


        train_loader = self.train_loader
        test_loader = self.test_loader

        for epoch in range(self.args.start_epoch,self.args.epochs):
            print(f"\033[94m=== Epoch {epoch+1}/{self.args.epochs} ===\033[0m")
            train_seg_loss = 0.0
            train_area_loss = 0.0
            model.train()

            for i,(images, masks,area) in enumerate(train_loader):
                if i % 100 == 0:  
                    print(f"Batch {i}: images {images.shape}, masks {masks.shape}, area {area.shape}")
                images = images.to(device)
                masks = masks.to(device)
                area = area.to(device)

                self.lr_scheduler(self.optimizer, i, epoch, self.best_pred)

                optimizer.zero_grad()

                seg_pred, _, area_pred = model(images)

                seg_prob = torch.softmax(seg_pred, dim=1)
                mask_prob = seg_prob[:, 1:2, :, :] + seg_prob[:, 2:3, :, :]


                seg_loss = seg_loss_fn(seg_pred, masks)
                area_loss_value = area_loss_fn(area_pred, area, mask_prob)

                

                total_loss = area_loss_value * beta + seg_loss * alpha
                total_loss.backward()

                optimizer.step()

                train_seg_loss += seg_loss.item()
                train_area_loss += area_loss_value.item()

                global_step = epoch * len(train_loader) + i
                if i % 100 == 0:  
                    self.writer.add_scalar("train/seg_loss_iter", seg_loss.item(), global_step)
                    self.writer.add_scalar("train/area_loss_iter", area_loss_value.item(), global_step)
                    self.writer.add_scalar("train/total_loss_iter", total_loss.item(), global_step)

                # log learning rate
                current_lr = optimizer.param_groups[0]['lr']
                self.writer.add_scalar("train/lr", current_lr, global_step)
            
            train_seg_loss /= len(self.train_loader)
            train_area_loss /= len(self.train_loader)
                
            model.eval()

            test_seg_loss = 0.0
            test_area_loss = 0.0

            self.evaluator.reset()
            
            with torch.no_grad():
                for images, masks, area in test_loader:
                    images = images.to(device)
                    masks = masks.to(device)
                    area = area.to(device)
                    
                    seg_pred, _, area_pred  = model(images)

                    seg_prob = torch.softmax(seg_pred, dim=1)
                    mask_prob = seg_prob[:, 1:2, :, :] + seg_prob[:, 2:3, :, :]

                    seg_loss = seg_loss_fn(seg_pred, masks)
                    area_loss_value = area_loss_fn(area_pred, area, mask_prob)

                    test_seg_loss += seg_loss.item()
                    test_area_loss += area_loss_value.item()

                    seg_pred_cpu = seg_pred.detach().cpu().numpy()
                    masks_cpu = masks.detach().cpu().numpy()

                    pred = np.argmax(seg_pred_cpu, axis=1)

                    final_area_pred = mask_prob * area_pred
                    final_area_pred_np = final_area_pred.detach().cpu().numpy()
                    area_target_np = area.detach().cpu().numpy()

                    self.evaluator.add_batch(masks_cpu, pred)
                    self.evaluator.add_area_batch(final_area_pred, area)
                    self.evaluator.area_rer_batch(final_area_pred_np, area_target_np, masks_cpu, pred, mode='pred')     
        
                    


            
            test_seg_loss /= len(self.test_loader)
            test_area_loss /= len(self.test_loader)
            
            self.writer.add_scalar("epoch/train_seg_loss", train_seg_loss, epoch)
            self.writer.add_scalar("epoch/train_area_loss", train_area_loss, epoch)
            self.writer.add_scalar("epoch/test_seg_loss", test_seg_loss, epoch)
            self.writer.add_scalar("epoch/test_area_loss", test_area_loss, epoch)

            self.writer.add_scalar("epoch/total_train_loss",
                       train_seg_loss + train_area_loss, epoch)

            self.writer.add_scalar("epoch/total_test_loss",
                                test_seg_loss * alpha + test_area_loss * beta, epoch)

     
            Acc = self.evaluator.Pixel_Accuracy()
            mIoU = self.evaluator.Mean_Intersection_over_Union()
            FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
            rer_stats = self.evaluator.area_rer_stats()

            rer_score = (
                rer_stats['leaf_mean'] +
                rer_stats['leaf_std'] +
                rer_stats['marker_mean'] +
                rer_stats['marker_std']
            )

            print(f"\033[96m[Epoch Summary]\033[0m "
                  f"Train Seg Loss: \033[91m{train_seg_loss:.4f}\033[0m, "
                  f"Train Area Loss: \033[92m{train_area_loss:.4f}\033[0m, "
                  f"Test Seg Loss: \033[91m{test_seg_loss:.4f}\033[0m, "
                  f"Test Area Loss: \033[92m{test_area_loss:.4f}\033[0m, "
                  f"mIoU: \033[93m{mIoU:.4f}\033[0m, Acc: \033[93m{Acc:.4f}\033[0m")

            # TensorBoard
            self.writer.add_scalar("val/Acc", Acc, epoch)
            self.writer.add_scalar("val/mIoU", mIoU, epoch)
            self.writer.add_scalar("val/fwIoU", FWIoU, epoch)
            self.writer.add_scalar("val/RER_score", rer_score, epoch)

            print("Val mIoU:", mIoU)
            print("Val Acc:", Acc)



            total_test_loss = test_seg_loss * alpha + test_area_loss * beta

            checkpoint = {
                    'epoch': epoch+1,
                    'state_dict': model.module.state_dict() if self.args.cuda else model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'best_pred': best_total_loss
                }
            os.makedirs("all_1st_crossv", exist_ok=True)           
            torch.save(checkpoint, f"all_1st_crossv/checkpoint_epoch_{epoch}.pth.tar")

               
            if total_test_loss < best_total_loss:
                best_total_loss = total_test_loss
                torch.save(checkpoint, "1st_best_model_crossv.pth.tar")
                print("\033[92mBest model updated!\033[0m")  

            print("best model saved")

        self.writer.close()
        

def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes', 'leaf', 'leaf-cross-validation'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--use-sbd', action='store_true', default=True,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=513,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'cityscapes': 200,
            'pascal': 50,
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.01,
            'pascal': 0.007,
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size


    if args.checkname is None:
        args.checkname = 'deeplab-'+str(args.backbone)

    cross_validation = CrossValidator(args)
    cross_validation.training_loop()

main()

    
