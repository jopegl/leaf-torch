import os
import csv
import numpy as np
import torch

from torch.utils.data import DataLoader
from utils.loss import SegmentationLosses
from utils.metrics import Evaluator
from torch.optim.lr_scheduler import PolynomialLR
from utils.color import C

from modeling.deeplab_seg import DeepLab
from dataloaders.datasets.multi_leaf import MultiLeafDataset


class CrossValidator:

    def __init__(self, args):

        self.args = args

        print(f"{C.CYAN}[DEBUG] Dataset: {args.dataset}{C.END}")

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        self.num_class = args.num_class

        print(
            f"{C.BOLD}{C.GREEN}✔ Number of classes:{C.END} "
            f"{C.YELLOW}{self.num_class}{C.END}"
        )

        # ===== CSV LOG =====
        os.makedirs("results", exist_ok=True)

        self.csv_path = "results/training_metrics.csv"

        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)

            header = [
                "fold", "epoch", "learning_rate",
                "train_loss", "val_loss",
                "val_miou", "val_accuracy"
            ]

            for i in range(self.num_class):
                header.append(f"iou_class_{i}")

            writer.writerow(header)

        os.makedirs("crossval_models", exist_ok=True)
        os.makedirs("all_models_no_area", exist_ok=True)

        # loss
        self.seg_loss = SegmentationLosses(
            weight=None,
            cuda=args.cuda
        ).build_loss('ce')


    def training_loop(self):

        NUM_FOLDS = 5
        device = 'cuda' if self.args.cuda else 'cpu'

        fold_results = {
            'MIoU': [],
            'IoU_per_class': []
        }

        for fold in range(1, NUM_FOLDS + 1):

            print(f"\n{C.BOLD}{C.HEADER}============================{C.END}")
            print(f"{C.BOLD}{C.HEADER}FOLD {fold}{C.END}")
            print(f"{C.BOLD}{C.HEADER}============================{C.END}")

            evaluator = Evaluator(self.num_class)

            train_set = MultiLeafDataset('train', fold, num_classes=self.num_class)
            val_set   = MultiLeafDataset('val', fold, num_classes=self.num_class)
            test_set  = MultiLeafDataset('test', fold, num_classes=self.num_class)

            print(
                f"{C.CYAN}Dataset sizes → "
                f"Train: {len(train_set)} | "
                f"Val: {len(val_set)} | "
                f"Test: {len(test_set)}{C.END}"
            )

            train_loader = DataLoader(
                train_set,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.workers
            )

            val_loader = DataLoader(
                val_set,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.workers
            )

            test_loader = DataLoader(
                test_set,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.workers
            )

            # ===== MODEL =====
            model = DeepLab(
                num_classes=self.num_class,
                backbone=self.args.backbone,
                output_stride=self.args.out_stride,
                sync_bn=self.args.sync_bn,
                freeze_bn=True
            )

            if self.args.cuda:
                model = torch.nn.DataParallel(model, self.args.gpu_ids).cuda()
            else:
                model = model.to(device)

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay
            )

            scheduler = PolynomialLR(
                optimizer,
                total_iters=self.args.epochs,
                power=0.9
            )

            best_val_miou = -1

            model_path = f"crossval_models/best_model_fold_{fold}.pth"

            # ===== EPOCHS =====
            for epoch in range(self.args.epochs):

                lr = optimizer.param_groups[0]['lr']

                print(f"\n{C.BOLD}{C.BLUE}Epoch {epoch+1}/{self.args.epochs}{C.END} | LR: {lr:.6f}")

                # ===== TRAIN =====
                model.train()
                train_loss = 0

                for batch_idx, (images, masks, filename) in enumerate(train_loader):

                    images = images.to(device)
                    masks = masks.to(device)

                    optimizer.zero_grad()

                    preds = model(images)
                    loss = self.seg_loss(preds, masks)

                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()

                train_loss /= len(train_loader)

                # ===== VALIDATION =====
                model.eval()
                val_loss = 0
                evaluator.reset()

                with torch.no_grad():

                    for images, masks, filename in val_loader:

                        images = images.to(device)
                        masks = masks.to(device)

                        preds = model(images)

                        loss = self.seg_loss(preds, masks)
                        val_loss += loss.item()

                        preds = np.argmax(preds.cpu().numpy(), axis=1)
                        masks = masks.cpu().numpy()

                        evaluator.add_batch(masks, preds)

                val_loss /= len(val_loader)

                val_miou = evaluator.Mean_Intersection_over_Union()
                val_acc = evaluator.Pixel_Accuracy()

                ious = np.nan_to_num(evaluator.IoU_per_class())

                for i, iou in enumerate(ious):
                    print(f"IoU class {i}: {iou:.4f}")

                print(
                    f"{C.GREEN}Train:{C.END} {train_loss:.4f} | "
                    f"{C.YELLOW}Val:{C.END} {val_loss:.4f} | "
                    f"{C.CYAN}mIoU:{C.END} {val_miou:.4f} | "
                    f"{C.BLUE}Acc:{C.END} {val_acc:.4f}"
                )

                # ===== CSV =====
                with open(self.csv_path, "a", newline="") as f:
                    writer = csv.writer(f)

                    row = [
                        fold,
                        epoch + 1,
                        lr,
                        train_loss,
                        val_loss,
                        val_miou,
                        val_acc
                    ]

                    for i in range(self.num_class):
                        row.append(ious[i])

                    writer.writerow(row)

                # save best
                if val_miou > best_val_miou:
                    best_val_miou = val_miou
                    torch.save(model.state_dict(), model_path)
                    print(f"{C.GREEN}✔ Best model updated{C.END}")

                scheduler.step()

            # ===== TEST =====
            print(f"\n{C.CYAN}Testing fold {fold}...{C.END}")

            model.load_state_dict(torch.load(model_path))
            model.eval()
            evaluator.reset()

            with torch.no_grad():

                for images, masks, filename in test_loader:

                    images = images.to(device)
                    masks = masks.to(device)

                    preds = model(images)

                    preds = np.argmax(preds.cpu().numpy(), axis=1)
                    masks = masks.cpu().numpy()

                    evaluator.add_batch(masks, preds)

            mIoU = evaluator.Mean_Intersection_over_Union()
            ious = np.nan_to_num(evaluator.IoU_per_class())

            print(f"{C.BOLD}{C.YELLOW}Fold {fold} mIoU:{C.END} {mIoU:.4f}")

            fold_results["MIoU"].append(mIoU)
            fold_results["IoU_per_class"].append(ious)

        # ===== FINAL =====
        print(f"\n{C.BOLD}{C.HEADER}CROSS VALIDATION RESULT{C.END}")

        for i, res in enumerate(fold_results["MIoU"]):
            print(f"Fold {i+1}: {res:.4f}")

        print(f"\nMean mIoU: {np.mean(fold_results['MIoU']):.4f}")


class Args:
    dataset = "multi_leaf"
    backbone = "xception"
    out_stride = 16

    workers = 0
    epochs = 70
    batch_size = 2

    lr = 0.0003
    weight_decay = 5e-4

    no_cuda = False
    gpu_ids = [0]
    seed = 1
    sync_bn = None
    num_class = 2


def main():

    args = Args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    trainer = CrossValidator(args)
    trainer.training_loop()


if __name__ == "__main__":
    main()