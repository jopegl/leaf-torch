import os
import csv
import numpy as np
import torch

from mypath import Path
from dataloaders.datasets import multi_leaf
from torch.utils.data import DataLoader

from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.metrics import Evaluator

from modeling.deeplab_seg import DeepLab

from utils.area_calc import calculate_leaf_area


# ===== Colored logs =====
class C:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    END = "\033[0m"


class CrossValidator:

    def __init__(self, args):

        self.args = args

        print(f"{C.CYAN}[DEBUG] Dataset: {args.dataset}{C.END}")

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        temp_dataset = multi_leaf.MultiLeafDataset('train', 1)
        self.num_class = temp_dataset.NUM_CLASSES

        print(
            f"{C.BOLD}{C.GREEN}✔ Number of classes detected:{C.END} "
            f"{C.YELLOW}{self.num_class}{C.END}"
        )

        # ===== CSV logging =====

        os.makedirs("results", exist_ok=True)

        self.csv_path = "results/training_metrics.csv"

        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "fold",
                "epoch",
                "train_loss",
                "val_loss",
                "val_miou",
                "val_accuracy"
            ])

        # ===== Loss =====

        self.seg_loss = SegmentationLosses(
            weight=None,
            cuda=args.cuda
        ).build_loss('ce')

        os.makedirs("crossval_models", exist_ok=True)
    
    def save_area_results_csv(self, fold, area_results):
        output_path = f"results/area_results_fold_{fold}.csv"

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "predicted_area_cm2", "target_area_cm2", "abs_error_cm2"])

            for filename, pred_area, target_area in area_results:
                abs_error = abs(pred_area - target_area)
                writer.writerow([filename, pred_area, target_area, abs_error])

        print(f"{C.GREEN}✔ Area results saved:{C.END} {output_path}")

    def training_loop(self):

        NUM_FOLDS = 5
        device = 'cuda' if self.args.cuda else 'cpu'

        fold_results = {
            'MIoU':[],
            'Area_MAE':[]
        }

        for fold in range(1, NUM_FOLDS + 1):

            print(f"\n{C.BOLD}{C.HEADER}============================{C.END}")
            print(f"{C.BOLD}{C.HEADER}STARTING FOLD {fold}{C.END}")
            print(f"{C.BOLD}{C.HEADER}============================{C.END}")

            evaluator = Evaluator(self.num_class)

            train_set = multi_leaf.MultiLeafDataset('train', fold)
            val_set   = multi_leaf.MultiLeafDataset('val', fold)
            test_set  = multi_leaf.MultiLeafDataset('test', fold)

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

            model = DeepLab(
                num_classes=self.num_class,
                backbone=self.args.backbone,
                output_stride=self.args.out_stride,
                sync_bn=self.args.sync_bn,
                freeze_bn=True
            )

            if self.args.cuda:
                model = torch.nn.DataParallel(model, self.args.gpu_ids)
                model = model.cuda()
            else:
                model = model.to(device)

            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=self.args.lr,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay,
                nesterov=self.args.nesterov
            )

            best_val_loss = float('inf')

            for epoch in range(self.args.epochs):

                print(f"\n{C.BOLD}{C.BLUE}Epoch {epoch+1}/{self.args.epochs}{C.END}")

                # ===== TRAIN =====

                model.train()
                train_loss = 0

                for batch_idx, (images, masks, orig_w, orig_h, filename, pattern_side, target_area) in enumerate(train_loader):

                    if batch_idx % 100 == 0:

                        print(
                            f"{C.YELLOW}[Batch {batch_idx}] "
                            f"Images shape: {images.shape} "
                            f"Masks shape: {masks.shape}{C.END}"
                        )

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

                    for (images, masks, orig_w, orig_h, filename, pattern_side, target_area) in val_loader:

                        images = images.to(device)
                        masks = masks.to(device)

                        preds = model(images)

                        loss = self.seg_loss(preds, masks)

                        val_loss += loss.item()

                        preds_np = preds.detach().cpu().numpy()
                        masks_np = masks.detach().cpu().numpy()

                        pred = np.argmax(preds_np, axis=1)

                        evaluator.add_batch(masks_np, pred)

                val_loss /= len(val_loader)

                val_miou = evaluator.Mean_Intersection_over_Union()
                val_acc = evaluator.Pixel_Accuracy()

                print(
                    f"{C.GREEN}Train Loss:{C.END} {train_loss:.4f} | "
                    f"{C.YELLOW}Val Loss:{C.END} {val_loss:.4f} | "
                    f"{C.CYAN}Val mIoU:{C.END} {val_miou:.4f} | "
                    f"{C.BLUE}Val Acc:{C.END} {val_acc:.4f}"
                )

                # ===== Save metrics =====

                with open(self.csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        fold,
                        epoch + 1,
                        train_loss,
                        val_loss,
                        val_miou,
                        val_acc
                    ])

                if val_loss < best_val_loss:

                    best_val_loss = val_loss

                    torch.save(
                        model.state_dict(),
                        f"crossval_models/best_model_fold_{fold}.pth"
                    )

                    print(f"{C.GREEN}✔ Best model updated{C.END}")

            # ===== TEST =====

            print(f"\n{C.CYAN}Testing best model...{C.END}")

            model.load_state_dict(
                torch.load(
                    f"crossval_models/best_model_fold_{fold}.pth"
                )
            )

            model.eval()
            evaluator.reset()

            with torch.no_grad():

                for (images, masks, orig_w, orig_h, filename, pattern_side, target_area) in test_loader:

                    images = images.to(device)
                    masks = masks.to(device)

                    preds = model(images)

                    preds = preds.detach().cpu().numpy()
                    masks = masks.detach().cpu().numpy()

                    pred = np.argmax(preds, axis=1)

                    evaluator.add_batch(masks, pred)

                    for i in range(pred.shape[0]):
                        pred_mask = pred[i]
                        ow = int(orig_w[i])
                        oh = int(orig_h[i])
                        ps = float(pattern_side[i])
                        ta = float(target_area[i])
                        fn = filename[i]

                        area_info = calculate_leaf_area(
                            mask=pred_mask,
                            pattern_side=ps,
                            orig_w=ow,
                            orig_h=oh
                        )

                        predicted_area = area_info["total_leaf_area_cm2"]

                        if predicted_area is not None and ta > 0:
                            evaluator.multileaf_area_results.append([fn, predicted_area, ta])

            mIoU = evaluator.Mean_Intersection_over_Union()
            area_mae = evaluator.calculate_mae()

            self.save_area_results_csv(fold, evaluator.multileaf_area_results)

            print(f"{C.BOLD}{C.YELLOW}Fold {fold} mIoU:{C.END} {mIoU:.4f}")
            print(f'{C.BOLD}{C.GREEN}Area MAE: {C.END}{area_mae}')

            fold_results["MIoU"].append(mIoU)
            fold_results["Area_MAE"].append(area_mae)

            
        # ===== FINAL RESULTS =====

        print(f"\n{C.BOLD}{C.HEADER}================================={C.END}")
        print(f"{C.BOLD}{C.HEADER}CROSS VALIDATION RESULTS{C.END}")
        print(f"{C.BOLD}{C.HEADER}================================={C.END}")

        for i, res in enumerate(fold_results['MIoU']):
            print(f"{C.YELLOW}Fold {i+1}:{C.END} {res:.4f}")

        for i, res in enumerate(fold_results["Area_MAE"]):
            print(f"{C.YELLOW}Fold {i+1}:{C.END} {res:.4f}")

        print(f"\n{C.BOLD}{C.GREEN}Mean mIoU:{C.END} {np.mean(fold_results['MIoU']):.4f}")
        print(f"\n{C.BOLD}{C.GREEN}Mean MAE:{C.END} {np.mean(fold_results['Area_MAE']):.4f}")


class Args:

    dataset = "multi_leaf"
    backbone = "xception"
    out_stride = 16

    workers = 0

    epochs = 50
    batch_size = 2

    lr = 0.01
    momentum = 0.9
    weight_decay = 5e-4
    nesterov = False

    no_cuda = False
    gpu_ids = [0]

    seed = 1

    sync_bn = None


def main():

    args = Args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    cross_validation = CrossValidator(args)

    cross_validation.training_loop()


if __name__ == "__main__":
    main()