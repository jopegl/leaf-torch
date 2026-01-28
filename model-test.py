from PIL import Image
from modeling.deeplab import DeepLab
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

image_path = os.path.join('complete_leaf_dataset\\train\\027\\images\\leaf_027_001.jpg')
area_path = os.path.join('complete_leaf_dataset\\train\\027\\area\\leaf_027_001_area.raw')
seg_path = os.path.join('complete_leaf_dataset\\train\\027\\segmentation\\leaf_027_001_segmentation.raw')

image = Image.open(image_path).convert('RGB')
seg_target = np.fromfile(seg_path, dtype=np.int8).reshape((512, 512))
seg_target = torch.from_numpy(seg_target)

transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
image = transform(image).unsqueeze(0)  # Adiciona dimensão de batch
checkpoint_path = os.path.join('always_saver\\checkpoint_epoch_56.pth.tar')

model = DeepLab(num_classes=3,
                        backbone='xception',
                        output_stride=16,
                        sync_bn=False,
                        freeze_bn=False)

checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint['state_dict'])

with torch.inference_mode():
    seg_out, _, area_out = model(image)
    seg_out = torch.softmax(seg_out, dim=1)
    pred_mask = seg_out.argmax(dim=1).squeeze().cpu().numpy()
    leaf_mask = seg_out[:,1:2] + seg_out[:,2:3]
    final_area_pred = leaf_mask * area_out

final_area_pred = final_area_pred.squeeze().cpu().numpy()
seg_out = seg_out.squeeze().cpu().numpy()
img_to_plot = Image.open(image_path).convert('RGB')
area_target = np.fromfile(area_path, dtype=np.float32).reshape((512, 512))
area_target = torch.from_numpy(area_target)

leaf_mask = seg_out[1]
marker_mask = seg_out[2]
leaf_area_pred = (final_area_pred * leaf_mask).sum()
marker_area_pred = (final_area_pred * marker_mask).sum()
leaf_area_target = area_target[seg_target == 1].sum()
marker_area_target = area_target[seg_target == 2].sum()
print(f"Predicted Leaf Area: {leaf_area_pred.item() / 1000:.4f}, Target Leaf Area: {leaf_area_target.item() / 1000:.4f}")
print(f"Predicted Marker Area: {marker_area_pred.item()/ 1000:.4f}, Target Marker Area: {marker_area_target.item()/ 1000:.4f}")


plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(img_to_plot)
plt.title('Input Image')
plt.subplot(1, 3, 2)
plt.imshow(pred_mask, cmap='jet', alpha=1)
plt.title('Predicted Segmentation Mask')
plt.subplot(1, 3, 3)
plt.imshow(final_area_pred, cmap='turbo', alpha=1)
plt.title('Predicted Area Map')
plt.show()