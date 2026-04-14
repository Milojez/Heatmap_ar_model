import torch
ckpt = torch.load('C:/Users/milos/Desktop/fixations_thesis/Adapt_yke_dataset/heatmap_ar_model/checkpoints/hm_ar_256_ur_dis_best/hm_ar_256_ur_dis_best.pth', map_location='cpu')
print({k: v for k, v in ckpt.items() if k != 'model_state'})
