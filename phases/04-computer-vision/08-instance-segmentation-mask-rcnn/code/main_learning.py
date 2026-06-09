"""
中文版本:
这个代码展示了如何实现一个简单的 RoI Align 函数，并与 torchvision 的实现进行比较。它还演示了如何加载预训练的 Mask R-CNN 模型，
检查其输出，并设置一个自定义的 Mask R-CNN 模型以适应新的类别数量，同时冻结骨干网络以进行微调。

"""

import torch
import torch.nn.functional as F
from torchvision.ops import roi_align


def roi_align_single(feature, box, output_size=7, spatial_scale=1 / 16.0):
    """
    实现一个简单的 RoI Align 函数，接受一个特征图和一个边界框，返回一个固定大小的输出。"""
    C, H, W = feature.shape
    x1, y1, x2, y2 = [c * spatial_scale - 0.5 for c in box]
    bin_w = (x2 - x1) / output_size # 计算每个输出单元格的宽度
    bin_h = (y2 - y1) / output_size # 计算每个输出单元格的高度

    grid_y = torch.linspace(y1 + bin_h / 2, y2 - bin_h / 2, output_size, device=feature.device) # 生成output_size的 y 坐标
    grid_x = torch.linspace(x1 + bin_w / 2, x2 - bin_w / 2, output_size, device=feature.device) # 生成output_size的 x 坐标
    yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij") # 生成一个网格，yy 和 xx 分别包含 y 和 x 坐标，形状为 (output_size, output_size)

    gx = 2 * (xx + 0.5) / W - 1 # 将 x 坐标转换为 [-1, 1] 范围
    gy = 2 * (yy + 0.5) / H - 1 # 将 y 坐标转换为 [-1, 1] 范围
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0) # 将 gx 和 gy 组合成一个 grid，形状为 (1, output_size, output_size, 2)
    sampled = F.grid_sample(feature.unsqueeze(0), grid, mode="bilinear",
                            align_corners=False) # 使用 grid_sample 从特征图中插值采样采样，得到形状为 (1, C, output_size, output_size) 的采样结果
    return sampled.squeeze(0)


def compare_with_torchvision_roi_align():
    '''
    比较自定义的 roi_align_single 函数与 torchvision 的 roi_align 实现的差异。
    返回每个框的最大绝对差异。


    '''
    torch.manual_seed(0)
    feature = torch.randn(1, 16, 50, 50) # 意味着 一张16channel的特征图，大小为50x50
    boxes = torch.tensor([[0, 10, 20, 100, 90],
                          [0, 5, 5, 80, 80],
                          [0, 30, 10, 120, 110]], dtype=torch.float32) # 3个框，每个框的格式为 [batch_index, x1, y1, x2, y2]

    diffs = []  # 存储每个框的最大绝对差异
    for b in boxes:
        # 目标: 从 feature map 中按照检测框 b 裁出对应区域，并通过双线性插值精确地缩放成一个固定的 7×7 特征块，供后续分类和边界框回归使用。
        ours = roi_align_single(feature[0], b[1:].tolist(), output_size=7, spatial_scale=1 / 4) # 使用自定义的 roi_align_single 函数对特征图进行采样
        theirs = roi_align(feature, b.unsqueeze(0), output_size=(7, 7), spatial_scale=1 / 4, sampling_ratio=1, aligned=True)[0] # 使用 torchvision 的 roi_align 函数进行采样,xiaoguo差不多. # 设置 aligned=True 使用半像素对齐,0.5,1.5,2.5,... 作为采样点坐标，而不是 0,1,2,... 这样更避免 RoI Pooling 取整误差
        diffs.append((ours - theirs).abs().max().item())
    return diffs


def load_pretrained_maskrcnn():
    from torchvision.models.detection import (
        maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights,
    )
    model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    model.eval() # 设置模型为评估模式，禁用 dropout 和 batch normalization 的训练行为
    return model


def build_custom_maskrcnn(num_classes):
    '''
    构建一个自定义的 Mask R-CNN 模型，适用于指定数量的类别。
    通过替换分类和掩码预测头来适应新的类别数量。
    '''
    from torchvision.models.detection import (
        maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features # 获取模型里面的box_predictor分类头的输入特征数量
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) # 替换分类预测头以适应新的类别数量
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels # 获取掩码预测头的输入特征数量
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes) # 替换掩码预测头以适应新的类别数量
    return model


def freeze_backbone(model):
    # torchvision Mask R-CNN's backbone includes the FPN (model.backbone.fpn),
    # so freezing model.backbone.parameters() also freezes the FPN parameters.
    for p in model.backbone.parameters():
        p.requires_grad = False
    return model


def main():
    print("[roi_align] comparing ours vs torchvision.ops.roi_align")
    diffs = compare_with_torchvision_roi_align()
    for i, d in enumerate(diffs):
        print(f"  box {i}: max|diff|={d:.2e}")

    try:
        print("\n[pretrained] loading maskrcnn_resnet50_fpn_v2 (downloads on first run)")
        model = load_pretrained_maskrcnn() # 加载预训练的 Mask R-CNN 模型
        with torch.no_grad():
            p = model([torch.randn(3, 200, 300)])[0] # 传入一个随机的图像张量，获取模型的输出
        print(f"  boxes:  {tuple(p['boxes'].shape)}")
        print(f"  labels: {tuple(p['labels'].shape)}")
        print(f"  masks:  {tuple(p['masks'].shape)}")

        print("\n[fine-tune setup] swap heads for 5-class dataset, freeze backbone") # 设置微调：为一个新的 5 类数据集替换预测头，并冻结骨干网络
        custom = build_custom_maskrcnn(num_classes=5) # 就改变了类别数量，其他部分都保持不变
        custom = freeze_backbone(custom) # 冻结骨干网络的参数，使其在训练过程中不更新
        trainable = sum(p.numel() for p in custom.parameters() if p.requires_grad) # 计算可训练参数的数量
        total = sum(p.numel() for p in custom.parameters()) # 计算模型的总参数数量
        print(f"  trainable: {trainable:,}   total: {total:,}")
    except Exception as e:
        print(f"[pretrained] skipped: {e}")


if __name__ == "__main__":
    main()
