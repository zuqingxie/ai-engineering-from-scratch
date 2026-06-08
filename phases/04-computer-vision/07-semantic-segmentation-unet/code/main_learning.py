import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam


class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3, padding=1, bias=False), # 这里使用了一个卷积层，输入通道数为in_c，输出通道数为out_c，卷积核大小为3x3，padding=1表示在输入图像的边缘添加1像素的零填充，以保持输出图像的空间尺寸不变，bias=False表示不使用偏置项。
            nn.BatchNorm2d(out_c), # 批归一化，对于这个batch里面的每一个channel 单独计算均值和方差， 然后每一个像素值都减去均值除以方差进行归一化，最后再乘以一个可学习的缩放参数并加上一个可学习的偏移参数，以增强模型的表达能力。所以在算模型参数的学习参数量时候这一层有2*out_c个参数。
            nn.ReLU(inplace=True), # inplace=True表示在原地进行ReLU操作，节省内存
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True), # inplace=True表示在原地进行ReLU操作，节省内存
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_c, out_c)) # maxpool 2x2 里面留最大的值，输出尺寸是输入的一半

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False) # 这里使用了双线性插值的方式进行上采样，将特征图的空间尺寸扩大一倍。align_corners=False表示在插值过程中不对齐角点，这通常会导致更平滑的结果。
        self.conv = DoubleConv(in_c, out_c)

    def forward(self, x, skip): # 前向传播的跳跃连接，输入x是上采样后的特征图，skip是对应的下采样阶段的特征图。首先对x进行上采样，如果上采样后的特征图尺寸与skip的尺寸不匹配，则使用双线性插值将x调整到与skip相同的空间尺寸。最后将skip和x在通道维度上进行拼接，并通过一个DoubleConv模块进行卷积处理，得到输出特征图。
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]: # 如果两个特征图的空间尺寸不匹配，则使用双线性插值将x调整到与skip相同的空间尺寸。
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False) # 这里使用了双线性插值的方式进行调整，mode="bilinear"表示使用双线性插值，align_corners=False表示在插值过程中不对齐角点，这通常会导致更平滑的结果。
        x = torch.cat([skip, x], dim=1) # 将两个在第二维度上面面拼接，也就是让channel叠加
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=3, base=32): # in_channels=3表示输入图像的通道数（RGB图像有3个通道），num_classes=3表示要分割的类别数量（背景、圆形、方形），base=32是UNet中卷积层的基准通道数，后续层的通道数将基于这个值进行倍增。
        super().__init__()
        self.inc = DoubleConv(in_channels, base) # 输入通道数为3，输出通道数为base（32）
        self.d1 = Down(base, base * 2)
        self.d2 = Down(base * 2, base * 4)
        self.d3 = Down(base * 4, base * 8)
        self.d4 = Down(base * 8, base * 16) # channel 逐渐变多
        self.u1 = Up(base * 16 + base * 8, base * 8) # 反过来channel 逐渐变少
        self.u2 = Up(base * 8 + base * 4, base * 4)
        self.u3 = Up(base * 4 + base * 2, base * 2)
        self.u4 = Up(base * 2 + base, base)
        self.outc = nn.Conv2d(in_channels=base, out_channels=num_classes, kernel_size=1) # 这个卷积通道数为base 16，输出通道数为num_classes，卷积核大小为1x1，用于将特征图映射到最终的类别空间。每个像素点的输出将包含num_classes个值，表示该像素属于每个类别的概率（在训练过程中通过softmax函数进行归一化）。

    def forward(self, x): # 输入数据， NCHW 这里属于前向传播函数，定义了数据如何通过网络进行计算。输入x是一个形状为(N, in_channels, H, W)的张量，其中N是批次大小，in_channels是输入图像的通道数（3），H和W是图像的高度和宽度。
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.d4(x4)
        x = self.u1(x5, x4)
        x = self.u2(x, x3)
        x = self.u3(x, x2)
        x = self.u4(x, x1)
        return self.outc(x)


def dice_loss(logits, targets, num_classes, eps=1e-6):
    probs = F.softmax(logits, dim=1)
    one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (probs * one_hot).sum(dim=dims)
    denom = probs.sum(dim=dims) + one_hot.sum(dim=dims)
    dice = (2 * inter + eps) / (denom + eps)
    return 1 - dice.mean()


def combined_loss(logits, targets, num_classes, lam=1.0):
    ce = F.cross_entropy(logits, targets)
    dc = dice_loss(logits, targets, num_classes)
    return ce + lam * dc, {"ce": ce.detach().item(), "dice": dc.detach().item()}


@torch.no_grad() # 这个装饰器表示在这个函数中不需要计算梯度，节省内存和计算资源，因为在评估阶段我们不需要进行反向传播。
def iou_counts(logits, targets, num_classes):
    preds = logits.argmax(dim=1) # 只提取里面预测值最大的索引,这里的索引也就是类别
    intersections = torch.zeros(num_classes, device=logits.device)
    unions = torch.zeros(num_classes, device=logits.device)
    for c in range(num_classes): # 0,1,2
        pred_c = (preds == c) #
        true_c = (targets == c)
        intersections[c] = (pred_c & true_c).sum().float()
        unions[c] = (pred_c | true_c).sum().float()
    return intersections, unions



def synthetic_segmentation(num_samples=120, size=64, seed=0):
    '''
        这里生成了一个简单的合成数据集，每个图像包含一个随机位置的圆形或方形，背景为绿色。圆形和方形分别对应不同的类别标签。图像中还添加了一些随机噪声以增加难度。
    '''
    rng = np.random.default_rng(seed)
    images = np.zeros((num_samples, size, size, 3), dtype=np.float32)
    masks = np.zeros((num_samples, size, size), dtype=np.int64)
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij") # 这里使用了meshgrid函数生成了两个二维数组xx和yy，分别表示图像中每个像素的x坐标和y坐标。这些坐标将用于后续生成圆形和方形的掩码。
    circle_color = np.array([0.9, 0.1, 0.1], dtype=np.float32) # 圆形的颜色为红色，表示类别1
    square_color = np.array([0.1, 0.2, 0.9], dtype=np.float32) # 方形的颜色为蓝色，表示类别2
    for i in range(num_samples):
        bg = np.array([0.3, 0.7, 0.3], dtype=np.float32) # 背景颜色为绿色
        images[i] = bg # 初始化图像为背景颜色
        cls = int(rng.integers(1, 3)) # 随机选择类别1（圆形）或类别2（方形）
        cx, cy = int(rng.integers(14, size - 14)), int(rng.integers(14, size - 14)) # 随机选择圆心或方形中心的位置，确保它们不会太靠近图像边缘，以避免生成的形状被裁剪。
        r = int(rng.integers(8, 14))
        if cls == 1:
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
            images[i][mask] = circle_color # 在图像中将圆形区域的像素值设置为circle_color，表示类别1
        else:
            mask = (np.abs(xx - cx) < r) & (np.abs(yy - cy) < r) # 这里生成了一个方形掩码，表示类别2
            images[i][mask] = square_color # 在图像中将方形区域的像素值设置为square_color，表示类别2
        masks[i][mask] = cls # 在掩码中将圆形或方形区域的像素值设置为对应的类别标签（cls），背景区域保持为0
        images[i] += rng.normal(0, 0.02, images[i].shape) # 在图像中添加一些随机噪声，以增加数据的多样性和难度。这里使用了正态分布的随机数生成器，均值为0，标准差为0.02，生成与图像形状相同的噪声数组，并将其加到原始图像上。
        images[i] = np.clip(images[i], 0, 1) # 将图像像素值限制在0到1之间，确保它们是有效的图像数据。最后返回生成的图像和对应的掩码。
    return images, masks # 图像是一个形状为(num_samples, size, size, 3)的数组，包含了RGB颜色值，值是0-1 normalized过后的；掩码是一个形状为(num_samples, size, size)的数组，包含了每个像素的类别标签（0表示背景，1表示圆形，2表示方形）。


class SegDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = torch.from_numpy(self.images[i]).permute(2, 0, 1).float()
        mask = torch.from_numpy(self.masks[i]).long()
        return img, mask


def main():
    torch.manual_seed(1)
    images, masks = synthetic_segmentation(num_samples=60, size=64)
    split = int(0.85 * len(images))
    train_ds = SegDataset(images[:split], masks[:split])   #  :3 包含第三个，2：不包含第二个。记忆方法，从：画直线向后
    val_ds = SegDataset(images[split:], masks[split:])
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True) # data loader是PyTorch中用于加载数据的工具，它可以将数据集分成小批量，并在训练过程中进行迭代。这里创建了一个train_loader，用于加载训练数据，batch_size=8表示每个批次包含8个样本，shuffle=True表示在每个epoch开始时会随机打乱数据顺序，以增加训练的随机性和泛化能力。
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = 3 # 背景0，圆形1，方形2
    model = UNet(in_channels=3, num_classes=num_classes, base=16).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}") # :, 是一种数字输出格式 11,123,324

    for epoch in range(40):
        model.train()
        loss_sum, total = 0.0, 0 # 损失总和，样本总数
        for x, y in train_loader: # train_loader 是迭代器
            x, y = x.to(device), y.to(device)
            logits = model(x) # 为什么这里是返回 8 3 64 64？因为输入x的形状是(8, 3, 64, 64)，经过UNet的前向传播后，输出logits的形状是(8, 3, 64, 64)，其中8是批次大小，3是类别数，64x64是图像的空间尺寸。这里预测的是对于三个类型的预测。
            loss, _ = combined_loss(logits, y, num_classes) # logits 是预测出来的，y 是ground truth
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(next(model.parameters()).grad) # 打印出gradients
            loss_sum += loss.detach().item() * x.size(0) # detach 是将loss分离出来 item 是将tensor转换成python的数值， x.size(0) 是当前批次的样本数量，将每个批次的损失乘以样本数量后累加到loss_sum中，以便在epoch结束时计算平均损失。
            total += x.size(0) # 计算样本总和

        model.eval() # 切换到评估/推理模式，关闭dropout和batchnorm的训练行为
        iou_intersections = torch.zeros(num_classes, device=device) # 初始化每个类别的交集累加器
        iou_unions = torch.zeros(num_classes, device=device) # 初始化每个类别的并集累加器
        with torch.no_grad():
            for x, y in val_loader: # validation也是按照批量处理来的
                x, y = x.to(device), y.to(device) # 推理也是使用GPU加速的
                y_pred = model(x)
                batch_intersections, batch_unions = iou_counts(y_pred, y, num_classes)
                iou_intersections += batch_intersections
                iou_unions += batch_unions
        iou_mean = torch.where(iou_unions > 0, iou_intersections / iou_unions, torch.full_like(iou_unions, float("nan")),).cpu().tolist() # 这里计算的是每个类别的平均IoU（交并比），如果某个类别的并集为0，则将其IoU设置为NaN（表示该类别在验证集中没有出现）。最后将结果转换为CPU上的列表，以便打印输出。
        print(f"epoch {epoch}  train_loss {loss_sum/total:.3f}  iou {[f'{v:.2f}' for v in iou_mean]}")


if __name__ == "__main__":
    main()
