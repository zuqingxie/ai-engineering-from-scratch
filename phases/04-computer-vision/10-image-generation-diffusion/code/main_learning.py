'''
中文:
这段代码实现了一个简单的图像生成扩散模型.
它包括了线性beta调度、预计算的调度参数、前向扩散过程、时间步嵌入、训练步骤以及DDPM和DDIM的采样函数。
最后，代码生成了一些合成的圆形图像，并训练模型以学习这些图像的分布，然后使用训练好的模型进行采样。
DDPM (Denoising Diffusion Probabilistic Models)
    输入 - > 前向扩散过程 (添加噪声) -> 模型预测噪声 -> 反向扩散过程 (去噪) -> 输出
    输入维度 - > (batch_size=16, channels=3, height=64, width=64)
    输出维度 - > (batch_size=16, channels=3, height=64, width=64)
UNet模型
    输入维度 (batch_size=16, channels=3, height=64, width=64)  
    输出维度 (batch_size=16, channels=3, height=64, width=64)

DDIM(Denoising Diffusion Implicit Models)
    输入维度 (batch_size=16, channels=3, height=64, width=64)  
    输出维度 (batch_size=16, channels=3, height=64, width=64)

'''
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def linear_beta_schedule(T=1000, beta_start=1e-4, beta_end=2e-2):
    return torch.linspace(beta_start, beta_end, T)


def precompute_schedule(betas):
    alphas = 1.0 - betas #
    alphas_cumprod = torch.cumprod(alphas, dim=0) # alpha_bar(t) = alpha_1 * alpha_2 * ... * alpha_t
    alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
        "sqrt_recip_alphas": torch.sqrt(1.0 / alphas),
        "posterior_variance": posterior_variance,
        "posterior_mean_coef1": betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        "posterior_mean_coef2": (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
    }


def q_sample(x0, t, noise, schedule):
    sqrt_a = schedule["sqrt_alphas_cumprod"].to(x0.device)[t].view(-1, 1, 1, 1) #
    sqrt_one_minus_a = schedule["sqrt_one_minus_alphas_cumprod"].to(x0.device)[t].view(-1, 1, 1, 1)
    return sqrt_a * x0 + sqrt_one_minus_a * noise


def show(images, title="images", path=None, max_images=8, columns=None):
    """Print image metadata and save a small PPM preview grid."""
    if images.dim() == 3:
        images = images.unsqueeze(0)
    if images.dim() != 4 or images.size(1) != 3:
        raise ValueError("show expects image tensors shaped (N, 3, H, W) or (3, H, W)")

    imgs = images.detach().cpu()[:max_images]
    count, channels, height, width = imgs.shape
    low = float(imgs.min())
    high = float(imgs.max())
    mean = float(imgs.mean())

    if path is None:
        safe_title = "".join(ch if ch.isalnum() else "_" for ch in title.lower()).strip("_")
        path = Path(__file__).resolve().parent / f"{safe_title or 'images'}.ppm"
    else:
        path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    columns = columns or min(4, count)
    rows = math.ceil(count / columns)
    gap = 2
    grid = np.full(
        (rows * height + (rows - 1) * gap, columns * width + (columns - 1) * gap, 3),
        255,
        dtype=np.uint8,
    )

    pixels = ((imgs.clamp(-1, 1) + 1.0) * 127.5).byte().permute(0, 2, 3, 1).numpy()
    for i, image in enumerate(pixels):
        row = i // columns
        col = i % columns
        y0 = row * (height + gap)
        x0 = col * (width + gap)
        grid[y0:y0 + height, x0:x0 + width] = image

    header = (
        f"P6\n# {title}; shape={(count, channels, height, width)}; "
        f"range=[{low:.3f}, {high:.3f}]\n{grid.shape[1]} {grid.shape[0]}\n255\n"
    )
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        f.write(grid.tobytes())

    print(f"\n[{title}]")
    print(f"  images: {count}  shape: ({channels}, {height}, {width})")
    print(f"  values: min={low:.3f}  max={high:.3f}  mean={mean:.3f}")
    print(f"  preview: {path}")
    return path


def timestep_embedding(t, dim=64):
    half = (dim + 1) // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([args.sin(), args.cos()], dim=-1)
    return emb[:, :dim]


class TinyUNet(nn.Module):
    def __init__(self, img_channels=3, base=16, t_dim=64):
        super().__init__()
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, base * 4),
            nn.SiLU(),
            nn.Linear(base * 4, base * 4),
        )
        self.t_dim = t_dim
        self.enc1 = nn.Conv2d(img_channels, base, 3, padding=1)
        self.enc2 = nn.Conv2d(base, base * 2, 4, stride=2, padding=1)
        self.mid = nn.Conv2d(base * 2, base * 2, 3, padding=1)
        self.dec1 = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)
        self.dec2 = nn.Conv2d(base * 2, img_channels, 3, padding=1)
        self.time_proj = nn.Linear(base * 4, base * 2)
    # 前向传播
    def forward(self, x, t):
        t_emb = self.t_mlp(timestep_embedding(t, self.t_dim))
        t_proj = self.time_proj(t_emb)[:, :, None, None]
        h1 = F.silu(self.enc1(x))
        h2 = F.silu(self.enc2(h1)) + t_proj
        h3 = F.silu(self.mid(h2))
        d1 = F.silu(self.dec1(h3))
        d2 = torch.cat([d1, h1], dim=1)
        return self.dec2(d2)


def train_step(model, data, schedule, optimizer, device, T=1000):
    model.train() # 启用训练模式.
    data_batch = data.to(device) # a batch data
    bs = data_batch.size(0) # batch size
    t = torch.randint(0, T, (bs,), device=device) # 随机选择一个时间步t，范围在[0, T-1]之间，大小为(batch_size,)，每个元素表示对应样本的时间步.
    noise = torch.randn_like(data_batch) # 生成与输入图像x0形状相同的随机噪声，分布为标准正态分布，用于模拟前向扩散过程中的噪声添加.
    x_t = q_sample(data_batch, t, noise, schedule) # 根据时间步t和预计算的调度参数，从原始图像x0中生成带有噪声的图像x_t，模拟前向扩散过程.
    pred = model(x_t, t)
    loss = F.mse_loss(pred, noise)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def sample_ddpm(model, schedule, shape, T=1000, device="cpu"):
    model.eval()
    x = torch.randn(shape, device=device)
    alphas_cumprod = schedule["alphas_cumprod"].to(device)
    sqrt_one_minus_a = schedule["sqrt_one_minus_alphas_cumprod"].to(device)
    posterior_variance = schedule["posterior_variance"].to(device)
    posterior_mean_coef1 = schedule["posterior_mean_coef1"].to(device)
    posterior_mean_coef2 = schedule["posterior_mean_coef2"].to(device)

    for t in reversed(range(T)):
        t_batch = torch.full((shape[0],), t, dtype=torch.long, device=device)
        eps = model(x, t_batch)
        x0_pred = (x - sqrt_one_minus_a[t] * eps) / torch.sqrt(alphas_cumprod[t])
        x0_pred = x0_pred.clamp(-1, 1)
        mean = posterior_mean_coef1[t] * x0_pred + posterior_mean_coef2[t] * x
        if t > 0:
            x = mean + torch.sqrt(posterior_variance[t]) * torch.randn_like(x)
        else:
            x = mean
    return x.clamp(-1, 1)


@torch.no_grad()
def sample_ddim(model, schedule, shape, steps=50, T=1000, device="cpu", eta=0.0):
    model.eval()
    x = torch.randn(shape, device=device)
    alphas_cumprod = schedule["alphas_cumprod"].to(device)

    ts = torch.linspace(T - 1, 0, steps + 1).long()
    for i in range(steps):
        t = int(ts[i])
        t_prev = int(ts[i + 1])
        t_batch = torch.full((shape[0],), t, dtype=torch.long, device=device)
        eps = model(x, t_batch)
        a_t = alphas_cumprod[t]
        a_prev = alphas_cumprod[t_prev]
        x0_pred = (x - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t)
        x0_pred = x0_pred.clamp(-1, 1)
        sigma = eta * torch.sqrt((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev).clamp_min(0))
        dir_xt = torch.sqrt((1 - a_prev - sigma ** 2).clamp_min(0)) * eps
        noise = sigma * torch.randn_like(x) if eta > 0 else 0
        x = torch.sqrt(a_prev) * x0_pred + dir_xt + noise
    return x.clamp(-1, 1)


def synthetic_circles(num=200, size=16, seed=0):
    rng = np.random.default_rng(seed)
    imgs = np.full((num, 3, size, size), -1.0, dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    for i in range(num): # 每张图像随机生成一个圆，圆心和半径随机分布在图像范围内，圆的颜色也随机分布在[-0.3, 1.0]范围内.
        r = rng.uniform(3, 5)
        cx, cy = rng.uniform(r, size - r, size=2)
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
        color = rng.uniform(-0.3, 1.0, size=3)
        for c in range(3):
            imgs[i, c][mask] = color[c]
    return torch.from_numpy(imgs)


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    T = 1000 # 扩散步骤数，越大理论上生成质量越好，但训练和采样时间也会增加. 常见选择是1000，但这里为了快速演示使用200.

    schedule = precompute_schedule(linear_beta_schedule(T=T, beta_start=1e-4, beta_end=0.02)) # 200 steps, beta从1e-4线性增加到0.02
    print(f"schedule: T={T}  alpha_bar[0]={float(schedule['alphas_cumprod'][0]):.4f}  "
          f"alpha_bar[-1]={float(schedule['alphas_cumprod'][-1]):.4f}")

    data = synthetic_circles(num=1000, size=16) # 生成1000张16x16的合成圆形图像，像素值范围在[-1, 1]之间，背景为-1，圆的颜色随机分布在[-0.3, 1.0]范围内.
    show(data[:8], title="training data examples")
    loader = DataLoader(TensorDataset(data), batch_size=64, shuffle=True) # 创建数据加载器，批量大小为64，数据顺序随机打乱.

    model = TinyUNet(img_channels=3, base=32).to(device)    
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(260):
        losses = []
        for (batch,) in loader: # batch 这里是16张图像的批次，维度为(batch_size=16, channels=3, height=16, width=16)，
            losses.append(train_step(model, batch, schedule, opt, device, T=T))
        print(f"epoch {epoch}  mse {np.mean(losses):.4f}")

    s_ddpm = sample_ddpm(model, schedule, shape=(8, 3, 16, 16), T=T, device=device) # 使用训练好的模型进行DDPM采样，生成8张16x16的图像，维度为(batch_size=8, channels=3, height=16, width=16)，
    s_ddim = sample_ddim(model, schedule, shape=(8, 3, 16, 16), steps=50, T=T, device=device) # 使用训练好的模型进行DDIM采样，生成8张16x16的图像，维度为(batch_size=8, channels=3, height=16, width=16)，采样步骤数为50.
    print(f"\nsampled DDPM: {tuple(s_ddpm.shape)}  range [{s_ddpm.min():.2f}, {s_ddpm.max():.2f}]")
    print(f"sampled DDIM: {tuple(s_ddim.shape)}  range [{s_ddim.min():.2f}, {s_ddim.max():.2f}]")
    show(s_ddpm, title="DDPM samples")
    show(s_ddim, title="DDIM samples")


if __name__ == "__main__":
    main()
