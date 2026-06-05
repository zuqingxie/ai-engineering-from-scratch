import time
import sys


def check_gpu():
    try:
        import torch
    except ImportError:
        print("PyTorch not installed. Run: pip install torch")
        return

    print("=== GPU Check ===\n")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("\nNo GPU detected. That's fine for most lessons.")
        print("For GPU-heavy lessons, use Google Colab (free).")
        return

    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    props = torch.cuda.get_device_properties(0)
    print(f"Memory: {props.total_memory / 1e9:.1f} GB")
    print(f"Compute capability: {props.major}.{props.minor}")

    print("\n=== CPU vs GPU Benchmark ===\n")
    size = 8000

    a = torch.randn(size, size) # Create random matrices on CPU
    b = torch.randn(size, size) # Create random matrices on CPU

    start = time.time()
    _ = a @ b  # Matrix multiplication on CPU
    cpu_time = time.time() - start
    print(f"CPU matrix multiply ({size}x{size}): {cpu_time:.3f}s")

    a_gpu = a.to("cuda") # Move matrices to GPU
    b_gpu = b.to("cuda") # Move matrices to GPU
    torch.cuda.synchronize() # Ensure GPU is ready before timing

    start = time.time()
    _ = a_gpu @ b_gpu
    torch.cuda.synchronize() # Ensure GPU has finished before stopping timer
    gpu_time = time.time() - start
    print(f"GPU matrix multiply ({size}x{size}): {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time / gpu_time:.0f}x")

    vram_gb = props.total_memory / 1e9
    params_fp16 = vram_gb * 1e9 / 2 # Each fp16 parameter takes 2 bytes
    params_billions = params_fp16 / 1e9
    print(f"\nEstimated max model size (fp16): ~{params_billions:.0f}B parameters")


if __name__ == "__main__":
    check_gpu()
