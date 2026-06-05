import tracemalloc

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# 这里我们演示如何使用 tracemalloc 来跟踪内存分配
def demo_memory_tracking():
    print("\n--- 3. Memory Tracking (tracemalloc) ---")
    tracemalloc.start() # 开始跟踪内存分配
    data = [torch.randn(100, 100) for _ in range(100)] # 创建一些张量来占用内存
    more_data = torch.randn(1000, 1000) # 创建一个更大的张量来占用更多内存
    snapshot = tracemalloc.take_snapshot() # 获取当前内存分配的快照
    top_stats = snapshot.statistics("lineno") # 获取按行号统计的内存分配情况
    print("  Top 5 memory allocations:") # 打印内存分配最多的前5行代码
    for stat in top_stats[:5]:
        print(f"    {stat}")
    del data, more_data # 删除占用内存的变量
    tracemalloc.stop() # 停止跟踪内存分配

def demo_logging():
    print("\n--- 1. Logging Tensor Info ---")
    x = torch.randn(2, 3)
    debug_print("input", x)

def debug_print(name, tensor):
    print(f"  {name}: shape={tensor.shape}, dtype={tensor.dtype}, "
          f"device={tensor.device}, "
          f"min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, "
          f"mean={tensor.mean().item():.4f}, "
          f"has_nan={tensor.isnan().any().item()}")

def demo_print_debugging():
    print("\n--- 1. Print Debugging for Tensors ---")
    x = torch.randn(32, 784)
    debug_print("input batch", x)

    w = torch.randn(784, 128)
    out = x @ w
    debug_print("after matmul", out)

    with_nan = out.clone()
    with_nan[0, 0] = float("nan")
    debug_print("with injected NaN", with_nan)




def demo_timing():
    print("\n--- 2. Timing Code Sections ---")
    pass



def demo_nan_detection():
    print("\n--- 5. NaN Detection ---")
    pass

def demo_device_checking():
    print("\n--- 6. Device Checking ---")
    pass

def demo_gradient_health():
    print("\n--- 7. Gradient Health ---")
    pass

def demo_gpu_memory():
    print("\n--- 8. GPU Memory ---")
    pass

def demo_conditional_breakpoint():
    print("\n--- 9. Conditional Breakpoint ---")
    pass

def demo_shape_checking():
    print("\n--- 4. Shape Checking Through Model ---")
    pass





def main():
    print("=" * 60)
    print("  AI Debugging and Profiling Toolkit")
    print("  Phase 0, Lesson 12")
    print("=" * 60)
    if not HAS_TORCH:
        print("\nPyTorch not installed. Install with:")
        print("  uv pip install torch")
        print("\nRunning non-PyTorch demos only...\n")

        demo_memory_tracking()
        demo_logging()
    demo_print_debugging()
    demo_timing()
    demo_memory_tracking()
    demo_shape_checking()
    demo_nan_detection()
    demo_device_checking()
    demo_gradient_health()
    demo_gpu_memory()
    demo_logging()
    demo_conditional_breakpoint()




if __name__=="__main__":
    main()
