def check_cuda_compatibility():
    """Check CUDA compatibility and return best available device"""
    try:
        if not torch.cuda.is_available():
            console.print("[yellow]CUDA not available. Using CPU.[/yellow]")
            return torch.device('cpu'), False
            
        # Test CUDA functionality
        device = torch.device('cuda')
        test_tensor = torch.randn(10, 10, device=device)
        test_result = test_tensor @ test_tensor.T  # Simple matrix multiplication test
        
        gpu_name = torch.cuda.get_device_name(0)
        console.print(f"[green]CUDA compatible! Using GPU: {gpu_name}[/green]")
        return device, True
        
    except Exception as e:
        console.print(f"[red]CUDA compatibility issue: {e}[/red]")
        console.print("[yellow]Falling back to CPU for stability.[/yellow]")
        return torch.device('cpu'), False

# Global configuration
CONFIG = TradingConfig()
DEVICE, CUDA_AVAILABLE = check_cuda_compatibility()