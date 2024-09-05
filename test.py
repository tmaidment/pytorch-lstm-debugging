import torch
import torch.nn as nn
import time
import numpy as np
import csv
import os
import psutil

# Set random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# LSTM Model
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# Generate sample data
def generate_data(num_samples, seq_length, input_size):
    return torch.randn(num_samples, seq_length, input_size)

# Inference function
def run_inference(model, data):
    model.eval()
    with torch.no_grad():
        return model(data)

def measure_performance(model, data, num_warmup_runs=10, num_hot_runs=90):
    # Warm-up phase
    warmup_times = []
    peak_memory_warmup = 0
    for _ in range(num_warmup_runs):
        start_time = time.time()
        _ = run_inference(model, data)
        end_time = time.time()
        warmup_times.append(end_time - start_time)
        peak_memory_warmup = max(peak_memory_warmup, torch.cuda.max_memory_allocated() if torch.cuda.is_available() else psutil.Process(os.getpid()).memory_info().rss)

    # Hot phase
    hot_times = []
    peak_memory_hot = 0
    for _ in range(num_hot_runs):
        start_time = time.time()
        _ = run_inference(model, data)
        end_time = time.time()
        hot_times.append(end_time - start_time)
        peak_memory_hot = max(peak_memory_hot, torch.cuda.max_memory_allocated() if torch.cuda.is_available() else psutil.Process(os.getpid()).memory_info().rss)

    avg_warmup_time = sum(warmup_times) / num_warmup_runs
    avg_hot_time = sum(hot_times) / num_hot_runs

    return avg_warmup_time, avg_hot_time, peak_memory_warmup, peak_memory_hot

# Get model size
def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

# Main test function
def run_test(torch_version, compile_model=False, backend=None):
    print(f"Testing PyTorch {torch_version}")
    
    # Model parameters
    input_size = 10
    hidden_size = 20
    output_size = 5
    
    # Data parameters
    num_samples = 1000
    seq_length = 50
    
    # Create model and data
    model = SimpleLSTM(input_size, hidden_size, output_size)
    data = generate_data(num_samples, seq_length, input_size)
    
    if compile_model and hasattr(torch, 'compile'):
        model = torch.compile(model, backend=backend)
    
    # Measure performance
    avg_warmup_time, avg_hot_time, peak_memory_warmup, peak_memory_hot = measure_performance(model, data)
    model_size = get_model_size(model)
    
    print(f"Average warmup inference time: {avg_warmup_time:.6f} seconds")
    print(f"Average hot inference time: {avg_hot_time:.6f} seconds")
    print(f"Peak memory usage (warmup): {peak_memory_warmup / 1024**2:.2f} MB")
    print(f"Peak memory usage (hot): {peak_memory_hot / 1024**2:.2f} MB")
    print(f"Model size: {model_size:.2f} MB")
    
    return {
        "torch_version": torch_version,
        "compiled": compile_model,
        "backend": backend if compile_model else "N/A",
        "avg_warmup_inference_time": avg_warmup_time,
        "avg_hot_inference_time": avg_hot_time,
        "peak_memory_warmup_mb": peak_memory_warmup / 1024**2,
        "peak_memory_hot_mb": peak_memory_hot / 1024**2,
        "model_size_mb": model_size
    }

def write_results_to_csv(results, filename="pytorch_performance_results.csv"):
    fieldnames = ["torch_version", "compiled", "backend", "avg_warmup_inference_time", "avg_hot_inference_time", "peak_memory_warmup_mb", "peak_memory_hot_mb", "model_size_mb"]
    
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        for result in results:
            writer.writerow(result)

if __name__ == "__main__":
    torch_version = torch.__version__
    results = []
    
    results.append(run_test(f"{torch_version}"))
    if torch_version.startswith("2."):
        results.append(run_test(f"{torch_version} (compiled)", compile_model=True))
        results.append(run_test(f"{torch_version} (compiled, inductor backend)", compile_model=True, backend="inductor"))
    
    write_results_to_csv(results)