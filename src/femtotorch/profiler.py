import numpy as np
import time
import tracemalloc
from femtotorch.tensor import Tensor
BYTES_PER_FLOAT32 = 4
MB_TO_BYTES = 1000000
class Profiler:
    
    def measure_latency(self, model, input_tensor, warmup = 10, iterations = 100):
        """
        Measure model inference latency,
        taking into account cold CPU cache with warm up iterations.
        """
        

        # Warmup runs
        for _ in range(warmup):
            _ = model.forward(input_tensor)

        times = []
        for _ in range(iterations):
            start_time = time.perf_counter()
            _ = model.forward(input_tensor)
            end_time = time.perf_counter()
            times.append((end_time - start_time)*1000) # *1000 to convert ot milisecond

        times = np.array(times)
        median_latency = np.median(times)
        mean_latency = np.mean(times)

        return float(median_latency), float(mean_latency)

    def measure_memory(self, model, input_shape):
        """
        Measures memory usage during forward pass
        """
        # start memory tracking
        tracemalloc.start()
        
        # Calculate parameter memory
        param_count = self.count_parameters(model)
        parameter_memory_bytes = param_count * BYTES_PER_FLOAT32
        parameter_memory_mb = parameter_memory_bytes / MB_TO_BYTES


        dummy_input = Tensor(np.random.randn(*input_shape), dtype=np.float32)
        input_memory_bytes = dummy_input.data.nbytes

        # Estimate activation memory (simplified)
        activation_memory_bytes = input_memory_bytes * 2  # Rough estimate
        activation_memory_mb = activation_memory_bytes / MB_TO_BYTES

        # Run forward pass to measure peak memory usage
        _ = model.forward(dummy_input)

        # Get peak memory
        _current_memory, peak_memory = tracemalloc.get_traced_memory()
        _baseline_memory = _current_memory # maybe ? not sure
        peak_memory_mb = (peak_memory - _baseline_memory) / MB_TO_BYTES

        tracemalloc.stop()

        # Calculate efficiency metrics
        useful_memory = parameter_memory_mb + activation_memory_mb
        memory_efficiency = useful_memory / max(peak_memory_mb, 0.001) # max(.., 0.001) to avoid division by zero
        return {
            'parameter_memory_mb': parameter_memory_mb,
            'activation_memory_mb': activation_memory_mb,
            'peak_memory_mb': max(peak_memory_mb, useful_memory),
            'memory_efficiency': min(memory_efficiency, 1.0)
        }
    
    def count_parameters(self, model):
        total_params = 0
        for parameters in model.parameters():
            total += parameters.data.size
        return total_params

    def count_flops(model, input_shape):
        pass
    
    
    def profile_forward_pass(model):
        pass
    def profile_backward_pass(model):
        pass