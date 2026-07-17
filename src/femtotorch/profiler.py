import numpy as np
import time

class Profiler:
    @staticmethod
    def measure_latency(model, input_tensor, warmup = 10, iterations = 100):
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




    def count_parameters(model):
        pass
    def count_flops(model, input_shape):
        pass
    def measure_memory(model, input_shape):
        pass
    
    def profile_forward_pass(model):
        pass
    def profile_backward_pass(model):
        pass