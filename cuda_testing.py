import torch
import time
print(torch.cuda.is_available())
print(torch.cuda.get_device_name())
print(torch.cuda.get_device_properties())
print(torch.version.cuda)


print("CUDA Available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

start = time.time()

x = torch.randn(10000, 10000, device="cuda")
y = torch.randn(10000, 10000, device="cuda")

z = torch.matmul(x, y)

torch.cuda.synchronize()

end = time.time()
print(z)
print("Matrix multiplication successful")
print("Time:", end - start, "seconds")