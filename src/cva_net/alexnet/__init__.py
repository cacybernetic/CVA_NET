"""
AlexNet model parameters:
	features.0.weight 	 torch.Size([96, 3, 11, 11])
	features.0.bias 	 torch.Size([96])
	features.1.weight 	 torch.Size([96])
	features.1.bias 	 torch.Size([96])
	features.1.running_mean 	 torch.Size([96])
	features.1.running_var 	 torch.Size([96])
	features.1.num_batches_tracked 	 torch.Size([])
	features.4.weight 	 torch.Size([256, 96, 5, 5])
	features.4.bias 	 torch.Size([256])
	features.5.weight 	 torch.Size([256])
	features.5.bias 	 torch.Size([256])
	features.5.running_mean 	 torch.Size([256])
	features.5.running_var 	 torch.Size([256])
	features.5.num_batches_tracked 	 torch.Size([])
	features.8.weight 	 torch.Size([384, 256, 3, 3])
	features.8.bias 	 torch.Size([384])
	features.9.weight 	 torch.Size([384])
	features.9.bias 	 torch.Size([384])
	features.9.running_mean 	 torch.Size([384])
	features.9.running_var 	 torch.Size([384])
	features.9.num_batches_tracked 	 torch.Size([])
	features.11.weight 	 torch.Size([384, 384, 3, 3])
	features.11.bias 	 torch.Size([384])
	features.12.weight 	 torch.Size([384])
	features.12.bias 	 torch.Size([384])
	features.12.running_mean 	 torch.Size([384])
	features.12.running_var 	 torch.Size([384])
	features.12.num_batches_tracked 	 torch.Size([])
	features.14.weight 	 torch.Size([256, 384, 3, 3])
	features.14.bias 	 torch.Size([256])
	features.15.weight 	 torch.Size([256])
	features.15.bias 	 torch.Size([256])
	features.15.running_mean 	 torch.Size([256])
	features.15.running_var 	 torch.Size([256])
	features.15.num_batches_tracked 	 torch.Size([])
	classifier.1.weight 	 torch.Size([4096, 6400])
	classifier.1.bias 	 torch.Size([4096])
	classifier.4.weight 	 torch.Size([4096, 4096])
	classifier.4.bias 	 torch.Size([4096])
	classifier.6.weight 	 torch.Size([1000, 4096])
	classifier.6.bias 	 torch.Size([1000])

"""

from .model import AlexNet, ModelConfig, ModelFactory, ModelRepository

__all__ = [
    'AlexNet', 'ModelConfig', 'ModelFactory', 'ModelRepository'
]
