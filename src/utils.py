import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

def normalize(train_imgs):
    """This function calculate the mean and standard
    deviation of a particular dataset

    input
    ---------
    train_imgs: list
    A list of corresponding path

    output
    ---------
    mean: np.array
    std: np.array
    """
    mean = np.zeros(3)
    std = np.zeros(3)

    for i in train_imgs:
        img = Image.open(i).convert("RGB")
        img = transforms.ToTensor()(img)

        mean = np.add(mean, img.mean(dim=(1, 2)).numpy())
        std = np.add(std, img.std(dim=(1, 2)).numpy())

    mean /= len(train_imgs)
    std /= len(train_imgs)

    print("Mean:", *mean)
    print("Std Deviation:", *std)

    return mean, std
    
    
    
    
    
def register_gradient_hooks(model):
    """
    Register hooks on convolutional and transpose convolutional layers of a PyTorch model to compute gradient norms.

    Args:
        model (nn.Module): The PyTorch model on which gradient hooks will be registered.

    Returns:
        dict: A dictionary containing gradient norms per layer, where keys are layer names and values are lists
        of gradient norms (L1 norm) for each output tensor of the layer.
    """
    gradient_norms_dict = {}  # Dictionary to store gradient norms per layer

    def hook_fn(module, grad_input, grad_output):
        """
        Hook function to calculate gradient norms (L1 norm) for each output tensor of a layer and store them.

        Args:
            module (nn.Module): The layer/module for which gradients are calculated.
            grad_input (tuple): Gradients with respect to the layer's input.
            grad_output (tuple): Gradients with respect to the layer's output.
        """
        # Calculate gradient norms (L1 norm) for each output tensor
        gradient_norms = [torch.norm(grad, p=1).item() for grad in grad_output]

        # Store the gradient norms in the dictionary
        gradient_norms_dict[module.__class__.__name__] = gradient_norms

    # Register hooks on convolutional and transpose convolutional layers
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.ConvTranspose2d):
            module.register_backward_hook(hook_fn)

    return gradient_norms_dict
