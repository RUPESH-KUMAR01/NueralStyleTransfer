import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# =======================
# Load and Preprocess Image
# =======================
def load_image(image_path, img_size=512, device="cpu"):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x[:3, :, :]),  # Ensure 3 channels
        transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], std=[1,1,1]),  # ImageNet mean
        transforms.Lambda(lambda x: x.mul(255))  # Multiply by 255
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)
    return image

# =======================
# Define Gram Matrix
# =======================
class GramMatrix(nn.Module):
    def forward(self, x):
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2)).div_(h * w)
        return gram

# =======================
# Define Gram MSE Loss
# =======================
class GramMSELoss(nn.Module):
    def forward(self, input, target):
        return nn.functional.mse_loss(GramMatrix()(input), target)

# =======================
# VGG Feature Extractor
# =======================
layer_mapping = {
    'r11': 0,  # conv1_1
    'r21': 5,  # conv2_1
    'r31': 10, # conv3_1
    'r41': 19, # conv4_1
    'r51': 28, # conv5_1
    'r42': 21  # conv4_2 (content layer)
}

class VGGFeatureExtractor(nn.Module):
    def __init__(self, target_layers):
        super().__init__()
        self.vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.target_layers = target_layers

    def forward(self, x):
        outputs = {}
        for idx, layer in self.vgg._modules.items():
            x = layer(x)
            for name in self.target_layers:
                if int(idx) == layer_mapping[name]:
                    outputs[name] = x
        return outputs

# =======================
# Post-processing function
# =======================
post_transform = transforms.Compose([
    transforms.Lambda(lambda x: x.detach().clone()),  # Detach to avoid modifying leaf tensors
    transforms.Lambda(lambda x: x * (1. / 255)),
    transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], std=[1, 1, 1]),
    transforms.Lambda(lambda x: x.clamp(0, 1)),
    transforms.ToPILImage()
])
