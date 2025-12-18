import torch.nn as nn
import torch
import torchvision.models as models
from torchvision import transforms
class CustomCNN(nn.Module):
  def __init__(self):
    super(CustomCNN, self).__init__()
    self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
    self.pool = nn.MaxPool2d(2,2)
    self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
    self.fc1 = nn.Linear(64*56*56, 128)
    self.fc2 = nn.Linear(128, 2)

  def forward(self, x):
    x = self.pool(torch.relu(self.conv1(x)))
    x = self.pool(torch.relu(self.conv2(x)))
    x = x.view(x.size(0), -1)
    x = torch.relu(self.fc1(x))
    x = self.fc2(x)
    return x
  
  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

custom_net = CustomCNN()
custom_net.load_state_dict(torch.load("./CustomCNN.pth",map_location=device))


res_net = models.resnet50(pretrained=False)
res_net.fc = nn.Linear(res_net.fc.in_features, 2)
res_net.load_state_dict(torch.load("./ResNet.pth",map_location=device))

viggl_net = models.vgg16(pretrained=False)
viggl_net.classifier[6] = nn.Linear(viggl_net.classifier[6].in_features, 2)
viggl_net.load_state_dict(torch.load("./VGG.pth",map_location=device))

mobil_net = models.mobilenet_v2(pretrained=False)
mobil_net.classifier[1] = nn.Linear(mobil_net.classifier[1].in_features, 2)
mobil_net.load_state_dict(torch.load("./MobileNetV2.pth",map_location=device))



torch.serialization.add_safe_globals([transforms.Compose])
transform = torch.load("./transform.pth", map_location=device,  weights_only=False ) # must be false to load full objects