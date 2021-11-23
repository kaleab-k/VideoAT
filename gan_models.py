from torch import nn
from torch.nn import functional as F


class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 1024)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)

class Generator(nn.Module):

    def __init__(self, in_ch):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 64, 4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, in_ch, 4, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        h = self.leaky_relu(self.bn1(self.conv1(x)))
        h = self.leaky_relu(self.bn2(self.conv2(h)))
        h = self.leaky_relu(self.bn3(self.deconv3(h)))
        h = self.tanh(self.deconv4(h))
        return h


class Discriminator(nn.Module):

    def __init__(self, in_ch):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 64, 3, stride=2)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2)
        self.bn3 = nn.BatchNorm2d(256)
        self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        if in_ch == 1:
            self.fc4 = nn.Linear(1024, 1)
        else:
            self.fc4 = nn.Linear(2304, 1)

    def forward(self, x):
        h = self.leaky_relu(self.conv1(x))
        h = self.leaky_relu(self.bn2(self.conv2(h)))
        h = self.leaky_relu(self.bn3(self.conv3(h)))
        h = self.sigmoid(self.fc4(h.view(h.size(0), -1)))
        return h

class GeneratorUCF(nn.Module):

    def __init__(self, in_ch):
        super(GeneratorUCF, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 64, 4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.deconv4 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.deconv5 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.deconv6 = nn.ConvTranspose2d(64, in_ch, 4, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.leaky_relu(self.bn1(self.conv1(x)))
        h = self.leaky_relu(self.bn2(self.conv2(h)))
        h = self.leaky_relu(self.bn3(self.conv3(h)))
        h = self.leaky_relu(self.bn4(self.deconv4(h)))
        h = self.leaky_relu(self.bn5(self.deconv5(h)))
        h = self.sigmoid(self.deconv6(h))
        return h


class DiscriminatorUCF(nn.Module):

    def __init__(self, in_ch):
        super(DiscriminatorUCF, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 64, 3, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 3, stride=2)
        self.bn4 = nn.BatchNorm2d(512)
        self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        if in_ch == 1:
            self.fc4 = nn.Linear(1024, 1)
        else:
            self.fc4 = nn.Linear(18432, 1024)
            self.fc5 = nn.Linear(1024, 1)

    def forward(self, x):
        h = self.leaky_relu(self.bn1(self.conv1(x)))
        h = self.leaky_relu(self.bn2(self.conv2(h)))
        h = self.leaky_relu(self.bn3(self.conv3(h)))
        h = self.leaky_relu(self.bn4(self.conv4(h)))
#         print(h.shape, h.reshape(h.size(0), -1).shape)
        h = self.fc4(h.reshape(h.size(0), -1))
        h = self.sigmoid(self.fc5(h.view(h.size(0), -1)))
        return h

## 3D-APE-GAN

class GeneratorUCF3D(nn.Module):

    def __init__(self, in_ch):
        super(GeneratorUCF3D, self).__init__()
        self.conv1 = nn.Conv3d(in_ch, 64, 4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.conv2 = nn.Conv3d(64, 128, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(128)
        self.conv3 = nn.Conv3d(128, 256, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(256)

        self.deconv4 = nn.ConvTranspose3d(256, 128, 4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm3d(128)
        self.deconv5 = nn.ConvTranspose3d(128, 64, 4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm3d(64)
        self.deconv6 = nn.ConvTranspose3d(64, in_ch, 4, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.leaky_relu(self.bn1(self.conv1(x)))
        h = self.leaky_relu(self.bn2(self.conv2(h)))
        h = self.leaky_relu(self.bn3(self.conv3(h)))
        h = self.leaky_relu(self.bn4(self.deconv4(h)))
        h = self.leaky_relu(self.bn5(self.deconv5(h)))
        h = self.sigmoid(self.deconv6(h))
        return h


class DiscriminatorUCF3D(nn.Module):

#     def __init__(self, in_ch):
#         super(DiscriminatorUCF3D, self).__init__()
#         self.conv1 = nn.Conv3d(in_ch, 64, 3, stride=2)
#         self.bn1 = nn.BatchNorm3d(64)
#         self.conv2 = nn.Conv3d(64, 128, 3, stride=2)
#         self.bn2 = nn.BatchNorm3d(128)
#         self.conv3 = nn.Conv3d(128, 256, 3, stride=2)
#         self.bn3 = nn.BatchNorm3d(256)
# #         self.conv4 = nn.Conv3d(256, 512, 3, stride=2)
# #         self.bn4 = nn.BatchNorm3d(512)
#         self.leaky_relu = nn.LeakyReLU()
#         self.sigmoid = nn.Sigmoid()
#         if in_ch == 1:
#             self.fc4 = nn.Linear(1024, 1)
#         else:
#             self.fc4 = nn.Linear(43264, 18432)
#             self.fc5 = nn.Linear(18432, 1024)
#             self.fc6 = nn.Linear(1024, 1)

#     def forward(self, x):
#         h = self.leaky_relu(self.bn1(self.conv1(x)))
#         h = self.leaky_relu(self.bn2(self.conv2(h)))
#         h = self.leaky_relu(self.bn3(self.conv3(h)))
# #         h = self.leaky_relu(self.bn4(self.conv4(h)))
# #         print(h.shape, h.reshape(h.size(0), -1).shape)
#         h = self.fc4(h.reshape(h.size(0), -1))
#         h = self.fc5(h)
#         h = self.sigmoid(self.fc6(h.view(h.size(0), -1)))
#         return h
    def __init__(self, in_ch=3, dim=112, out_conv_channels=512):
        super(DiscriminatorUCF3D, self).__init__()
        conv1_channels = int(out_conv_channels / 8)
        conv2_channels = int(out_conv_channels / 4)
        conv3_channels = int(out_conv_channels / 2)
        self.out_conv_channels = out_conv_channels
        self.out_dim = int(dim / 16)

        self.conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels=in_ch, out_channels=conv1_channels, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm3d(conv1_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(
                in_channels=conv1_channels, out_channels=conv2_channels, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm3d(conv2_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(
                in_channels=conv2_channels, out_channels=conv3_channels, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm3d(conv3_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(
                in_channels=conv3_channels, out_channels=out_conv_channels, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm3d(out_conv_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.out = nn.Sequential(
            nn.Linear(out_conv_channels * self.out_dim * self.out_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # Flatten and apply linear + sigmoid
#         print(x.shape)
        x = x.view(-1, self.out_conv_channels * self.out_dim * self.out_dim)
        x = self.out(x)
        return x


    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)