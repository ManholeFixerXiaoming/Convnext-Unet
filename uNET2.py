# @修井盖的小铭同学
import numpy
import torch
from torch import nn
from torchsummary import summary
from torchviz import make_dot
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model


class cqm_unetLayer(nn.Module):  # ：B,C,H,W
    def __init__(self, input_channel, padd=1, padd2=0):
        super(cqm_unetLayer, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=(3, 3), padding=padd, stride=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=padd, stride=1)
        self.down1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=padd2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=padd, stride=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=padd, stride=1)
        self.down2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=padd2)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=padd, stride=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=padd, stride=1)
        self.down3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=padd2)
        self.conv7 = nn.Conv2d(256, 512, kernel_size=(3, 3), padding=padd, stride=1)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=padd, stride=1)
        self.down4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=padd2)
        self.conv9 = nn.Conv2d(512, 1024, kernel_size=(3, 3), padding=padd, stride=1)
        self.conv10 = nn.Conv2d(1024, 1024, kernel_size=(3, 3), padding=padd, stride=1)
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, padding=padd2)
        self.up_conv1 = nn.Conv2d(1024, 512, kernel_size=(3, 3), padding=padd, stride=1)
        self.up_conv2 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=padd, stride=1)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=padd2)
        self.up_conv3 = nn.Conv2d(512, 256, kernel_size=(3, 3), padding=padd, stride=1)
        self.up_conv4 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=padd, stride=1)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=padd2)
        self.up_conv5 = nn.Conv2d(256, 128, kernel_size=(3, 3), padding=padd, stride=1)
        self.up_conv6 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=padd, stride=1)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=padd2)
        self.up_conv7 = nn.Conv2d(128, 64, kernel_size=(3, 3), padding=padd, stride=1)
        self.up_conv8 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=padd, stride=1)
        self.up_conv9 = nn.Conv2d(64, 10, kernel_size=(1, 1), padding=padd2, stride=1)
        self.jh = nn.ReLU()

    def forward(self, x):

        x = x[:, :, :, :]
        x1 = self.conv1(x)
        x1 = self.jh(x1)
        x2 = self.conv2(x1)
        x2 = self.jh(x2)
        x_down1 = self.down1(x2)
        x3 = self.conv3(x_down1)
        x3 = self.jh(x3)
        x4 = self.conv4(x3)
        x4 = self.jh(x4)
        x_down2 = self.down2(x4)
        x5 = self.conv5(x_down2)
        x5 = self.jh(x5)
        x6 = self.conv6(x5)
        x6 = self.jh(x6)
        x_down3 = self.down3(x6)
        x7 = self.conv7(x_down3)
        x7 = self.jh(x7)
        x8 = self.conv8(x7)
        x8 = self.jh(x8)
        x_down4 = self.down4(x8)
        x9 = self.conv9(x_down4)
        x9 = self.jh(x9)
        x10 = self.conv10(x9)
        x10 = self.jh(x10)
        x_up1 = self.up1(x10)
        x_up1_cat = torch.cat((x_up1, x8), dim=1)  #  x8  x_up1
        x_up1_conv1 = self.up_conv1(x_up1_cat)
        x_up1_conv1 = self.jh(x_up1_conv1)
        x_up1_conv2 = self.up_conv2(x_up1_conv1)
        x_up1_conv2 = self.jh(x_up1_conv2)
        x_up2 = self.up2(x_up1_conv2)
        x_up2_cat = torch.cat((x_up2, x6), dim=1)  #  x6  x_up2
        x_up2_conv3 = self.up_conv3(x_up2_cat)
        x_up2_conv3 = self.jh(x_up2_conv3)
        x_up2_conv4 = self.up_conv4(x_up2_conv3)
        x_up2_conv4 = self.jh(x_up2_conv4)
        x_up3 = self.up3(x_up2_conv4)
        x_up3_cat = torch.cat((x_up3, x4), dim=1)  #  x4  x_up3
        x_up3_conv5 = self.up_conv5(x_up3_cat)
        x_up3_conv5 = self.jh(x_up3_conv5)
        x_up3_conv6 = self.up_conv6(x_up3_conv5)
        x_up3_conv6 = self.jh(x_up3_conv6)
        x_up4 = self.up4(x_up3_conv6)
        x_up4_cat = torch.cat((x_up4, x2), dim=1)  #  x2  x_up4
        x_up4_conv7 = self.up_conv7(x_up4_cat)
        x_up4_conv7 = self.jh(x_up4_conv7)
        x_up4_conv8 = self.up_conv8(x_up4_conv7)
        x_up4_conv8 = self.jh(x_up4_conv8)
        x_up4_conv9 = self.up_conv9(x_up4_conv8)
        return x_up4_conv9




# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model2 = cqm_unetLayer(input_channel=1, padd=1, padd2=0)
# # model2.to('cuda')
# # summary(model2, input_size=(1, 512, 512))
# # cc = torch.randn(1, 1, 512, 512).to(device)
# # input_tensor = model2(cc)
# cc = torch.randn(100, 5, 1, 512, 512)
# y = model2(cc)

# viz_graph = make_dot(y.mean(), params=dict(model2.named_parameters()))
# viz_graph.format = "png"
# viz_graph.directory = "../data.png"
# viz_graph.view()
