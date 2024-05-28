import os
from prefetch_generator import BackgroundGenerator
import pickle
import time
import numpy as np
import torch
from torch.optim import lr_scheduler
from torch.optim.swa_utils import SWALR
from torch.utils.data import DataLoader, Dataset, Subset
import Uconvnext_CQM_BNX2

from torch import nn, optim
from torchsummary import summary
from tqdm import tqdm



class CustomDataset(Dataset):
    def __init__(self, data_folder1, data_folder2):
        self.data_folder1 = data_folder1
        self.data_folder2 = data_folder2

        self.file_list1 = sorted(os.listdir(data_folder1), key=lambda x: int(x.split('.')[0]))
        self.file_list2 = sorted(os.listdir(data_folder2), key=lambda x: int(x.split('.')[0]))

    def __len__(self):
        return len(self.file_list1)

    def __getitem__(self, idx):
        file_path1 = os.path.join(self.data_folder1, self.file_list1[idx])
        file_path2 = os.path.join(self.data_folder2, self.file_list2[idx])
        data1 = np.load(file_path1)
        data1 = torch.tensor(data1, dtype=torch.float32)
        data2 = np.load(file_path2)
        data2 = torch.tensor(data2, dtype=torch.float32)

        return data1, data2


# 数据集的路径，包含了 1.npy 到 10000.npy
data_folder1 = '..\\X'  #
data_folder2 = '..\\Y'


custom_dataset = CustomDataset(data_folder1, data_folder2)


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())



batch_size = 1
pici = 2
if pici == 2:
    train_dataset = Subset(custom_dataset, range(0, 6250))  # 1,1251 #0,6251
    val_dataset = Subset(custom_dataset, range(6251, 7555))  # 6251,7555a
    yzjz = 1304

elif pici == 10:
    train_dataset = Subset(custom_dataset, range(0, 1251))  # 1,1251 #0,6251
    val_dataset = Subset(custom_dataset, range(1251, 1511))  # 6251,7555
    yzjz = 260
val_loader = DataLoaderX(val_dataset, batch_size=batch_size, shuffle=False)

tra_loader = DataLoaderX(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

net2 = Uconvnext_CQM_BNX2.cqm_unetLayer(10)


start_time = []


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()

        self.transconv1 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=10, out_channels=10, kernel_size=(2, 3, 3), stride=(2, 1, 1),
                               padding=(0, 1, 1)),
            nn.BatchNorm3d(10,affine=False),
            nn.ReLU()
        )

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=10, out_channels=10, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(10,affine=False),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=10, out_channels=10, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(10,affine=False),
            nn.ReLU()
        )

        self.final_conv = nn.Conv3d(in_channels=10, out_channels=10, kernel_size=(1, 1, 1), stride=(1, 1, 1))

    def forward(self, x):
        x = self.transconv1(x)

        x = self.conv1(x)

        x = self.conv2(x)

        x = self.final_conv(x)
        return x.squeeze(dim=2)


net = nn.Sequential(CustomModel(), net2)
net = net.to(torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = net.to('cuda')
losses = []
losses2 = []
times = []
start_time = time.time()

optimizer = optim.Adam(net.parameters(), lr=0.0001)
criterion = torch.nn.MSELoss()

criterion = criterion.to(device)

num_epochs = 300
val_interval = 4
best_rmse = 0.052
start_epoch = 0



swa_model = torch.optim.swa_utils.AveragedModel(net)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

swa_start = 160
swa_scheduler = SWALR(optimizer, swa_lr=0.00005)
for epoch in range(start_epoch + 1, num_epochs):
    net.train()
    running_loss = 0.0
    for ii, (inputs, targets) in tqdm(enumerate(tra_loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = inputs.squeeze(), targets.squeeze()  # 移动到GPU
        inputs, targets = torch.permute(inputs, [4, 3, 2, 0, 1]), torch.permute(targets, [3, 2, 0, 1])
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    checkpoint = {
        "net": net.state_dict(),
        'optimizer': optimizer.state_dict(),
        "epoch": epoch
    }
    if epoch > swa_start:
        swa_model.update_parameters(net)
        swa_scheduler.step()
    else:
        scheduler.step()
    losses.append(loss.item())
    times.append(time.time() - start_time)
    end_time = time.time()
    total_time = end_time - start_time
    print(f'Epoch [{epoch}] - Total Time: {total_time:.2f} seconds - Loss: {loss.item():.10f} ',
          optimizer.state_dict()['param_groups'][0]['lr'])

    scheduler.step()

    if epoch % val_interval == 0 and epoch > 0:
        val_loss = 0.0
        with torch.no_grad():
            for ii, (inputs, targets) in tqdm(enumerate(val_loader)):
                inputs, targets = inputs.squeeze(), targets.squeeze()  # 移动到GPU
                inputs, targets = torch.permute(inputs, [4, 3, 2, 0, 1]), torch.permute(targets, [3, 2, 0, 1])
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                val_loss += criterion(outputs, targets)

        val_rmse = torch.sqrt(val_loss / 1304)  # 1304 or 260
        current_rmse = val_rmse.item()
        losses2.append(val_loss.item())
        print(f'Validation RMSE: {val_rmse:.6f}')
        if current_rmse < best_rmse:
            best_rmse = current_rmse
            torch.save(checkpoint, 'UconvBNX2'+str(pici)+'_GN205_SWA_cheakpoint_E' + str(epoch) + '_R' + str(best_rmse) + '.pth')
        with open('UconvBNX2_Batch_d0.7'+str(pici)+'_GN205_swa_losses.pkl', 'wb') as file:
            pickle.dump(losses, file)
        with open('UconvBNX2_Batch_d0.7'+str(pici)+'_GN205_swa_losses2.pkl', 'wb') as file:
            pickle.dump(losses2, file)


torch.optim.swa_utils.update_bn(tra_loader, swa_model)
torch.save(swa_model.state_dict(), 'Swamodel_Batch'+str(pici)+'_BN_model_Convnext.pth')
