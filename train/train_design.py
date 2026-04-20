import os
import time
import torch
import torch.utils.data as Data
from os.path import join
import numpy as np
import torch.nn as nn
from torch import optim
from torch.utils.data import sampler
from torch.utils.data import DataLoader
from networks.NET import NET
from data.SdfSamples.Self_Design.generate import  marching_cube, save_mesh_as_html, save_mesh_as_stl
from Utils import _cartesian_product



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root = join(os.path.dirname(os.path.realpath(__file__)), '..')
log_dir = root + '/pretrained/ellipsoid_' + time.strftime('%m_%d_%H%M%S')
os.makedirs(log_dir, exist_ok=True)



def train(xx, yy, epoch_num=200):
    latent_size = int(xx.shape[1])

    net = NET(latent_size).to(device)


    batch_size = 512
    validation_split = .2
    shuffle_dataset = True
    random_seed = 42



    xx = xx.double().to(device)
    yy = yy.double().to(device)
    xx.requires_grad = False
    yy.requires_grad = False
    dataset = Data.TensorDataset(xx, yy)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = sampler.SubsetRandomSampler(train_indices)
    valid_sampler = sampler.SubsetRandomSampler(val_indices)

    # 在DataLoader中使用num_workers和pin_memory
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)
    # print(yy.t().detach().cpu())


    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.1)
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # 设置每处理多少批次后计算并输出误差的百分比
    report_percentage = 0.1  # 例如，每处理10%的批次后报告一次

    for epoch in range(epoch_num):
        net.train()  # 确保网络处于训练模式
        total_train_loss = 0.0
        total_train_samples = 0
        batch_count = 0

        # 计算间隔批次数
        total_batches = len(train_loader)
        report_interval = int(total_batches * report_percentage)
        if report_interval == 0:
            report_interval = 1  # 防止report_interval为0

        for data in train_loader:
            inputs, labels = data
            inputs.requires_grad = False
            labels.requires_grad = False

            optimizer.zero_grad()
            outputs = net(inputs.double())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 累计训练误差
            total_train_loss += loss.item() * inputs.size(0)
            total_train_samples += inputs.size(0)
            batch_count += 1

            # 每处理report_interval个批次后计算并输出当前的训练误差和验证误差
            if batch_count % report_interval == 0 or batch_count == total_batches:
                current_train_loss = total_train_loss / total_train_samples if total_train_samples else 0

                # 验证过程
                net.eval()  # 设置为评估模式
                total_val_loss = 0.0
                total_val_samples = 0
                with torch.no_grad():
                    for val_data in validation_loader:
                        val_inputs, val_labels = val_data
                        val_outputs = net(val_inputs.double())
                        val_loss = criterion(val_outputs, val_labels)
                        total_val_loss += val_loss.item() * val_inputs.size(0)
                        total_val_samples += val_inputs.size(0)

                current_val_loss = total_val_loss / total_val_samples if total_val_samples else 0
                print(f'Epoch {epoch}, Batch {batch_count // report_interval}, '
                      f'Current Training Loss: {current_train_loss}, '
                      f'Current Validation Loss: {current_val_loss}')

                net.train()  # 重新设置为训练模式

        # 保存模型
        torch.save({'weights': net.state_dict()}, join(log_dir, f'ellipsoid_{epoch}.pth'))

        # 可选的预测功能
        if epoch > 0:
            predict(latent_size, log_dir, epoch)

    print('Finished Training')





def data():
    file_dir = '/home/hbliu/Desktop/DeepSDF_EIT/data/SdfSamples/Self_Design/ellipsoid/ellipsoid/'
    filenames = os.listdir(file_dir)

    xx = torch.tensor([])
    yy = torch.tensor([])

    for file in filenames:
        data = np.load(file_dir + file)
        para_points = data['arr_0'][:, :-1]
        sdf = data['arr_0'][:, -1]

        xx = torch.cat([xx, torch.from_numpy(para_points)], dim=0)
        yy = torch.cat([yy, torch.from_numpy(sdf)], dim=0)

    xx = xx.double()
    yy = yy.double().unsqueeze(1)

    return xx,yy




def predict(latent_size, log_dir, epoch):


    # spheres = []
    # spheres.append(np.array([0.6, 0, 0, 0.4]))
    # spheres.append(np.array([0, 0.6, 0, 0.4]))
    # spheres.append(np.array([0, 0, 0.6, 0.4]))
    # para = np.array(spheres)

    # x = np.random.uniform(0.01, 0.9)
    # y = np.random.uniform(0.01, 0.9)
    # z = np.random.uniform(0.01, 0.9)
    x = 0.2
    y = 0.4
    z = 0.6
    para = np.array((x, y, z))
    saved_model_dir = log_dir + '/' + 'ellipsoid_' + str(epoch) + '.pth'



    bound = ((-1.2, -1.2, -1.2), (1.2, 1.2, 1.2))
    Nr = 32
    (x0, y0, z0), (x1, y1, z1) = bound
    X = np.linspace(x0, x1, Nr)
    Y = np.linspace(y0, y1, Nr)
    Z = np.linspace(z0, z1, Nr)
    P = _cartesian_product(X, Y, Z)

    net = NET(latent_size).to(device)
    saved_model_state = torch.load(saved_model_dir)
    net.load_state_dict(saved_model_state['weights'])

    para = para.reshape(1, -1)
    para = para.repeat(P.shape[0], 0)
    input = torch.from_numpy(np.hstack((para, P))).to(device)

    volume = net(input.double()).detach().cpu().numpy()

    verts, faces = marching_cube(volume, bound, Nr, level = 0.05)

    save_mesh_as_html(verts, faces, log_dir + '/' + 'predict_' + str(epoch) + '.html')

    save_mesh_as_stl(verts, faces, log_dir + '/' + 'predict_' + str(epoch) + ".stl")





if __name__ == "__main__":

    # xx, yy = data()
    #
    # print("data loaded")

    # train(xx, yy, epoch_num = 100)

    predict(6, '/home/hbliu/Desktop/DeepSDF_EIT/pretrained/ellipsoid', 6)