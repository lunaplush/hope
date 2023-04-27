import skimage.io as io
#https://www.dropbox.com/s/k88qukc20ljnbuo/PH2Dataset.rar
#io.use_plugin("pillow")
from PIL import Image
import os
import matplotlib.pyplot as plt
from skimage.io import imread

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.optim as optim
from time import time
from IPython.display import clear_output
import pickle
import gc
# Очистка памяти gpu - используется насколько я поняла, если прогоны обучения сети неуспешные,   то начинаются проблемы с памятью
#import gc
# def clear_gpu_memory():
#     torch.cuda.empty_cache()
#     variables = gc.collect()
#     del variables
n = gc.collect()
print("Number of unreachable objects collected by GC:", n)
from matplotlib import rcParams
rcParams['figure.figsize'] = (15, 4)


# Путь к папке с изображениями
path = 'PH2Dataset/PH2 Dataset images/IMD140/IMD140_lesion/'

# for filename in os.listdir(path):
#     if filename.endswith('.bmp'):
#         with Image.open(os.path.join(path, filename)) as im:
#             im.convert('RGB').save(os.path.join(path, filename[:-4] + '.jpg'))
# p_image='PH2Dataset/PH2 Dataset images/IMD002/IMD002_lesion/IMD002_lesion.bmp'
# plt.subplot(1, 1, 1)
# plt.imshow(imread(p_image))
# plt.show()

import torchvision, torch
from score_segmentation import iou_pytorch
from losses_segmentation import bce_loss, dice_loss, focal_loss
from net_SegNetSmall import SegNetSmall
from net_SegNet import SegNet
from net_Unet import UNet
import pandas as pd


images = []
lesions = []
from skimage.io import imread
import os
root = 'PH2Dataset'

for root, dirs, files in os.walk(os.path.join(root, 'PH2 Dataset images')):
    if root.endswith('_Dermoscopic_Image'):
        images.append(imread(os.path.join(root, files[0])))
    if root.endswith('_lesion'):
        lesions.append(imread(os.path.join(root, files[0])))


from skimage.transform import resize
size = (256, 256)

X = [resize(x, size, mode='constant', anti_aliasing=True,) for x in images]
Y = [resize(y, size, mode='constant', anti_aliasing=False) > 0.5 for y in lesions]

import numpy as np
X = np.array(X, np.float32)
Y = np.array(Y, np.float32)
print(f'Loaded {len(X)} images')


ix = np.random.default_rng(seed=10).choice(len(X), len(X), False)

tr, val, ts = np.split(ix, [100, 150])

from torch.utils.data import DataLoader
# 0 - Утонить batch size
batch_size = 10
data_tr = DataLoader(list(zip(np.rollaxis(X[tr], 3, 1), Y[tr, np.newaxis])),
                     batch_size=batch_size, shuffle=True)
data_val = DataLoader(list(zip(np.rollaxis(X[val], 3, 1), Y[val, np.newaxis])),
                      batch_size=batch_size, shuffle=True)
data_ts = DataLoader(list(zip(np.rollaxis(X[ts], 3, 1), Y[ts, np.newaxis])),
                     batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict(model, data):
    """не используется пока"""
    model.eval()  # testing mode
    Y_pred = [model(X_batch) for X_batch, _ in data]
    Y_pred = torch.sigmoid(Y_pred)#>0.5
    return np.array(Y_pred)



def score_model(model, metric, data):
    model.eval()  # testing mode
    scores = 0
    for X_batch, Y_label in data:
        Y_pred = model(X_batch.to(device))
        Y_pred = Y_pred > 0.5
        scores += metric(Y_pred, Y_label.to(device)).mean().item()

    return scores/len(data)



def train(model, opt, loss_fn, epochs, start_epoch_num, data_tr, data_val, data_ts):
    X_val, Y_val = next(iter(data_val))
    X_test, Y_test = next(iter(data_ts))
    next_epoch_num = start_epoch_num
    analysis_dict = {"loss_train": [], "loss_test": [], "loss_val": [],
                     "score_train": [], "score_test": [], "score_val": []
                     }  # Для сохранения лосов и метрик

    for epoch in range(epochs):
        tic = time()
        print('* Epoch %d/%d' % (next_epoch_num+1, start_epoch_num+epochs))

        avg_loss = 0
        model.train()  # train mode
        for X_batch, Y_batch in data_tr:
            # data to device
            X_batch.to(device)
            Y_batch.to(device)

            # set parameter gradients to zero
            opt.zero_grad()

            # forward
            Y_pred = model(X_batch)
            loss = loss_fn(y_real=Y_batch, y_pred=Y_pred)# forward-pass
            loss.backward()  # backward-pass
            opt.step()  # update weights

            # calculate loss to show the user
            avg_loss += loss / len(data_tr)
        toc = time()
        print('loss: %f' % avg_loss)
        analysis_dict["loss_train"].append(avg_loss.detach().numpy())


        # show intermediate results
        model.eval()  # testing mode
        Y_hat_val = model(X_val).to("cpu")# detach and put into cpu
        score_val = score_model(model, iou_pytorch, data_val)
        print(f"\tIoU metric {score_val} on epoch {next_epoch_num+1}")        # Visualize tools

        analysis_dict["loss_val"].append(loss_fn(y_real=Y_val, y_pred=Y_hat_val).detach().numpy())
        Y_hat_val = Y_hat_val.detach().numpy()
        analysis_dict["score_val"].append(score_val)

        Y_hat_test = model(X_test).to("cpu")
        score_test = score_model(model, iou_pytorch, data_ts)
        analysis_dict["score_test"].append(score_test)
        analysis_dict["loss_test"].append(loss_fn(y_real=Y_test, y_pred=Y_hat_test).detach().numpy())

        score_train = score_model(model, iou_pytorch, data_tr)
        analysis_dict["score_train"].append(score_train)


        clear_output(wait=True)
        for k in range(6):
            plt.subplot(2, 6, k+1)
            plt.imshow(np.rollaxis(X_val[k].numpy(), 0, 3), cmap='gray')
            plt.title('Real')
            plt.axis('off')
            plt.subplot(2, 6, k+7)
            plt.imshow(Y_hat_val[k, 0], cmap='gray')
            plt.title('Output')
            plt.axis('off')
        plt.suptitle('%d / %d - loss: %f' % (next_epoch_num+1, start_epoch_num+epochs, avg_loss))

        plt.savefig("epoch{}.jpg".format(next_epoch_num+1))
        next_epoch_num += 1
        gc.collect()

    return pd.DataFrame(analysis_dict)

# ---- Настройка следующего цикла обучения -----------
# 1 - Установить название модели
#model_name = "segnetsmall_bce"
#model_name = "segnetsmall_dice"
#model_name = "segnetsmall_focal"
model_name = "unet_bce"
#model_name = "unet_bce"
# 2 -  В соответсвии с названием модели выбрать класс нейронной сети
#model = SegNetSmall().to(device)
model = UNet().to(device)
# 3 - Выбрать функцию ошибки для обучения
criterion = bce_loss
#criterion = dice_loss
#criterion= focal_loss
# 4 - Установить флаг - начала обучения FIRST_STEP True или False -  если продолжаем обучение
FIRST_STEP = True
# 5 - Установить сколько эпох за этот подход хотим обучить
max_epochs = 1

if FIRST_STEP:
    f = open("epoch.num", "wb")
    epoch_num = 0
    pickle.dump(epoch_num, f)
    if not os.path.exists(model_name+"_epochs"):
        os.mkdir(model_name+"_epochs")
else:
    f = open("epoch.num", "rb")
    epoch_num = pickle.load(f)
    model.load_state_dict(torch.load(model_name+".net"))
f.close()

lr = 1e-4
if epoch_num > 20:
    lr *= 0.1
if epoch_num > 40:
    lr *= 0.01
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
work_path = os.getcwd()
os.chdir(model_name+"_epochs")
analysis_data = train(model, optimizer, criterion, max_epochs, epoch_num, data_tr, data_val, data_ts)
os.chdir(work_path)


#----------------Завершение цикла обучения
f = open("epoch.num", "wb")
step = epoch_num + max_epochs
pickle.dump(step, f)
f.close()
torch.save(model.state_dict(), model_name+".net")
if FIRST_STEP:
    analysis_data.to_csv(model_name + ".csv")
else:
    analysis_data.to_csv(model_name+".csv", mode="a", header=False) #index_label=range()


if False:
    # Убедимся, что сеть корректна, размеры слоев согласованы между собой и на выходне картинка размером 256х256
    model = SegNet()
    model.eval()
    a = model(torch.FloatTensor(np.rollaxis(X[[0]], 3, 1)))
    print(a[0].shape)
    im = a[0].detach().numpy()
    im = np.rollaxis(im, 0, 3)
    plt.imshow(im)
    plt.show()
    # Без обучания пока ничего не сегментируется



