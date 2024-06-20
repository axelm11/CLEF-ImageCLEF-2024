
#import packages
import os, time
import pandas as pd
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import autocast
import torchvision.transforms as transforms
from timm.models import create_model
from sklearn.metrics import f1_score
from livelossplot import PlotLosses

DATA_DIR=os.getcwd()+"/"
MODEL_DIR = DATA_DIR +"results/"
filenameCSV=MODEL_DIR+"performaces.csv"

data_train=DATA_DIR+"path_to/train/"
data_valid=DATA_DIR+"path_to/valid/"



#load dataset with IDs
df_train=pd.read_csv(DATA_DIR+"path_to/train_concepts.csv",sep=",")
df_valid=pd.read_csv(DATA_DIR+"path_to/valid_concepts.csv",sep=",")

df_train["image_path"]=data_train+df_train.ID+".jpg"
df_valid["image_path"]=data_valid+df_valid.ID+".jpg"

df_train_merged=pd.concat([df_train,df_valid])

cuis_list=[]
for (i,row) in df_train_merged.iterrows():
    for cui in row["CUIs"].split(";"):
        if not cui in cuis_list:
            cuis_list.append(cui)
##np.save('my_list.npy', cuis_list)

#cuis_list = np.load('my_list.npy').tolist()

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

NUM_CLASSES=len(cuis_list)

class ROCOv2Dataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform # Image augmentation pipeline

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        obj = self.data.iloc[index] # get instance
        label = obj.CUIs # get label
        label_enc=torch.zeros((NUM_CLASSES))
        for cui in label.split(";"):
            label_enc[cuis_list.index(cui)]=1
        # img. augmentation
        img = Image.open(obj.image_path).convert("RGB") # load image
        img = self.transform(img)

        return (img, label_enc)

class IdentityTransform:
    def __call__(self, x):
        return x


# train data augmentation/ preprocessing pipeline
def get_train_augmentation_preprocessing(img_size, rand_aug=False):
    print(f'IMG_SIZE_TRAIN: {img_size}, RandAug: {rand_aug}')
    return transforms.Compose([
                transforms.Resize(int(img_size * 1.25)), # Expand IMAGE_SIZE before random crop
                #RandomGridShuffle(grid=TRANSFORMS['n_grid']),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomCrop((img_size, img_size)), # Random Crop to IMAGE_SIZE
                #transforms.RandAugment(num_ops=2, magnitude=9) if rand_aug else IdentityTransform(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

# shuffle
df_train_merged = df_train_merged.sample(frac=1, random_state=1).reset_index(drop=True)

imgsize_train=224
imgsize_val=224

train_aug_preprocessing = get_train_augmentation_preprocessing(imgsize_train, True)

train_dataset= ROCOv2Dataset(df_train_merged, transform = train_aug_preprocessing)

BATCH_SIZE=128

train_loader = DataLoader(dataset = train_dataset, shuffle = True, batch_size = BATCH_SIZE, num_workers = 9, drop_last = True, pin_memory = True)

m = nn.Sigmoid()
lr=0.001
opt="adam"
use_amp = True
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
batchsize_factor=1
val_interval = 1
epoch_loss_values = []
max_epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=create_model('efficientnet_b0', pretrained=True, num_classes=NUM_CLASSES, drop_path_rate=0.2)
loss_function = torch.nn.MultiLabelSoftMarginLoss()
if opt =="adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
elif opt=="sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
else:
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
model = model.to(device)
for epoch in range(max_epochs):
    start_time = time.time()
    model.train()
    logs = {}
    epoch_loss = 0
    epoch_loss_val = 0
    step = 0
    labels_sum = np.empty([0, NUM_CLASSES])
    pred_sum = np.empty([0, NUM_CLASSES])
    labels_sum_val = np.empty([0, NUM_CLASSES])
    pred_sum_val = np.empty([0, NUM_CLASSES])
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        step += 1
        inputs = inputs.cuda()
        labels = labels.cuda()
        with autocast(device_type = 'cuda', enabled = True):
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss = loss / batchsize_factor
            scaler.scale(loss).backward()
            output_sig = m(outputs)
            output_sig_class = (output_sig >= 0.5).long()
        if (step + 1) % batchsize_factor == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        epoch_loss += (loss.item() * batchsize_factor)
        epoch_len = len(train_dataset) // train_loader.batch_size
        labels_sum = np.append(labels_sum, labels.detach().cpu().numpy(), axis = 0)

        pred_sum = np.append(pred_sum, output_sig_class.detach().cpu().numpy(), axis = 0)
        print("--- %s seconds ---" % (time.time() - start_time))

    logs['log loss'] = epoch_loss / len(train_dataset)
    logs['F1 macro'] = f1_score(labels_sum.T, pred_sum.T, average = 'macro')
    epoch_loss_values.append(epoch_loss)
    torch.save(model.state_dict(), MODEL_DIR + "model_" + str(opt) + "_" + str(lr) + "_" + str(epoch) + ".pth")
    print("--- %s seconds ---" % (time.time() - start_time))