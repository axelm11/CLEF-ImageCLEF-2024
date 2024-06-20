
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


MODEL_DIR = "/path_to/directory/"
filenameCSV="EDV2performaces_filt.csv"
data_train= "/path_to/train_filt/"
data_valid= "/path_to//valid_filt/"



df_train=pd.read_csv("/path_to//train_filt.csv",sep=",")
df_valid=pd.read_csv("/path_to/valid_filt.csv",sep=",")

df_train

df_valid

df_train["image_path"]=data_train+df_train.ID+".jpg"
df_valid["image_path"]=data_valid+df_valid.ID+".jpg"

cuis_list=[]
for (i,row) in df_train.iterrows():
    for cui in row["CUIs"].split(";"):
        if not cui in cuis_list:
            cuis_list.append(cui)

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
        img = Image.open(obj.image_path).convert("RGB") # load image
        img = self.transform(img)

        return (img, label_enc)

def get_val_preprocessing(img_size):
    return transforms.Compose([
                transforms.Resize(int(img_size * 1.25)), # Expand IMAGE_SIZE before center crop
                transforms.CenterCrop(int(img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

class IdentityTransform:
    def __call__(self, x):
        return x


def get_train_augmentation_preprocessing(img_size, rand_aug=False):
    return transforms.Compose([
                transforms.Resize(int(img_size * 1.25)), # Expand IMAGE_SIZE before random crop
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomCrop((img_size, img_size)), # Random Crop to IMAGE_SIZE
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

# shuffle
train_data = df_train.sample(frac=1, random_state=1).reset_index(drop=True)
valid_data = df_valid.sample(frac=1, random_state=1).reset_index(drop=True)

imgsize_train=224
imgsize_val=224

train_aug_preprocessing = get_train_augmentation_preprocessing(imgsize_train, True)
val_preprocessing = get_val_preprocessing(imgsize_val)

train_dataset= ROCOv2Dataset(df_train, transform=train_aug_preprocessing)
valid_dataset = ROCOv2Dataset(df_valid, transform=val_preprocessing)

BATCH_SIZE=128

train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=9, drop_last=True, pin_memory=True)
valid_loader = DataLoader(dataset=valid_dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=9, drop_last=False, pin_memory=True)

liveloss = PlotLosses()
m = nn.Sigmoid()



#for lr in [1e-1,1e-2,1e-3,1e-4,1e-5]:
for lr in [1e-3]:
#    for opt in ["adam","sgd","rmsprop"]:
    for opt in ["adam"]:
        use_amp = True
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        batchsize_factor=1
        val_interval = 1
        epoch_loss_values = []
        max_epochs = 40
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model=create_model('efficientnet_b0', pretrained=True, num_classes=NUM_CLASSES, drop_path_rate=0.2)
        loss_function = nn.MultiLabelSoftMarginLoss()
        if opt == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif opt== "sgd":
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
            labels_sum = np.empty([0,NUM_CLASSES])
            pred_sum = np.empty([0,NUM_CLASSES])
            labels_sum_val = np.empty([0,NUM_CLASSES])
            pred_sum_val = np.empty([0,NUM_CLASSES])
            for batch_idx,(inputs, labels) in enumerate(train_loader):
                step += 1
                inputs = inputs.cuda()
                labels = labels.cuda()
                with autocast(device_type = 'cuda', enabled = True): 
                    outputs = model(inputs)
                    loss = loss_function(outputs,labels)
                    loss = loss / batchsize_factor
                    scaler.scale(loss).backward()
                    output_sig = m(outputs)
                    output_sig_class = (output_sig>=0.5).long()
                if (step+1) % batchsize_factor == 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none = True)
                epoch_loss += (loss.item() * batchsize_factor)
                epoch_len = len(train_dataset) // train_loader.batch_size
                labels_sum = np.append(labels_sum, labels.detach().cpu().numpy(), axis = 0)

                pred_sum = np.append(pred_sum, output_sig_class.detach().cpu().numpy(), axis = 0)

            logs['log loss'] = epoch_loss / len(train_dataset)
            logs['F1 macro'] = f1_score(labels_sum.T, pred_sum.T, average='macro')
            epoch_loss_values.append(epoch_loss)
            if (epoch + 1) % val_interval == 0:
                model.eval()
                with torch.no_grad():
                    num_correct = 0.0
                    metric_count = 0
                    for batch_idx, (inputs, labels) in enumerate(valid_loader):
                        inputs = inputs.cuda()
                        labels = labels.cuda()
                        with autocast(device_type = 'cuda', enabled = True):
                            outputs = model(inputs)
                            loss = loss_function(outputs, labels)
                            loss = loss / batchsize_factor
                            output_sig = m(outputs)
                            output_sig_class = (output_sig >= 0.5).long()
                        labels_sum_val = np.append(labels_sum_val, labels.detach().cpu().numpy(), axis = 0)
                        pred_sum_val = np.append(pred_sum_val, output_sig_class.detach().cpu().numpy(), axis = 0)
                        epoch_loss_val += loss.item()
                    logs['val_log loss'] = epoch_loss_val / len(valid_dataset)
                    logs['val_F1 macro'] = f1_score(labels_sum_val.T, pred_sum_val.T, average='macro')
            #liveloss.update(logs)
            #liveloss.send()
            torch.save(model.state_dict(), MODEL_DIR+"EDV2_model_"+str(opt)+"_"+str(lr)+"_"+str(epoch)+".pth")
            d = {'optimizer': [opt], 'LR': [lr], 'Epoch':[epoch], "Epoch-F1": [f1_score(labels_sum_val.T, pred_sum_val.T, average='macro') * 100],"Epoch-Loss":[epoch_loss_val / len(valid_dataset)]}

            df = pd.DataFrame(data = d)

            if os.path.isfile(filenameCSV):
                df.to_csv(filenameCSV, mode = 'a', header = False)
            else:
                df.to_csv(filenameCSV, mode = 'w', header = True)
            print("--- %s seconds ---" % (time.time() - start_time))