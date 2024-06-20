
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
from sklearn.metrics import classification_report
import shutil

DATA_DIR=os.getcwd()+"/"
MODEL_DIR = DATA_DIR +"path_to_directory/"
data_train=DATA_DIR+"path_to/train/"
data_valid=DATA_DIR+"path_to/valid/"
data_test=DATA_DIR+"path_to_test/"
model_path = MODEL_DIR+"/path_to_weights_original_model/EDV2_model_adam_0.001.pth"
model_path1 = DATA_DIR + "/path_to_weights/EDV2_model_adam_0.001.pth"
model_path2 = DATA_DIR + "/path_to_weights2/EDV2_model_adam_0.001.pth"
model_path3 = DATA_DIR + "/path_to_weights3/EDV2_model_adam_0.001.pth"
model_path4 = DATA_DIR + "/path_to_weights4/EDV2_model_adam_0.001.pth"


#load dataset with IDs
df_train=pd.read_csv(DATA_DIR+"path_to/train_concepts.csv",sep=",")
df_valid=pd.read_csv(DATA_DIR+"path_to/valid_concepts.csv",sep=",")

df_train["image_path"]=data_train+df_train.ID+".jpg"
df_valid["image_path"]=data_valid+df_valid.ID+".jpg"

cuis_list=[]
for (i,row) in df_train.iterrows():
    for cui in row["CUIs"].split(";"):
        if not cui in cuis_list:
            cuis_list.append(cui)
cuiscomplete=np.array(cuis_list)
            
df_train1=pd.read_csv(DATA_DIR+"path_to/train_1_filt.csv",sep=",")
df_valid1=pd.read_csv(DATA_DIR+"path_to/valid_1_filt.csv",sep=",")
cuis_list1 = []
cuis_index = np.array([1])
for (i,row) in df_train1.iterrows():
    for cui in row["CUIs"].split(";"):
        if not cui in cuis_list1:
            cuis_list1.append(cui)
NUM_CLASSES1=len(cuis_list1)

df_train2=pd.read_csv(DATA_DIR+"path_to/train_2_filt.csv",sep=",")
df_valid2=pd.read_csv(DATA_DIR+"path_to/valid_2_filt.csv",sep=",")
cuis_list2 = []
cuis_index2 = np.array([2])
for (i,row) in df_train2.iterrows():
    for cui in row["CUIs"].split(";"):
        if not cui in cuis_list2:
            cuis_list2.append(cui)
NUM_CLASSES2=len(cuis_list2)

df_train3=pd.read_csv(DATA_DIR+"path_to/train_3_filt.csv",sep=",")
df_valid3=pd.read_csv(DATA_DIR+"path_to/valid_3_filt.csv",sep=",")
cuis_list3 = []
cuis_index3 = np.array([3])
for (i,row) in df_train3.iterrows():
    for cui in row["CUIs"].split(";"):
        if not cui in cuis_list3:
            cuis_list3.append(cui)
NUM_CLASSES3=len(cuis_list3)


df_train4=pd.read_csv(DATA_DIR+"path_to/train_4_filt.csv",sep=",")
df_valid4=pd.read_csv(DATA_DIR+"path_to/valid_4_filt.csv",sep=",")
cuis_list4 = []
cuis_index4 = np.array([4])
for (i,row) in df_train4.iterrows():
    for cui in row["CUIs"].split(";"):
        if not cui in cuis_list4:
            cuis_list4.append(cui)
NUM_CLASSES4=len(cuis_list4)

#para sacar el histograma 

df_train1['CUIs'] = df_train1['CUIs'].str.split(';')
CUIs_counts = [g for gen in df_train1['CUIs'] for g in gen]
c= pd.Series(CUIs_counts).value_counts()
d= df_train['CUIs'].str.len().plot.hist(bins=60)


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
        pp=obj.ID
        label_enc=torch.zeros((NUM_CLASSES))
        for cui in label.split(";"):
            label_enc[cuis_list.index(cui)]=1
        img = Image.open(obj.image_path).convert("RGB") # load image
        img = self.transform(img)

        return (img, label_enc,pp)

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

BATCH_SIZE=1

train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=9, drop_last=True, pin_memory=True)
valid_loader = DataLoader(dataset=valid_dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=9, drop_last=False, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=create_model('efficientnet_b0', pretrained=True, num_classes=NUM_CLASSES, drop_path_rate=0.2)  
state_dict = torch.load(model_path)
model.load_state_dict(state_dict)
model = model.to(device)             
model.eval()


model1=create_model('efficientnet_b0', pretrained=True, num_classes=NUM_CLASSES1, drop_path_rate=0.2)
state_dict1 = torch.load(model_path1)
model1.load_state_dict(state_dict1)
model1 = model1.to(device)
model1.eval()

model2=create_model('efficientnet_b0', pretrained=True, num_classes=NUM_CLASSES2, drop_path_rate=0.2)
state_dict2 = torch.load(model_path2)
model2.load_state_dict(state_dict2)
model2 = model2.to(device)
model2.eval()

model3=create_model('efficientnet_b0', pretrained=True, num_classes=NUM_CLASSES3, drop_path_rate=0.2)
state_dict3 = torch.load(model_path3)
model3.load_state_dict(state_dict3)
model3 = model3.to(device)
model3.eval()

model4=create_model('efficientnet_b0', pretrained=True, num_classes=NUM_CLASSES4, drop_path_rate=0.2)
state_dict4 = torch.load(model_path4)
model4.load_state_dict(state_dict4)
model4 = model4.to(device)
model4.eval()

loss_function = nn.MultiLabelSoftMarginLoss()
batchsize_factor=1
m = nn.Sigmoid()
labels_sum_val = np.empty([0,NUM_CLASSES])
pred_sum_val = np.empty([0,NUM_CLASSES])
epoch_loss_val = 0
logs = {}

output_cuis=[]
with torch.no_grad():
    num_correct = 0.0
    metric_count = 0
    for batch_idx, (inputs, labels,pp) in enumerate(valid_loader):
        inputs = inputs.cuda()
        labels = labels.cuda()         
        with autocast(device_type = 'cuda', enabled = True):                            
            outputs = model(inputs)                            
            loss = loss_function(outputs, labels)                            
            loss = loss / batchsize_factor                            
            output_sig = m(outputs)                            
            o1 = (output_sig >= 0.5).long().detach().cpu().numpy()     
            
            ones = np.where(o1 == 1)[1]
            if len(ones) > 0:
                for index in ones:   
                    if index in cuis_index:
                        outputs1 = model1(inputs)
                        output_sig1 = m(outputs1)
                        om1 = (output_sig1 >= 0.5).long().detach().cpu().numpy()
                        ones1 = np.where(om1 == 1)[1]
                        if len(ones1)>0:
                            for ind in ones1:
                                CUIadd=cuis_list1[ind]
                                ind1 = np.where( cuiscomplete == CUIadd)[0]
                                o1[:,ind1]=1
                                
                    if index in cuis_index2:
                        outputs2 = model2(inputs)
                        output_sig2 = m(outputs2)
                        o2 = (output_sig2 >= 0.5).long().detach().cpu().numpy()
                        ones2 = np.where(o2 == 1)[1]
                        if len(ones2)>0:
                            for ind in ones2:
                                CUIadd=cuis_list2[ind]
                                ind2 = np.where( cuiscomplete == CUIadd)[0]
                                o1[:,ind2]=1
                                
                    if index in cuis_index3:
                        outputs3 = model3(inputs)
                        output_sig3 = m(outputs3)
                        o3 = (output_sig3 >= 0.5).long().detach().cpu().numpy()
                        ones3 = np.where(o3 == 1)[1]
                        if len(ones3)>0:
                            for ind in ones3:
                                CUIadd=cuis_list3[ind]
                                ind3 = np.where( cuiscomplete == CUIadd)[0]
                                o1[:,ind3]=1
                                
                    if index in cuis_index4:
                        outputs4 = model4(inputs)
                        output_sig4= m(outputs4)
                        o4 = (output_sig4 >= 0.5).long().detach().cpu().numpy()
                        ones4 = np.where(o4 == 1)[1]
                        if len(ones1)>0:
                            for ind in ones4:
                                CUIadd=cuis_list4[ind]
                                ind4 = np.where( cuiscomplete == CUIadd)[0]
                                o1[:,ind4]=1
                                    
        selected_indices = np.where(o1 == 1)[1]
        selected_names = [cuis_list[i] for i in selected_indices]
        output_cuis.append((pp, selected_names))
        
        pred_sum_val = np.append(pred_sum_val, o1, axis = 0)                   
        labels_sum_val = np.append(labels_sum_val, labels.detach().cpu().numpy(), axis = 0)                    
        
    report = classification_report(labels_sum_val, pred_sum_val, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv('classification_report_94_4_1_25_decay.csv')
    #df2 = pd.DataFrame(output_cuis, columns=['ID', 'CUIs'])
    #df2.to_csv('output_94_4_1_25_decay.csv')
                        
