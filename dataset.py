
from PIL import Image
import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoConfig
from torch.utils.data import DataLoader


class MyDataset(Dataset):
    def __init__(self, root_dir, tokenizer):
        self.root_dir = root_dir
        self.data = pd.read_csv(self.root_dir, sep='\t')
        self.tokenizer = tokenizer
        self.maxlen = 50
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sen = self.data['sentence'][idx]
        sen = self.tokenizer(sen, return_tensors="pt", max_length=self.maxlen, padding='max_length',  truncation=True)

        input_ids = sen['input_ids'].squeeze()
        token_type_ids = sen['token_type_ids'].squeeze()
        attention_mask = sen['attention_mask'].squeeze()
        label = self.data['label'][idx]
        return input_ids, token_type_ids, attention_mask, label


class MyDataset_CV(Dataset):
    def __init__(self, data_path, label_path):
        self.test_transforms = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )

        self.data = []
        imglist = os.listdir(data_path)
        self.path2label = self.get_labels(label_path=label_path)
        for img in imglist:
            self.data.append((os.path.join(data_path, img), self.path2label[img]))
        print('len(data) is ', len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        (img_path, label) = self.data[index]

        img = Image.open(img_path).convert("RGB")
        img = self.test_transforms(img)
        return img, label

    def get_labels(self, label_path):
        path = label_path
        path2label = {}
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                arrs = line.split()
                key, value = arrs[0], int(arrs[1])
                path2label[key] = value
        return path2label


def get_dataSST_2():
    BERT_MODEL_DIR = "./model_bert"
    DataPathDev = './data/SST-2/dev.tsv'
    DataPathTrain = './data/SST-2/train.tsv'
    config = AutoConfig.from_pretrained(BERT_MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_DIR)
    dataTrain = MyDataset(DataPathTrain, tokenizer)
    dataDev = MyDataset(DataPathDev, tokenizer)
    Train_dataloader = DataLoader(dataTrain, batch_size=16)
    Dev_dataloader = DataLoader(dataDev, batch_size=16)
    return Train_dataloader, Dev_dataloader

def load_dataloader(batch_size):
    data_path = 'data/dataset/imagenet/images'
    label_path = 'data/dataset/imagenet/meta.txt'
    train_data = MyDataset_CV(data_path, label_path)
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader, len(train_data)
