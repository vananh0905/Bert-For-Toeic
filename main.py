import numpy as np
import pandas as pd
import json, time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn import metrics


import transformers
from transformers import AutoModel, AutoTokenizer
from transformers import AdamW


# specify GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_data():
    print('Get data ...')
    with open("data.json") as data_file:
        file = json.load(data_file)
        data = []
        for value in file.values():
            data.append(value)

    return data

def get_data_for_datatype(id):
    print('Get dataset type ' + id)
    filename = 'dataset_type_' + id + '.json'
    with open(filename) as data_file:
        file = json.load(data_file)
        data = []
        for value in file.values():
            data.append(value)

    return data

def preprocess_data(data, max_seq_len, batch_size):

    print('Preprocess data ...')
    # split train set, test set
    D_train, D_test = train_test_split(data, random_state = 2020, test_size = 0.2)

    def get(data):
        X, Y = [], []
        for item in data:
            for id in range(1,5):
                sentence = item['question'].replace('___', item[str(id)])
                target = int(item[str(id)] == item['answer'])
                X.append(sentence)
                Y.append(target)

        return X, Y

    train_X, train_Y = get(D_train)

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    tokens_train = tokenizer.batch_encode_plus(
        train_X,
        max_length = max_seq_len,
        padding = True,
        truncation=True,
        return_tensors='pt'
    )

    # convert to Tensor
    train_seq = torch.tensor(tokens_train['input_ids'])
    train_mask = torch.tensor(tokens_train['attention_mask'])
    train_Y = torch.tensor(train_Y)

    train_data = TensorDataset(train_seq, train_mask, train_Y)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    print('Done')

    return train_dataloader, D_test


class Net(nn.Module):
    def __init__(self, bert):
        super(Net, self).__init__()
        self.bert = bert

        # dropout Layer
        self.dropout = nn.Dropout(p=0.2)

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 2)
        # self.fc2 = nn.Linear(512, 2)
        # self.softmax = nn.LogSoftmax(dim=1)


    def forward(self, sent_id, masks):
    
        q = self.bert(sent_id, attention_mask=masks)
        cls = q[1]
        x = self.dropout(cls)

        x = self.fc1(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        
        # # output Layer
        # x = self.fc2(x)

        return x


class TOEICBert:

    def __init__(self, model, lr, n_epoches, train_loader):
        super(TOEICBert, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(self.device)
        self.model = model
        self.lr = lr
        self.n_epoches = n_epoches

        self.ce=torch.nn.CrossEntropyLoss()
        self.optimizer = self._get_optimizer()
       

    def _get_optimizer(self, lr=None):
        if lr is None: lr = self.lr
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
                {'params': [param for name, param in param_optimizer if not any(nd in name for nd in no_decay)],
                    'weight_decay': 0.01},
                {'params': [param for name, param in param_optimizer if any(nd in name for nd in no_decay)],
                'weight_decay': 0.0}
        ]

        return AdamW(self.model.parameters(), lr=lr)

    
    def criterion(self,ouput, target):
        return self.ce(ouput, target)

    
    def train_epoch(self, train_dataloader):
        
        self.model.train()
        total_loss = 0.0
        for step,batch in enumerate(train_dataloader):
            

            batch = [r.to(self.device) for r in batch]
            sent_id, mask, labels = batch

            self.optimizer.zero_grad()      

            preds = self.model.forward(sent_id, mask)

            loss = self.criterion(preds, labels)
            total_loss += loss.item()

            loss.backward()
            self.optimizer.step()
            
            if step % 100 == 0:
              print('\tbatch: {}/{} loss: {}'.format(step, len(train_dataloader), total_loss/(step+1)))


    def train(self, train_dataloader, test_dataloader, datatype):
        self.model = self.model.to(self.device)

        print('Start training...')

        for e in range(self.n_epoches):
            print('Epoch {:3d} '.format(e+1))
            self.train_epoch(train_dataloader)

            train_acc, train_f1, train_prec = self.validate(train_dataloader)
            print('train acc: {:.3f} | train f1: {:.3f} | train precision: {:.3f}'.format(train_acc, train_f1, train_prec))

            test_acc, time = self.evaluate(test_dataloader)
            print('test acc: {:.3f} time each question: {:.3f}'.format(test_acc, time))

            type1_acc, _ = self.evaluate(datatype['1'])
            print('type1 acc: {:.3f}'.format(type1_acc))

            type2_acc, _ = self.evaluate(datatype['2'])
            print('type2 acc: {:.3f}'.format(type2_acc))

            type3_acc, _ = self.evaluate(datatype['3'])
            print('type3 acc: {:.3f}'.format(type3_acc))

            type4_acc, _ = self.evaluate(datatype['4'])
            print('type4 acc: {:.3f}'.format(type4_acc))

            type5_acc, _ = self.evaluate(datatype['5'])
            print('type5 acc: {:.3f}'.format(type5_acc))


    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        total_num = 0
        predicted_labels, target_labels = list(), list()
        with torch.no_grad():
            for step, batch in enumerate(dataloader):
                batch = [r.to(self.device) for r in batch]
                sent_id, mask, labels = batch

                preds = self.model(sent_id, mask)
                loss = self.criterion(preds, labels)
                total_loss += loss.item()
                
                target_labels.extend(labels.cpu().detach().numpy())
                predicted_labels.extend(torch.argmax(preds, dim=-1).cpu().detach().numpy())

        train_loss = total_loss/len(dataloader)
        predicted_labels, target_labels = np.array(predicted_labels), np.array(target_labels)
        accuracy = metrics.accuracy_score(target_labels, predicted_labels)
        f1 = metrics.f1_score(target_labels, predicted_labels)
        precision = metrics.precision_score(target_labels, predicted_labels)
        return accuracy, f1, precision

    
    def evaluate(self, dataloader):
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        def test(item):
            self.model.eval()
            sents = []
            for id in range(1, 5):
                question  = item['question'].replace("___", item[str(id)])
                sents.append(question)
            inputs = tokenizer.batch_encode_plus(sents, padding=True, truncation=True, max_length=64, return_tensors='pt') 
            with torch.no_grad():
                output = self.model(inputs['input_ids'].to(device), inputs['attention_mask'].to(device))
            prediction = torch.softmax(output, dim=-1)
            return torch.argmax(prediction[:, 1], dim=-1).item()+1
        tstart = time.time()
        sent = 0
        for item in dataloader:
            answer = test(item)
            if item[str(answer)] == item['answer']:
              sent += 1
        tend = time.time()
        return sent/len(dataloader), (tend-tstart)/len(dataloader)


if __name__ == "__main__":
    bert = AutoModel.from_pretrained('bert-base-uncased')
    model = Net(bert)
    data = get_data()
    set1 = get_data_for_datatype('1')
    set2 = get_data_for_datatype('2')
    set3 = get_data_for_datatype('3')
    set4 = get_data_for_datatype('4')
    set5 = get_data_for_datatype('5')
    datatype = {}
    datatype['1'] = set1
    datatype['2'] = set2 
    datatype['3'] = set3
    datatype['4'] = set4 
    datatype['5'] = set5
    train_dataloader, D_test = preprocess_data(data, max_seq_len = 64, batch_size=32)
    appr = TOEICBert(model, lr=1e-5, n_epoches=20, train_loader=train_dataloader)
    appr.train(train_dataloader, D_test, datatype)
