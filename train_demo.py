import os
import torch
import torch.nn.functional as F
import numpy as np
import random
import yaml
import argparse
from sklearn import metrics
from munch import *
from torch.utils.data import DataLoader
parser = argparse.ArgumentParser(description="code")
parser.add_argument('--gpu', '-g', type=int, default=5, help='gpu id')
parser.add_argument('--config', '-c', dest='config', help='config file', type=argparse.FileType('r', encoding="utf-8"), required=True)
parser.add_argument('--seed', '-s', type=int,  default=100, help='seed')
parser.add_argument('--alpha', '-a', type=float,  default=0.5, help='alpha')

args = parser.parse_args()
gpu_id = str(args.gpu)
configs = DefaultMunch.fromDict(yaml.safe_load(args.config.read()))
# set seed
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

from model.HMCAN import HMCAN

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
from data.dataset import Dataset
from sklearn.model_selection import train_test_split

alpha = args.alpha
print('seed: ', seed)
print('alpha: ', alpha)
print('batch_size: ', configs.batch_size)

print(torch.cuda.is_available())

def load_data(mode):
    dataset_dir = 'data/processed/' + configs.dataset
    data_label_list_path = os.path.join(dataset_dir, 'shuffle_{}_{}_label.pt'.format(configs.dataset, mode))
    data_vgg_list_path = os.path.join(dataset_dir, 'shuffle_{}_{}_resnet.pt'.format(configs.dataset, mode))
    data_bert_list_path = os.path.join(dataset_dir, 'shuffle_{}_{}_bert.pt'.format(configs.dataset, mode))

    datasets = Dataset(data_label_list_path, data_vgg_list_path, data_bert_list_path)

    return datasets


model = HMCAN(configs, alpha).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_func = F.cross_entropy

def test(x_feature, f_feature):
    model.eval()
    # label = np.array(label)

    logit = model(x_feature, f_feature)
    logit = F.softmax(logit, dim=1)
    pred = torch.argmax(logit, dim=1)
    a = pred.cpu().detach().numpy()
    return a

def evaluate(a, label, max_acc):
    acc = metrics.accuracy_score(label, a)
    precision = metrics.precision_score(label, a, average='weighted')
    recall = metrics.recall_score(label, a, average='weighted')
    F1 = metrics.f1_score(label, a, average='weighted')
    if max_acc < acc:
        print(metrics.classification_report(label, a, digits=4))

    return acc, precision, recall, F1

def trainer(configs):
    # load dataset
    train_dataset = load_data('train')
    test_dataset = load_data('test')

    train_dataloader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=True)

    test_dataloader = DataLoader(test_dataset, batch_size=configs.batch_size, shuffle=True)


    best_test_acc = best_precision = 0
    for epoch in range(configs.max_epoch):
        model.train()
        total_loss = 0
        for batch_idx, (index, y_train, f_train, x_train) in enumerate(train_dataloader):
            x_train = torch.tensor(x_train, dtype=torch.float32).cuda()
            f_train = torch.tensor(f_train, dtype=torch.float32).cuda()
            y_train = torch.LongTensor(y_train.long()).cuda()
            optimizer.zero_grad()
            logits = model(x_train, f_train)
            loss = loss_func(logits, y_train)
            loss.backward()
            optimizer.step()
            total_loss += loss

        pred_list = []
        label = []
        for batch_idx, (index, y_test, f_test, x_test) in enumerate(test_dataloader):
            x_test = torch.tensor(x_test, dtype=torch.float32).cuda()
            f_test = torch.tensor(f_test, dtype=torch.float32).cuda()
            pred = test(x_test, f_test)
            pred_list.append(pred)
            label.append(y_test)
        pred_result = np.concatenate(pred_list)
        label_result = np.concatenate(label)
        acc, precision, recall, F1 = evaluate(pred_result, label_result, best_test_acc)
        if best_test_acc < acc:
            best_test_acc = acc
            best_precision = precision
        print("epoch={}\ttotal_loss={}\tmax_acc={}\tacc={}\tprecision={}".format(epoch, total_loss, best_test_acc, acc, best_precision))




if __name__ == "__main__":
    trainer(configs)
