import matplotlib.pyplot as plt
from model import ResNet18
from dataset import Dataset
import torch
import os, sys
import numpy as np

import scikitplot as skplt
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

root = sys.path[0]

train_dataset = Dataset(mode='train')
test_dataset = Dataset(mode='test')


batch_size = 8
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


epochs = 10 
model = ResNet18(len(train_dataset.classes)).to(device) 
optim = torch.optim.AdamW(model.parameters(), lr=2e-4) 
loss_func = torch.nn.CrossEntropyLoss() 

train_loss = []
test_loss = []
train_acc = []
test_acc = []

best_acc = 0
for epoch in range(epochs):
    print('\n*****************\n\nepoch', epoch+1)
    _loss = 0
    _acc = 0
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        out = model(images) 
        _acc += (out.argmax(dim=1) == labels).sum().item() / images.size(0)
        loss = loss_func(out, labels) 
        _loss += loss.item()
        optim.zero_grad()
        loss.backward() 
        optim.step() 


    train_loss.append(_loss/len(train_loader))
    train_acc.append(_acc/len(train_loader))
    print('train loss:', train_loss[-1], 'acc:', train_acc[-1])

    with torch.no_grad():
        _loss = 0
        _acc = 0
        model.eval()
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            out = model(images) 
            _acc += (out.argmax(dim=1) == labels).sum().item() / images.size(0) 
            loss = loss_func(out, labels) 
            _loss += loss.item()

        test_loss.append(_loss/len(test_loader))
        test_acc.append(_acc/len(test_loader))
        print('test loss:', test_loss[-1], 'acc:', test_acc[-1])

    if test_acc[-1] >= best_acc:
        best_acc = test_acc[-1]
        torch.save(model, os.path.join(root, 'classifier.pth'))

    l = list(range(epoch+1))
    plt.plot(l, train_loss, color='r', label='train')
    plt.plot(l, test_loss, color='y', label='test')
    plt.title('loss')
    plt.legend(loc='best')
    plt.savefig(os.path.join(root, 'loss.png'), format='png') 
    plt.clf()

    plt.plot(l, train_acc, color='r', label='train')
    plt.plot(l, test_acc, color='b', label='test')
    plt.legend(loc='best')
    plt.title('accuracy')
    plt.savefig(os.path.join(root, 'acc.png'), format='png') 
    plt.clf()


model = torch.load(os.path.join(root, 'classifier.pth'), map_location=device).eval()
y_true = []
predict_proba = []
with torch.no_grad():
    model.eval()
    for i, (images, labels) in enumerate(test_loader):
        print(f'TEST... {i+1}/{len(test_loader)}')
        images = images.to(device)
        y_true.append(labels)
        labels = labels.to(device)

        out = model(images)
        predict_proba.append(out.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    predict_proba = torch.cat(predict_proba, dim=0).numpy()
    y_pred = np.argmax(predict_proba, axis=1)

    skplt.metrics.plot_confusion_matrix(y_true, y_pred, normalize=True)
    plt.savefig(os.path.join(root, f'confusion matrix.png'), format='png')

    skplt.metrics.plot_roc(y_true, predict_proba)
    plt.savefig(os.path.join(root, f'ROC.png'), format='png')

    skplt.metrics.plot_precision_recall(y_true, predict_proba)
    plt.savefig(os.path.join(root, f'PR.png'), format='png')

    print("Accuracy score:")
    print(accuracy_score(y_true, y_pred))

    print("Recall score:")
    print(recall_score(y_true, y_pred, average='macro'))

    print("Precision score:")
    print(precision_score(y_true, y_pred, average='macro'))

    print("F1 score:")
    print(f1_score(y_true, y_pred, average='macro'))
