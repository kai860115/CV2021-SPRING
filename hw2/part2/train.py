import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim
import matplotlib.pyplot as plt
from model import ConvNet, MyNet
from data import get_dataloader

if __name__ == "__main__":
    # Specifiy data folder path and model type(fully/conv)
    folder, model_type = sys.argv[1], sys.argv[2]
    
    # Get data loaders of training set and validation set
    train_loader, val_loader = get_dataloader(folder, batch_size=32)

    # Specify the type of model
    if model_type == 'conv':
        model = ConvNet()
    elif model_type == 'mynet':
        model = MyNet()

    # Set the type of gradient optimizer and the model it update 
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Choose loss function
    criterion = nn.CrossEntropyLoss()

    # Check if GPU is available, otherwise CPU is used
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    
    loss_record, acc_record = {"train": [], "val": []}, {"train": [], "val": []}
    best_acc = 0.0
    # Run any number of epochs you want
    ep = 20
    for epoch in range(ep):
        print('Epoch:', epoch)
        ##############
        ## Training ##
        ##############
        model.train()
        
        # Record the information of correct prediction and loss
        correct_cnt, total_loss, total_cnt = 0, 0, 0
        
        # Load batch data from dataloader
        for batch, (x, label) in enumerate(train_loader,1):
            # Set the gradients to zero (left by previous iteration)
            optimizer.zero_grad()
            # Put input tensor to GPU if it's available
            if use_cuda:
                x, label = x.cuda(), label.cuda()
            # Forward input tensor through your model
            out = model(x)
            # Calculate loss
            loss = criterion(out, label)
            # Compute gradient of each model parameters base on calculated loss
            loss.backward()
            # Update model parameters using optimizer and gradients
            optimizer.step()

            # Calculate the training loss and accuracy of each iteration
            total_loss += loss.item()
            _, pred_label = torch.max(out, 1)
            total_cnt += x.size(0)
            correct_cnt += (pred_label == label).sum().item()

            # Show the training information
            if batch % 500 == 0 or batch == len(train_loader):
                acc = correct_cnt / total_cnt
                ave_loss = total_loss / batch           
                print ('Training batch index: {}, train loss: {:.6f}, acc: {:.3f}'.format(
                    batch, ave_loss, acc))

        loss_record['train'].append(total_loss / len(train_loader))
        acc_record['train'].append(correct_cnt / total_cnt)

        ################
        ## Validation ##
        ################
        model.eval()
        # TODO
        correct_cnt, total_loss, total_cnt = 0, 0, 0
        with torch.no_grad():
            for batch, (x, label) in enumerate(val_loader):
                if use_cuda:
                    x, label = x.cuda(), label.cuda()
                out = model(x)
                loss = criterion(out, label)
                total_loss += loss.item()
                _, pred_label = torch.max(out, 1)
                total_cnt += x.size(0)
                correct_cnt += (pred_label == label).sum().item()
        print('Val loss: {:.6f}, acc: {:.3f}'.format(total_loss / len(val_loader), correct_cnt / total_cnt))

        loss_record['val'].append(total_loss / len(val_loader))
        acc_record['val'].append(correct_cnt / total_cnt)

        # Save trained model
        if best_acc < correct_cnt / total_cnt:
            print('save model')
            torch.save(model.state_dict(), './checkpoint/%s.pth' % model.name())
            best_acc = correct_cnt / total_cnt


    # Plot Learning Curve
    # TODO
    plt.figure(figsize=(6, 4))
    plt.plot(range(ep), loss_record['train'], c='tab:red', label='training loss')
    plt.plot(range(ep), loss_record['val'], c='tab:cyan', label='validation loss')
    plt.xlabel('epoches')
    plt.ylabel('Loss')
    plt.ylim((0.0, 0.5))
    plt.title('Loss_{}'.format(model_type))
    plt.legend()
    plt.savefig('Loss_{}.png'.format(model_type))

    plt.figure(figsize=(6, 4))
    plt.plot(range(ep), acc_record['train'], c='tab:red', label='training acc')
    plt.plot(range(ep), acc_record['val'], c='tab:cyan', label='validation acc')
    plt.xlabel('epoches')
    plt.ylabel('Accuracy')
    plt.ylim((0.8, 1.0))
    plt.title('Accuracy_{}'.format(model_type))
    plt.legend()
    plt.savefig('Accuracy_{}.png'.format(model_type))


    