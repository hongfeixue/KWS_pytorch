import argparse
import torch
from command_loader import CommandLoader
from model import DNN, TDNN, CNN, DFSMN
from train import train, test
import os
import numpy as np

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    torch.backends.cudnn.enabled = False
    parser = argparse.ArgumentParser(description='Deep netural network for keyword spotting')
    parser.add_argument('--train_path', default='./mobvoi_hotword_dataset_resources/p_train.json', help='path to the train data folder')
    parser.add_argument('--test_path', default='./mobvoi_hotword_dataset_resources/p_test.json', help='path to the test data folder')
    parser.add_argument('--valid_path', default='./mobvoi_hotword_dataset_resources/p_dev.json', help='path to the valid data folder')
    parser.add_argument('--batch_size', type=int, default=100, metavar='N', help='training and valid batch size')
    parser.add_argument('--test_batch_size', type=int, default=100, metavar='N', help='batch size for testing')
    parser.add_argument('--arc', default='TDNN', help='network architecture: ConvNet,FcNet')
    parser.add_argument('--epochs', type=int, default=5, metavar='N', help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.005, metavar='LR', help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum, for SGD only')
    parser.add_argument('--optimizer', default='adam', help='optimization method: sgd | adam')
    parser.add_argument('--cuda', default=True, help='enable CUDA')
    parser.add_argument('--seed', type=int, default=1234, metavar='S', help='random seed')
    parser.add_argument('--patience', type=int, default=5, metavar='N',
                        help='how many epochs of no loss improvement should we wait before stop training')
    parser.add_argument('--savepath', type=str, default='./checkpoint/epoch', help='path to save model')
    
    args = parser.parse_args()

    args.cuda = args.cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    train_dataset = CommandLoader(args.train_path)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=args.cuda, sampler=None
    )

    valid_dataset = CommandLoader(args.valid_path)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=None,
        num_workers=4, pin_memory=args.cuda, sampler=None
    )

    test_dataset = CommandLoader(args.test_path)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=None,
        num_workers=4, pin_memory=args.cuda, sampler=None
    )

    if args.arc == 'DNN':
        model = DNN()
    elif args.arc == 'TDNN':
        model = TDNN()
    elif args.arc == 'DFSMN':
        model = DFSMN()
    elif args.arc == 'CNN':
        model = CNN()
    else:
        model = DNN()

    if args.cuda:
        print('Using CUDA with {0} GPUs'.format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model).cuda()

    if args.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    best_valid_loss = np.inf
    iteration = 0
    epoch = 1

    while (epoch < args.epochs+1) and (iteration < args.patience):
        train(train_loader, model, optimizer, epoch, args.cuda)
        vaild_loss = test(valid_loader, model, args.cuda)
        if vaild_loss > best_valid_loss:
            iteration += 1
            print('Loss was not improved, iteration {0}'.format(str(iteration)))
        else:
            print('Saving model...')
            iteration = 0
            best_valid_loss = vaild_loss
            state = {
                'net' : model.module if args.cuda else model,
                'acc' : vaild_loss,
                'epoch' : epoch
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, args.savepath+str(epoch))
        epoch += 1

    test(test_loader, model, args.cuda)