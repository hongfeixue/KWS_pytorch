import os
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm
def train(loader, model, optimizer, epoch, cuda, verbose=True):
    model.train()
    train_loss = 0
    correct = 0
    Loss = []
    acc = []
    loop = tqdm(enumerate(loader), total =len(loader))
    for index, (data, target) in loop:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        prediction = model(data)
        train_loss = F.nll_loss(prediction, target, reduction='sum')
        pred = prediction.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        batchsize=prediction.shape[0]
        Loss.append(train_loss/batchsize)
        acc.append(correct / ((index+1)*batchsize))
        loop.set_description(f'Epoch [{epoch}/{epoch}]')
        loop.set_postfix(loss = train_loss/batchsize, correct = correct / ((index+1)*batchsize)) 
    loss = torch.tensor(Loss)
    acc = torch.tensor(acc)
    if not os.path.isdir('loss'):
        os.mkdir('loss')
    torch.save(loss,'./loss/epochl_{}'.format(epoch))
    torch.save(acc,'./loss/epocha_{}'.format(epoch))
    return train_loss/batchsize

def test(loader, model, cuda, verbose=True):
    model.eval()
    test_loss=0
    correct=0
    loop = tqdm(enumerate(loader), total =len(loader))
    for index, (data, target) in loop:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum')
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        loop.set_description(f'Test: ')
        loop.set_postfix(loss = test_loss/(index+1), correct = correct/(index+1))
    test_loss /= len(loader.dataset)
    if verbose:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(loader.dataset), 100. * correct / len(loader.dataset)))
    return test_loss