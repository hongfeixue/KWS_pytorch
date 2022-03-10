import os
import os.path
import torch
import torchaudio
import json
import numpy

AUDIO_EXTENSIONS = '.wav'
AUDIO_PATH = '/home/disk1/xuehongfei/project/wake_up/mobvoi_hotword_dataset/'

def find_classes():
    classes = ["HIW", "other"] # (hi wenwen), (你好文文，其它杂音)
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(dir, class_to_idx):
    paths = []
    file = open(dir, 'r', encoding='utf-8')
    data = json.load(file)
    for i in range(len(data)):
        if data[i]["utt_id"] == "0":
            continue
        path = data[i]["utt_id"]
        path = AUDIO_PATH + path + AUDIO_EXTENSIONS
        # try:
        #     path = os.path.expanduser(path)
        #     y, sr = torchaudio.load(path)
        # except RuntimeError:
        #     data[i]["utt_id"] = "0"
        #     continue
        target = data[i]["keyword_id"]
        if  target == -1:
            target = 1
        item = (path, target)
        paths.append(item)
    # with open(dir, "w", encoding='utf-8') as jsonFile:
    #     json.dump(data, jsonFile)
    return paths

def spect_loader(path, normalize, max_frames=98):
    try:
        path = os.path.expanduser(path)
        y, sr = torchaudio.load(path)
    except RuntimeError:
        spect = torch.zeros((max_frames,1640))
        return spect

    # 加载单个音频文件，获取fbank特征，其中帧移为10ms，帧长25ms，梅尔数为40
    fbank = torchaudio.compliance.kaldi.fbank(y,frame_length=25.0,frame_shift=10.0,num_mel_bins=40)
    spect = fbank
    # 对特征进行处理，将过去30帧和未来10帧堆叠
    S = []
    for x in range(len(fbank)):
        if x-30<0:  # 过去30帧
            k1=0
        else:
            k1=x-30
        for last in range(1,30): 
            if x-30+last<0:
                k2=0
            else:
                k2=x-30+last
            if last==1 :
                s = torch.cat([spect[k1],fbank[k2]],0)
            else:
                s = torch.cat([s,fbank[k2]],0)
        s = torch.cat([s,fbank[k2]],0)  # 此帧
        if x+1>=len(fbank): # 未来10帧
            t1=len(fbank)-1
        else:
            t1=x+1
        s = torch.cat([s,fbank[t1]],0)
        for future in range(2,11): 
            if x+future>=len(fbank):
                t2=len(fbank)-1
            else:
                t2=x+future
            s = torch.cat([s,fbank[t2]],0)
        S.append(s)
    # 得到堆叠后的特征，形状为 max_frames*1640
    spect= torch.tensor([item.cpu().detach().numpy() for item in S]).cuda() 
    # 设置所有音频文件的帧数目相同
    if spect.shape[0] < max_frames:
        pad = torch.zeros((max_frames - spect.shape[0],spect.shape[1]))
        spect = spect.cpu()
        pad = pad.cpu()
        spect = torch.cat([spect, pad],0)
    elif spect.shape[0] > max_frames:
        spect = spect[:max_frames, ]

    # 特征归一化
    if normalize:
        mean = spect.mean()
        std = spect.std()
        if std != 0:
            spect.add_(-mean)
            spect.div_(std)
    spect = spect.cpu()
    return spect    

class CommandLoader(torch.utils.data.Dataset):
    def __init__(self, root, normalize=True, max_frames=98):
        classes, class_to_idx = find_classes()
        paths = make_dataset(root, class_to_idx)
        if len(paths) == 0:
            print("Dataset is None!")
            raise (RuntimeError)
        
        self.root = root
        self.paths = paths
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.loader = spect_loader
        self.normalize = normalize
        self.max_frames = max_frames

    def __getitem__(self, index):
        path, target = self.paths[index]
        spect = self.loader(path, self.normalize, self.max_frames)
        return spect, target
    
    def __len__(self):
        return len(self.paths)

# 用于测试可以正确加载数据
if __name__ == '__main__':
    train_dataset = CommandLoader('./mobvoi_hotword_dataset_resources/p_test.json', max_frames=200)
    for i in range(21282):
        print(train_dataset[i])
            # 200 * 1640