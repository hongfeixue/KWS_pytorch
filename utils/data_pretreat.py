from posixpath import dirname
import re
import os
import librosa
import soundfile

def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if file.split('.')[-1] == 'ogg':
                L.append(os.path.join(root, file))
        return L

def cut_to_5s(src_dir,des_dir,seconds_per_split_file):

    # 获取该文件夹下所有语音数据
    filenames = file_name(src_dir)
   
    # 对每一个语音数据进行切片
    for filename in filenames:

        # 获取文件名字
        print("当前切割语音文件：    ", filename)
#         sound = AudioSegment.from_ogg(filename)
        y, sr = librosa.load(filename)
#         print(y.shape[0]) #1036800

        # 获取音频持续时间（单位为秒s）,并计算可以切割多少段？

        seconds_of_file = y.shape[0]
        print("该音频特征为：", int(seconds_of_file), "点")


        times = int(int(seconds_of_file) / seconds_per_split_file)
        print("当前语音共可切割：  ", times, " 次")

        # # 语音切割,以毫秒为单位
        start_time = 0
        internal = seconds_per_split_file
        end_time = seconds_per_split_file
        name=re.split(r'[/ .]', filename)[-2]
        print(name)


        for i in range(times):
            # 语音文件切割
            part = y[start_time:end_time]
            data_split_filename = des_dir+ '/' + name + '_' + str(i) + '.wav'
            print(data_split_filename)
            # 保存切割文件
#             part.export(data_split_filename, format="wav")
            soundfile.write(data_split_filename, part, sr)

            start_time += internal
            end_time += internal
dir = '../input/birdclef-2022/train_audio'
classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
i = 0
for dirname in classes:
    if not os.path.exists('output/' + dirname):
        os.makedirs('output/' + dirname)
    i = i + 1
    print('=' * 50)
    print(i)
    print('=' * 50)
    cut_to_5s(dir+'/'+dirname, 'output/'+dirname, 22800)
