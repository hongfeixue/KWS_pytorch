import os
import json
dir = "./output"
outdir = "./audiopath"
if not os.path.isdir(outdir):
    os.makedirs(outdir)
def listdir(path, list_name):  # 传入存储的list
    for file in os.listdir(path):
        file_path = path+'/'+file
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)
 
classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
dir_dict={}
for c in classes:
    list_name=[]
    class_path = dir + '/' + c   #文件夹路径
    listdir(class_path,list_name)
    # file_name = [d for d in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, d))]
    dir_dict[c] = list_name
item = json.dumps(dir_dict)
with open("./audio_dir.json", "w", encoding='utf-8') as f:
    f.write(item + ",\n")
    print("^_^ write success")