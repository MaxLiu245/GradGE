import time
start_time = time.time()
print(f"🚀 任务开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
import torch
import torch.nn.functional as F
import esm
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold
import os
import sys
script_dir = os.path.abspath(__file__)
project_dir = os.path.dirname(os.path.dirname(script_dir))
src_dir = os.path.join(project_dir, 'src')
if src_dir not in sys.path:
    sys.path.append(src_dir)
print(f"Looking for model in: {src_dir}")
from model import covid_prediction_model_chenningning
from dataset import DMS_data
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seq', type=str, default='/users/zmliu/fastscratch/covid-predict/data/pVNT_seq.csv',help='csv file of input sequences')
parser.add_argument('--out', type=str, default='/users/zmliu/fastscratch/covid-predict/data',help='Output dir')
parser.add_argument('--prediction', type=bool, default=False,help='whether to output the prediction results')
parser.add_argument('--embeddings', type=bool, default=True,help='whether to output the prediction embeddings')
args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
_,alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
batch_converter = alphabet.get_batch_converter()
model=covid_prediction_model_chenningning(freeze_bert=True)
model.load_state_dict(torch.load(os.path.join(project_dir, 'trained_model/pytorch_model.bin'),map_location=device)) ## load the trained model 

print("model in device:",next(model.parameters()).device)
model.eval()

data=pd.read_csv(args.seq)
idx,strs,tokens=batch_converter([(1,s) for s in data.seq.to_list()])
all_result=np.empty((0,9,2))
#embedding=np.empty((0,9,2580))
f_mean=[]
f_all = [[] for _ in range(9)]
batches = torch.split(tokens, 10)
pbar = tqdm(enumerate(batches), total=len(batches), desc="Inference")
# for batch_token in torch.split(tokens,10):
for i, batch_token in pbar:
    predict=model(batch_token.cuda(),labels=None)
    #embedding=np.append(embedding,predict.embedding.cpu().detach().numpy(),axis=0)
    all_result=np.append(all_result,predict.logits.softmax(axis=-1).cpu().detach().numpy(),axis=0)

    class_score=F.softmax(predict.logits,dim=-1)[:,:,1]
    class_score_all=np.array(class_score.cpu().detach(), dtype=float)
    class_score=torch.mean(class_score,dim=-1)# 1/9 fitness
    class_score=np.array(class_score.cpu().detach(), dtype=float)
    f_mean.extend(class_score)
    for i in range(9):
        f_all[i].extend(class_score_all[:, i])

# labels = ["ACE2", "COV2-2096", "COV2-2832", "COV2-2094", "COV2-2050", "COV2-2677", "COV2-2479", "COV2-2165","COV2-2499"]
labels=['ace2_bind',
        'COV2-2096_400', 'COV2-2832_400', 'COV2-2094_400', 'COV2-2050_400',
        'COV2-2677_400', 'COV2-2479_400', 'COV2-2165_400', 'COV2-2499_400']
pred_label_columns = [f"pred_{col}" for col in labels]
fitness_columns = [f"f_{col}" for col in labels]
arr = all_result
# Select "bind" values for ACE2 and "escape" values for the rest
# values = np.hstack([arr[:, 0, 1].reshape(-1, 1), arr[:, 1:, 1]]) 这里是有大问题的，不能对ACE2选择第一列
values = arr[:,:,1]
# Format values as "(class)value"
'''
formatted_values = np.empty_like(values, dtype=object)
formatted_values[:, 0] = ["bind" if val>0.5 else "non-bind" for val in values[:, 0]]
for i in range(1, 9):
    formatted_values[:, i] = ["escape" if val>0.5 else "non-escape"for val in values[:, i]]
'''
formatted_values = values
# Convert to DataFrame
data['f_mean'] = values.mean(axis = 1)   # 这里计算出来的是9个概率的均值
data[fitness_columns] = formatted_values
#df["id"] = data.id
#df = df.set_index("id")

# data['f_mean_'] = f_mean
for i, col_name in enumerate(fitness_columns[:]):
    data[col_name+'_'] = f_all[i]


data.to_csv(f"{args.out}/pVNT_preprocess.csv")


end_time = time.time() 
total_duration = end_time - start_time
minutes = int(total_duration // 60)
seconds = int(total_duration % 60)

print(f"✅ 处理完成！")
print(f"⏱️ 总耗时: {minutes} 分 {seconds} 秒 (共 {total_duration:.2f} 秒)")