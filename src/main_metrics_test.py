from main_mrp_test import Tee
import os
import sys
import glob
import numpy as np
import pandas as pd
import torch
import esm
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
from polyleven import levenshtein

#TASK_NAME = 'result_3_vocs_20rounds'
#THRESHOLD = 0.5

TASK_NAME = 'result_3_vocs_20rounds_30'
THRESHOLD = 0.6

#TASK_NAME = 'result_2_wt_50rounds'
#THRESHOLD = 0.1

#TASK_NAME = 'result_6_om_50rounds'
#THRESHOLD = 0.1

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
save_log_path = os.path.join(f'./{TASK_NAME}', 'log_metrics.log')
os.makedirs(os.path.dirname(save_log_path), exist_ok=True)
sys.stdout = Tee(save_log_path)

SAVE_DIR = f'./{TASK_NAME}'
HISTORY_DIR = os.path.join(SAVE_DIR, 'evolution_history') # 自定义保存目录
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


csv_path_pattern = os.path.join(HISTORY_DIR, "round_*.csv")
csv_files = glob.glob(csv_path_pattern)

try:
    csv_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
except Exception as e:
    print(f"文件名排序出错，请检查文件名格式是否为 round_x.csv: {e}")

print(f"运行环境: {DEVICE}")
print(f"数据目录: {HISTORY_DIR}")
print(f"共找到 {len(csv_files)} 个轮次文件:")
if len(csv_files) > 0:
    print(f"  Start: {os.path.basename(csv_files[0])}")
    print(f"  End:   {os.path.basename(csv_files[-1])}")
else:
    print("  警告: 未找到任何 CSV 文件，请检查路径！")





print("正在加载 ESM-2 模型 (esm2_t6_8M_UR50D)...")
esm_model, esm_alphabet = esm.pretrained.esm2_t6_8M_UR50D()
esm_batch_converter = esm_alphabet.get_batch_converter()
esm_model.eval().to(DEVICE)

def calculate_ppl_batch(sequences, batch_size=32):
    batch_ppls = []
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i + batch_size]
        data = [(str(j), seq) for j, seq in enumerate(batch_seqs)]
        batch_labels, batch_strs, batch_tokens = esm_batch_converter(data)
        batch_tokens = batch_tokens.to(DEVICE)
        
        with torch.no_grad():
            results = esm_model(batch_tokens, repr_layers=[6], return_contacts=False)
            logits = results["logits"] # Shape: [Batch, SeqLen, Vocab]
            
            log_probs = torch.log_softmax(logits, dim=-1)
            
            token_log_probs = torch.gather(log_probs, 2, batch_tokens.unsqueeze(-1)).squeeze(-1)
            
            mask = (batch_tokens != esm_alphabet.padding_idx) & \
                   (batch_tokens != esm_alphabet.cls_idx) & \
                   (batch_tokens != esm_alphabet.eos_idx)

            seq_log_prob_sum = (token_log_probs * mask).sum(dim=1)
            seq_len = mask.sum(dim=1)

            seq_nll = - seq_log_prob_sum / seq_len
            seq_ppl = torch.exp(seq_nll)
            
            batch_ppls.extend(seq_ppl.cpu().numpy())
    return batch_ppls

ppl_history = []
rounds = []

print("开始计算每一轮的 PPL 并更新文件...")
for f in tqdm(csv_files):
    df = pd.read_csv(f)
    round_num = int(os.path.basename(f).split('_')[1].split('.')[0])
    
    seqs = df['Sequence'].tolist()

    ppl_values = calculate_ppl_batch(seqs, batch_size=64)

    df['Perplexity'] = ppl_values
    df.to_csv(f, index=False) # 覆盖原文件
    
    avg_ppl = np.mean(ppl_values)
    ppl_history.append(avg_ppl)
    rounds.append(round_num)

print("所有文件的 PPL 计算完毕并已保存。")

plt.figure(figsize=(8, 5))
plt.plot(rounds, ppl_history, 'r-s', linewidth=2, markersize=6, label='Avg Perplexity')
plt.title('Evolution of Sequence Perplexity (Lower is Better)')
plt.xlabel('Round')
plt.ylabel('Average PPL (ESM-2)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(f'./{TASK_NAME}', 'metric_ppl.png'))
plt.show()




fitness_history = []
rounds = []

print("开始统计每一轮的 Fitness...")
for f in tqdm(csv_files):
    df = pd.read_csv(f)
    round_num = int(os.path.basename(f).split('_')[1].split('.')[0])

    avg_fit = df['Fitness_Score'].mean()
    
    fitness_history.append(avg_fit)
    rounds.append(round_num)

plt.figure(figsize=(8, 5))
plt.plot(rounds, fitness_history, 'b-o', linewidth=2, markersize=6, label='Avg Fitness')
plt.title('Evolution of Population Fitness (Higher is Better)')
plt.xlabel('Round')
plt.ylabel('Average Fitness Score')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(f'./{TASK_NAME}', 'metric_fs.png'))
plt.show()



csv_files = glob.glob(os.path.join(HISTORY_DIR, "round_*.csv"))
csv_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))

max_fitness_list = []
mean_fitness_list = []
rounds = []

print(f"正在分析 MFS (Max Fitness Score)... 共 {len(csv_files)} 轮")

for f in tqdm(csv_files):
    df = pd.read_csv(f)
    round_num = int(os.path.basename(f).split('_')[1].split('.')[0])
    
    fitness_values = df['Fitness_Score'].values
    
    mfs = np.max(fitness_values)  # Max Fitness Score
    avg_fit = np.mean(fitness_values)
    
    max_fitness_list.append(mfs)
    mean_fitness_list.append(avg_fit)
    rounds.append(round_num)

plt.figure(figsize=(8, 5))

plt.plot(rounds, max_fitness_list, 'r-o', linewidth=2, label='MFS (Max Fitness Score)')
plt.plot(rounds, mean_fitness_list, 'b--s', alpha=0.6, label='Mean Fitness')

plt.title('Evolution of MFS (Max Fitness Score)')
plt.xlabel('Round')
plt.ylabel('Fitness')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(f'./{TASK_NAME}', 'metric_mfs.png'))
plt.show()

print(f"最终轮 MFS: {max_fitness_list[-1]:.4f}")






def levenshtein_distance(s1, s2):
    return levenshtein(s1, s2)

# [核心修改] 移除采样逻辑，强制使用全排列
def calculate_diversity(sequences):
    if len(sequences) < 2: return 0.0
    
    pairs = list(itertools.combinations(sequences, 2))
            
    distances = [levenshtein_distance(p[0], p[1]) for p in pairs]
    return np.mean(distances)

csv_files = glob.glob(os.path.join(HISTORY_DIR, "round_*.csv"))
csv_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))

diversity_list = []
rounds = []

print("正在分析 Diversity (Polyleven全排列版)...")

for f in tqdm(csv_files):
    df = pd.read_csv(f)
    round_num = int(os.path.basename(f).split('_')[1].split('.')[0])
    
    seqs = df['Sequence'].tolist()
    div = calculate_diversity(seqs)
    
    diversity_list.append(div)
    rounds.append(round_num)

plt.figure(figsize=(8, 5))
plt.plot(rounds, diversity_list, 'g-^', linewidth=2, label='Diversity (Intra-population)')

plt.title('Evolution of Population Diversity')
plt.xlabel('Round')
plt.ylabel('Avg Levenshtein Distance')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(f'./{TASK_NAME}', 'metric_diveristy.png'))
plt.show()





INITIAL_SEQ_FILE = '../data/Covid19_RBD_seq.txt' 

def levenshtein_distance(s1, s2):
    return levenshtein(s1, s2)

def calculate_novelty(seq, baseline_seqs):
    dists = [levenshtein_distance(seq, base) for base in baseline_seqs]
    return np.min(dists)

baseline_seqs = []
if os.path.exists(INITIAL_SEQ_FILE):
    with open(INITIAL_SEQ_FILE, 'r') as f:
        content = f.read().strip()
        baseline_seqs = content.split('\n') if '\n' in content else [content]
    print(f"已加载基准序列: {len(baseline_seqs)} 条")
else:
    print("未找到初始文件，使用 Round 1 作为基准...")
    first_round = os.path.join(HISTORY_DIR, "round_1.csv")
    if os.path.exists(first_round):
        baseline_seqs = pd.read_csv(first_round)['Sequence'].tolist()

csv_files = glob.glob(os.path.join(HISTORY_DIR, "round_*.csv"))
csv_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))

novelty_list = []
rounds = []

print("正在分析 Novelty (Polyleven版)...")

for f in tqdm(csv_files):
    df = pd.read_csv(f)
    round_num = int(os.path.basename(f).split('_')[1].split('.')[0])
    
    seqs = df['Sequence'].tolist()
    
    current_round_novelties = [calculate_novelty(s, baseline_seqs) for s in seqs]
    avg_novelty = np.mean(current_round_novelties)
    
    novelty_list.append(avg_novelty)
    rounds.append(round_num)

plt.figure(figsize=(8, 5))
plt.plot(rounds, novelty_list, 'orange', marker='d', linewidth=2, label='Novelty (Dist to Initial)')

plt.title('Evolution of Novelty')
plt.xlabel('Round')
plt.ylabel('Avg Levenshtein Distance to WildType')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(f'./{TASK_NAME}', 'metric_novelty.png'))
plt.show()