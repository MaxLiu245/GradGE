import torch
import torch.nn as nn
import torch.nn.functional as F
import esm
import numpy as np
import random
import copy
import os
import sys
import glob
import pandas as pd
import shutil
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
import requests
import time
from my_mutations import KNOWN_RISKY_MUTATIONS

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, 'src')
if src_dir not in sys.path:
    sys.path.append(src_dir)
from model import covid_prediction_model_chenningning

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
CHAR_TO_INT = {char: i for i, char in enumerate(AMINO_ACIDS)}
for k, v in list(CHAR_TO_INT.items()):
    pass 
VOCAB_SIZE = 33 
SEQ_LEN = 201


def set_seed(seed=42):
    print(f"Setting random seed to {seed}...")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def calculate_identity(seq1, seq2):
    if len(seq1) != len(seq2): return 0
    matches = sum(c1 == c2 for c1, c2 in zip(seq1, seq2))
    return matches / len(seq1)


class FGSM_GA_Optimizer:
    def __init__(self, model, device):

    def _hook_fn(self, module, input, output):
        if self.attack_embedding is not None:
            return self.attack_embedding
        return output

    def get_fitness(self, seqs, batch_size=16):
        all_fitness = []
        if len(all_fitness) > 0:
            return np.concatenate(all_fitness)
        return np.array([])

    def reconstruct_topk_constrained(self, adv_embed_tensor, original_seq_str, k=1):
        _, _, batch_tokens = self.batch_converter([("id", original_seq_str)])
        
        return "".join(new_seq_list)

    def fgsm_generate(self, seq_str, epsilon_list, target_fitness=0.12):
        data = [("seq", seq_str)]
        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)
        
        return generated_seqs

    def mutate(self, seq, prob=0.1):
        if random.random() > prob: return seq
        seq_list = list(seq)
        pos = random.randint(0, len(seq_list)-1)
        seq_list[pos] = random.choice(AMINO_ACIDS)
        return "".join(seq_list)
    
    def crossover(self, seq1, seq2):
        return False


def calculate_kl_divergence_and_plot(initial_seqs, evolved_seqs, save_path):
    return 0

MIN_PROPORTION = 0.05

BASE_SEQ_FILE = "../data/Covid19_RBD_seq.txt"
# ===========================================

REAL_START_POS = 331
RBD_END_POS = 531

def download_and_generate(lineage, MIN_PROPORTION = 0.05, BASE_SEQ_FILE = "../data/Covid19_RBD_seq.txt", REAL_START_POS = 331, RBD_END_POS = 531):
    print(f"\n🔵 正在处理谱系: {lineage}")
    
    url = "https://lapis.cov-spectrum.org/open/v2/sample/aminoAcidMutations"
    params = {
        "pangoLineage": lineage,
        "minProportion": MIN_PROPORTION,
        "dataFormat": "csv",
        "downloadAsFile": "true"
    }
    
    csv_filename = f"{lineage}_Mutations.csv"
    
    try:
        print(f"   📡 请求下载突变表...")
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code != 200:
            print(f"   ❌ 下载失败 (状态码 {response.status_code})")
            print(f"   可能原因: 数据库中没有 {lineage} 的数据")
            return

        csv_content = response.content.decode('utf-8')
 
        if not csv_content.strip():
            print("   ⚠️ 下载的文件内容为空。")
            return

        with open(csv_filename, "w", encoding="utf-8") as f:
            f.write(csv_content)
        print(f"   💾 已保存突变表: {csv_filename}")

        process_csv_to_sequence(csv_filename, lineage)

    except Exception as e:
        print(f"   ❌ 发生错误: {e}")

def process_csv_to_sequence(csv_path, lineage):
    try:
        with open(BASE_SEQ_FILE, 'r') as f:
            base_rbd = list(f.read().strip())
    except FileNotFoundError:
        print("   ❌ 找不到原始序列文件。")
        return

    try:
        df = pd.read_csv(csv_path)
    except:
        print("   ❌ CSV 格式错误。")
        return
    
    col_map = {c.lower(): c for c in df.columns}
    mut_col = col_map.get('mutation') or col_map.get('substitutions')
    
    if not mut_col:
        print(f"   ⚠️ 找不到突变列。")
        return

    applied_logs = []
    valid_mutations = []
    skipped_indels = []

    for _, row in df.iterrows():
        mut_str = row[mut_col]
        if not str(mut_str).startswith("S:"):
            continue
            
        try:
            clean_str = mut_str.replace("S:", "")
            
            import re
            match = re.search(r'([A-Z])(\d+)([A-Z])', clean_str)
            if match:
                ref, pos_str, alt = match.groups()
                pos = int(pos_str)
                
                if REAL_START_POS <= pos <= RBD_END_POS:

                    idx = pos - REAL_START_POS
                    if 0 <= idx < len(base_rbd):
                        current_aa = base_rbd[idx]
                        if current_aa != ref:
                            pass
                        
                        valid_mutations.append((pos, ref, alt))
            else:
                pos_match = re.search(r'(\d+)', clean_str)
                if pos_match:
                    p = int(pos_match.group(1))
                    if REAL_START_POS <= p <= RBD_END_POS:
                        skipped_indels.append(clean_str)
        except:
            continue
    
    valid_mutations.sort(key=lambda x: x[0])
    for pos, ref, alt in valid_mutations:
        idx = pos - REAL_START_POS
        if 0 <= idx < len(base_rbd):
            base_rbd[idx] = alt
            applied_logs.append(f"{ref}{pos}{alt}")
            
    final_seq = "".join(base_rbd)
    
    seq_filename = f"{lineage}_Standard_RBD.txt"
    with open(seq_filename, "w") as f:
        f.write(final_seq)
        
    print(f"   🧬 标准序列生成成功!")
    print(f"   ✅ 应用 S-RBD 突变 ({len(applied_logs)}个): {', '.join(applied_logs)}")
    if len(skipped_indels) > 0: 
        print(f"   ⚠️ 提示: S蛋白中有 {len(skipped_indels)} 个非替换突变(缺失/插入)已被省略: {', '.join(skipped_indels)}")
    print(f"   💾 已保存: {seq_filename}")





def calculate_mrp(gen_seq, standard_seq, wt_seq, alpha=0.5):

    s_pred = {(i, aa) for i, aa in enumerate(gen_seq) if aa != wt_seq[i]}
    s_known = {(i, aa) for i, aa in enumerate(standard_seq) if aa != wt_seq[i]}

    if not s_known or not s_pred:
        return 0.0
    
    intersection = s_pred.intersection(s_known)
    
    mr = len(intersection) / len(s_known)
    
    mp = len(intersection) / len(s_pred)
    
    mrp = alpha * mr + (1 - alpha) * mp
    
    return mrp, mr, mp


def check_similarity_to_known_variants(generated_seqs, target_lineage_names, wt_seq=None, threshold=0.7, save_path='./'):
    
    return results



def run_experiment(model, device, wt_seq, 
                   initial_population=None,
                   task_name='Exp_MLAEP_Style',
                   INITIAL_SAMPLE_SIZE=500,
                   ROUNDS=10,
                   POP_SIZE=128,
                   TARGET_SCORE=0.12,
                   LOWER_BOUND=0.08,
                   UPPER_BOUND=0.15,
                   MAX_MUTATIONS=15
                   ):
    
    return high_risk_list

# ==========================================
# 5. 主程序入口
# ==========================================
if __name__ == "__main__":

    start_time = time.time()
    print(f"🚀 任务开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

    import gc
    gc.collect()
    torch.cuda.empty_cache()

    set_seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading model...")
    model = covid_prediction_model_chenningning(freeze_bert=True)
    
    model_path = os.path.join(parent_dir, 'trained_model', 'pytorch_model.bin')
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)
        print("Weights loaded.")
    else:
        print(f"Error: Model weights not found at {model_path}")
        sys.exit(1)
        
    model.to(device)
    
    wt_path = '../data/Covid19_RBD_seq.txt'
    if os.path.exists(wt_path):
        with open(wt_path, 'r') as f:
            wt_seq = f.read().strip()
    else:
        print("Warning: WT file not found, using placeholder.")
        wt_seq = "NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKST"


    ini_path = '../data/pVNT_seq.csv'
    ini_seqs=pd.read_csv(ini_path).seq.tolist()


    task_name = 'result_3_vocs_20rounds_30'
    INITIAL_SAMPLE_SIZE = 2000
    ROUNDS = 20
    POP_SIZE = 2048
    run_experiment(
        model, 
        device, 
        wt_seq,
        initial_population = ini_seqs, 
        task_name=task_name,
        INITIAL_SAMPLE_SIZE = INITIAL_SAMPLE_SIZE,
        ROUNDS = ROUNDS,
        POP_SIZE = POP_SIZE,
        TARGET_SCORE = 1.00,
        LOWER_BOUND = 0.70,
        UPPER_BOUND = 1.05,
        MAX_MUTATIONS=30
    )



    target_vocs = sorted(list(set(lineage for lineages in KNOWN_RISKY_MUTATIONS.values() for lineage in lineages)))
    print(f"成功提取 {len(target_vocs)} 个目标谱系进行 MRP 比对分析。")
    
    last_round_file = f'./{task_name}/evolution_history/round_{ROUNDS}.csv'
    
    if os.path.exists(last_round_file):
        last_gen_seqs = pd.read_csv(last_round_file)['Sequence'].tolist()
        
        check_similarity_to_known_variants(
            generated_seqs = last_gen_seqs,
            target_lineage_names = target_vocs,
            wt_seq = wt_seq,
            threshold = 0.5,
            save_path = f'./{task_name}'
        )



    end_time = time.time()
    total_duration = end_time - start_time
    hours = int(total_duration // 3600)
    minutes = int((total_duration % 3600) // 60)
    seconds = int(total_duration % 60)
    print(f"✅ 处理完成！")
    print(f"⏱️ 总耗时: {hours} 小时 {minutes} 分 {seconds} 秒 (共 {total_duration:.2f} 秒)")


    results = check_similarity_to_known_variants(
            generated_seqs = last_gen_seqs,
            target_lineage_names = target_vocs,
            wt_seq = wt_seq,
            threshold = 0.7,
            save_path = f'./{task_name}'
        )
    
import numpy as np

def calculate_mrp_adaptive(gen_seq, standard_seq, wt_seq):
    s_pred = {(i, aa) for i, aa in enumerate(gen_seq) if aa != wt_seq[i]}
    s_known = {(i, aa) for i, aa in enumerate(standard_seq) if aa != wt_seq[i]}
    
    len_p, len_k = len(s_pred), len(s_known)
    if len_k == 0 or len_p == 0: return 0.0, 0.0, 0.0
    
    inter = len(s_pred.intersection(s_known))
    mr = inter / len_k  # Recall
    mp = inter / len_p  # Precision

    alpha = 1 / (1 + np.exp(-0.5 * (len_k - 10)))

    new_muts_count = max(0, len_p - inter)
    
    penalty_lambda = 1.0 / (len_k + 1)
    precision_penalty = np.exp(-penalty_lambda * new_muts_count)

    mrp_score = alpha * mr + (1 - alpha) * (mp * precision_penalty)
    
    return round(mrp_score, 4), round(mr, 4), round(mp, 4)

def check_similarity_to_known_variants_adaptive(generated_seqs, target_lineage_names, wt_seq=None, threshold=0.7, save_path='./'):

    print(f"\n[Analysis] 正在检索与已知谱系 ({', '.join(target_lineage_names)}) 的相似性...")
    
    def get_standard_sequence_by_name(name):
        filename = f"{name}_Standard_RBD.txt"
        file_path = os.path.join(parent_dir, 'analysis', filename)
        
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                sequence = f.read().strip()
                if sequence:
                    return sequence
        else:
            print(f"  📥 本地未找到 {filename}，正在调用 download_and_generate({name})...")
            try:
                download_and_generate(name)
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return f.read().strip()
            except Exception as e:
                print(f"  ❌ 下载或读取 {name} 失败: {e}")
        
        return None

    results = []
    
    for lineage in target_lineage_names:
        standard_seq = get_standard_sequence_by_name(lineage)
        if not standard_seq:
            print(f"  ⚠️ 警告: 无法获取 {lineage} 的标准序列，跳过对比。")
            continue
            
        s_known_set = {(i, aa) for i, aa in enumerate(standard_seq) if aa != wt_seq[i]}
            
        for i, gen_seq in enumerate(generated_seqs):
            s_pred_set = {(i, aa) for i, aa in enumerate(gen_seq) if aa != wt_seq[i]}
            intersection_set = s_pred_set.intersection(s_known_set)

            START_POS=331
            common_muts_list = sorted([f"{wt_seq[idx]}{idx + START_POS}{aa}" for idx, aa in intersection_set])
            common_muts_str = ", ".join(common_muts_list)

            mrp, mr, mp = calculate_mrp_adaptive(gen_seq, standard_seq, wt_seq)#, alpha=0.5)
            
            if mrp >= threshold:
                results.append({
                    'Generated_ID': f"Gen_{i}",
                    'S_pred': len(s_pred_set),
                    'Generated_Sequence': gen_seq,
                    'Target_Lineage': lineage,
                    'S_known': len(s_known_set),
                    'Intersection': len(intersection_set),
                    'Common_Mutations': common_muts_str,
                    'Similarity_Score': round(mrp, 4),
                    'MR': round(mr, 4),
                    'MP': round(mp, 4)
                })
    
    if results:
        df_sim = pd.DataFrame(results)
        cols = ['Generated_ID', 'S_pred', 'Target_Lineage', 'S_known', 'Intersection', 'Common_Mutations', 'MR', 'MP', 'Similarity_Score', 'Generated_Sequence']
        save_file = os.path.join(save_path, 'known_variants_similarity_match.csv')
        df_sim.to_csv(save_file, index=False)
        
        print(f"  ✅ 发现 {len(results)} / {len(generated_seqs)} 条生成与已知高危序列高度相似的记录！")
        print(f"  详细结果已保存至: {save_file}")
        
        print("-" * 85)
        print(f"{'Gen_ID':<10} | {'S_pred':<6} | {'Target':<12} | {'S_knw':<6} | {'Inter':<6} | {'MR':<6} | {'MP':<6} | {'MRP':<8} | {'Common Mutations'}")
        print("-" * 85)
        for r in results[:10]:
            display_muts = (r['Common_Mutations'][:30] + '...') if len(r['Common_Mutations']) > 30 else r['Common_Mutations']
            print(f"{r['Generated_ID']:<10} | {r['S_pred']:<6} | {r['Target_Lineage']:<12} | {r['S_known']:<6} | {r['Intersection']:<6} | {r['MR']:<6} | {r['MP']:<6} | {r['Similarity_Score']:<8} | {display_muts}")
    else:
        print(f"  ℹ️ 未发现相似度高于 {threshold} 的生成序列。")
    
    return results


from my_mutations import KNOWN_RISKY_MUTATIONS
import os
import pandas as pd
import numpy as np

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)

wt_path = '../data/Covid19_RBD_seq.txt'
if os.path.exists(wt_path):
    with open(wt_path, 'r') as f:
        wt_seq = f.read().strip()

target_vocs = sorted(list(set(lineage for lineages in KNOWN_RISKY_MUTATIONS.values() for lineage in lineages)))
print(f"成功提取 {len(target_vocs)} 个目标谱系进行 MRP 比对分析。")

task_name = 'result_3_vocs_20rounds'# result_trial_test
ROUNDS=20
last_round_file = f'./{task_name}/evolved_high_risk_population.csv'

if os.path.exists(last_round_file):
    last_gen_seqs = pd.read_csv(last_round_file)['Sequence'].tolist()

results = check_similarity_to_known_variants_adaptive(
    generated_seqs = last_gen_seqs,
    target_lineage_names = target_vocs,
    wt_seq = wt_seq,
    threshold = 0.5,
    save_path = f'./{task_name}'
)