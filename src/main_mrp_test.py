from my_mutations import KNOWN_RISKY_MUTATIONS
import os
import sys
import pandas as pd
import numpy as np

class Tee(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

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

    if len_k <5:
        mrp_score = alpha * mr + (1 - alpha) * (mp * precision_penalty)
    else:
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
        print(df_sim['Generated_ID'].value_counts().head(5))
    else:
        print(f"  ℹ️ 未发现相似度高于 {threshold} 的生成序列。")
    
    return results

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
save_log_path = os.path.join(f'./{TASK_NAME}', 'known_variants_similarity_match.log')
os.makedirs(os.path.dirname(save_log_path), exist_ok=True)
sys.stdout = Tee(save_log_path)

wt_path = '../data/Covid19_RBD_seq.txt'
if os.path.exists(wt_path):
    with open(wt_path, 'r') as f:
        wt_seq = f.read().strip()

target_vocs = sorted(list(set(lineage for lineages in KNOWN_RISKY_MUTATIONS.values() for lineage in lineages)))
print(f"成功提取 {len(target_vocs)} 个目标谱系进行 MRP 比对分析。")


task_name = TASK_NAME
ROUNDS=50
last_round_file = f'./{task_name}/evolved_high_risk_population.csv'# f'./{task_name}/evolution_history/round_{ROUNDS}.csv'

if os.path.exists(last_round_file):
    last_gen_seqs = pd.read_csv(last_round_file)['Sequence'].tolist()

results = check_similarity_to_known_variants_adaptive(
    generated_seqs = last_gen_seqs,
    target_lineage_names = target_vocs,
    wt_seq = wt_seq,
    threshold = THRESHOLD,
    save_path = f'./{task_name}'
)