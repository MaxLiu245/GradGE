import torch
from torch._C import device
import torch.nn as nn
import os
import sys
import esm
from transformers.file_utils import ModelOutput
from typing import Optional
import pandas as pd
import numpy as np
dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(dir) 
from utils.struct2vec_utils import StructEmbed
from utils.struct2vec_utils import StructureDataset
from utils.struct2vec_utils import featurize
import os
from dataclasses import dataclass

@dataclass
class output(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: tuple = None
    embedding: Optional[torch.FloatTensor] =None


class ClassPredictionHead(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hid_dim: int,
                 out_dim: int,
                 droupout: float=0.1) -> None:
        super().__init__()
        self.fc_layer = nn.Sequential(
            nn.Linear(in_dim,hid_dim,bias=True),
            nn.ReLU(),
            nn.Dropout(droupout,inplace=False),# True???
            nn.Linear(hid_dim,out_dim)
            )
   
    def forward(self,pooled_output):
        value_pred = self.fc_layer(pooled_output)
        outputs = value_pred
        return(outputs)

class covid_prediction_model(nn.Module):
    def __init__(self,
                 jsonl_path: str=os.path.join(dir,'data/merged_all.jsonl'),
                 freeze_bert: bool=False,
                 seq_embedding_size: int=1280,
                 stru_embedding_size: int=5,
                 stru_seq_len:int=130,
                 dropout_prob: float=0.1,
                 pooling: str="first_token") -> None:
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#torch.device("cpu")#
        self.esm_model = esm.pretrained.esm2_t33_650M_UR50D()[0].to(device)# esm.pretrained.esm1b_t33_650M_UR50S()[0].to(device)
        self.model_name ="esm2_t33_650M_UR50D"# "esm1b_t33_650M_UR50S"
        self.model_layers = self.model_name.split("_")[1][1:]
        self.repr_layers = int(self.model_layers)
        self.predict_class = ClassPredictionHead(seq_embedding_size+stru_embedding_size*stru_seq_len*2,512,2,dropout_prob).to(device)
        self.pooling = pooling
        self.structEmbed = StructEmbed(node_features=128,edge_features=128,hidden_dim=128,out_dim=stru_embedding_size).to(device)
        self.antibody_data = StructureDataset(jsonl_file=jsonl_path, truncate=None, max_length=500)
        X, S, mask, lengths = featurize(self.antibody_data, device=device, shuffle_fraction=0)
        self.X=X
        self.S=S
        self.mask=mask
        self.lengths=lengths

        # 为了attack看logits，加点料
        self.batch_converter = esm.pretrained.esm2_t33_650M_UR50D()[1].get_batch_converter()
        self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()[1]

        if freeze_bert:
            for p in self.esm_model.parameters():
                p.requires_grad = False
        
    def forward(self,input_ids,labels):
        outputs = self.esm_model(input_ids, repr_layers=[self.repr_layers])["representations"][self.repr_layers]
        b=outputs.shape[0]
        device=outputs.device
        if self.pooling == "first_token":
            outputs = outputs[:,0,:]
        elif self.pooling == "average" :
            outputs = outputs.mean(1)
        else:
            ValueError("Not implemented! Please choose the pooling methods from first_token/average")
        outputs = torch.repeat_interleave(outputs.unsqueeze(dim=1),repeats=9,dim=1)

        Struct_embedding = self.structEmbed(self.X,self.S,self.lengths,self.mask)
        Struct_embedding = torch.cat((Struct_embedding[0::2,:,:],Struct_embedding[1::2,:,:]),dim=1)
        Struct_embedding = torch.flatten(Struct_embedding,start_dim=1)

        Struct_embedding = torch.repeat_interleave(Struct_embedding.unsqueeze(dim=0),repeats=b,dim=0)

        combined_embedding = torch.cat((Struct_embedding,outputs),dim=2)

        class_logits = self.predict_class(combined_embedding)
        loss = None
        
        if labels is not None:
            pos_weight=torch.tensor([[0.08,0.92],[0.18,0.82],[0.06,0.94],[0.1,0.9],[0.08,0.92],[0.1,0.9],[0.05,0.95],[0.04,0.96],[0.17,0.83]],device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            if labels.shape[1]==10:
                labels=labels[:,0:9,:]
            loss = criterion(class_logits,labels)
        # print(class_logits[0,:,:])
        return(output(
        loss = loss,
        logits = class_logits,
        embedding = combined_embedding
        ))

class covid_prediction_model_chenningning(nn.Module):
    def __init__(self,
                 jsonl_path: str=os.path.join(dir,'data/merged_all.jsonl'),
                 freeze_bert: bool=False,
                 seq_embedding_size: int=1280,
                 stru_embedding_size: int=5,
                 stru_seq_len:int=130,
                 dropout_prob: float=0.1,
                 pooling: str="first_token") -> None:
        super().__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.esm_model = esm.pretrained.esm1b_t33_650M_UR50S()[0].to(device)
        self.alphabet = esm.pretrained.esm1b_t33_650M_UR50S()[1]# 再单独取 alphabet（不会加载第二个模型）需要加这个
        self.model_name ="esm1b_t33_650M_UR50S"
        self.model_layers = self.model_name.split("_")[1][1:]
        self.repr_layers = int(self.model_layers)
        self.predict_class = ClassPredictionHead(seq_embedding_size+stru_embedding_size*stru_seq_len*2,512,2,dropout_prob).to(device)
        self.pooling = pooling
        self.structEmbed = StructEmbed(node_features=128,edge_features=128,hidden_dim=128,out_dim=stru_embedding_size).to(device)
        self.antibody_data = StructureDataset(jsonl_file=jsonl_path, truncate=None, max_length=500)
        X, S, mask, lengths = featurize(self.antibody_data, device=device, shuffle_fraction=0)
        self.X=X
        self.S=S
        self.mask=mask
        self.lengths=lengths

        if freeze_bert:
            for p in self.esm_model.parameters():
                p.requires_grad = False
        
    def forward(self,input_ids,labels):
        outputs = self.esm_model(input_ids, repr_layers=[self.repr_layers])["representations"][self.repr_layers]
        b=outputs.shape[0]
        device=outputs.device
        if self.pooling == "first_token":
            outputs = outputs[:,0,:]
        elif self.pooling == "average" :
            outputs = outputs.mean(1)
        else:
            ValueError("Not implemented! Please choose the pooling methods from first_token/average")
        outputs = torch.repeat_interleave(outputs.unsqueeze(dim=1),repeats=9,dim=1)

        Struct_embedding = self.structEmbed(self.X,self.S,self.lengths,self.mask)
        Struct_embedding = torch.cat((Struct_embedding[0::2,:,:],Struct_embedding[1::2,:,:]),dim=1)
        Struct_embedding = torch.flatten(Struct_embedding,start_dim=1)

        Struct_embedding = torch.repeat_interleave(Struct_embedding.unsqueeze(dim=0),repeats=b,dim=0)

        combined_embedding = torch.cat((Struct_embedding,outputs),dim=2)

        class_logits = self.predict_class(combined_embedding)
        loss = None
        
        if labels is not None:
            pos_weight=torch.tensor([[0.08,0.92],[0.18,0.82],[0.06,0.94],[0.1,0.9],[0.08,0.92],[0.1,0.9],[0.05,0.95],[0.04,0.96],[0.17,0.83]],device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            if labels.shape[1]==10:
                labels=labels[:,0:9,:]
            loss = criterion(class_logits,labels)
        return(output(
        loss = loss,
        logits = class_logits,
        embedding = combined_embedding
        ))



class LabelDifference(nn.Module):
    def __init__(self, distance_type='l1'):
        super(LabelDifference, self).__init__()
        self.distance_type = distance_type

    def forward(self, labels):
        # labels: [bs, label_dim]
        # output: [bs, bs]
        if self.distance_type == 'l1':
            return torch.abs(labels[:, None] - labels[None, :])
        else:
            raise ValueError(self.distance_type)

class FeatureSimilarity(nn.Module):
    def __init__(self, similarity_type='l2'):
        super(FeatureSimilarity, self).__init__()
        self.similarity_type = similarity_type

    def forward(self, features):
        # labels: [bs, feat_dim]
        # output: [bs, bs]
        if self.similarity_type == 'l2':
            return - (features[:, None, :] - features[None, :, :]).norm(2, dim=-1)
        else:
            raise ValueError(self.similarity_type)

class RnCLoss(nn.Module):
    def __init__(self, temperature=2, label_diff='l1', feature_sim='l2', augmentation=True):
        super(RnCLoss, self).__init__()
        self.t = temperature
        self.augmentation = augmentation
        self.label_diff_fn = LabelDifference(label_diff)
        self.feature_sim_fn = FeatureSimilarity(feature_sim)

    def forward(self, features, labels):
        # features: [bs, 2, feat_dim] if augmentation, else [bs, feat_dim]
        # labels: [bs, label_dim]
        
        if self.augmentation:
            features = torch.cat([features[:, 0], features[:, 1]], dim=0)  # [2bs, feat_dim]
            labels = labels.repeat(2, 1)  # [2bs, label_dim]
        else:
            pass

        label_diffs = self.label_diff_fn(labels)
        logits = self.feature_sim_fn(features).div(self.t)
        # stable softmax
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits -= logits_max.detach()
        exp_logits = logits.exp()

        n = logits.shape[0]  # n = 2bs if augmentation else bs

        # remove diagonal, needed for both augmentation and non-augmentation
        logits = logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        exp_logits = exp_logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        label_diffs = label_diffs.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)

        loss = 0.
        for k in range(n - 1):
            pos_logits = logits[:, k]  # 2bs
            pos_label_diffs = label_diffs[:, k]  # 2bs
            neg_mask = (label_diffs >= pos_label_diffs.view(-1, 1)).float()  # [2bs, 2bs - 1]
            pos_log_probs = pos_logits - torch.log((neg_mask * exp_logits).sum(dim=-1))  # 2bs
            loss += - (pos_log_probs / (n * (n - 1))).sum()

        return loss
class LabelDifferenceNumPy:
    def __init__(self, distance_type='l1'):
        self.distance_type = distance_type

    def __call__(self, labels):
        # labels: [n, label_dim] -> 支持多维标签
        if self.distance_type == 'l1':
            # 利用广播机制计算 [n, 1, dim] - [1, n, dim]
            diff = np.abs(labels[:, np.newaxis] - labels[np.newaxis, :])
            # 如果标签是多维的，通常取均值或求和转为 [n, n]
            if diff.ndim > 2:
                diff = np.mean(diff, axis=-1)
            return diff
        else:
            raise ValueError(self.distance_type)

class FeatureSimilarityNumPy:
    def __init__(self, similarity_type='l2'):
        self.similarity_type = similarity_type

    def __call__(self, features):
        # features: [n, feat_dim]
        if self.similarity_type == 'l2':
            # 计算欧式距离：- sqrt(sum((a-b)^2))
            diff = features[:, np.newaxis, :] - features[np.newaxis, :, :]
            dist = np.linalg.norm(diff, ord=2, axis=-1)
            return -dist
        else:
            raise ValueError(self.similarity_type)

class RnCLossNumPy:
    def __init__(self, temperature=2, label_diff='l1', feature_sim='l2', augmentation=False):
        self.t = temperature
        self.augmentation = augmentation
        self.label_diff_fn = LabelDifferenceNumPy(label_diff)
        self.feature_sim_fn = FeatureSimilarityNumPy(feature_sim)

    def __call__(self, features, labels):
        # 确保输入是 NumPy 数组
        if hasattr(features, 'detach'): features = features.detach().cpu().numpy()
        if hasattr(labels, 'detach'): labels = labels.detach().cpu().numpy()

        # 这里的 labels 形状如果是 [n, 9]，建议循环外面只传一个任务的列进来
        if labels.ndim == 1:
            labels = labels[:, np.newaxis]

        if self.augmentation:
            # 对应原代码 [bs, 2, dim] 的处理
            features = np.concatenate([features[:, 0], features[:, 1]], axis=0)
            labels = np.tile(labels, (2, 1))

        n = features.shape[0]
        
        # 1. 计算相似度矩阵和标签差异矩阵 [n, n]
        label_diffs = self.label_diff_fn(labels)
        logits = self.feature_sim_fn(features) / self.t

        # 2. Stable Softmax 技巧
        logits_max = np.max(logits, axis=1, keepdims=True)
        logits = logits - logits_max
        exp_logits = np.exp(logits)

        # 3. 移除对角线元素 (Mask self-similarity)
        # 创建一个对角线为 False 的 mask
        mask = ~np.eye(n, dtype=bool)
        
        # 将矩阵平铺并重新 view 成 [n, n-1]
        logits = logits[mask].reshape(n, n - 1)
        exp_logits = exp_logits[mask].reshape(n, n - 1)
        label_diffs = label_diffs[mask].reshape(n, n - 1)

        loss = 0.0
        # 4. 核心计算循环
        for k in range(n - 1):
            pos_logits = logits[:, k]  # 当前作为正样本的 logit [n]
            pos_label_diffs = label_diffs[:, k]  # 当前正样本的标签距离 [n]
            
            # neg_mask: 只有标签距离大于等于当前正样本标签距离的才作为分母 (Rank-based)
            # 形状: [n, n-1]
            neg_mask = (label_diffs >= pos_label_diffs[:, np.newaxis]).astype(np.float32)
            
            # 计算分母的 log-sum-exp
            denominator = np.log(np.sum(neg_mask * exp_logits, axis=-1) + 1e-12)
            
            # log(exp(pos)/sum(exp(neg))) = pos - log(sum(exp(neg)))
            pos_log_probs = pos_logits - denominator
            loss += - np.sum(pos_log_probs / (n * (n - 1)))

        return loss
class covid_prediction_model_rnc_version(nn.Module):
    def __init__(self,
                 jsonl_path: str=os.path.join(dir,'data/merged_all.jsonl'),
                 freeze_bert: bool=False,
                 seq_embedding_size: int=1280,
                 stru_embedding_size: int=5,
                 stru_seq_len:int=130,
                 dropout_prob: float=0.1,
                 pooling: str="first_token",
                 temp: float=2,# add rnc setting
                 label_diff: str='l1',# add rnc setting
                 feature_sim: str='l2'# add rnc setting
                 ) -> None:
        super().__init__()
        torch.device("cuda" if torch.cuda.is_available() else "cpu")#device = torch.device("cpu")# 
        self.esm_model = esm.pretrained.esm2_t33_650M_UR50D()[0].to(device)# esm.pretrained.esm1b_t33_650M_UR50S()[0].to(device)
        self.model_name ="esm2_t33_650M_UR50D"# "esm1b_t33_650M_UR50S"
        self.model_layers = self.model_name.split("_")[1][1:]
        self.repr_layers = int(self.model_layers)
        self.predict_class = ClassPredictionHead(seq_embedding_size+stru_embedding_size*stru_seq_len*2,512,2,dropout_prob).to(device)
        self.pooling = pooling
        self.structEmbed = StructEmbed(node_features=128,edge_features=128,hidden_dim=128,out_dim=stru_embedding_size).to(device)
        self.antibody_data = StructureDataset(jsonl_file=jsonl_path, truncate=None, max_length=500)
        X, S, mask, lengths = featurize(self.antibody_data, device=device, shuffle_fraction=0)
        self.X=X
        self.S=S
        self.mask=mask
        self.lengths=lengths

        # rnc loss
        self.temp=temp
        self.label_diff=label_diff
        self.feature_sim=feature_sim

        # 为了attack看logits，加点料
        self.batch_converter = esm.pretrained.esm2_t33_650M_UR50D()[1].get_batch_converter()
        self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()[1]

        if freeze_bert:
            for p in self.esm_model.parameters():
                p.requires_grad = False
        
    def forward(self,input_ids,labels,return_embedding=True):
        outputs = self.esm_model(input_ids, repr_layers=[self.repr_layers])["representations"][self.repr_layers]
        b=outputs.shape[0]
        device=outputs.device
        if self.pooling == "first_token":
            outputs = outputs[:,0,:]
        elif self.pooling == "average" :
            outputs = outputs.mean(1)
        else:
            ValueError("Not implemented! Please choose the pooling methods from first_token/average")
        outputs = torch.repeat_interleave(outputs.unsqueeze(dim=1),repeats=9,dim=1)

        Struct_embedding = self.structEmbed(self.X,self.S,self.lengths,self.mask)
        Struct_embedding = torch.cat((Struct_embedding[0::2,:,:],Struct_embedding[1::2,:,:]),dim=1)
        Struct_embedding = torch.flatten(Struct_embedding,start_dim=1)

        Struct_embedding = torch.repeat_interleave(Struct_embedding.unsqueeze(dim=0),repeats=b,dim=0)

        combined_embedding = torch.cat((Struct_embedding,outputs),dim=2)

        # 如果是 RnC 阶段，分别对 9 个任务计算 Loss
        if return_embedding:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            self.current_task_losses = {} # 用于日志记录
            if labels is not None:
                # 注意：RnC 应该输入原始连续分数标签
                # print(labels.shape)# torch.Size([16, 9, 2])原文是这个，我的原始分数是torch.Size([16, 9])，对了

                criterion = RnCLoss(
                    temperature=self.temp, 
                    label_diff=self.label_diff, 
                    feature_sim=self.feature_sim,
                    augmentation=False
                )

                # combined_embedding 形状: [Batch, 9, 2580]
                # labels 形状: [Batch, 9]
                total_loss = 0.0
                # 遍历 9 个任务分别计算
                for i in range(9):
                    task_embedding = combined_embedding[:, i, :] # [Batch, 2580]
                    task_labels = labels[:, i]                  # [Batch]
                    
                    # 对第 i 个任务计算 RnC Loss
                    # print(task_embedding.shape, task_labels.shape)
                    #原文是torch.Size([16, 2580]) torch.Size([16, 2])
                    # 我的原始分数是torch.Size([16, 2580]) torch.Size([16])，对了
                    task_loss = criterion(task_embedding, task_labels)
                    total_loss += task_loss
                    # 记录每一个任务的 loss，方便 RnCTrainer.log 打印
                    self.current_task_losses[f"rnc_{i+1}"] = task_loss.detach().item()
                
                # 取平均值作为最终 loss
                loss = total_loss / 9.0
            else:
                # 如果没有标签，给一个 0 值的 tensor 以防 Trainer 报错
                loss = torch.tensor(0.0, device=device, requires_grad=True)

            return output(
                loss = loss,
                logits = None,
                embedding = combined_embedding
            )
        return combined_embedding # 形状为 [Batch, 9, 2580]

class covid_prediction_model_score_version(nn.Module):
    def __init__(self,
                 jsonl_path: str=os.path.join(dir,'data/merged_all.jsonl'),
                 freeze_bert: bool=False,
                 seq_embedding_size: int=1280,
                 stru_embedding_size: int=5,
                 stru_seq_len:int=130,
                 dropout_prob: float=0.1,
                 pooling: str="first_token") -> None:
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#torch.device("cpu")#
        self.esm_model = esm.pretrained.esm2_t33_650M_UR50D()[0].to(device)# esm.pretrained.esm1b_t33_650M_UR50S()[0].to(device)
        self.model_name ="esm2_t33_650M_UR50D"# "esm1b_t33_650M_UR50S"
        self.model_layers = self.model_name.split("_")[1][1:]
        self.repr_layers = int(self.model_layers)
        self.predict_class = ClassPredictionHead(seq_embedding_size+stru_embedding_size*stru_seq_len*2,512,2,dropout_prob).to(device)
        self.predict_reg = nn.Sequential(
            nn.Linear(seq_embedding_size + stru_embedding_size * stru_seq_len * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(512, 1) # 最后一个维度从 2 改为 1
        )
        self.pooling = pooling
        self.structEmbed = StructEmbed(node_features=128,edge_features=128,hidden_dim=128,out_dim=stru_embedding_size).to(device)
        self.antibody_data = StructureDataset(jsonl_file=jsonl_path, truncate=None, max_length=500)
        X, S, mask, lengths = featurize(self.antibody_data, device=device, shuffle_fraction=0)
        self.X=X
        self.S=S
        self.mask=mask
        self.lengths=lengths

        # 为了attack看logits，加点料
        self.batch_converter = esm.pretrained.esm2_t33_650M_UR50D()[1].get_batch_converter()
        self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()[1]

        if freeze_bert:
            for p in self.esm_model.parameters():
                p.requires_grad = False
        
    def forward(self,input_ids,labels):
        outputs = self.esm_model(input_ids, repr_layers=[self.repr_layers])["representations"][self.repr_layers]
        b=outputs.shape[0]
        device=outputs.device
        if self.pooling == "first_token":
            outputs = outputs[:,0,:]
        elif self.pooling == "average" :
            outputs = outputs.mean(1)
        else:
            ValueError("Not implemented! Please choose the pooling methods from first_token/average")
        outputs = torch.repeat_interleave(outputs.unsqueeze(dim=1),repeats=9,dim=1)

        Struct_embedding = self.structEmbed(self.X,self.S,self.lengths,self.mask)
        Struct_embedding = torch.cat((Struct_embedding[0::2,:,:],Struct_embedding[1::2,:,:]),dim=1)
        Struct_embedding = torch.flatten(Struct_embedding,start_dim=1)

        Struct_embedding = torch.repeat_interleave(Struct_embedding.unsqueeze(dim=0),repeats=b,dim=0)

        combined_embedding = torch.cat((Struct_embedding,outputs),dim=2)

        pred = self.predict_reg(combined_embedding)
        loss = None
        
        if labels is not None:
            # 确保 labels 形状为 [B, 9, 1] 以匹配 class_logits
            if labels.dim() == 2: # 如果 labels 是 [B, 9]
                labels = labels.unsqueeze(-1)
                
            # 回归任务不再使用 pos_weight
            criterion = nn.L1Loss() 
            # 或者为了更稳健：criterion = nn.HuberLoss()
            
            loss = criterion(pred, labels)
        return(output(
        loss = loss,
        logits = pred,
        embedding = combined_embedding
        ))

    
class covid_prediction_model_without_GCN(nn.Module):
    def __init__(self,
                 freeze_bert: bool=False,
                 esm_path: str="./model",
                 seq_embedding_size: int=1280,
                 dropout_prob: float=0.1,
                 pos_weight:torch.tensor=torch.tensor([0.1,0.9]),
                 pooling: str="first_token") -> None:
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.esm_model = esm.pretrained.load_model_and_alphabet_local(esm_path)[0].to(device)
        self.model_name = os.path.basename(esm_path)
        self.model_layers = self.model_name.split("_")[1][1:]
        self.repr_layers = int(self.model_layers)
        self.predict_class = ClassPredictionHead(seq_embedding_size,512,2,dropout_prob).to(device)
        self.pooling = pooling
        self.pos_weight=pos_weight
        if freeze_bert:
            for p in self.esm_model.parameters():
                p.requires_grad = False
    def forward(self,input_ids,labels):
        outputs = self.esm_model(input_ids, repr_layers=[self.repr_layers])["representations"][self.repr_layers]
        b=outputs.shape[0]
        device=outputs.device
        if self.pooling == "first_token":
            outputs = outputs[:,0,:]
        elif self.pooling == "average" :
            outputs = outputs.mean(1)
        else:
            ValueError("Not implemented! Please choose the pooling methods from first_token/average")
        class_logits = self.predict_class(outputs) 
        loss = None
        if labels is not None:
            criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
            loss = criterion(class_logits,labels)
        return(output(
        loss = loss,
        logits = class_logits,
        embedding = outputs
        ))


class covid_prediction_model_without_GCN_add_noise(nn.Module):
    def __init__(self,
                 freeze_bert: bool=False,
                 esm_path: str="./model",
                 seq_embedding_size: int=1280,
                 dropout_prob: float=0.1,
                 pooling: str="first_token") -> None:
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.esm_model = esm.pretrained.load_model_and_alphabet_local(esm_path)[0].to(device)
        self.model_name = os.path.basename(esm_path)
        self.model_layers = self.model_name.split("_")[1][1:]
        self.repr_layers = int(self.model_layers)
        self.predict_class = ClassPredictionHead(seq_embedding_size+1300,512,2,dropout_prob).to(device)
        self.pooling = pooling
        if freeze_bert:
            for p in self.esm_model.parameters():
                p.requires_grad = False
    def forward(self,input_ids,labels):
        outputs = self.esm_model(input_ids, repr_layers=[self.repr_layers])["representations"][self.repr_layers]
        b=outputs.shape[0]
        device=outputs.device
        if self.pooling == "first_token":
            outputs = outputs[:,0,:]
        elif self.pooling == "average" :
            outputs = outputs.mean(1)
        else:
            ValueError("Not implemented! Please choose the pooling methods from first_token/average")
        outputs = torch.repeat_interleave(outputs.unsqueeze(dim=1),repeats=9,dim=1)
        torch.manual_seed(3)
        noise=torch.randn((b,9,1300),device=device)
        combined_embedding = torch.cat((noise,outputs),dim=2)
        class_logits = self.predict_class(combined_embedding) 
        loss = None
        if labels is not None:
            pos_weight=torch.tensor([[0.08,0.92],[0.18,0.82],[0.06,0.94],[0.1,0.9],[0.08,0.92],[0.1,0.9],[0.05,0.95],[0.04,0.96],[0.17,0.83]],device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            if labels.shape[1]==10:
                labels=labels[:,0:9,:]
            loss = criterion(class_logits,labels)
        return(output(
        loss = loss,
        logits = class_logits,
        embedding = combined_embedding
        ))




# 从这里，我添加ved代码，以及直接把ved加到mlaep原始模型中的代码
import argparse
from esm import ESM2
import torch.nn.functional as F
def config_rep(device, protein, level, reduce_dim=None):
    assert protein in ['GFP', 'AAV', 'RBD']
    args = argparse.Namespace()
    args.name = protein
    args.device = device 
    args.level = level 
    args.embed_dim = 1280
    args.num_layers = 33
    args.hidden_dim = 256
    if protein == 'GFP':
        args.length = 237
        args.num_tokens = 237 + 2
        args.reduce_dim = 32
        args.num_trainable_layers = 4
    elif protein == 'AAV':
        args.length = 28
        args.num_tokens = 28 + 2
        args.reduce_dim = 16
        args.num_trainable_layers = 4
    elif protein == 'RBD':
        args.length = 201
        args.num_tokens = 201 + 2
        args.reduce_dim = 32
        args.num_trainable_layers = 4
    else:
        raise NotImplementedError()
    if reduce_dim != None:
        args.reduce_dim = reduce_dim
    return args

os.environ["TORCH_HOME"] = "/mnt/fastscratch/users/zmliu/.cache/torch"# "/home/ubuntu/.cache/torch"
current_dir = os.getcwd()
print('current_dir:', current_dir)
checkpoint_path = os.path.join(
    os.environ["TORCH_HOME"],
    "hub", "checkpoints", "esm2_t33_650M_UR50D.pt"# esm1b_t33_650M_UR50S
)
if not os.path.exists(checkpoint_path):
    print(f"Checkpoint not found at {checkpoint_path}")
    print("Downloading pretrained ESM2 model (this may take several minutes)...")
    esm_model,alphabet = esm.pretrained.esm2_t33_650M_UR50D()# esm1b_t33_650M_UR50S
    
    if os.path.exists(checkpoint_path):
        print(f"Download completed and saved to {checkpoint_path}")
    else:
        print("Download failed. Please check your network or TORCH_HOME permissions.")
else:
    print(f"Found existing checkpoint: {checkpoint_path}")

class VED(nn.Module):
    def __init__(self, cfg, pretrained=None, esm_pretrained=None):
        super().__init__()
        assert pretrained == None or esm_pretrained == None, 'Both VED checkpoint and ESM-2 checkpoint are given'

        # print(cfg.reduce_dim, cfg.embed_dim)

        self.alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
        self.batch_converter = self.alphabet.get_batch_converter()
        self.device = cfg.device
        self.esm_num_layers = cfg.num_layers
        self.num_tokens = cfg.num_tokens
        self.cfg = cfg

        self.encoder = ESM2(num_layers=cfg.num_layers, embed_dim=cfg.embed_dim, attention_heads=20, alphabet=self.alphabet, token_dropout=False)
        self.encoder.load_state_dict(self.load_esm_ckpt(checkpoint_path), strict=False)
        self.encoder.requires_grad_(False)
        self.reduce = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.reduce_dim),
            nn.Tanh()
        )

        self.lm = ESM2(num_layers=cfg.num_layers, embed_dim=cfg.embed_dim, attention_heads=20, alphabet=self.alphabet, token_dropout=False)
        self.lm.requires_grad_(False)
        for i, layer in enumerate(self.lm.layers):
            if i < cfg.num_trainable_layers:
                layer.requires_grad_ = True 
                
        self.rep_recover = nn.Sequential(
            nn.Linear(cfg.reduce_dim, cfg.embed_dim),
            nn.LeakyReLU(),
            nn.Linear(cfg.embed_dim, cfg.embed_dim)
        )

        if pretrained == None:
            self.lm.load_state_dict(self.load_esm_ckpt(esm_pretrained), strict=False)
        else:
            self.load_state_dict(self.load_ckpt(pretrained))

    def load_esm_ckpt(self, esm_pretrained):
        ckpt = {}
        model_data = torch.load(esm_pretrained, weights_only=False)["model"]
        for k in model_data:
            if 'lm_head' in k:
                ckpt[k.replace('encoder.','')] = model_data[k]
            else:
                ckpt[k.replace('encoder.sentence_encoder.','')] = model_data[k]
        return ckpt

    def load_ckpt(self, pretrained):
        ckpt = {}
        model_data = torch.load(pretrained, map_location=self.device)
        for k in model_data:
            ckpt[k.replace('module.','')] = model_data[k]
        return ckpt
    
    def compose_input(self, list_tuple_seq):
        _, _, batch_tokens = self.batch_converter(list_tuple_seq)
        batch_tokens = batch_tokens.to(self.device)
        return batch_tokens 

    def set_wt_tokens(self, wt_seq):
        tokens = self.compose_input([('protein', wt_seq)])
        with torch.no_grad():
            encoded = self.encoder(tokens, set([self.esm_num_layers])) 
            encoded = encoded["representations"][self.esm_num_layers]
            encoded = encoded[:, 0]
            # self.wt_encoded = encoded
            self.wt_encoded = nn.Parameter(encoded, requires_grad=False)# 尝试兼容多卡
            
            padding_mask = tokens.eq(self.lm.padding_idx)  # B, T
            self.padding_mask = padding_mask
            x = self.lm.embed_scale * self.lm.embed_tokens(tokens)
            if padding_mask is not None:
                x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
                x = x.transpose(0, 1)
            self.wt_embed = x

    def encode(self, input):
        if isinstance(input, str):
            tokens = self.compose_input([('protein', input)])
        elif isinstance(input, list):
            tokens = self.compose_input([('protein', seq) for seq in input])
        else:
            tokens = input
            
        with torch.no_grad():
            output = self.encoder(tokens, set([self.esm_num_layers])) 
        x = output["representations"][self.esm_num_layers]
        x = x[:, 0]
        x = x - self.wt_encoded
        x = self.reduce(x)
        # print(f'input shape {input.shape}, wt_encoded shape {self.wt_encoded.shape}, reduced x shape {x.shape}')
        # input shape torch.Size([4, 203]), wt_encoded shape torch.Size([1, 1280]), reduced x shape torch.Size([4, 32])
        return x

    def decode(self, repr, to_seq=False, template=None, topk=None):
        sos = self.rep_recover(repr)
        x = torch.clone(self.wt_embed).repeat(1, repr.size(0), 1).to(repr.device)
        x[0] = sos
        for _, layer in enumerate(self.lm.layers):
            x, _ = layer(
                x,
                self_attn_padding_mask=self.padding_mask.to(repr.device).repeat(repr.size(0), 1),
                need_head_weights=False,
            )
        x = self.lm.emb_layer_norm_after(x)
        x = x.transpose(0, 1)
        logits = self.lm.lm_head(x)
        if to_seq: # constrained decoding
            tokens = torch.argmax(logits[:,1:-1,4:24], dim=-1)  
            if topk != None: 
                indices = torch.topk(torch.max(logits[:,1:-1,4:24], dim=-1).values, topk).indices
                indices = indices.flatten().tolist()
                sequences = [''.join([self.alphabet.all_toks[i+4] if i in indices else t for t, i in zip(list(template),sequence.tolist())]) for sequence in list(tokens)]
            else:
                sequences = [''.join([self.alphabet.all_toks[i+4] for i in sequence.tolist()]) for sequence in list(tokens)]
            if len(sequences) == 1:                
                sequences = sequences[0]
            return sequences
        return logits

    def forward(self, input, return_rep=False):
        repr = self.encode(input)
        logits = self.decode(repr)
        if return_rep:
            return logits, repr
        # print(input.shape, repr.shape, logits.shape)
        return logits
protein = 'RBD'# args.protein
level = 'all'
return_rep=True
REFSEQ = {
    'AAV': {
        'medium': 'DEEEIRTTNPVATEQYGSVETPDEVGNC',
        'hard': 'DEEEIRTTNPFATEQYGSVEEGECQGDF'
    },
    'GFP': {
        'medium': 'SKGEELFTGVVPILVELDGDVNGHKSSVSGEGEGDATYGKLTLKFICTTGKLPVPRPTLATTLSYGVQCLSRYPDHMRQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVSFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK',
        'hard': 'SKGEELFTGVVPILVELDGDVDGHKFSVSGEGEGDATYGKLTLKSICTTGKLPVPWPALVTTLSYGVQCFSRYPDHMKQHDFFKSAMPVGYVQERTIFLKDDGNYKTRAEVRFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEGGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK'
    },
    'RBD': {
        'all': 'NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKST'
    }
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = config_rep(device, protein, level)# 
ved = VED(config, esm_pretrained=checkpoint_path)
# "/home/ubuntu/lzm/Covid-predict/model/RBD_all_15-09:54_epoch_32.pt"
ved_params = torch.load("/users/zmliu/fastscratch/covid-predict/model/RBD_all_15-09_54_epoch_32.pt", map_location="cpu", weights_only=True)
ved.load_state_dict(ved_params, strict=True)
ved.to(device)
ved.set_wt_tokens(REFSEQ[protein][level])
ved.lm.to('cpu')# 必须先set_wt_tokens再送lm去cpu，否则报错
class LocalCNN640(nn.Module):
    """
    输入: x [B, 203]
    输出: feat [B, 640]
    """
    def __init__(self, kernels=(1,3,5,7,9,11,13,15), out_per_kernel=80):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=1,
                out_channels=out_per_kernel,
                kernel_size=k,
                padding=(k - 1) // 2  # 保持长度
            ) for k in kernels
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(out_per_kernel) for _ in kernels
        ])
        self.out_dim = out_per_kernel * len(kernels)  # 80 × 8 = 640

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, 203]
        feats = []
        for conv, bn in zip(self.convs, self.bns):
            h = F.relu(bn(conv(x)))  # [B, 80, 203]
            h = F.max_pool1d(h, h.size(-1)).squeeze(-1)  # [B, 80]
            feats.append(h)
        feat = torch.cat(feats, dim=1)  # [B, 640]
        return feat
class SimpleCNN640(nn.Module):
    def __init__(self, kernel_size=9):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(1, 640, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(640)

    def forward(self, x):
        x = x.unsqueeze(1)        # [B, 1, 203]
        h = F.relu(self.bn(self.conv(x)))   # [B, 640, 203]
        h = F.max_pool1d(h, h.size(-1)).squeeze(-1)  # [B, 640]
        return h
@dataclass
class output_with_ved(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: tuple = None
    embedding: Optional[torch.FloatTensor] =None
    ved_embeddings: Optional[torch.FloatTensor] =None
    ved_embedding: Optional[torch.FloatTensor] =None
class covid_prediction_model_with_ved(nn.Module):
    def __init__(self,
                 ved: nn.Module=ved,
                 jsonl_path: str=os.path.join(dir,'data/merged_all.jsonl'),
                 freeze_bert: bool=False,
                 use_ved: bool=True,
                 seq_embedding_size: int=1280,
                 stru_embedding_size: int=5,
                 stru_seq_len:int=130,
                 dropout_prob: float=0.1,
                 pooling: str="first_token",
                 use_localcnn: bool=True,
                 vocab_size: int=33,
                 emb_dim: int=128) -> None:
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_ved = use_ved
        self.use_localcnn = use_localcnn
        self.ved = ved if self.use_ved else None
        self.esm_model = esm.pretrained.esm2_t33_650M_UR50D()[0].to(device)# esm.pretrained.esm1b_t33_650M_UR50S()[0].to(device)
        self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()[1]
        self.batch_converter = esm.pretrained.esm2_t33_650M_UR50D()[1].get_batch_converter()
        self.model_name ="esm2_t33_650M_UR50D"# "esm1b_t33_650M_UR50S"
        self.model_layers = self.model_name.split("_")[1][1:]
        self.repr_layers = int(self.model_layers)
        if self.use_ved and self.use_localcnn:
            self.predict_class = ClassPredictionHead(seq_embedding_size+1280,512,2,dropout_prob).to(device)
        elif self.use_ved or self.use_localcnn:
            self.predict_class = ClassPredictionHead(seq_embedding_size+640,512,2,dropout_prob).to(device)
        else:
            self.predict_class = ClassPredictionHead(seq_embedding_size+0,512,2,dropout_prob).to(device)
        self.pooling = pooling

        
        # self.structEmbed = StructEmbed(node_features=128,edge_features=128,hidden_dim=128,out_dim=stru_embedding_size).to(device)
        # self.antibody_data = StructureDataset(jsonl_file=jsonl_path, truncate=None, max_length=500)
        # X, S, mask, lengths = featurize(self.antibody_data, device=device, shuffle_fraction=0)
        # self.X=X
        # self.S=S
        # self.mask=mask
        # self.lengths=lengths

        self.expand = nn.Sequential(
            nn.Linear(32, 256),
            nn.ReLU(),
            nn.Linear(256, 640),
            nn.ReLU()
        )
        self.token2embedding = nn.Sequential(
            nn.Embedding(vocab_size, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 1)
            )
        self.localCNN = LocalCNN640() if use_localcnn else None # SimpleCNN640()

        if freeze_bert:
            for p in self.esm_model.parameters():
                p.requires_grad = False
    
        
    def forward(self,input_ids,labels):
        outputs = self.esm_model(input_ids, repr_layers=[self.repr_layers])["representations"][self.repr_layers]
        b=outputs.shape[0]
        device=outputs.device
        if self.pooling == "first_token":
            outputs = outputs[:,0,:]
        elif self.pooling == "average" :
            outputs = outputs.mean(1)
        else:
            ValueError("Not implemented! Please choose the pooling methods from first_token/average")
        outputs = torch.repeat_interleave(outputs.unsqueeze(dim=1),repeats=9,dim=1)

        if self.use_ved:
            ved_embedding = self.ved.encode(input_ids)# [b, 32]
            ved_embeddings = self.expand(ved_embedding)# [b, 32] to [b, 640]
            ved_embeddings = torch.repeat_interleave(ved_embeddings.unsqueeze(dim=1),repeats=9,dim=1)# [b, 640] to [b, 9, 640]
        else:
            ved_embedding = torch.tensor(0.)
            ved_embeddings = torch.tensor(0.)

        # Struct_embedding = self.structEmbed(self.X,self.S,self.lengths,self.mask)
        # Struct_embedding = torch.cat((Struct_embedding[0::2,:,:],Struct_embedding[1::2,:,:]),dim=1)
        # Struct_embedding = torch.flatten(Struct_embedding,start_dim=1)
        # Struct_embedding = torch.repeat_interleave(Struct_embedding.unsqueeze(dim=0),repeats=b,dim=0)

        if self.use_localcnn:
            conti_tokens = self.token2embedding(input_ids)# torch.Size([4, 203, 1])
            local_embedding = self.localCNN(conti_tokens.squeeze(-1))
            local_embeddings = torch.repeat_interleave(local_embedding.unsqueeze(dim=1),repeats=9,dim=1)# [b, 640] to [b, 9, 640]
        else:
            conti_tokens = torch.tensor(0.)
            local_embedding = torch.tensor(0.)
            local_embeddings = torch.tensor(0.)

        if self.use_ved and self.use_localcnn:
            combined_embedding = torch.cat((ved_embeddings, local_embeddings, outputs),dim=2)
        elif not self.use_ved and self.use_localcnn:
            combined_embedding = torch.cat((local_embeddings, outputs),dim=2)
        elif self.use_ved and not self.use_localcnn:
            combined_embedding = torch.cat((ved_embeddings, outputs),dim=2)
        else:
            combined_embedding = outputs

        class_logits = self.predict_class(combined_embedding)
        loss = None
        
        if labels is not None:
            pos_weight=torch.tensor([[0.08,0.92],[0.18,0.82],[0.06,0.94],[0.1,0.9],[0.08,0.92],[0.1,0.9],[0.05,0.95],[0.04,0.96],[0.17,0.83]],device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            if labels.shape[1]==10:
                labels=labels[:,0:9,:]
            loss = criterion(class_logits,labels)
        # print(class_logits[0,:,:])
        return(output_with_ved(
        loss = loss,
        logits = class_logits,
        embedding = combined_embedding,
        ved_embeddings = ved_embeddings,
        ved_embedding = ved_embedding,
        ))