import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# loss.py (请添加或替换)
import torch
import torch.nn as nn
import torch.nn.functional as F

class NaNMSELoss():
    def __init__(self, cfg):
        super().__init__()
        self.model_name = cfg["modelname"]

    def fit(self, y_pred,y_true,lossmse):
        mask = y_true == y_true
        y_true = y_true[mask]
        y_pred = torch.squeeze(y_pred[mask])
        loss = torch.sqrt(lossmse(y_true, y_pred))
        return loss
class ImprovedTeacherLoss(nn.Module):
    def __init__(self, product_weights=None, lambda_consensus=1.0, lambda_product=0.05, lambda_diversity=1.0, diversity_threshold=0.2):
        super().__init__()
        if product_weights is None:
            product_weights = {'era5': 0.6, 'colm': 0.2, 'smci': 0.2}
        self.product_weights = product_weights
        self.lambda_consensus = lambda_consensus
        self.lambda_product = lambda_product
        self.lambda_diversity = lambda_diversity
        self.threshold = diversity_threshold

    def forward(self, teacher_outputs, product_targets):
        """
        teacher_outputs: 模型输出字典，需包含 'consensus_predictions', 'predictions', 'features'
        product_targets: 各产品未来7天真实值，字典
        """
        # 提取预测
        consensus_preds = teacher_outputs['consensus_predictions']
        product_preds = teacher_outputs['predictions']
        product_feats = teacher_outputs['features']['product']  # dict of [B, 64]

        # 1. 共识预测损失（主任务）
        cons_loss = 0.0
        for name in ['era5', 'colm', 'smci']:
            pred = consensus_preds[name]
            target = product_targets[name]
            mask = ~torch.isnan(target)
            if mask.any():
                cons_loss += self.product_weights[name] * F.mse_loss(pred[mask], target[mask])
        cons_loss = cons_loss / 3.0

        # 2. 产品预测损失（辅助）
        prod_loss = 0.0
        for name in ['era5', 'colm', 'smci']:
            pred = product_preds[name]
            target = product_targets[name]
            mask = ~torch.isnan(target)
            if mask.any():
                prod_loss += self.product_weights[name] * F.mse_loss(pred[mask], target[mask])
        prod_loss = prod_loss / 3.0

        # 3. 产品特征多样性损失（阈值惩罚）
        feat_stack = torch.stack([product_feats[n] for n in ['era5', 'colm', 'smci']], dim=1)  # [B, 3, 64]
        feat_norm = F.normalize(feat_stack, dim=2)
        sim_matrix = torch.bmm(feat_norm, feat_norm.transpose(1, 2))  # [B, 3, 3]
        # 提取非对角线元素
        mask = torch.eye(3, device=sim_matrix.device).bool()
        off_diag = sim_matrix[:, ~mask].view(sim_matrix.size(0), -1)  # [B, 6]
        # 只惩罚超过阈值的部分
        diversity_loss = torch.relu(off_diag - self.threshold).mean()

        total = (self.lambda_consensus * cons_loss +
                 self.lambda_product * prod_loss +
                 self.lambda_diversity * diversity_loss)

        loss_components = {
            'consensus_pred': cons_loss,
            'product_pred': prod_loss,
            'diversity': diversity_loss
        }
        return total, loss_components

class TeacherLoss(nn.Module):
    def __init__(self, product_weights=None, lambda_consensus=1.0, lambda_product=0.1, lambda_diversity=1.0):
        super().__init__()
        if product_weights is None:
            product_weights = {'era5': 0.6, 'colm': 0.2, 'smci': 0.2}
        self.product_weights = product_weights
        self.lambda_consensus = lambda_consensus
        self.lambda_product = lambda_product
        self.lambda_diversity = lambda_diversity

    def forward(self, teacher_outputs, product_targets):
        # 提取数据
        consensus_preds = teacher_outputs['consensus_predictions']
        product_preds = teacher_outputs['predictions']
        product_feats = teacher_outputs['features']['product']

        # 1. 共识预测损失（主任务）
        cons_loss = 0.0
        for name in ['era5', 'colm', 'smci']:
            pred = consensus_preds[name]
            target = product_targets[name]
            mask = ~torch.isnan(target)
            if mask.any():
                cons_loss += self.product_weights[name] * F.mse_loss(pred[mask], target[mask])
        cons_loss = cons_loss / 3.0

        # 2. 产品预测损失（辅助，权重较低）
        prod_loss = 0.0
        for name in ['era5', 'colm', 'smci']:
            pred = product_preds[name]
            target = product_targets[name]
            mask = ~torch.isnan(target)
            if mask.any():
                prod_loss += self.product_weights[name] * F.mse_loss(pred[mask], target[mask])
        prod_loss = prod_loss / 3.0

        # 3. 产品特征多样性损失（鼓励正交）
        # 将三个产品特征堆叠 [B, 3, 64]
        feat_stack = torch.stack([product_feats[n] for n in ['era5', 'colm', 'smci']], dim=1)
        # 计算特征间的余弦相似度矩阵 [B, 3, 3]
        feat_norm = F.normalize(feat_stack, dim=2)
        sim_matrix = torch.bmm(feat_norm, feat_norm.transpose(1, 2))  # [B, 3, 3]
        # 惩罚非对角线元素的绝对值（使相似度趋于0）
        mask = torch.eye(3, device=sim_matrix.device).bool()
        off_diag = sim_matrix[:, ~mask].view(sim_matrix.size(0), -1)  # [B, 6]
        #diversity_loss = off_diag.abs().mean()
        threshold = 0.3
        diversity_loss = torch.relu(off_diag - threshold).mean()   # 惩罚超过阈值的相似度
        # 总损失
        total_loss = (self.lambda_consensus * cons_loss +
                      self.lambda_product * prod_loss +
                      self.lambda_diversity * diversity_loss)

        loss_components = {
            'consensus_pred': cons_loss,
            'product_pred': prod_loss,
            'contrast': diversity_loss
        }

        return total_loss, loss_components
'''class TeacherLoss(nn.Module):
    def __init__(self, product_weights=None, lambda_consensus_pred=1.0, lambda_product_pred=0.2, lambda_contrast=0.5):
        super().__init__()
        self.product_weights = product_weights or {'era5':0.6, 'colm':0.2, 'smci':0.2}
        self.lambda_consensus_pred = lambda_consensus_pred
        self.lambda_product_pred = lambda_product_pred
        self.lambda_contrast = lambda_contrast

    def forward(self, outputs, targets):
        # outputs 应包含 consensus_predictions, predictions（产品预测）, features
        consensus_preds = outputs['consensus_predictions']
        product_preds = outputs['predictions']
        consensus_feat = outputs['features']['consensus']
        product_feats = outputs['features']['product']

        # 共识预测损失
        cons_loss = 0
        for name in ['era5', 'colm', 'smci']:
            pred = consensus_preds[name]
            target = targets[name]
            mask = ~torch.isnan(target)
            if mask.any():
                cons_loss += self.product_weights[name] * F.mse_loss(pred[mask], target[mask])
        cons_loss = cons_loss / 3.0

        # 产品预测损失（辅助）
        prod_loss = 0
        for name in ['era5', 'colm', 'smci']:
            pred = product_preds[name]
            target = targets[name]
            mask = ~torch.isnan(target)
            if mask.any():
                prod_loss += self.product_weights[name] * F.mse_loss(pred[mask], target[mask])
        prod_loss = prod_loss / 3.0

        # 对比损失（拉近共识与产品特征）
        contrast_loss = self.compute_contrast_loss(consensus_feat, product_feats)

        total = self.lambda_consensus_pred * cons_loss + self.lambda_product_pred * prod_loss + self.lambda_contrast * contrast_loss
        return total, {'consensus_pred': cons_loss, 'product_pred': prod_loss, 'contrast': contrast_loss}

    def compute_contrast_loss(self, consensus, product_dict):
        # 实现简单的 InfoNCE 或余弦相似度拉近
        B = consensus.size(0)
        consensus_norm = F.normalize(consensus, dim=1)
        loss = 0
        for name, feat in product_dict.items():
            prod_norm = F.normalize(feat, dim=1)
            pos_sim = (consensus_norm * prod_norm).sum(dim=1)  # [B]
            # 构造负例：其他样本的产品特征
            logits = torch.mm(consensus_norm, prod_norm.T) / 0.1  # [B, B]
            labels = torch.arange(B, device=logits.device)
            loss += F.cross_entropy(logits, labels)
        return loss / 3.0
        '''
'''class TeacherLoss(nn.Module):
    """
    改进的教师网络损失函数（不改变模型结构）
    主要改动：
    1. 移除特征多样性损失（避免与一致性冲突）
    2. 特征一致性损失改为适度相似（0.3~0.8），防止产品特征趋同或发散
    3. 新增对比学习损失：拉近共识特征与同一样本的产品特征，推开与其他样本的产品特征
    4. 保留产品预测损失，但调整权重
    """
    def __init__(self, product_weights=None, temp=0.2, 
                 lambda_pred=0.5, lambda_consistency=0.5, lambda_contrast=1.0):
        super().__init__()
        if product_weights is None:
            # 默认产品权重（与原有保持一致）
            product_weights = {'smci': 0.2, 'era5': 0.6, 'colm': 0.2}
        self.product_weights = product_weights
        self.temp = temp  # 对比学习温度
        self.lambda_pred = lambda_pred
        self.lambda_consistency = lambda_consistency
        self.lambda_contrast = lambda_contrast

    def cosine_similarity(self, x1, x2):
        """计算两个张量之间的余弦相似度（与原有保持一致）"""
        x1_norm = F.normalize(x1, dim=-1)
        x2_norm = F.normalize(x2, dim=-1)
        cos_sim = torch.sum(x1_norm * x2_norm, dim=-1)
        return cos_sim  # 返回 [batch]，不取平均，以便后续逐样本操作

    def compute_prediction_loss(self, predictions, targets):
        """产品预测损失（MSE，处理NaN，加权）"""
        total_loss = 0.0
        for product_name in ['era5', 'colm', 'smci']:
            pred = predictions[product_name]
            target = targets[product_name]
            mask = ~torch.isnan(target)
            if mask.any():
                loss = F.mse_loss(pred[mask], target[mask])
                total_loss += self.product_weights[product_name] * loss
        return total_loss / 3.0  # 平均

    def compute_consistency_loss(self, product_features):
        """适度特征一致性损失：惩罚相似度低于0.3或高于0.8"""
        feat_era5 = product_features['era5']
        feat_colm = product_features['colm']
        feat_smci = product_features['smci']

        sim_ec = self.cosine_similarity(feat_era5, feat_colm)
        sim_es = self.cosine_similarity(feat_era5, feat_smci)
        sim_cs = self.cosine_similarity(feat_colm, feat_smci)

        low, high = 0.4, 0.7
        # 分别计算低于low和高于high的惩罚，然后平均
        loss_ec = torch.relu(low - sim_ec).mean() + torch.relu(sim_ec - high).mean()
        loss_es = torch.relu(low - sim_es).mean() + torch.relu(sim_es - high).mean()
        loss_cs = torch.relu(low - sim_cs).mean() + torch.relu(sim_cs - high).mean()
        consistency_loss = (loss_ec + loss_es + loss_cs) / 3.0
        return consistency_loss

    def compute_contrast_loss(self, consensus_feat, product_features):
        """
        对比学习损失：以共识特征为锚点，同一样本的三个产品特征为正例，
        其他样本的所有产品特征为负例。采用InfoNCE的简化实现（每个样本一个正例，正例为三个产品的平均logits）
        """
        B = consensus_feat.size(0)
        # 堆叠三个产品特征 [B, 3, 64]
        product_stack = torch.stack([product_features['era5'], 
                                     product_features['colm'], 
                                     product_features['smci']], dim=1)
        # 归一化
        consensus_norm = F.normalize(consensus_feat, dim=1)  # [B, 64]
        product_norm = F.normalize(product_stack, dim=2)    # [B, 3, 64]

        # 计算每个样本的共识与自身三个产品的相似度 [B, 3]
        pos_sim = torch.bmm(product_norm, consensus_norm.unsqueeze(2)).squeeze(2)  # [B, 3]

        # 计算共识与所有产品特征的相似度矩阵 [B, B*3]
        # 将所有产品展平 [B*3, 64]
        all_product = product_stack.reshape(-1, 64)  # [B*3, 64]
        all_product_norm = F.normalize(all_product, dim=1)
        sim_matrix = torch.mm(consensus_norm, all_product_norm.T)  # [B, B*3]

        # 温度缩放
        pos_logits = pos_sim / self.temp          # [B, 3]
        all_logits = sim_matrix / self.temp       # [B, B*3]

        # 为每个样本构建正例logits（取三个正例的平均）和负例logits（排除自己的三个正例）
        pos_logits_mean = pos_logits.mean(dim=1, keepdim=True)  # [B, 1]

        # 生成负例掩码：对于第i个样本，排除索引 [i*3, i*3+1, i*3+2]
        mask = torch.ones(B, B*3, dtype=torch.bool, device=consensus_feat.device)
        for i in range(B):
            mask[i, i*3:(i+1)*3] = False
        # 提取负例logits [B, B*3-3]
        neg_logits = all_logits[mask].view(B, -1)

        # 拼接正例和负例
        final_logits = torch.cat([pos_logits_mean, neg_logits], dim=1)  # [B, 1 + (B*3-3)]
        # 目标类别为0（第一个位置对应正例）
        target = torch.zeros(B, dtype=torch.long, device=consensus_feat.device)
        contrast_loss = F.cross_entropy(final_logits, target)
        return contrast_loss

    def forward(self, teacher_outputs, product_targets):
        """
        teacher_outputs: 教师网络的输出，需包含：
            - 'features': 包含 'product' (字典) 和 'consensus' (张量)
            - 'predictions': 各产品的预测 (字典)
        product_targets: 各产品的真实值 (字典)
        """
        # 提取必要数据
        features = teacher_outputs['features']
        product_features = features['product']          # dict of [B,64]
        consensus_feat = features['consensus']          # [B,64]
        predictions = teacher_outputs['predictions']    # dict of [B,7]

        # 1. 产品预测损失
        pred_loss = self.compute_prediction_loss(predictions, product_targets)

        # 2. 特征一致性损失（适度相似）
        consistency_loss = self.compute_consistency_loss(product_features)

        # 3. 对比学习损失（共识特征与产品特征）
        contrast_loss = self.compute_contrast_loss(consensus_feat, product_features)

        # 总损失
        total_loss = (self.lambda_pred * pred_loss +
                      self.lambda_consistency * consistency_loss +
                      self.lambda_contrast * contrast_loss)

        # 损失组件（用于日志记录）
        loss_components = {
            'prediction': pred_loss,
            'consistency': consistency_loss,
            'contrast': contrast_loss
        }

        return total_loss, loss_components
'''
class StudentLoss(nn.Module):
    """学生网络损失函数"""

    def __init__(self, w_site=0.7, w_distill=0.3, w_attention=0.1):
        super().__init__()
        self.weights = {
            'site_prediction': w_site,
            'feature_distillation': w_distill,
            'attention_regularization': w_attention
        }

        # 不同产品的蒸馏权重
        self.product_weights = {
            'smci': 0.5,
            'era5': 0.3,
            'colm': 0.2,
            'consensus': 0.5
        }

    def cosine_similarity(self, x1, x2):
        """计算余弦相似度"""
        x1_norm = F.normalize(x1, dim=-1)
        x2_norm = F.normalize(x2, dim=-1)
        cos_sim = torch.sum(x1_norm * x2_norm, dim=-1)
        return cos_sim.mean()

    def compute_distillation_loss(self, adapted_features, teacher_features):
        """计算特征蒸馏损失"""
        total_loss = 0.0

        for key in ['era5', 'colm', 'smci', 'consensus']:
            # 计算特征相似度
            cos_sim = self.cosine_similarity(
                adapted_features[key],
                teacher_features[key]
            )

            # 蒸馏损失：希望特征相似度高
            distill_loss = 1 - cos_sim

            # 加权
            weighted_loss = self.product_weights[key] * distill_loss
            total_loss += weighted_loss

        return total_loss / 4.0

    def compute_attention_regularization(self, attention_weights):
        """注意力正则化：鼓励使用所有特征源"""
        # 计算注意力权重的熵（鼓励多样性）
        eps = 1e-8
        entropy = -torch.sum(
            attention_weights * torch.log(attention_weights + eps),
            dim=-1
        )

        # 最大熵：log(特征源数量)
        max_entropy = torch.log(
            torch.tensor(
                attention_weights.shape[-1],
                dtype=torch.float32,
                device=attention_weights.device
            )
        )

        regularization_loss = 1 - (entropy.mean() / max_entropy)
        return regularization_loss

    def forward(self, student_outputs, site_targets):
        """
        计算学生网络的总损失

        Args:
            student_outputs: 学生网络的输出（包含各种特征）
            site_targets: 站点真实值 [batch, 7]
        """
        # 1. 站点预测损失（MSE）
        site_pred = student_outputs['site_prediction']
        site_mask = ~torch.isnan(site_targets)  # 处理缺失值

        if site_mask.any():
            site_loss = F.mse_loss(
                site_pred[site_mask],
                site_targets[site_mask]
            )
        else:
            site_loss = torch.tensor(0.0, device=site_pred.device)

        # 2. 特征蒸馏损失
        distill_loss = self.compute_distillation_loss(
            student_outputs['adapted_features'],
            student_outputs['teacher_features']
        )

        # 3. 注意力正则化损失
        attention_loss = self.compute_attention_regularization(
            student_outputs['attention_weights']
        )

        # 4. 总损失
        total_loss = (
                self.weights['site_prediction'] * site_loss +
                self.weights['feature_distillation'] * distill_loss +
                self.weights['attention_regularization'] * attention_loss
        )

        # 返回损失和各个分量
        return total_loss, {
            'site_loss': site_loss,
            'distill_loss': distill_loss,
            'attention_loss': attention_loss
        }
'''

class TeacherLoss(nn.Module):
    def __init__(self):
        super().__init__()

        # 各损失项的权重
        self.weights = {
            'feature_consistency':  0.6,
            'product_prediction': 0.2,
            'feature_diversity': 0.2,
            # 'physics_constraint': config.get('w_physics', 0.1),
            # 'temporal_coherence': config.get('w_temporal', 0.1)
        }

        # 不同产品的权重（基于与站点的相关性）
        self.product_weights = {
            'smci': 0.2,  # 与站点相关性最高
            'era5': 0.6,
            'colm': 0.2
        }

    def cosine_similarity(self, x1, x2):
        """计算两个张量之间的余弦相似度"""
        # 归一化
        x1_norm = F.normalize(x1, dim=-1)
        x2_norm = F.normalize(x2, dim=-1)

        # 计算余弦相似度
        cos_sim = torch.sum(x1_norm * x2_norm, dim=-1)

        return torch.nanmean(cos_sim)
    def compute_feature_consistency_loss(self, product_features):
        """
        计算不同产品特征间的一致性
        目标：让不同产品提取的特征尽可能相似
        """
        # 获取三个产品的特征
        era5_feat = product_features['era5']  # [batch, 64]
        colm_feat = product_features['colm']  # [batch, 64]
        smci_feat = product_features['smci']  # [batch, 64]

        # 计算两两之间的相似度
        sim_era5_colm = self.cosine_similarity(era5_feat, colm_feat)
        sim_era5_smci = self.cosine_similarity(era5_feat, smci_feat)
        sim_colm_smci = self.cosine_similarity(colm_feat, smci_feat)

        # 我们希望相似度高，所以损失 = 1 - 相似度
        loss = (
                       torch.nanmean(1 - sim_era5_colm) +
                       torch.nanmean(1 - sim_era5_smci) +
                       torch.nanmean(1 - sim_colm_smci)
               ) / 3.0

        return loss

    def compute_prediction_loss(self, predictions, targets):
        """
        计算产品预测损失（使用模式相似性而非绝对数值）
        """
        total_loss = 0.0

        for product_name in ['era5', 'colm', 'smci']:
            pred = predictions[product_name]  # [batch, 7]
            target = targets[product_name]  # [batch, 7]
            mask = ~torch.isnan(target)
            pred_valid = pred[mask]
            target_valid = target[mask]
            mse_loss = F.mse_loss(pred_valid, target_valid)
            weighted_mse = self.product_weights[product_name] * mse_loss
            total_loss += weighted_mse
            # # 计算序列相关系数（关注趋势而非数值）
            # correlation = self.sequence_correlation(pred, target)
            #
            # # 模式损失：1 - 相关系数（鼓励趋势一致）
            # pattern_loss = 1 - correlation
            #
            # # 加入产品权重
            # weighted_loss = self.product_weights[product_name] * pattern_loss
            #
            # total_loss += weighted_loss

        return total_loss

    def sequence_correlation(self, pred, target):
        """
        计算两个序列的相关系数（对异常值鲁棒）
        """
        # 批次维度处理
        batch_size = pred.shape[0]

        correlations = []
        for i in range(batch_size):
            pred_seq = pred[i]  # [7]
            target_seq = target[i]  # [7]

            # 计算相关系数（安全版本）
            pred_mean = pred_seq.mean()
            target_mean = target_seq.mean()

            numerator = ((pred_seq - pred_mean) * (target_seq - target_mean)).sum()
            denominator = torch.sqrt(
                ((pred_seq - pred_mean) ** 2).sum() *
                ((target_seq - target_mean) ** 2).sum()
            ) + 1e-8

            corr = numerator / denominator
            # 裁剪到合理范围
            corr = torch.clamp(corr, -1.0, 1.0)

            correlations.append(corr)

        return torch.nanmean(torch.stack(correlations))

    def compute_feature_diversity_loss(self, features):
        """
        鼓励特征多样性，避免所有特征趋同
        """
        consensus = features['consensus']  # [batch, 32]
        product_features = [features['product'][name] for name in ['era5', 'colm', 'smci']]  # 每个 [batch, 64]
        cos_sim_era5 = F.cosine_similarity(consensus, features['product']['era5'], dim=-1).abs().mean()
        cos_sim_colm = F.cosine_similarity(consensus, features['product']['colm'], dim=-1).abs().mean()
        cos_sim_smci = F.cosine_similarity(consensus, features['product']['smci'], dim=-1).abs().mean()

        # 我们希望共识特征与产品特征不相关，即相似度绝对值接近0
        diversity_loss = (cos_sim_era5 + cos_sim_colm + cos_sim_smci) / 3.0
        return diversity_loss
        # # 收集所有特征
        # all_features = []
        #
        # # 产品特定特征
        # for product_name in ['era5', 'colm', 'smci']:
        #     all_features.append(features['product'][product_name])
        #
        # # 共识特征
        # all_features.append(features['consensus'])
        #
        # # 堆叠 [n_features, batch, feature_dim]
        # stacked = torch.stack(all_features, dim=0)  # [4, batch, feature_dim]
        #
        # # 计算特征间的相关性矩阵
        # # 首先展平批次维度
        # flat_features = stacked.view(stacked.shape[0], -1)  # [4, batch*feature_dim]
        #
        # # 计算相关性矩阵 [4, 4]
        # correlation_matrix = torch.corrcoef(flat_features)
        #
        # # 我们希望非对角线元素小（特征间低相关）
        # # 对角线是自相关，应为1
        # mask = torch.eye(correlation_matrix.shape[0],
        #                  device=correlation_matrix.device).bool()
        #
        # off_diag = correlation_matrix[~mask]
        # off_diag_abs = torch.abs(off_diag)
        #
        # # 处理可能的NaN值
        # valid_values = off_diag_abs[~torch.isnan(off_diag_abs)]
        # if len(valid_values) == 0:
        #     return torch.tensor(0.0, device=flat_features.device)
        #
        # diversity_loss = torch.nanmean(valid_values)
        #
        #
        # return diversity_loss

    def compute_temporal_coherence_loss(self, temporal_features):
        """
        确保时序特征具有合理的时间连续性
        temporal_features: [batch, seq_len, feature_dim]
        """
        # 计算相邻时间步的特征差异
        diff = temporal_features[:, 1:, :] - temporal_features[:, :-1, :]

        # 差异应平滑（不是突变）
        # 计算差异的二阶范数
        diff_norm = torch.norm(diff, dim=2)  # [batch, seq_len-1]

        # 我们希望差异适度（既不完全相同，也不突变）
        # 使用目标范围约束
        target_min = 0.01
        target_max = 0.1

        # 过小惩罚
        too_small = torch.nanmean(F.relu(target_min - diff_norm))

        # 过大惩罚
        too_large = torch.nanmean(F.relu(diff_norm - target_max))

        return (too_small + too_large) * 0.1

    def forward(self, teacher_outputs, product_targets):
        """
        teacher_outputs: 教师网络的输出（包含特征和预测）
        product_targets: 各产品的真实值
        inputs: 输入数据（用于物理约束等）
        """
        total_loss = 0.0
        loss_components = {}

        # 1. 特征一致性损失（最重要）
        # 鼓励不同产品提取相似的特征（寻找共识）
        consistency_loss = self.compute_feature_consistency_loss(
            teacher_outputs['features']['product']
        )
        total_loss += self.weights['feature_consistency'] * consistency_loss
        loss_components['consistency'] = consistency_loss
        #
        # # 2. 产品预测损失（辅助，使用模式相似性）
        prediction_loss = self.compute_prediction_loss(
            teacher_outputs['predictions'], product_targets
        )
        total_loss += self.weights['product_prediction'] * prediction_loss
        loss_components['prediction'] = prediction_loss

        # # 3. 特征多样性损失
        # # 避免所有特征趋同，鼓励提取多样化信息
        diversity_loss = self.compute_feature_diversity_loss(
            teacher_outputs['features']
        )
        total_loss += self.weights['feature_diversity'] * diversity_loss
        loss_components['diversity'] = diversity_loss

        # # 4. 时间相干性损失
        # temporal_loss = self.compute_temporal_coherence_loss(
        #     teacher_outputs['features']['temporal']
        # )
        # total_loss += self.weights['temporal_coherence'] * temporal_loss
        # loss_components['temporal'] = temporal_loss

        return total_loss, loss_components
'''