#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
def _move_to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    elif isinstance(x, dict):
        return {k: _move_to_device(v, device) for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        return type(x)(_move_to_device(v, device) for v in x)
    else:
        return x

from loralib.loralib.layers import LoRALayer
from typing import Optional, List 


class SVDLinear(nn.Linear, LoRALayer):
    # SVD-based adaptation implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, 
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r, in_features))
            )
            self.lora_E = nn.Parameter(
                self.weight.new_zeros(r, 1)
            ) 
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features, r))
            )
            self.ranknum = nn.Parameter(
                self.weight.new_zeros(1), requires_grad=False
            )
            self.ranknum.data.fill_(float(r))
            self.scaling = self.lora_alpha if self.lora_alpha>0 else float(self.r)   
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            self.ranknum.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.zeros_(self.lora_E)
            nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
            nn.init.normal_(self.lora_B, mean=0.0, std=0.02)

    def train(self, mode: bool = True):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            if self.r > 0:
                self.weight.data -= T(
                    self.lora_B @ (self.lora_A*self.lora_E)
                ) * self.scaling / (self.ranknum+1e-5)
            self.merged = False
    
    def eval(self):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            if self.r > 0:
                self.weight.data += T(
                    self.lora_B @ (self.lora_A * self.lora_E)
                ) * self.scaling / (self.ranknum+1e-5)
            self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            result += (
                self.lora_dropout(x) @ (self.lora_A * self.lora_E).T @ self.lora_B.T
            ) * self.scaling / (self.ranknum+1e-5)
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


# ----------------- 原本 RankAllocator 用於 EMA+mask ----------------- #
class RankAllocator(object):
    """
    The RankAllocator for AdaLoRA Model that will be called every training step. 
    Paper: https://openreview.net/pdf?id=lq62uWRJjiY

    Args:
        model: the model that we apply AdaLoRA to.
        lora_r (`int`): The initial rank for each incremental matrix.
        target_rank (`int`): The target average rank of incremental matrix.
        init_warmup (`int`): The steps of initial fine-tuning warmup.
        final_warmup (`int`): The step of final fine-tuning.
        mask_interval (`int`): The time internval between two budget allocations.
        beta1 (`float`): The hyperparameter of EMA for sensitivity smoothing.
        beta2 (`float`): The hyperparameter of EMA for undertainty quantification.
        total_step (`int`): The total training steps, correctly configured before training.
        target_total_rank (`Optinal[int]`): The speficified final total rank. 
        tb_writter (`SummaryWriter`): Tensorboard SummaryWriter. 
        tb_writter_loginterval (`int`): The logging interval of SummaryWriter. 
    """
    def __init__(
        self, model, 
        lora_r:int,
        target_rank:int, 
        init_warmup:int, 
        final_warmup:int,
        mask_interval:int,
        beta1:float, 
        beta2:float, 
        total_step:Optional[int]=None, 
        target_total_rank:Optional[int]=None,
        tb_writter=None,
        tb_writter_loginterval:int=500, 
    ):
        self.ave_target_rank = target_rank 
        self.target_rank = target_total_rank
        self.lora_init_rank = lora_r 
        self.initial_warmup = init_warmup
        self.final_warmup = final_warmup 
        self.mask_interval = mask_interval
        self.beta1 = beta1
        self.beta2 = beta2
        self.total_step = total_step

        self.model = model
        self.ipt = {} 
        self.exp_avg_ipt = {}
        self.exp_avg_unc = {}
        self.cat_ipt = {}
        self.rank_pattern = {} 
        self.get_lora_param_name()

        self.tb_writter = tb_writter
        self.log_interval = tb_writter_loginterval 

        assert (self.beta1<1 and self.beta1>0)
        assert (self.beta2<1 and self.beta2>0)

    def set_total_step(self, total_step:int): 
        # Set total step number 
        self.total_step = total_step
        assert self.total_step>self.initial_warmup+self.final_warmup

    def get_rank_pattern(self):
        # Return rank pattern 
        return self.rank_pattern

    def get_lora_param_name(self):
        # Prepare the budget scheduler 
        self.name_set = set() 
        self.total_rank = 0 
        self.shape_dict = {}
        for n,p in self.model.named_parameters():
            if "lora_A" in n: 
                name_mat = n.replace("lora_A", "%s")
                self.name_set.add(name_mat)
                self.total_rank += p.size(0) 
                self.shape_dict[n] = p.shape
            if "lora_B" in n:
                self.shape_dict[n] = p.shape
        self.name_set = list(sorted(self.name_set)) 
        if self.target_rank is None:
            self.target_rank = self.ave_target_rank * len(self.name_set) 

    def schedule_threshold(self, step:int):
        # Global budget schedule
        mask_ind = False 
        target_rank = self.target_rank 
        initial_warmup = self.initial_warmup 
        final_warmup = self.final_warmup 
        total_step = self.total_step 
        self.global_step = step
        if step <= initial_warmup: 
            # Initial warmup 
            curr_rank = self.total_rank 
            mask_ind = False 
        elif step > total_step - final_warmup: 
            # Final fine-tuning 
            curr_rank = self.target_rank 
            # Fix the rank pattern by 
            # always masking the same unimportant singluar values 
            mask_ind = True 
        else: 
            # Budget decreasing 
            mul_coeff = 1-(step-initial_warmup)/(total_step-final_warmup-initial_warmup)
            curr_rank = target_rank + (self.total_rank-target_rank)*(mul_coeff**3)
            curr_rank = int(curr_rank)
            mask_ind = True if step % self.mask_interval == 0 else False 
        return curr_rank, mask_ind 


    def update_ipt(self, model): 
        for n,p in model.named_parameters():
            if "lora_" in n: 
                if n not in self.ipt:
                    self.ipt[n] = torch.zeros_like(p)
                    self.exp_avg_ipt[n] = torch.zeros_like(p) 
                    self.exp_avg_unc[n] = torch.zeros_like(p) 
                with torch.no_grad():
                    # Calculate sensitivity 
                    self.ipt[n] = (p * p.grad).abs().detach()
                    # Update sensitivity 
                    self.exp_avg_ipt[n] = self.beta1 * self.exp_avg_ipt[n] + \
                                        (1-self.beta1)*self.ipt[n]
                    # Update uncertainty 
                    self.exp_avg_unc[n] = self.beta2 * self.exp_avg_unc[n] + \
                                        (1-self.beta2)*(self.ipt[n]-self.exp_avg_ipt[n]).abs()

    def calculate_score(self, n, p=None, metric="ipt"):
        if metric == "ipt":
            # Combine the senstivity and uncertainty 
            ipt_score = self.exp_avg_ipt[n] * self.exp_avg_unc[n]
        elif metric == "mag":
            ipt_score = p.abs().detach().clone() 
        else:
            raise ValueError("Unexcptected Metric: %s"%metric)
        return ipt_score 

    def _combine_ipt(self, ipt_E, ipt_AB):
        ipt_AB = ipt_AB.sum(dim=1, keepdim=False)
        sum_ipt = ipt_E.view(-1) + ipt_AB.view(-1)
        return sum_ipt

    def mask_to_target_rank(self, model, curr_rank): 
        is_dict = {}
        combine_dict = {} 
        singular_dict = {}
        # Calculate the importance score for each sub matrix 
        for n,p in model.named_parameters(): 
            if "lora_A" in n: 
                rdim, hdim_a = p.shape
                ipt_score = self.calculate_score(n, metric="ipt")
                comb_ipt = torch.mean(ipt_score, dim=1, keepdim=True)
                name_mat = n.replace("lora_A", "%s")
                if name_mat not in combine_dict: 
                    combine_dict[name_mat] = [comb_ipt]
                else:
                    combine_dict[name_mat].append(comb_ipt)
            if "lora_B" in n: 
                hdim_b, rdim = p.shape 
                ipt_score = self.calculate_score(n, metric="ipt")
                comb_ipt = torch.mean(ipt_score, dim=0, keepdim=False).view(-1, 1)
                name_mat = n.replace("lora_B", "%s")
                if name_mat not in combine_dict: 
                    combine_dict[name_mat] = [comb_ipt]
                else:
                    combine_dict[name_mat].append(comb_ipt)
            if "lora_E" in n:
                ipt_score = self.calculate_score(n, p=p, metric="ipt")                
                name_mat = n.replace("lora_E", "%s")
                singular_dict[name_mat] = ipt_score

        # Combine the importance scores 
        all_is = []
        for name_mat in combine_dict: 
            ipt_E = singular_dict[name_mat] 
            ipt_AB = torch.cat(combine_dict[name_mat], dim=1)
            sum_ipt = self._combine_ipt(ipt_E, ipt_AB)
            name_E = name_mat%"lora_E"
            is_dict[name_E] = sum_ipt.view(-1, 1)
            all_is.append(sum_ipt.view(-1))

        # Calculate the masking threshold 
        mask_threshold = torch.kthvalue(torch.cat(all_is), (self.total_rank-curr_rank))[0].item()

        # Mask out unimportant singular values 
        with torch.no_grad():
            curr_sum_rank = 0
            sum_param = 0
            for n,p in model.named_parameters():
                if "lora_E" in n: 
                    p.data.masked_fill_(is_dict[n]<=mask_threshold, 0.0)
                    ranknum = (is_dict[n]>mask_threshold).sum().item() 

                    if self.tb_writter is not None and self.global_step%self.log_interval==0:
                        self.tb_writter.add_scalar("Ranknum/%s"%(n,), ranknum, self.global_step) 
                        self.rank_pattern[n] = ranknum 
                        curr_sum_rank += ranknum 
                        sum_param += ranknum*self.shape_dict[n.replace("lora_E", "lora_A")][1]  
                        sum_param += ranknum*self.shape_dict[n.replace("lora_E", "lora_B")][0]  

            if self.tb_writter is not None and self.global_step%self.log_interval==0:
                self.tb_writter.add_scalar("Budget/total_rank", curr_sum_rank, self.global_step)
                self.tb_writter.add_scalar("Budget/mask_threshold", mask_threshold, self.global_step)
                self.tb_writter.add_scalar("Budget/sum_param", sum_param, self.global_step)

        return mask_threshold


    def update_and_mask(self, model, global_step):
        if global_step<self.total_step-self.final_warmup:
            # Update importance scores element-wise 
            self.update_ipt(model)
            # do not update ipt during final fine-tuning 
        # Budget schedule
        curr_rank, mask_ind = self.schedule_threshold(global_step)
        if mask_ind:
            # Mask to target budget 
            mask_threshold = self.mask_to_target_rank(model, curr_rank) 
        else:
            mask_threshold = None 
        self._maybe_tb_writter_log(model)
        return curr_rank, mask_threshold

    def _maybe_tb_writter_log(self, model):
        if self.tb_writter is not None and self.global_step%self.log_interval==0:
            with torch.no_grad():
                regu_loss = []
                for n,p in model.named_parameters():
                    if "lora_A" in n or "lora_B" in n:
                        mat = p.data.detach().clone()
                        mat_cov = mat @ mat.T if "lora_A" in n else mat.T @ mat 
                        I = torch.eye(*mat_cov.size(), out=torch.empty_like(mat_cov))
                        I.requires_grad = False
                        orth_regu = torch.norm(mat_cov-I, p="fro")
                        regu_loss.append(orth_regu.item())
                        self.tb_writter.add_scalar(
                            "Orth_regu_loss/%s"%n, orth_regu.item(), self.global_step
                        )
                self.tb_writter.add_scalar(
                    "train/orth_regu_loss", sum(regu_loss)/len(regu_loss), self.global_step
                )
# ----------------- PSO 版本 RankAllocator ----------------- #
class PSORankAllocator(RankAllocator):
    def __init__(self, model,num_particles=10, w=0.5, c1=0.8, c2=0.9, criterion=None, data_loader=None, device=None, **kwargs):
        super().__init__(model, **kwargs)
        self.num_particles = num_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.criterion = criterion  # loss function
        self.data_loader = data_loader  # data for fitness evaluation
        self.device = device if device is not None else next(model.parameters()).device
        # initialize particles
        self.layer_names = self.name_set
        self.particles = []
        self.velocities = []
        self.pbest = []
        self.pbest_score = []
        self.gbest = None
        self.gbest_score = float("inf")
        for _ in range(num_particles):
            ranks = torch.randint(1, self.ave_target_rank*2, (len(self.layer_names),),device=self.device)
            self.particles.append(ranks.clone())
            self.velocities.append(torch.zeros_like(ranks, dtype=torch.float, device=self.device))
            self.pbest.append(ranks.clone())
            self.pbest_score.append(float("inf"))

    def evaluate_particle(self, particle, model):
        # ===== 保存原始 lora_E（避免污染模型）=====
        backup_E = {}
        for n, p in model.named_parameters():
            if "lora_E" in n:
                backup_E[n] = p.data.clone()

        # ===== 套用 particle rank mask =====
        for i, name_mat in enumerate(self.layer_names):
            rank_val = int(particle[i].item())
            for n, p in model.named_parameters():
                if n.startswith(name_mat % "lora_E"):
                    with torch.no_grad():
                        vals = p.abs().view(-1)
                        sorted_vals, _ = torch.sort(vals, descending=True)
                        threshold = sorted_vals[min(rank_val, len(sorted_vals) - 1)]
                        p.data.masked_fill_(p.abs() < threshold, 0.0)

        # ===== 計算 loss（只用一個 batch）=====
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in self.data_loader:
                # ✅ 檢查 batch 是否有必要欄位
                if "input_ids" not in batch:
                    raise ValueError(f"Batch missing input_ids. Available keys: {batch.keys()}. "
                                     "請確保資料集已經過 tokenization 處理。")
                if "labels" not in batch:
                    raise ValueError(f"Batch missing labels: {batch.keys()}")

                labels = batch["labels"].to(self.device)
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}

                outputs = model(**inputs)
                logits = outputs.logits
                loss = self.criterion(logits, labels)
                total_loss = loss.item()
                break

        # ===== 正交正則 =====
        total_loss += compute_orth_regu(model, regu_weight=0.1).item()

        # ===== 還原模型 =====
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in backup_E:
                    p.data.copy_(backup_E[n])

        return total_loss



    def update_particles(self, model):
        for i in range(self.num_particles):
            fitness = self.evaluate_particle(self.particles[i], model)
            # 更新 pbest
            if fitness < self.pbest_score[i]:
                self.pbest_score[i] = fitness
                self.pbest[i] = self.particles[i].clone()
            # 更新 gbest
            if fitness < self.gbest_score:
                self.gbest_score = fitness
                self.gbest = self.particles[i].clone()
        # 更新速度和位置
        for i in range(self.num_particles):
            r1 = torch.rand(len(self.layer_names), device=self.device)
            r2 = torch.rand(len(self.layer_names), device=self.device)
            self.velocities[i] = (self.w*self.velocities[i] +
                                  self.c1*r1*(self.pbest[i]-self.particles[i].float()) +
                                  self.c2*r2*(self.gbest-self.particles[i].float()))
            self.particles[i] = (self.particles[i].float() + self.velocities[i]).clamp(min=1).int()

    def get_best_rank_mask(self, model):
        """
        將 gbest 的 rank 映射到模型並返回 threshold
        """
        is_dict = {}
        for i, name_mat in enumerate(self.layer_names):
            rank_val = int(self.gbest[i].item())
            for n, p in model.named_parameters():
                if n.startswith(name_mat % "lora_E"):
                    sorted_vals, _ = torch.sort(p.abs().view(-1), descending=True)
                    threshold = sorted_vals[min(rank_val, len(sorted_vals)-1)]
                    p.data.masked_fill_(p.abs() < threshold, 0.0)
                    is_dict[n] = threshold
        return is_dict


# ----------------- 正交正則化函數 ----------------- #
def compute_orth_regu(model, regu_weight=0.1):
    regu_loss, num_param = 0., 0
    for n,p in model.named_parameters():
        if "lora_A" in n or "lora_B" in n:
            para_cov = p @ p.T if "lora_A" in n else p.T @ p 
            I = torch.eye(*para_cov.size(), out=torch.empty_like(para_cov))
            I.requires_grad = False
            regu_loss += torch.norm(para_cov-I, p="fro")
            num_param += 1
    return regu_weight*regu_loss/num_param
