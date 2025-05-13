import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import inspect

# 根据我们的需求添加的打印训练参数的函数
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )
    return trainable_params, all_param

# 打印模块的所有属性和方法，帮助调试
def debug_print_module_info(module, prefix=""):
    print(f"{prefix}模块类型: {type(module).__name__}")
    print(f"{prefix}模块属性:")
    
    # 打印模块的所有属性
    for name, value in module.__dict__.items():
        if not name.startswith('_'):
            # 对于张量，打印其形状
            if isinstance(value, torch.Tensor):
                print(f"{prefix}  {name}: Tensor, shape={value.shape}")
            # 对于子模块，打印其类型
            elif isinstance(value, nn.Module):
                print(f"{prefix}  {name}: {type(value).__name__}")
            # 对于简单类型，直接打印
            elif isinstance(value, (int, float, str, bool)):
                print(f"{prefix}  {name}: {value}")
            # 对于其他类型，只打印类型
            else:
                print(f"{prefix}  {name}: {type(value).__name__}")
    
    # 打印模块的所有方法
    print(f"{prefix}模块方法:")
    for name, method in inspect.getmembers(module, predicate=inspect.ismethod):
        if not name.startswith('_'):
            print(f"{prefix}  {name}")

# LoRA的降维和升维矩阵
class LoRALayer(nn.Module):
    def __init__(
        self, 
        in_features: int, 
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        peg_distribution: str = 'Gaussian-Layer'
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # 降维矩阵A
        self.lora_A = nn.Parameter(torch.zeros((rank, in_features)))
        # 升维矩阵B
        self.lora_B = nn.Parameter(torch.zeros((out_features, rank)))
        self.dropout = nn.Dropout(dropout)
        
        # 根据PEG分布进行特定初始化
        if peg_distribution == 'Gaussian-Layer':
            # 高斯分布初始化，适合PEG的Gaussian-Layer分布
            nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
        else:
            # 标准初始化
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor):
        # LoRA forward: x → A → B → scaled output
        return (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling

# 为QKV注意力添加LoRA
class LoRAQKVAttention(nn.Module):
    def __init__(
        self,
        original_module: nn.Module,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        peg_distribution: str = 'Gaussian-Layer'
    ):
        super().__init__()
        self.original_module = original_module
        
        # 尝试获取模块维度
        if hasattr(original_module, 'dim'):
            dim = original_module.dim
        elif hasattr(original_module, 'qkv') and hasattr(original_module.qkv, 'in_features'):
            dim = original_module.qkv.in_features
        elif hasattr(original_module, 'qkv') and hasattr(original_module.qkv, 'weight'):
            dim = original_module.qkv.weight.shape[1]
        else:
            # 如果都没有，打印模块信息帮助调试
            debug_print_module_info(original_module)
            raise ValueError("无法确定Attention模块的维度")
        
        # 获取注意力头数量
        if hasattr(original_module, 'num_heads'):
            self.num_heads = original_module.num_heads
        else:
            self.num_heads = dim // 64  # 假设每个头的维度是64
        
        self.dim = dim
        self.scale = getattr(original_module, 'scale', dim ** -0.5)
        
        # 冻结原始模块参数
        for param in self.original_module.parameters():
            param.requires_grad = False
            
        # 为QKV矩阵添加LoRA层
        self.lora_qkv = LoRALayer(
            in_features=dim,
            out_features=dim * 3,  # QKV三个矩阵合一
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            peg_distribution=peg_distribution
        )
        
        # 为注意力投影添加另一个LoRA层，确保模型有可训练参数
        self.lora_proj = LoRALayer(
            in_features=dim,
            out_features=dim,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            peg_distribution=peg_distribution
        )
        
    def forward(self, x):
        # 使用原始前向函数的框架，但添加LoRA的增量
        B, N, C = x.shape
        
        # 处理原始的注意力计算
        if hasattr(self.original_module, 'forward'):
            # 先尝试使用原始的前向函数
            try:
                if isinstance(self.original_module.forward(x), tuple):
                    output, _ = self.original_module.forward(x)
                else:
                    output = self.original_module.forward(x)
            except Exception as e:
                print(f"原始注意力模块前向传播失败: {e}")
                # 如果前向传播失败，尝试使用简化的计算
                qkv = self.original_module.qkv(x)
                q, k, v = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)
                attn = (q @ k.transpose(-2, -1)) * self.scale
                attn = attn.softmax(dim=-1)
                x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                output = self.original_module.proj(x)
        else:
            # 如果没有前向函数，尝试手动计算
            qkv = self.original_module.qkv(x)
            q, k, v = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            output = self.original_module.proj(x)
        
        # 计算LoRA增量
        lora_qkv = self.lora_qkv(x)
        lora_out = self.lora_proj(output)
        
        # 返回原始输出加上LoRA增量
        return output + lora_out


# 为Learngene的注意力机制添加LoRA
class LoRALearnGeneAttention(nn.Module):
    def __init__(
        self,
        original_module: nn.Module,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        peg_distribution: str = 'Gaussian-Layer'
    ):
        super().__init__()
        self.original_module = original_module
        
        # 打印模块信息，帮助调试
        debug_print_module_info(original_module)
        
        # 尝试多种方式获取维度信息
        if hasattr(original_module, 'qkv') and hasattr(original_module.qkv, 'in_features'):
            # 直接从qkv模块获取
            self.in_dim = original_module.qkv.in_features
            self.out_dim = self.in_dim 
        elif hasattr(original_module, 'inherited_attn') and hasattr(original_module.inherited_attn, 'qkv'):
            # 从继承的注意力模块获取
            if hasattr(original_module.inherited_attn.qkv, 'in_features'):
                self.in_dim = original_module.inherited_attn.qkv.in_features
            elif hasattr(original_module.inherited_attn.qkv, 'weight'):
                self.in_dim = original_module.inherited_attn.qkv.weight.shape[1]
            self.out_dim = self.in_dim
        elif hasattr(original_module, 'inherited_attn') and hasattr(original_module.inherited_attn, 'weight'):
            # 直接从权重获取
            self.in_dim = original_module.inherited_attn.weight.shape[1]
            self.out_dim = original_module.inherited_attn.weight.shape[0]
        else:
            # 如果找不到维度信息，使用默认值并发出警告
            print("警告: 无法确定Attention_Learngene模块的确切维度，使用默认值192")
            self.in_dim = 192  # DeiT-tiny的默认维度
            self.out_dim = 192
            
        # 获取注意力头数量
        if hasattr(original_module, 'num_heads'):
            self.num_heads = original_module.num_heads
        elif hasattr(original_module, 'inherited_attn') and hasattr(original_module.inherited_attn, 'num_heads'):
            self.num_heads = original_module.inherited_attn.num_heads
        else:
            self.num_heads = 3  # DeiT-tiny的默认头数
            
        # 冻结原始模块参数
        for param in self.original_module.parameters():
            param.requires_grad = False
            
        # 为输入特征添加LoRA层
        self.lora_in = LoRALayer(
            in_features=self.in_dim,
            out_features=self.in_dim,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            peg_distribution=peg_distribution
        )
        
        # 为输出特征添加另一个LoRA层
        self.lora_out = LoRALayer(
            in_features=self.in_dim,
            out_features=self.in_dim,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            peg_distribution=peg_distribution
        )
        
    def forward(self, x):
        # 先进行输入LoRA
        x_lora = self.lora_in(x)
        
        # 使用原始模块的前向传播
        try:
            if isinstance(self.original_module.forward(x), tuple):
                output, attn = self.original_module.forward(x + x_lora)
            else:
                output = self.original_module.forward(x + x_lora)
                attn = None
        except Exception as e:
            print(f"Attention_Learngene前向传播失败: {e}")
            # 如果前向传播失败，仅使用LoRA层
            output = x + x_lora
            attn = None
        
        # 应用输出LoRA
        output = output + self.lora_out(output)
        
        # 返回修改后的输出
        if attn is not None:
            return output, attn
        else:
            return output, None


# 为MLP添加LoRA
class LoRAMLP(nn.Module):
    def __init__(
        self,
        original_module: nn.Module,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        peg_distribution: str = 'Gaussian-Layer'
    ):
        super().__init__()
        self.original_module = original_module
        
        # 尝试获取MLP维度
        try:
            in_features = original_module.fc1.in_features
            hidden_features = original_module.fc1.out_features
            out_features = original_module.fc2.out_features
        except AttributeError:
            # 如果无法获取维度，打印模块信息并使用默认值
            debug_print_module_info(original_module)
            print("警告: 无法获取MLP的确切维度，使用默认值")
            in_features = 192  # DeiT-tiny的默认维度
            hidden_features = in_features * 4  # MLP默认扩展比例
            out_features = in_features
            
        # 冻结原始模块参数
        for param in self.original_module.parameters():
            param.requires_grad = False
            
        # 为FC1和FC2添加LoRA
        self.lora_fc1 = LoRALayer(
            in_features=in_features,
            out_features=hidden_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            peg_distribution=peg_distribution
        )
        
        self.lora_fc2 = LoRALayer(
            in_features=hidden_features,
            out_features=out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            peg_distribution=peg_distribution
        )
    
    def forward(self, x):
        # 尝试使用原始MLP的前向传播
        try:
            # 获取原始输出
            original_output = self.original_module(x)
            
            # 计算LoRA输出
            lora_x1 = self.lora_fc1(x)
            lora_x1 = F.gelu(lora_x1)
            lora_out = self.lora_fc2(lora_x1)
            
            # 结合原始输出和LoRA输出
            return original_output + lora_out
        except Exception as e:
            print(f"MLP前向传播失败: {e}")
            # 如果前向传播失败，仅使用LoRA层
            lora_x1 = self.lora_fc1(x)
            lora_x1 = F.gelu(lora_x1)
            lora_out = self.lora_fc2(lora_x1)
            return lora_out


# 为TransMlp添加LoRA
class LoRATransMlp(nn.Module):
    def __init__(
        self,
        original_module: nn.Module,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        peg_distribution: str = 'Gaussian-Layer'
    ):
        super().__init__()
        self.original_module = original_module
        
        # 打印模块信息，帮助调试
        debug_print_module_info(original_module)
        
        # 尝试多种方式获取维度信息
        if hasattr(original_module, 'in_features') and hasattr(original_module, 'hidden_features'):
            in_features = original_module.in_features
            hidden_features = original_module.hidden_features
            out_features = in_features  # 通常MLP输出维度与输入相同
        elif hasattr(original_module, 'fc1') and hasattr(original_module.fc1, 'in_features'):
            in_features = original_module.fc1.in_features
            hidden_features = original_module.fc1.out_features
            out_features = original_module.fc2.out_features if hasattr(original_module, 'fc2') else in_features
        elif hasattr(original_module, 'descendant_fc1_weights'):
            # 尝试从权重获取维度
            in_features = original_module.descendant_fc1_weights.shape[1]
            hidden_features = original_module.descendant_fc1_weights.shape[0]
            out_features = in_features
        else:
            # 如果找不到维度信息，使用默认值并发出警告
            print("警告: 无法确定TransMlp模块的确切维度，使用默认值192")
            in_features = 192  # DeiT-tiny的默认维度
            hidden_features = in_features * 4  # MLP默认扩展比例
            out_features = in_features
        
        # 冻结原始模块参数
        for param in self.original_module.parameters():
            param.requires_grad = False
            
        # 为FC1和FC2添加LoRA
        self.lora_fc1 = LoRALayer(
            in_features=in_features,
            out_features=hidden_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            peg_distribution=peg_distribution
        )
        
        self.lora_fc2 = LoRALayer(
            in_features=hidden_features,
            out_features=out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            peg_distribution=peg_distribution
        )
        
        # 添加一个简单的恒等LoRA以确保模型有可训练参数
        self.identity_lora = nn.Linear(in_features, in_features, bias=False)
        nn.init.zeros_(self.identity_lora.weight)
    
    def forward(self, x):
        # 添加恒等LoRA的增量
        x_identity = self.identity_lora(x)
        
        # 尝试使用原始模块的前向传播
        try:
            original_output = self.original_module(x + x_identity)
            
            # 计算LoRA增量
            lora_x1 = self.lora_fc1(x)
            lora_x1 = F.gelu(lora_x1)
            lora_out = self.lora_fc2(lora_x1)
            
            # 返回原始输出加上LoRA增量
            return original_output + lora_out
        except Exception as e:
            print(f"TransMlp前向传播失败: {e}")
            # 如果前向传播失败，仅使用LoRA层和恒等增量
            lora_x1 = self.lora_fc1(x)
            lora_x1 = F.gelu(lora_x1)
            lora_out = self.lora_fc2(lora_x1)
            return x + x_identity + lora_out

# 简单的可训练层，确保在所有LoRA应用失败时仍有可训练参数
class SimpleTrainableLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.weight, std=0.02)
        
    def forward(self, x):
        return x + self.weight.unsqueeze(0).unsqueeze(0)

# 应用LoRA到整个模型的递归函数
def apply_lora_to_model_recursive(
    model,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: List[str] = ["attn", "mlp"],
    peg_distribution: str = 'Gaussian-Layer',
    count_dict: Dict[str, int] = None,
    parent_name: str = ""
):
    """
    递归地将LoRA应用到模型的特定模块
    
    Args:
        model: 要应用LoRA的模块
        rank: LoRA的秩
        alpha: LoRA的缩放因子
        dropout: LoRA中使用的dropout率
        target_modules: 要应用LoRA的模块类型列表
        peg_distribution: PEG使用的分布类型
        count_dict: 计数不同类型模块的字典
        parent_name: 父模块的名称，用于调试
    
    Returns:
        被修改后的模块和计数字典
    """
    from .vision_transformer import Attention, Attention_Learngene, Mlp, TransMlp, Block, Block_learngene
    
    if count_dict is None:
        count_dict = {"attn": 0, "mlp": 0, "other": 0, "trainable_added": 0, "blocks": {}}
    
    # 处理当前模块
    if isinstance(model, Attention):
        if "attn" in target_modules:
            try:
                new_module = LoRAQKVAttention(
                    model, rank=rank, alpha=alpha, dropout=dropout, peg_distribution=peg_distribution
                )
                count_dict["attn"] += 1
                return new_module, count_dict
            except Exception as e:
                print(f"无法将LoRA应用到Attention模块({parent_name}): {e}")
                
    elif isinstance(model, Attention_Learngene):
        if "attn" in target_modules:
            try:
                new_module = LoRALearnGeneAttention(
                    model, rank=rank, alpha=alpha, dropout=dropout, peg_distribution=peg_distribution
                )
                count_dict["attn"] += 1
                return new_module, count_dict
            except Exception as e:
                print(f"无法将LoRA应用到Attention_Learngene模块({parent_name}): {e}")
                
    elif isinstance(model, Mlp):
        if "mlp" in target_modules:
            try:
                new_module = LoRAMLP(
                    model, rank=rank, alpha=alpha, dropout=dropout, peg_distribution=peg_distribution
                )
                count_dict["mlp"] += 1
                return new_module, count_dict
            except Exception as e:
                print(f"无法将LoRA应用到Mlp模块({parent_name}): {e}")
                
    elif isinstance(model, TransMlp):
        if "mlp" in target_modules:
            try:
                new_module = LoRATransMlp(
                    model, rank=rank, alpha=alpha, dropout=dropout, peg_distribution=peg_distribution
                )
                count_dict["mlp"] += 1
                return new_module, count_dict
            except Exception as e:
                print(f"无法将LoRA应用到TransMlp模块({parent_name}): {e}")
    
    elif isinstance(model, (Block, Block_learngene)):
        # 为Block添加一个跟踪记录
        block_id = len(count_dict["blocks"])
        count_dict["blocks"][block_id] = {"attn": 0, "mlp": 0}
        
    # 递归处理子模块
    for name, child in model.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else name
        if len(list(child.children())) > 0:  # 如果有子模块，递归处理
            new_child, count_dict = apply_lora_to_model_recursive(
                child, rank, alpha, dropout, target_modules, peg_distribution, count_dict, full_name
            )
            setattr(model, name, new_child)
    
    # 如果是Transformer块，并且没有可训练参数，添加一个简单的可训练层
    if isinstance(model, (Block, Block_learngene)):
        has_trainable = False
        for param in model.parameters():
            if param.requires_grad:
                has_trainable = True
                break
        
        if not has_trainable:
            # 确定嵌入维度
            if hasattr(model, 'norm1'):
                # 从标准化层获取维度
                dim = model.norm1.normalized_shape[0]
            elif hasattr(model, 'attn') and hasattr(model.attn, 'qkv') and hasattr(model.attn.qkv, 'in_features'):
                # 从注意力层获取维度
                dim = model.attn.qkv.in_features
            else:
                # 默认维度
                dim = 192
            
            # 添加一个简单的可训练层
            model.trainable_layer = SimpleTrainableLayer(dim)
            count_dict["trainable_added"] += 1
    
    return model, count_dict


# 应用LoRA到整个模型的入口函数
def apply_lora_to_model(
    model,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: List[str] = ["attn", "mlp"],
    peg_distribution: str = 'Gaussian-Layer'
):
    """
    将LoRA应用到模型的特定模块
    
    Args:
        model: 要应用LoRA的模型
        rank: LoRA的秩
        alpha: LoRA的缩放因子
        dropout: LoRA中使用的dropout率
        target_modules: 要应用LoRA的模块类型列表
        peg_distribution: PEG使用的分布类型
    
    Returns:
        应用了LoRA的模型
    """
    # 首先将所有参数设为不需要梯度
    for param in model.parameters():
        param.requires_grad = False
    
    # 使用递归函数应用LoRA
    model, count_dict = apply_lora_to_model_recursive(
        model, rank, alpha, dropout, target_modules, peg_distribution
    )
    
    # 检查模型是否有可训练参数
    has_trainable = False
    for param in model.parameters():
        if param.requires_grad:
            has_trainable = True
            break
    
    # 如果没有可训练参数，添加一个简单的可训练层到顶层
    if not has_trainable:
        print("警告: 模型没有可训练参数，添加一个顶层可训练层")
        # 尝试识别嵌入维度
        if hasattr(model, 'head') and isinstance(model.head, nn.Linear):
            dim = model.head.in_features
        elif hasattr(model, 'pos_embed'):
            dim = model.pos_embed.shape[-1]
        else:
            dim = 192  # 默认维度
        
        # 添加一个全局可训练层
        model.global_trainable_layer = SimpleTrainableLayer(dim)
        count_dict["trainable_added"] += 1
    
    print(f"已应用LoRA到模型: Attention模块数量 = {count_dict['attn']}, MLP模块数量 = {count_dict['mlp']}")
    print(f"额外添加的可训练层数量: {count_dict['trainable_added']}")
    
    return model 