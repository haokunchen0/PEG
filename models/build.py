from .vision_transformer import DeiT_mixture_learngene
from .lora import apply_lora_to_model, print_trainable_parameters


def build_model(config, ancestry_model):
    model_type = config.MODEL.TYPE
    if  model_type == 'deit':
        hdp = config.MODEL.HDP.HDP
        # if hdp:
        #     assert isinstance(hdp, str)
        #     assert hdp in ['q', 'k', 'qk', 'qkv'], hdp
        model = DeiT_mixture_learngene(ancestry_model, 
                    config.MODEL.HDP.DISTRIBUTION, config.MODEL.HDP.FFN_RATIOS, config.MODEL.HDP.DESCENDANT_FFN_RATIOS, config.MODEL.HDP.FFN_INHERIT, config.MODEL.DEIT.NUM_HEADS, config.MODEL.DEIT.NUM_HEADS_LEARNGENE,
                    config.MODEL.DEIT.NUM_HEADS_DESCENDANT, embed_dim=config.MODEL.DEIT.EMBED_DIM, patch_size=16, num_classes=config.MODEL.NUM_CLASSES, depth=config.MODEL.DEIT.DEPTHS, 
                    hdp=config.MODEL.HDP.HDP,
                    hdp_ratios=config.MODEL.HDP.HDP_RATIOS,
                    hdp_non_linear=config.MODEL.HDP.NON_LINEAR,)

        # 检查是否使用LoRA
        if hasattr(config.MODEL, 'LORA') and config.MODEL.LORA.ENABLED:
            print("Applying LoRA to the model...")
            
            # 获取LoRA配置
            lora_rank = getattr(config.MODEL.LORA, 'RANK', 8)
            lora_alpha = getattr(config.MODEL.LORA, 'ALPHA', 16.0)
            lora_dropout = getattr(config.MODEL.LORA, 'DROPOUT', 0.0)
            
            # 确定要应用LoRA的模块
            target_modules = getattr(config.MODEL.LORA, 'TARGET_MODULES', ["attn", "mlp"])
            
            # 应用LoRA
            model = apply_lora_to_model(
                model,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                target_modules=target_modules,
                peg_distribution=config.MODEL.HDP.DISTRIBUTION
            )
            
            # 打印可训练参数信息
            trainable_params, all_params = print_trainable_parameters(model)
            print(f"LoRA parameters: rank={lora_rank}, alpha={lora_alpha}")

    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
