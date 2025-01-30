# new_llm_script.py
import os
import warnings
import logging
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import MSELoss
from torch.cuda.amp import autocast as autocast

from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from model_deepseek.sources.model_config import Qwen2_5_Config
# from model.sources.modeling_internvideo2_vit import pretrain_internvideo2_giant_patch14_224_clean
# from model.sources.modeling_qformer import build_qformer

# 시스템 경로를 추가하여 상위 경로 접근 가능하도록 변경
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

logger = logging.getLogger(__name__)

try:
    token = os.environ['HF_TOKEN']
except:
    warnings.warn("The HF_TOKEN was not found in the system variables. Please ensure that it is filled out correctly and that you have requested access to the model. If you haven't applied, please visit https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3 to request access.")
    token=None

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def freeze_module(module):
    for _, param in module.named_parameters():
        param.requires_grad = False
    module = module.eval()
    module.train = disabled_train
    return module

class BaseMLLM(PreTrainedModel):
    config_class = Qwen2_5_Config
    def __init__(self, config):
        self.model_config = config.model_config
        config.model_config = None
        super().__init__(config)
        self.lm = None
        self.build_llm()

    def build_llm(self):
        self.lm_name = self.model_config.llm.name
        if self.model_config.llm.name == 'mistral_7b':
            config = AutoConfig.from_pretrained(
                self.model_config.llm.pretrained_llm_path,
                torch_dtype=torch.bfloat16,
                token=token,
                # attn_implementation="flash_attention_2",
            )
            lm = AutoModelForCausalLM.from_config(config)
        elif self.model_config.llm.name == 'internlm_20b':
            lm = AutoModelForCausalLM.from_pretrained(
                self.model_config.llm.pretrained_llm_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            lm.gradient_checkpointing = True
            lm._set_gradient_checkpointing()
        elif self.model_config.llm.name == 'internlm2_5_7b':
            lm = AutoModelForCausalLM.from_pretrained(
                self.model_config.llm.pretrained_llm_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                local_files_only=True,
            )
        elif self.model_config.llm.name == 'qwen2':
            lm = AutoModelForCausalLM.from_pretrained(
                self.model_config.llm.pretrained_llm_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
        else:
            raise NotImplementedError(self.model_config.llm.name)

        freeze_llm = self.model_config.get("freeze_llm", True) # True->False->True: OOM 문제

        logger.info(f'freeze_llm: {freeze_llm}')
        if freeze_llm:
            logger.info("freeze llm")
            for _, param in lm.named_parameters(): # freeze_module 함수 내용을 여기에 직접 적용
                param.requires_grad = False
            lm.eval()
            def disabled_train(self, mode=True): # disabled_train 함수 내용을 여기에 직접 정의
                return self
            lm.train = disabled_train

        elif not freeze_llm:
            lm.gradient_checkpointing_enable()  # 메모리 효율을 위한 설정
            lm.enable_input_require_grads()

        # if model_config.llm.use_lora:
        #     use_lora = True
        #     logger.info("Use lora")
        #     if model_config.llm.name == 'internlm_20b':
        #         peft_config = LoraConfig(
        #             task_type=TaskType.CAUSAL_LM, inference_mode=False,
        #             r=model_config.llm.lora_r, lora_alpha=model_config.llm.lora_alpha, lora_dropout=model_config.llm.lora_dropout,
        #             target_modules=['wqkv', 'wo', 'w1', 'w2', 'w3', 'output']
        #         )
        #     else:
        #         peft_config = LoraConfig(
        #             task_type=TaskType.CAUSAL_LM, inference_mode=False,
        #             r=model_config.llm.lora_r, lora_alpha=model_config.llm.lora_alpha, lora_dropout=model_config.llm.lora_dropout,
        #             target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
        #                             "gate_proj", "up_proj", "down_proj", "lm_head"]
        #         )
            # lm = get_peft_model(lm, peft_config)
            # lm.enable_input_require_grads()
            # lm.print_trainable_parameters()
        else:
            use_lora = False
        self.lm = lm
        
        
    def forward(self, inputs):
        return self.lm(inputs)  # 순전파 메서드 구현
    @property
    def dtype(self):
        return self.lm.dtype


    @property
    def device(self):
        return self.lm.device
