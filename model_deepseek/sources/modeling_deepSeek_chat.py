import io
import logging
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import MSELoss
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
)
from typing import List, Optional, Tuple, Union
from torch.cuda.amp import autocast as autocast
from .modeling_base_addDeepSeek import BaseMLLM
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class DeepSeekChat(BaseMLLM):
    def __init__(
        self,
        config
    ):
        super().__init__(config=config)
        # 언어 모델 헤드 추가
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
    def forward(self, **kwargs):
        # 기존 forward 로직 유지
        outputs = super().forward(**kwargs)
        
        # 언어 모델 헤드 추가
        logits = self.lm_head(outputs.last_hidden_state)
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )