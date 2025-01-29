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
from .modeling_base import BaseMLLM
from .modeling_internvideo2_vit import pretrain_internvideo2_giant_patch14_224_clean, interpolate_pos_embed_internvideo2_new
from .modeling_qformer import build_qformer
# from .flash_attention_class import FlashAttention
import torch.nn.functional as F

logger = logging.getLogger(__name__)

IMG_TOKEN = "[<IMG_PLH>]"
VID_TOKEN = "[<VID_PLH>]"

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_BOS_TOKEN = '<s>'
DEFAULT_EOS_TOKEN = '</s>'
DEFAULT_UNK_TOKEN = "<unk>"

DEFAULT_IMAGE_TOKEN = "[IMAGETOKEN]"
DEFAULT_VIDEO_TOKEN = "[VIDEOTOKEN]"

DEFAULT_IMG_PLACEHOLDER = "[<IMG_PLH>]"
DEFAULT_VID_PLACEHOLDER = "[<VID_PLH>]"

class InternVideo2_VideoChat2(BaseMLLM):
    
    def __init__(
        self,
        config
    ):
        super().__init__(config=config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        instruction = None,
        video_idx = None,
        image_idx = None,
    ):  
        
        """
        text token과 video token을 하나의 vector로 align 진행(self.pad_text_embeds),
        이후 'Mistral-7B' 모델에 입력하는 부분 (Transformer.AutoModelForCausalLM.from_config을 통해 정의한 모델)
        
        Mistral-7B 모델의 forward 함수 설명    
        outputs = self.lm(
            inputs_embeds=text_embeds,      # 토큰 임베딩 텐서 (batch_size, sequence_length, hidden_size)
            attention_mask=attention_mask,   # 어텐션 마스크 (batch_size, sequence_length)
            labels=labels,                   # 정답 레이블 (batch_size, sequence_length)
            output_hidden_states=True,       # 모든 레이어의 hidden states 반환 여부
            return_dict=True,               # 딕셔너리 형태로 반환할지 여부
        )
        """

        if self.use_vision_regression_loss:
            text_embeds, visual, visual_idx = self.pad_text_embeds(input_ids=input_ids, image=image,video=video, return_visual=True, video_idx=video_idx, image_idx=image_idx, instruction = instruction)
        else:
            text_embeds = self.pad_text_embeds(input_ids=input_ids, image=image, video=video, return_visual=False, video_idx=video_idx, image_idx=image_idx,  instruction = instruction)

        outputs = self.lm(
            inputs_embeds=text_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )

        return outputs, text_embeds

    def pad_to_size(self,
                    tensor: torch.Tensor,
                    target_dim: int,
                    target_size: int,
                    pad_value: int = 0):
        """
        tensor의 크기를 target_size로 맞춰 padding 후 반환하는 함수
        
        Args:
            tensor: torch.Tensor
            target_dim: int
            target_size: int
            pad_value: int
            
        Returns:
            padded_tensor: torch.Tensor
        """

        if tensor.shape[target_dim] >= target_size:
            raise ValueError(f"tensor shape {tensor.shape} is not less than target_size {target_size}")
        else:
            max_dim = len(tensor.shape)*2
            padding_size = [0] * max_dim
            padding_size[-(2*target_dim+1)] = target_size - tensor.shape[target_dim]
            padding_size = tuple(padding_size)
            padded_tensor = F.pad(tensor, padding_size, mode='constant', value=pad_value)

            return padded_tensor
    
    def pad_text_embeds(
        self,
        input_ids: torch.LongTensor = None,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        image_idx = None,
        video_idx = None,
        return_visual: bool = False,
        instruction = None,
    ):
        """
        text token과 video token을 하나의 vector로 align 진행
        """
        self.lm.resize_token_embeddings(input_ids.max() + 1) 
        text_embeds = self.lm.get_input_embeddings()(input_ids.long()).detach()

        visual = None
        visual_idx = None

        if image is not None:
            B, T, C, H, W = image.shape
            image = image.permute(0, 2, 1, 3, 4)
            prompt_image_embeds = self.encode_vision(image, instruction=instruction)

            visual = prompt_image_embeds
            prompt_image_embeds = self.project_up(prompt_image_embeds)
            
            prompt_image_embeds = prompt_image_embeds.view(-1, prompt_image_embeds.shape[-1])
            
            visual_idx = image_idx
            text_embeds[image_idx == 1] = text_embeds[image_idx == 1] * 0 + prompt_image_embeds.to(text_embeds.device)

        elif video is not None:
            if len(video.shape) == 5:
                B, T, C, H, W = video.shape
                N = 1
            else:
                B, N, T, C, H, W = video.shape
            
            video = video.reshape(B*N, T, C, H, W).permute(0, 2, 1, 3, 4)
            prompt_video_embeds = self.encode_vision(video, instruction=instruction)

            visual = prompt_video_embeds
            prompt_video_embeds = self.project_up(prompt_video_embeds)
            prompt_video_embeds = prompt_video_embeds.view(-1, prompt_video_embeds.shape[-1])

            seq_length = text_embeds.shape[1]  # 150
            padded_video_idx = torch.zeros(B, seq_length).to(video_idx.device)
            padded_video_idx[:, :video_idx.shape[1]] = video_idx  # 처음 96개 위치에 원래 video_idx 복사
            
            # 입력으로 받은 text_embedds에 video_idx에 vision encoder 결과를 더함.
            # text_embeds에서 padding_video_idx == 1 은 원래 video_idx == 1 인 것이므로, 
            # 해당 위치에 *0을 해주면서 원래 있던 값을 0으로 만들어주고, Vision encoder 결과를 더해줌.
            text_embeds[padded_video_idx == 1] = text_embeds[padded_video_idx == 1] * 0 + prompt_video_embeds.to(text_embeds.device).to(text_embeds.dtype)
            print(f"[Debug] text_embeds shape: {text_embeds.shape}")
        else:   
            logger.warn(f"don't get visual input, input_ids: {input_ids}")
            
        if return_visual:
            return text_embeds, visual, visual_idx
        
        return text_embeds

    def encode_vision(
        self,
        image,
        instruction
    ):
        """
        instruction option에 따라 encode_vision 함수 내부의 Q-former 연산이 다름. 
        해당 값이 있으면, text와 video에 대한 Q-former를 연산함

        Args:
            image: (B, T, C, H, W)
            instruction: str
        Returns:
            query_output: (B, T*L, C)
        """
        device = image.device
        B = image.shape[0]
        T = image.shape[2]
        use_image = True if T == 1 else False
        image_embeds = self.vision_encoder(image, use_image=use_image)
        C = image_embeds.shape[-1]
        image_embeds = image_embeds.reshape(B, -1, C)
        image_embeds = self.vision_layernorm(image_embeds).to(device)  # [B, T*L, C]
        
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
        if self.extra_num_query_token > 0:
            query_tokens = torch.cat([self.query_tokens, self.extra_query_tokens], dim=1)
        query_tokens = query_tokens.expand(image_embeds.shape[0], -1, -1)


        if instruction is not None:
            text_Qformer = self.qformer_tokenizer(
                instruction,
                padding='longest',
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(image_embeds.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image_embeds.device)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)
            query_output = self.qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        else:
            query_output = self.qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        
        return query_output.last_hidden_state[:, :query_tokens.size(1), :]

    def generate_caption(
        self,
        input_ids,
        attention_mask: Optional[torch.Tensor] = None,
        image_idx = None,
        video_idx = None,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        num_beams=1,
        max_new_tokens=200,
        do_sample=True,
        top_p=0.9,
        top_k=None,
        temperature=1.0,
        length_penalty=1,
        repetition_penalty=1.0,
    ):
        """
        llm의 출력을 caption으로 변환하는 함수
    
        Args:
            input_ids: (B, T)
            attention_mask: (B, T)
            image_idx: (B, T)
            video_idx: (B, T)
            image: (B, T, C, H, W)
            video: (B, T, C, H, W)

        Returns:
            outputs: (B, T)
        """
        text_embeds = self.pad_text_embeds(input_ids=input_ids, image=image, video=video, image_idx=image_idx, video_idx=video_idx)
        outputs = self.lm.generate(
            inputs_embeds=text_embeds,
            attention_mask=attention_mask,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            min_length=1,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
        )

        return outputs
    
    def build_input_ids(
            self, 
            tokenizer, 
            conversation,
            max_length,
            add_special_tokens,
            truncation,
            image = None, 
            video = None, 
            padding = "longest", 
            return_tensors = "pt",
            image_placeholder: str = DEFAULT_IMG_PLACEHOLDER,
            video_placeholder: str = DEFAULT_VID_PLACEHOLDER,
    ):
        input_ids = []
        indexs = []
        attention_mask = []
        start, total_len = 0, 0
        while True:
            index1 = conversation.find(image_placeholder, start)
            index2 = conversation.find(video_placeholder, start)
            if index1 == -1 and index2 == -1:
                index = -1
            elif index1 == -1:
                index = index2
            elif index2 == -1:
                index = index1
            else:
                index = min(index1, index2)
                assert index != -1
            if index == -1:
                inputs = tokenizer(conversation[start:], max_length=max_length-total_len, truncation=truncation, padding=padding, return_tensors=return_tensors)
            else:
                inputs = tokenizer(conversation[start:index], max_length=max_length,  truncation=truncation, padding='longest', return_tensors=return_tensors)
            
            input_ids += inputs.input_ids
            attention_mask += inputs.attention_mask
            total_len += inputs.input_ids[0].shape[0]
            indexs += torch.zeros_like(inputs.input_ids)
            
            if index != -1:
                input_ids += [torch.zeros(96).long()]
                attention_mask += [torch.ones(96).long()]
                indexs += [torch.ones(96)]
            
            if index == -1:
                return {
                    'input_ids': torch.cat(input_ids),
                    'attention_mask': torch.cat(attention_mask),
                    'index': torch.cat(indexs).to(torch.bool),
                }
            start = index + len(DEFAULT_IMG_PLACEHOLDER)
            
    def chat(
      self,
      tokenizer,
      msg,
      user_prompt,
      media_type,
      media_tensor, 
      instruction=None,
      chat_history =[],
      return_history =False,
      generation_config={}
    ):
        input_ids, attention_masks, labels = [], [], []

        conversation = ""
        if instruction:
            conversation += instruction
        conversation += (
                    "[INST]" + " "
                )

        if media_type == 'image':
            conversation +=( "<Image>" + IMG_TOKEN + "</Image>")#*ilen
        else:
            conversation += ("<Video>" + VID_TOKEN + "</Video>")#*ilen


        conversation += (
                    msg.rstrip() + "[/INST]"
                )

        for q,a in chat_history:
            conversation += (" [INST] " + q + " [/INST]")
            conversation += (a + "</s>")

        conversation += (" [INST] " + user_prompt + " [/INST]")
        conversation += ("")


        total_len = 0
        indexs = []
        tokenized = self.build_input_ids(
            tokenizer,
            conversation,
            max_length=248,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_tensors='pt'
        )
        
        if media_type == 'image':
            generation_output = self.generate_caption(
                tokenized['input_ids'].unsqueeze(0).to(self.device), 
                tokenized['attention_mask'].unsqueeze(0).to(self.device), 
                image_idx = tokenized['index'].unsqueeze(0),
                image = media_tensor.unsqueeze(0), 
                **generation_config)
        else:
            generation_output = self.generate_caption(
                tokenized['input_ids'].unsqueeze(0).to(self.device), 
                tokenized['attention_mask'].unsqueeze(0).to(self.device), 
                video_idx = tokenized['index'].unsqueeze(0),
                video = media_tensor.unsqueeze(0), 
                **generation_config)
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        if return_history:
            chat_history.append((user_prompt,response))
            return response, chat_history
        return response