import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class DeepSeek:
    def __init__(self):
        model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype="auto"
        ).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        self.tokenizer.padding_side = "left"
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
    def generate_fusion(self, stt, caption):
        prompt = f"""
        You are a text analysis expert. About one video, here is 1 vision caption: {caption}, 1 audio caption: {stt}. 
        You need to understand and encode them into 1 sentence. Do not simply concatenate them together. The weights of video/audio are equaled. 
        Considering dropping audio caption if it is incomprehensible. And keep detail information. The sentence is:
        """
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_config = self.model.generation_config
        generated_config.update(
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.5
        )
        try:
            generated_ids = self.model.generate(
                **model_inputs,
                generation_config=generated_config
            )
        except Exception as e:
            print(f"DeepSeek 생성 오류 발생: {str(e)}")
            exit(1)

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        # 출력 처리 개선
        response = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True  # 불필요한 공백 제거
        )[0].strip()  # 앞뒤 공백 정리  
        return response