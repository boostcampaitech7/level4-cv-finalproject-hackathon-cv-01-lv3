import os
import torch
from transformers import AutoTokenizer, AutoConfig
from model.sources.model_config import VideoChat2Config
from model.sources.modeling_videochat2 import InternVideo2_VideoChat2
from decord import VideoReader, cpu
import torch.nn.functional as F
import torchvision.transforms as T
from model.utils.data_utils_from_json import InternVideo2_VideoChat2_Dataset, InternVideo2_VideoChat2_DataLoader
from tqdm import tqdm
import transformers

def train(
    model_path,
    video_path,
    data_path="../../data",
    num_epochs=50,
    train_batch_size=2,
    test_batch_size=1,
    train_num_workers=4,
    test_num_workers=4,
    learning_rate=1e-4,
    weight_decay=1e-2,
    validation_interval=25,
    device='cuda'
):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config = VideoChat2Config.from_json_file(
        os.path.join(current_dir,'model','configs', 'config.json')
    )

    # 토크나이저 초기화 (Mistral-7B)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_config.llm.pretrained_llm_path,
        trust_remote_code=True,
        use_fast=False,
        token=os.getenv('HF_TOKEN')
    )
    tokenizer.add_special_tokens({'pad_token': '<unk>'})

    # 모델 초기화
    if torch.cuda.is_available():
        model = InternVideo2_VideoChat2.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).to(device)
    else:
        model = InternVideo2_VideoChat2.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )

    train_dataset = InternVideo2_VideoChat2_Dataset(
        data_path=data_path,
        use_segment=True,
        use_audio=False,
        train=True
    )

    test_dataset = InternVideo2_VideoChat2_Dataset(
        data_path=data_path,
        use_segment=True,
        use_audio=False,
        train=False
    )
    
    train_loader = InternVideo2_VideoChat2_DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        num_workers=train_num_workers,
        shuffle=False,
        pin_memory=True,
        use_audio=False
    )
    
    test_loader = InternVideo2_VideoChat2_DataLoader(
        dataset=test_dataset,
        batch_size=test_batch_size,
        num_workers=test_num_workers,
        shuffle=False,
        pin_memory=True,
        use_audio=False
    )
    
    optimizer, scheduler = model.prepare_for_training(
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )

    from rouge_score import rouge_scorer
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    query_embedding_size = model.query_tokens.shape[1] + model.extra_query_tokens.shape[1]
    # video_index = torch.ones(train_batch_size, query_embedding_size).to(device)

    # 학습 도중에 Text와 Video의 각 embedding vector를 합치는 과정이 있습니다.
    # 현재 알고리즘은, video_index 크기를 text embedding vector보다 작게 유지시켜야 합니다.
    # (내부 알고리즘에서 video_index를 text에 맞춤)
    # 따라서 추후 논의 전까지, raise 검증문을 하나 추가합니다.
    if train_batch_size * query_embedding_size > 300:
        raise ValueError("upper bound of Video index is 300, video_index=train_batch_size*query_embedding_size")

    for epoch in range(num_epochs):
        train_loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, batch in enumerate(train_loop):
            if torch.cuda.is_available():
                frames = batch['frames'].to(device)
                annotations = batch['annotations']  # 텍스트 데이터는 그대로 유지
            else:
                frames = batch['frames']
                annotations = batch['annotations']
            
            # 텍스트 토큰화
            print(f"[Debug train]: Tokenizer를 forward 합니다")
            text_inputs = tokenizer(
                annotations,
                padding='max_length',
                truncation=True,
                max_length=96,
                return_tensors="pt"
            ).to(device)
            print(f"[Debug train]: Tokenizer를 실행하여 text_inputs를 생성하였습니다: {text_inputs}")

            optimizer.zero_grad()
            # forward 패스 수행
            # 현재는 LLM 출력이 Outputs에 해당하고, LoRA를 건들면 제대로 값이 나오질 않으니, 이대로 갑니다. 
            # 추후 가중치 변경을 잘 끝내면 Q-former에 대한 logit 계산을 위해 validation처럼 text_embeds를 사용하겠습니다. 
            # forward 함수를 호출하면서, attention_mask는 text에 대한 logit을 만들 때 사용함
            outputs, _ = model(
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask,
                video=frames,
                labels=text_inputs.input_ids,
                video_idx=torch.ones(frames.shape[0], query_embedding_size).to(device)
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            # print(f"outputs: {outputs[0]}")
            # tqdm description 업데이트하여 loss 표시
            train_loop.set_postfix(loss=f'{loss.item():.4f}, batch_idx: {batch_idx}')
            
            del frames
            del annotations
            del text_inputs
            del outputs
            del loss

            # Cache는 내부에서 사용해서, 지울 수 없음.

            # if epoch % validation_interval == 0:
            #     print("--------------------------------")
            #     print(f"validation start, epoch: {epoch+1}")
            #     validation(model, test_loader, tokenizer, device, query_embedding_size)
            #     print(f"validation end, epoch: {epoch+1}")
            #     print("--------------------------------")
            #     model.train()
            

def validation(model, dataloader, tokenizer, device, query_embedding_size):
    model.eval()
    total_loss = 0
    # validation loop에도 tqdm 적용
    val_loop = tqdm(dataloader, desc='Validation')
    with torch.no_grad():
        for batch in val_loop:
            frames = batch['frames'].to(device)
            annotations = batch['annotations']
            
            # text_inputs = tokenizer(
            #     annotations,
            #     padding='max_length',
            #     truncation=True,
            #     max_length=96,
            #     return_tensors="pt"
            # ).to(device)
            
            # outputs, _ = model(
            #     input_ids=text_inputs.input_ids,
            #     attention_mask=text_inputs.attention_mask,
            #     video=frames,
            #     labels=text_inputs.input_ids,
            #     video_idx=torch.ones(frames.shape[0], query_embedding_size).to(device)
            # )
            
            #print(f"sucess extract text_embeds from vision encoder and q-former")

            generation_config = {
                'num_beams': 1,            # 빔 서치 크기
                'max_new_tokens': 200,     # 최대 생성 길이
                'do_sample': False,         # 샘플링 사용
                'top_p': 0.9,
                'top_k': None,
                'temperature': 1.0,
                'length_penalty': 1,
                'repetition_penalty': 1.0
            }

            # generate를 사용하기에는 decoding에 관한 이해가 전혀 없어 pass. 다시 chat 기반으로 수정함.
            # 캡션 생성
            response, _ = model.chat(
                tokenizer=tokenizer,
                msg='',
                user_prompt='Describe the video step by step',
                instruction="Carefully watch the video and describe what is happening in detail.",
                media_type='video',
                media_tensor=frames,
                chat_history=[],
                return_history=True,
                generation_config=generation_config
            )
            
            print(f"Validation Output: {response}")
            print(f"Annotation: {batch['annotations']}")

    avg_loss = total_loss / len(dataloader)
    #print(f"Validation Loss: {avg_loss}")
    return avg_loss

def main():
    # 현재 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 모델 경로 설정
    model_path = os.path.join(current_dir, "model/weights")
    
    # 비디오 경로 설정
    video_path = os.path.join(current_dir, "../../data")

    train(model_path, video_path)

if __name__ == "__main__":
    main()
