from GroundingDINO.groundingdino.util.inference import load_model, predict, annotate
import os
import cv2
from tqdm import tqdm
import torch
import clip
from PIL import Image
import torchvision.transforms as T
import numpy as np

#깃 클론을 먼저 해야 Grounding_DINO를 사용할 수 있습니다.
#!git clone https://github.com/IDEA-Research/GroundingDINO.git
CONFIG_PATH = '/workspace/level4-cv-finalproject-hackathon-cv-01-lv3/v2t/exp2/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py'# Grounding DINO GitHubd에서 다운로드 한 위치
#weights 다운로드
#!wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth      <- SwinT
#!wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth <- SwinB
WEIGHTS_NAME = 'groundingdino_swint_ogc.pth'
WEIGHTS_PATH = '/workspace/level4-cv-finalproject-hackathon-cv-01-lv3/v2t/exp2/GroundingDINO/groundingdino_swinb_cogcoor.pth'#weights 다운 받은 경로

def process_image(frame, model, text_prompt: str) -> tuple:
    """
    Grounding DINO 모델을 사용하여 입력 프레임에서 텍스트 프롬프트에 해당하는 객체를 탐지합니다.

    Args:
        frame (np.ndarray): BGR 형식의 이미지 프레임.
        model: Grounding DINO 모델 객체.
        text_prompt (str): 탐지할 객체에 대한 텍스트 설명.

    Returns:
        Tuple:
            boxes (List[List[float]]): 탐지된 객체들의 bounding box 좌표 리스트.
            phrases (List[str]): 각 bounding box에 대응하는 텍스트 프레이즈 리스트.
    """
    # 필요에 따라 Threshold 조정 가능
    BOX_THRESHOLD = 0.1
    TEXT_TRESHOLD = 0.1
    
    tensor_frame = np_to_tensor(frame)

    boxes, _, phrases = predict(
        model = model,
        image = tensor_frame,
        caption = text_prompt,
        box_threshold = BOX_THRESHOLD,
        text_threshold = TEXT_TRESHOLD,
    )

    return boxes, phrases

def np_to_tensor(image_np: np.ndarray) -> torch.Tensor:
    """
    BGR numpy array (OpenCV image)를 RGB PyTorch Tensor로 변환합니다.

    Args:
        image_np(np.ndarray)

    Returns: torch.tensor -> Transform이 적용된 image_rgb(np.ndarray) 
    """
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    transform = T.Compose(
        [
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return transform(image_rgb)

def box_convert(boxes: torch.Tensor, in_fmt: str, out_fmt: str) -> torch.Tensor:
    """
    bbox를 다른 format으로 변경합니다.

    Supported formats:
    - 'cxcywh': (center_x, center_y, width, height)
    - 'xyxy': (x_min, y_min, x_max, y_max)

    Args:
        boxes (torch.Tensor): shape (N, 4)
        in_fmt (str): input format ('cxcywh' or 'xyxy')
        out_fmt (str): output format ('cxcywh' or 'xyxy')

    Returns:
        torch.Tensor: converted boxes, shape (N, 4)
    """
    if in_fmt == out_fmt:
        return boxes

    if in_fmt == "cxcywh" and out_fmt == "xyxy":
        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        print("point:",x1, y1, x2, y2)
        return torch.stack((x1, y1, x2, y2), dim=-1)

    elif in_fmt == "xyxy" and out_fmt == "cxcywh":
        x1, y1, x2, y2 = boxes.unbind(-1)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        return torch.stack([cx, cy, w, h], dim=-1)


    else:
        raise ValueError(f"Unsupported conversion: {in_fmt} -> {out_fmt}")

def find_best_bbox(frame, boxes, text_prompt: str) -> tuple:
    """
    탐지된 여러 객체 중 텍스트 프롬프트와 가장 유사한 하나의 객체를 선택합니다.

    Args:
        frame (np.ndarray): BGR 이미지 프레임.
        boxes (List[List[float]]): Grounding DINO로부터 받은 bounding box 좌표들.
        text_prompt (str): 객체를 설명하는 텍스트 프롬프트.

    Returns:
        Tuple:
            best_bbox (List[float]): 가장 유사도가 높은 객체의 bounding box.
            best_crop (np.ndarray): 해당 bounding box 영역으로 자른 이미지.
            best_score (float): CLIP 기반 유사도 점수.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess = clip.load("ViT-B/32", device=device)
    text_tokens = clip.tokenize([text_prompt]).to(device)
    
    best_score = -1
    best_bbox = None
    best_crop = None
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
    for box in boxes:
        h, w, _ = frame.shape
        box_tensor = torch.tensor(box).float()
        box_tensor*=torch.tensor([w, h, w, h])
        box_xyxy = box_convert(box_tensor.unsqueeze(0), "cxcywh", "xyxy").squeeze(0)

        x0, y0, x1, y1 = map(int, box_xyxy.tolist())
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(w, x1)
        y1 = min(h, y1)
  
        print("point:",x0, y0, x1, y1)
        if x1 <= x0 or y1 <= y0:
            print(f"Skipped invalid box: {box}")
            continue

        cropped_frame = frame[y0:y1, x0:x1]
        if cropped_frame.size == 0:
            print(f"Skipped empty crop: {box}")
            continue
        
        pil_image = Image.fromarray(cropped_frame[:, :, ::-1]) # BGR to RGB
        image_input = preprocess(pil_image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            similarity = torch.nn.functional.cosine_similarity(text_features, image_features).item()
            
        if similarity > best_score:
            best_score = similarity
            best_crop = cropped_frame
            best_bbox = box
            
    return best_bbox, best_crop, best_score

def run_detection(frame, model, categories:list) -> list:
    """
    여러 텍스트 카테고리에 대해 객체를 탐지하고, CLIP을 통해 가장 관련 있는 객체를 선택합니다.

    Args:
        frame (np.ndarray): BGR 이미지 프레임.
        model: Grounding DINO 모델 객체.
        categories (List[str]): 객체 탐지를 위한 텍스트 프롬프트 리스트.

    Returns:
        List[Dict]: 각 카테고리별로 탐지 결과를 포함하는 딕셔너리 리스트.
            - "label": Grounding DINO가 반환한 프레이즈.
            - "description": 입력된 텍스트 프롬프트.
            - "bbox": CLIP 유사도가 가장 높은 객체의 bounding box.
            - "crop": 해당 영역의 이미지 crop (numpy 배열).
            - "clip_score": CLIP 기반 유사도 점수.
    """
    results=[]
    
    for category in categories:
        boxes, phrases = process_image(frame, model, category)
        bbox, crop, score = find_best_bbox(frame, boxes, category)
        results.append({"label": phrases,
                        "description": category,
                        "bbox": bbox,
                        "crop": crop,  # numpy array or PIL.Image
                        "clip_score": score})
    
    return results    
        
    

#===============================================================실험용 코드입니다==================================================================
if __name__ == "__main__":
    model = load_model(CONFIG_PATH, WEIGHTS_PATH)

    frame = cv2.imread('/workspace/level4-cv-finalproject-hackathon-cv-01-lv3/v2t/exp2/pexels-jenny-uhling-2262740-31857462.jpg')

    categories = [
        "The man is wearing a beige t-shirt, black pants, and glasses.",
        "The child is wearing an orange shirt and dark-colored jeans." 
    ]

    output_path = "/workspace/output_result.jpg"  # 필요시 절대경로로 수정


results = run_detection(frame, model, categories)


if frame is None:
    raise ValueError("❌ 이미지 프레임이 None입니다. cv2.imread 경로를 확인하세요.")

for idx, res in enumerate(results):
    print(f"[{idx}] label: {res['label']}")
    print(f"   desc : {res['description']}")
    print(f"   bbox : {res['bbox']}")
    print(f"   CLIP score: {res['clip_score']:.4f}")

    if res['bbox'] is not None:
        try:
            if len(res['bbox']) == 4:
                h, w, _ = frame.shape

                box_tensor = torch.tensor(res['bbox']).float()

                box_tensor*=torch.tensor([w, h, w, h])
                box_xyxy = box_convert(box_tensor.unsqueeze(0), "cxcywh", "xyxy").squeeze(0)
                x0, y0, x1, y1 = map(int, box_xyxy.tolist())
                x0 = max(0, x0)
                y0 = max(0, y0)
                x1 = min(w, x1)
                y1 = min(h, y1)

                if 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h:
                    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 5)
                    cv2.putText(frame, res['description'], (x0, max(0, y0 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                else:
                    print(f"⚠️ 잘못된 bbox 좌표: {(x0, y0, x1, y1)} → 이미지 사이즈 벗어남")
            else:
                print(f"⚠️ bbox 포맷이 이상함: {res['bbox']}")
        except Exception as e:
            print(f"❌ bbox 처리 중 오류: {e}")
    else:
        print("⚠️ bbox가 유효하지 않아 스킵됨.")

saved = cv2.imwrite(output_path, frame)
if saved:
    print(f"✅ 결과 이미지 저장 완료: {output_path}")
else:
    print(f"❌ 이미지 저장 실패! 경로 쓰기 권한이나 파일 경로 확인 필요: {output_path}")
