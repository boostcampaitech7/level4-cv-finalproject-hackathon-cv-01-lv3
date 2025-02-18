# 🎥 영화 속 장면 검색을 위한 Video-to-Text / Text-to-Frame 프로젝트 
<img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white"> <img src ="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white"> <img src ="https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi">
<img src= "https://img.shields.io/badge/elasticsearch-%230377CC.svg?style=for-the-badge&logo=elasticsearch&logoColor=white">
<img  src="https://img.shields.io/badge/jira-%230A0FFF.svg?style=for-the-badge&logo=jira&logoColor=white"> <img src= "https://img.shields.io/badge/confluence-%23172BF4.svg?style=for-the-badge&logo=confluence&logoColor=white">
>본 프로젝트는 **7기 TVING 기업 연계 프로젝트**로,  
>자연어를 활용한 **동영상 속 특정 장면 검색**을 목표로 하는 **Video Retrieval** 프로젝트 입니다. <br>
> 기존 **메타데이터(제목, 태그, 키워드) 기반의 영상 검색** 방식은 영상 **구간별 검색**이 불가하다는 한계를 해결하고자,
> 본 프로젝트는 **장면별 텍스트 변환(V2T, Video-to-Text)** 과 **텍스트 기반 장면 검색 기능(T2V , Text-to-Video)** 을 통해
>사용자가 원하는 특정 장면 검색이 가능하도록 하였습니다.   


## 🎯 주요 기능  

### 📺 **Video-to-Text (V2T)**   

- 동영상을 **장면 기준으로 자동 분할**하고, 장면 별 **설명문을 생성**하여 저장합니다.



### 🔍 **Text-to-Video (T2V)**  
- 사용자가 입력한 **자연어 쿼리**를 기반으로, **가장 적절한 장면을 검색**하여 제공합니다.  
- 영상 속 **대사(Speech-to-Text) 및 장면 설명**을 활용하여 더욱 정교한 검색을 수행합니다.


**💡특징**  

✅ 빠른 검색을 위해 Vector DB 사용 및 병렬 MSA(Micro Service Architecture) 패턴 적용 <br>
✅ 최신 멀티모달 AI 모델(IntrenVL2.5, InternVideo2.5 등)과 Whisper STT, Vector DB 등 최신 기술 채용<br>
✅ 영상 내 Speech 정보 + 전체 영상 summary + 장면 캡션의 결합(Cap fusion)으로 검색 정확도 향상<br>
✅ MLLM의 Hallucination 문제 해결을 위한 자체 정성 평가 체크리스트 및 Prompt Engineering 수행<br>




## 📜 프로젝트 아키텍처  
### 모델 아키텍쳐 
![image](https://github.com/user-attachments/assets/703495ee-092c-48eb-87dc-02d09282363d)

### 서비스 아키텍쳐 
![image](https://github.com/user-attachments/assets/8330ef98-a8ed-424c-8cb9-88bd0135050f)

## 🎬 Demo 

### Demo Page (GPU 리소스로 인해 ~2/28 까지만 제공됩니다)
🌐 [실시간 데모 체험하기](https://affecting-rl-tend-kg.trycloudflare.com/)



### V2T 
![image](https://github.com/user-attachments/assets/8d586e7d-092c-42f2-9979-e1c8789e5661)
### T2V 
![image](https://github.com/user-attachments/assets/8384e9d7-234f-4028-90e7-785c6a6711a7)

---
## 👥 팀원 소개  

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/9a15231a-b69d-447f-9070-f58b29ccdcec"
" width="150px;" alt="이한성"/><br />
      <b>이한성 (T7232)</b><br />
      PM, Speech-to-Text, <br />T2V(Vector DB) 구축, <br />Demo 페이지 (Back-End)
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/7c44b0c5-927a-4c65-8d21-8e240bcf1618" width="150px;" alt="강대민"/><br />
      <b>강대민 (T7101)</b><br />
      모델 구축 및 환경 설정, <br />Prompt Engineering, <br />Fine-tuning, V2T 구축
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/fc431d0d-51d5-4774-b900-67bc6a2bb2b5" width="150px;" alt="김홍주"/><br />
      <b>김홍주 (T7142)</b><br />
      Video Trimming, <br />데이터 수집 및 라벨링, <br />Prompt Engineering, V2T 구축
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/b17ce868-5498-4acf-8831-31829f8f7cbd" width="150px;" alt="서승환"/><br />
      <b>서승환 (T7161)</b><br />
      Video Trimming, <br />T2V 구축, Fine-Tuning, V2T 구축
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/ddebfbe1-317d-4bf7-915c-524e51e5bd69" width="150px;" alt="박나영"/><br />
      <b>박나영 (T7147)</b><br />
      번역 모델, <br />Demo 페이지 (Front-End), <br />데이터 수집 및 라벨링, V2T 구축
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/d155ec79-8d03-45d4-b703-44a848b9b463" width="150px;" alt="이종서"/><br />
      <b>이종서 (T7171)</b><br />
      데이터 수집 및 라벨링, <br />T2V 구축, 평가 방법 제시
    </td>
  </tr>
</table>


## 📅 프로젝트 타임 라인 
![image](https://github.com/user-attachments/assets/81361036-72ed-4d82-92b9-06dc9ea01bff)

### 📚 추가 자료
- 발표 영상
- [랩업 리포트 ](https://docs.google.com/document/d/1TtDpcJWyHGGwEDV9qbvnlARkm2NByundthXcdKrACmc/edit?usp=sharing)

