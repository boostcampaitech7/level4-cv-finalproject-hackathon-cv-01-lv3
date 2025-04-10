#pip install nltk pycocoevalcap
#pip install pycocoevalcap
#pip install rouge_score
#pip install bert_score

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate import meteor_score
import nltk
nltk.download('wordnet')
from pycocoevalcap.cider.cider import Cider
from rouge_score import rouge_scorer
from bert_score import score

class InferenceEvaluator:
    def __init__(self, reference, hypothesis):
        """
        :param reference: 정답 문장 (문자열 또는 단어 토큰 리스트)
        :param hypothesis: 모델 출력 문장 (문자열 또는 단어 토큰 리스트)
        """
        # 입력이 문자열인 경우 공백 기준으로 토큰화합니다.
        if isinstance(reference, str):
            self.reference_tokens = [reference.split()]
            self.reference_str = reference
        else:
            self.reference_tokens = reference
            self.reference_str = " ".join(reference)
        
        if isinstance(hypothesis, str):
            self.hypothesis_tokens = hypothesis.split()
            self.hypothesis_str = hypothesis
        else:
            self.hypothesis_tokens = hypothesis
            self.hypothesis_str = " ".join(hypothesis)


    def compute_bleu(self, use_smoothing=True):
        """
        BLEU 점수를 계산합니다.
        
        :param use_smoothing: 스무딩 함수 사용 여부 (기본 True)
        :return: BLEU 점수 (0~1 사이의 값)
        """
        if use_smoothing:
            smoothing_fn = SmoothingFunction().method1
            score = sentence_bleu(self.reference_tokens, self.hypothesis_tokens, smoothing_function=smoothing_fn)
        else:
            score = sentence_bleu(self.reference_tokens, self.hypothesis_tokens)
        return score

    def compute_meteor(self):
        """
        METEOR 점수를 계산합니다.
        NLTK의 meteor_score 모듈은 참조 문장을 문자열 리스트와 가설 문장을 문자열로 받습니다.
        
        :return: METEOR 점수 (0~1 사이의 값)
        """
        score = meteor_score.meteor_score(self.reference_tokens, self.hypothesis_tokens)
        return score
    
    def compute_rouge(self):
        """
        ROUGE 점수를 계산합니다.
        rouge_score 라이브러리를 이용하여 ROUGE-1, ROUGE-2, ROUGE-L 점수를 반환합니다.
        
        :return: dict 형태의 ROUGE 점수 {'rouge1': {...}, 'rouge2': {...}, 'rougeL': {...}}
        """
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(self.reference_str, self.hypothesis_str)
        return scores
    
    # def compute_cider(self):
    #     """
    #     CIDEr 점수를 계산합니다.
    #     pycocoevalcap의 CIDEr 구현은 참조 문장과 가설 문장을 딕셔너리 형태로 받습니다.
    #     예를 들어, 한 이미지에 대해 정답 캡션과 모델 출력 캡션을 아래와 같이 구성합니다.
    #         gts = {0: [정답 캡션 문자열]}
    #         res = {0: [모델 출력 캡션 문자열]}
        
    #     :return: CIDEr 점수 (평균 점수, 한 이미지의 경우 단일 값)
    #     """
    #     # pycocoevalcap의 CIDEr는 참조와 가설이 각각 문자열 리스트 형태로 주어져야 합니다.
    #     gts = {0: [self.reference_str]}
    #     res = {0: [self.hypothesis_str]}
        
    #     cider_scorer = Cider()
    #     score, _ = cider_scorer.compute_score(gts, res)
    #     return score
    
    def compute_bert(self):
        """
        BERT score 계산합니다. 
        """
        gts = [self.reference_str]
        res = [self.hypothesis_str]
        P, R, F1 = score(res, gts, lang="en")
        # F1는 tensor 형태일 수 있으므로, 숫자로 변환합니다.
        return F1.item()


if __name__ == "__main__":
    reference = "The video shows a baseball game between the New York Yankees and the Houston Astros, with the Astros’ Carlos Correa at bat. The Astros’ team color is orange, and Correa is wearing an orange Astros jersey with the number “1” on the back and an orange and navy blue helmet. In the upper right corner, “FS1 ALCS GAME 1” is written in white. Near the bottom of the screen, the score is displayed. The Yankees have 0 runs and the Astros have 2 runs, and Correa has 1 hit out of 2 at bats. The inning is the top of the 6th with 1 out. Correa has a chin guard/face protector attached to his helmet, which is covering the lower half of his face. As Correa looks down at the ground, he is touching the chin guard with his gloved left hand as if adjusting it. In the background, a website for the Astros is displayed as well as other members of the crowd. At the bottom of the screen, information about the Astros win percentage for different years (2011-2014 and 2015-2017) and their MLB rank are shown. Correa lifts his head slightly and continues touching the chin guard/face protector. Then, he picks up his bat with his right hand as he seems to prepare for the next pitch"
    # hypothesis = "The video shows a baseball game between the New York Yankees and the Houston Astros, with the Astros’ Carlos Correa at bat. The Astros’ team color is orange, and Correa is wearing an orange Astros jersey with the number “1” on the back and an orange and navy blue helmet. In the upper right corner, “FS1 ALCS GAME 1” is written in white. Near the bottom of the screen, the score is displayed. The Yankees have 0 runs and the Astros have 2 runs, and Correa has 1 hit out of 2 at bats. The inning is the top of the 6th with 1 out. Correa has a chin guard/face protector attached to his helmet, which is covering the lower half of his face. As Correa looks down at the ground, he is touching the chin guard with his gloved left hand as if adjusting it. In the background, a website for the Astros is displayed as well as other members of the crowd. At the bottom of the screen, information about the Astros win percentage for different years (2011-2014 and 2015-2017) and their MLB rank are shown. Correa lifts his head slightly and continues touching the chin guard/face protector. Then, he picks up his bat with his right hand as he seems to prepare for the next pitch"
    hypothesis = "The video captures a baseball game between the Boston Red Sox and the Los Angeles Dodgers, featuring the Dodgers’ star Mookie Betts at bat. The Dodgers’ primary color is blue, and Betts is sporting a blue Dodgers jersey with the number “50” on the back, along with a blue and white helmet. In the upper left corner of the screen, “ESPN MLB GAME 5” is displayed in white. Near the bottom, the current score is shown: the Red Sox have 1 run, while the Dodgers lead with 3 runs, and Betts has recorded 2 hits out of 3 at bats. It’s the bottom of the 7th inning with 2 outs. Betts’s helmet features a visor that slightly obscures his eyes; as he concentrates on the pitcher, he uses his gloved right hand to adjust the visor. In the background, a Dodgers merchandise website is visible, along with excited fans in the stands. Additionally, at the bottom of the screen, details about Betts’s batting average for the season and the Dodgers’ win-loss record are provided. Betts narrows his eyes for a moment as he readjusts his helmet before gripping his bat firmly with both hands, clearly preparing for the upcoming pitch."
    
    evaluator = InferenceEvaluator(reference, hypothesis)
    bleu = evaluator.compute_bleu()
    meteor = evaluator.compute_meteor()
    # cider = evaluator.compute_cider()
    rouge = evaluator.compute_rouge()
    bert = evaluator.compute_bert()
    
    
    print(f"BLEU Score: {bleu:.4f}")
    print(f"Meteor Score: {meteor:.4f}")
    # print(f"CIDEr Score: {cider:.4f}")
    print(f"BERT Score: {bert:.4f}")
    print("ROUGE Scores:")
    for key, value in rouge.items():
        print(f"  {key}: Precision: {value.precision:.4f}, Recall: {value.recall:.4f}, F1: {value.fmeasure:.4f}")