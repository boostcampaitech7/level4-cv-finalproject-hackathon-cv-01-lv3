from eval_metric import InferenceEvaluator
from inference import inference, sec_to_time
import pandas as pd
import json
import os

def find_file(root_dir, target_filename):
    for root, dirs, files in os.walk(root_dir):
        if target_filename in files:
            return os.path.join(root, target_filename)
    return None

def main():
    data_path = "../../data"
    model_path = "./model/weights"
    #inference.py에서 inference 함수 불러와서 결과주는 코드 
    inference(data_path, model_path)
    
    csv_file_path = "./v2t_submission.csv"
    inference_result = pd.read_csv(csv_file_path)

    results = []
    
    for index, row in inference_result.iterrows():
        target_filename = str(row['segment_name']) + ".json"
        
       
        gt_file_path = find_file("../../data", target_filename)
        
        with open(gt_file_path, 'r', encoding='utf-8') as f:
            gt = json.load(f)
            reference = gt.get('caption')
        
        hypothesis = row['caption']
        evaluator = InferenceEvaluator(reference, hypothesis)
        bleu = evaluator.compute_bleu()
        meteor = evaluator.compute_meteor()
        rouge = evaluator.compute_rouge()
        rouge_str = {key: {
                        "precision": value.precision,
                        "recall": value.recall,
                        "f1": value.fmeasure
                      } for key, value in rouge.items()}
        bert = evaluator.compute_bert()

        results.append({
            "segment_name": row['segment_name'],
            "reference": reference,
            "hypothesis": hypothesis,
            "BLEU Score": bleu,
            "Meteor Score": meteor,
            "BERT Score": bert,
            "ROUGE Score": json.dumps(rouge_str) 
        })

        print(row['segment_name'])
        print(f"BLEU Score: {bleu:.4f}")
        print(f"Meteor Score: {meteor:.4f}")
        print(f"BERT Score: {bert:.4f}")
        print("ROUGE Scores:")
        for key, value in rouge.items():
            print(f"  {key}: Precision: {value.precision:.4f}, Recall: {value.recall:.4f}, F1: {value.fmeasure:.4f}")
        
        results_df = pd.DataFrame(results)
        output_csv_path = "./evaluation_score.csv"
        results_df.to_csv(output_csv_path, index=False, encoding='utf-8')
        print(f"평가 결과가 {output_csv_path}에 저장되었습니다.")

if __name__ == "__main__":
    main()
