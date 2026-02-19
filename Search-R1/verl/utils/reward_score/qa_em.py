import re
import string
from typing import List


def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction: str, golden_answers: List[str]) -> bool:
    if not prediction:
        return False

    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]

    pred_norm = normalize_answer(prediction)
    for ga in golden_answers:
        if normalize_answer(ga) == pred_norm:
            return True
    return False


def is_valid_sequence(text: str) -> bool:
    return "<answer>" in text and "</answer>" in text


def extract_solution(solution_str: str):
    """更稳健的答案提取逻辑"""
    pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(pattern, solution_str, re.DOTALL)
    if not matches:
        return None
    return matches[-1].strip()


def extract_information_blocks(text: str):
    return re.findall(r"<information>(.*?)</information>", text, re.DOTALL)


def is_retrieval_correct(text: str, golden_answers: List[str]) -> bool:
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]

    infos = extract_information_blocks(text)
    for info in infos:
        info_norm = normalize_answer(info)
        for ga in golden_answers:
            if normalize_answer(ga) in info_norm:
                return True
    return False



def compute_score_em(solution_str, ground_truth, **kwargs):
    try:
        if 'ground_truth' in ground_truth:
            actual_gt = ground_truth['ground_truth']['target']
        else:
            actual_gt = ground_truth.get('target', [])
            
        if hasattr(actual_gt, 'tolist'):
            actual_gt = actual_gt.tolist()
    except Exception:
        actual_gt = []

    score = kwargs.get('score', 1.0)
    fmt_score = kwargs.get('format_score', 0.1) 
    retrieval_bonus = kwargs.get('retrieval_bonus', 0.1)

    has_answer_tags = "<answer>" in solution_str and "</answer>" in solution_str
    
    if not has_answer_tags:
        return 0.0

    current_reward = fmt_score
    answer = extract_solution(solution_str)

    if answer and em_check(answer, actual_gt):
        current_reward += score
    else:
        if is_retrieval_correct(solution_str, actual_gt):
            current_reward += retrieval_bonus

    return current_reward
