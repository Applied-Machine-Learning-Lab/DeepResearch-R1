# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import string
import random
import requests
import time

JUDGE_API_KEY = 
JUDGE_API_BASE = 
JUDGE_MODEL = 

def call_llm_judge(reference, prediction, max_retries=5):

    if not prediction or not str(prediction).strip():
        return False

    prompt = f"""Role: Strict Factual Auditor
Task: Compare the Predicted Answer against the Reference Answer.

[Reference Answer]: {reference}
[Predicted Answer]: {prediction}

Verdict: [[CORRECT]] if the core facts match, otherwise [[INCORRECT]].
Output only the verdict."""

    payload = {
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0
    }
    headers = {"Authorization": f"Bearer {JUDGE_API_KEY}"}

    for attempt in range(max_retries):
        try:
            response = requests.post(JUDGE_API_BASE, headers=headers, json=payload, timeout=20)
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                return "[[CORRECT]]" in content
            elif response.status_code == 429:  
                time.sleep(2 ** attempt + random.uniform(0, 1))
            else:
                time.sleep(1)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"LLM Judge API 最终失败: {e}")
            time.sleep(1)
    return False

def normalize_answer(s):
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


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def is_valid_sequence(text):
    # Find the position of "<|im_start|>assistant" with potential whitespace
    assistant_pattern = r"<\|im_start\|>assistant\s*"
    assistant_match = re.search(assistant_pattern, text)
    
    if not assistant_match:
        return False, "Missing assistant marker"
    
    # Extract the content after the assistant marker
    start_pos = assistant_match.end()
    content = text[start_pos:]
    
    # Check for balanced tags
    tags_to_check = ["think", "search", "information", "answer"]
    for tag in tags_to_check:
        opening_count = len(re.findall(f"<{tag}>", content))
        closing_count = len(re.findall(f"</{tag}>", content))
        if opening_count != closing_count:
            return False, f"Mismatch in {tag} tags: {opening_count} opening vs {closing_count} closing tags"
    
    # Now check for proper sequence pattern and no extraneous content
    
    # 1. First split the content by any tags we recognize
    split_pattern = r"(</?(?:think|search|information|answer)>)"
    parts = re.split(split_pattern, content)
    
    # 2. Keep track of the current position in the expected sequence
    state = "start"  # start -> think -> search -> information -> think -> ... -> answer -> end
    
    # 3. Check each part
    for i, part in enumerate(parts):
        # Skip empty parts
        if not part.strip():
            continue
            
        # Check if this is a tag
        if re.match(r"</?(?:think|search|information|answer)>", part):
            # This is a tag, check if it's valid in the current state
            if part == "<think>" and state in ["start", "information"]:
                state = "in_think"
            elif part == "</think>" and state == "in_think":
                state = "after_think"
            elif part == "<search>" and state == "after_think":
                state = "in_search"
            elif part == "</search>" and state == "in_search":
                state = "after_search"
            elif part == "<information>" and state == "after_search":
                state = "in_information"
            elif part == "</information>" and state == "in_information":
                state = "information"
            elif part == "<answer>" and state == "after_think":
                state = "in_answer"
            elif part == "</answer>" and state == "in_answer":
                state = "end"
            else:
                return False, f"Unexpected tag {part} in state {state}"
        else:
            # This is content, check if it's valid in the current state
            if state in ["in_think", "in_search", "in_information", "in_answer"]:
                # Content is allowed inside tags
                pass
            elif state in ["start", "after_think", "after_search", "information"]:
                # Only whitespace is allowed between tags
                if part.strip():
                    return False, f"Unexpected content '{part.strip()}' between tags (state: {state})"
            else:
                return False, f"Unexpected content in state {state}"
    
    # Check final state
    if state != "end":
        return False, f"Incomplete sequence, ended in state {state}"
        
    return True, "Valid sequence format"


def extract_solution(solution_str):
    """Extract the equation from the solution string."""

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    
    # If there are 0 or exactly 1 matches, return None
    if len(matches) <= 1:
        return None
    
    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def extract_information_blocks(text: str) -> list[str]:
    pattern = r"<information>(.*?)</information>"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]


def is_retrieval_correct(text: str, golden_answers: list[str]) -> list[str]:
    seqs = extract_information_blocks(text)
    for seq in seqs:
        for golden_answer in golden_answers:
            if normalize_answer(golden_answer) in normalize_answer(seq):
                return True
    return False


def compute_score_em(solution_str, ground_truth, **kwargs):

    try:
        actual_ground_truth = ground_truth['ground_truth']['target']
        if hasattr(actual_ground_truth, 'tolist'):
            actual_ground_truth = actual_ground_truth.tolist()
        judge_reference = actual_ground_truth[0] if isinstance(actual_ground_truth, list) and len(actual_ground_truth) > 0 else actual_ground_truth
    except (KeyError, TypeError):
        actual_ground_truth = ground_truth.get('target', [])
        judge_reference = actual_ground_truth

    is_valid_format, _ = is_valid_sequence(solution_str)
    retrieval_correct = False
    if is_valid_format:
        retrieval_correct = is_retrieval_correct(solution_str, actual_ground_truth)
    
    answer = extract_solution(solution_str=solution_str)
    
    if answer is None:
        if is_valid_format:
            return kwargs.get('structure_format_score', 0) + (kwargs.get('retrieval_score', 0) if retrieval_correct else 0)
        return 0
    else:
        is_correct = em_check(answer, actual_ground_truth)
        
        if not is_correct:
            is_correct = call_llm_judge(judge_reference, answer[:500])

        if is_correct:
            return kwargs.get('score', 1.0) if is_valid_format else (kwargs.get('score', 1.0) - kwargs.get('structure_format_score', 0))
        return kwargs.get('final_format_score', 0.2)