import os
import shutil
import re
import time
import torch
import inspect
from typing import List
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

def _find_subsequence_indices(sequence: List[int], subsequence: List[int]) -> List[int]:
    if not subsequence or not sequence:
        return []
    sub_len = len(subsequence)
    return [
        i
        for i in range(len(sequence) - sub_len + 1)
        if sequence[i : i + sub_len] == subsequence
    ]


class CompletionOnlyDataCollator:
    def __init__(
        self,
        tokenizer,
        response_template: str,
        role_start_template: str,
        max_length: int,
    ):
        self.tokenizer = tokenizer
        self.response_template_ids = tokenizer.encode(
            response_template, add_special_tokens=False
        )
        self.role_start_ids = tokenizer.encode(
            role_start_template, add_special_tokens=False
        )
        self.max_length = max_length

    def __call__(self, features):
        if "text" in features[0]:
            texts = [f["text"] for f in features]
            batch = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
        else:
            truncated = []
            for f in features:
                item = dict(f)
                if "input_ids" in item:
                    item["input_ids"] = item["input_ids"][: self.max_length]
                if "attention_mask" in item:
                    item["attention_mask"] = item["attention_mask"][: self.max_length]
                truncated.append(item)
            batch = self.tokenizer.pad(
                truncated,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]

        labels = torch.full_like(input_ids, fill_value=-100)

        for i, seq in enumerate(input_ids.tolist()):
            response_starts = _find_subsequence_indices(
                seq, self.response_template_ids
            )
            if not response_starts:
                continue
            role_starts = _find_subsequence_indices(seq, self.role_start_ids)
            for start_idx in response_starts:
                content_start = start_idx + len(self.response_template_ids)
                next_role = next(
                    (r for r in role_starts if r > start_idx), len(seq)
                )
                labels[i, content_start:next_role] = input_ids[i, content_start:next_role]

        labels[attention_mask == 0] = -100
        for i in range(labels.size(0)):
            if (labels[i] != -100).any():
                continue
            valid_len = int(attention_mask[i].sum().item())
            last_idx = max(valid_len - 1, 1)
            labels[i, last_idx] = input_ids[i, last_idx]

        batch["labels"] = labels
        return batch

def _infer_chat_templates(tokenizer):
    if getattr(tokenizer, "chat_template", None) is None:
        tokenizer.chat_template = (
            "{% set bos_token = '<|begin_of_text|>' %}"
            "{% set eos_token = '<|eot_id|>' %}"
            "{% for message in messages %}"
            "{% if loop.first %}{{ bos_token }}{% endif %}"
            "<|start_header_id|>{{ message['role'] }}<|end_header_id|>\\n\\n"
            "{{ message['content'] }}{{ eos_token }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "<|start_header_id|>assistant<|end_header_id|>\\n\\n"
            "{% endif %}"
        )
    sample_messages = [{"role": "user", "content": "hi"}]
    prompt_no_gen = tokenizer.apply_chat_template(
        sample_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_with_gen = tokenizer.apply_chat_template(
        sample_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    response_template = prompt_with_gen[len(prompt_no_gen) :]
    role_start_template = None
    for role in ("system", "user", "assistant", "tool"):
        idx = prompt_no_gen.find(role)
        if idx != -1:
            role_start_template = prompt_no_gen[:idx]
            break
    if not response_template:
        raise RuntimeError("Failed to infer assistant response template.")
    if not role_start_template:
        role_start_template = response_template
    return response_template, role_start_template

MODEL_ID = ""
DATA_PATH = ""
OUTPUT_DIR = ""
PREPROCESSED_DIR = ""
PREPROCESS_DONE = os.path.join(PREPROCESSED_DIR, ".done")


def train():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"

    response_template, role_start_template = _infer_chat_templates(tokenizer)

    max_length = 2048
    response_template_ids = tokenizer.encode(
        response_template, add_special_tokens=False
    )
    role_start_ids = tokenizer.encode(role_start_template, add_special_tokens=False)
    tool_response_pattern = re.compile(
        r"<tool_response>.*?</tool_response>",
        flags=re.DOTALL | re.IGNORECASE,
    )

    def mask_tool_response(content: str) -> str:
        return tool_response_pattern.sub(
            "<tool_response>[TOOL_RESPONSE_OMITTED]</tool_response>",
            content,
        )

    def formatting_prompts_func(example):
        messages = []
        for msg in example["messages"]:
            if not isinstance(msg, dict):
                messages.append(msg)
                continue
            content = msg.get("content")
            if isinstance(content, str):
                content = mask_tool_response(content)
                msg = {**msg, "content": content}
            if msg.get("role") == "tool":
                msg = {**msg, "content": "<tool_response>[TOOL_RESPONSE_OMITTED]</tool_response>"}
            messages.append(msg)

        output_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        token_ids = tokenizer.encode(
            output_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )
        if len(token_ids) < 2:
            return {"text": output_text, "has_response": False}
        response_starts = _find_subsequence_indices(token_ids, response_template_ids)
        if not response_starts:
            return {"text": output_text, "has_response": False}
        role_starts = _find_subsequence_indices(token_ids, role_start_ids)
        has_response_content = False
        for start_idx in response_starts:
            content_start = start_idx + len(response_template_ids)
            next_role = next((r for r in role_starts if r > start_idx), len(token_ids))
            if next_role > content_start:
                has_response_content = True
                break
        return {
            "text": output_text,
            "has_response": has_response_content,
        }

    def build_and_save_dataset():
        if os.path.exists(PREPROCESSED_DIR):
            shutil.rmtree(PREPROCESSED_DIR, ignore_errors=True)
        os.makedirs(PREPROCESSED_DIR, exist_ok=True)
        raw_dataset = load_dataset("json", data_files=DATA_PATH, split="train")
        dataset = raw_dataset.map(
            formatting_prompts_func,
            remove_columns=raw_dataset.column_names,
        )
        dataset = dataset.filter(lambda x: x["has_response"])
        dataset = dataset.remove_columns(["has_response"])
        dataset.save_to_disk(PREPROCESSED_DIR)
        with open(PREPROCESS_DONE, "w", encoding="utf-8") as f:
            f.write("done\n")

    if local_rank == 0:
        if os.path.exists(PREPROCESS_DONE):
            try:
                _ = load_from_disk(PREPROCESSED_DIR)
            except Exception:
                try:
                    os.remove(PREPROCESS_DONE)
                except OSError:
                    pass
                build_and_save_dataset()
        else:
            build_and_save_dataset()
    else:
        while not os.path.exists(PREPROCESS_DONE):
            time.sleep(2)

    dataset = None
    for _ in range(5):
        try:
            dataset = load_from_disk(PREPROCESSED_DIR)
            break
        except Exception:
            time.sleep(2)
    if dataset is None:
        raise RuntimeError(f"Failed to load dataset from {PREPROCESSED_DIR}")

    split_dataset = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    sft_kwargs = dict(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        packing=False,
        max_seq_length=max_length,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        num_train_epochs=15,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        tf32=True,
        gradient_checkpointing=False,
        ddp_find_unused_parameters=True,
        remove_unused_columns=False,
        logging_steps=1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="tensorboard",
    )
    sft_params = set(inspect.signature(SFTConfig).parameters.keys())
    if "evaluation_strategy" not in sft_params and "eval_strategy" in sft_params:
        sft_kwargs["eval_strategy"] = sft_kwargs["evaluation_strategy"]
    sft_config = SFTConfig(**{k: v for k, v in sft_kwargs.items() if k in sft_params})

    collator = CompletionOnlyDataCollator(
        tokenizer=tokenizer,
        response_template=response_template,
        role_start_template=role_start_template,
        max_length=max_length,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
    )

    model.requires_grad_(True)
    model.config.use_cache = False

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
        data_collator=collator,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)


if __name__ == "__main__":
    train()
