# train_lora_llm.py

import os
import json
import argparse
from dataclasses import dataclass
from typing import Dict, List
import boto3
import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from peft import LoraConfig, get_peft_model


# =========================
# 1. Dataset 정의
# =========================

class JsonlSftDataset(Dataset):
    """
    JSONL 형식의 SFT 데이터셋:
      {"prompt": "...", "completion": "...", "date": "..."}
    를 받아서
      prompt + "\n\n### 답변:\n" + completion
    구조의 텍스트를 만들고, prompt 부분은 loss에서 제외(-100)한다.
    """

    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.records: List[Dict] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                prompt = obj["prompt"]
                completion = obj["completion"]
                self.records.append(
                    {
                        "prompt": prompt,
                        "completion": completion,
                        "date": obj.get("date", None),
                    }
                )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        prompt = rec["prompt"]
        completion = rec["completion"]

        # 프롬프트 텍스트 / 전체 텍스트 구성
        prompt_text = prompt.rstrip() + "\n\n### 답변:\n"
        full_text = prompt_text + completion.rstrip()

        # 각각 따로 토크나이즈해서, prompt 부분 길이를 구함
        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
        )["input_ids"]

        encoded = self.tokenizer(
            full_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = encoded["input_ids"]
        attn_mask = encoded["attention_mask"]

        # labels = input_ids 복사 후, prompt 구간을 -100으로 마스킹
        labels = input_ids.copy()
        # special tokens 때문에 실제 prompt 길이가 약간 다를 수 있음 → 최소/최대 보호
        prompt_len = min(len(prompt_ids), len(labels))
        for i in range(prompt_len):
            labels[i] = -100

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# =========================
# 2. 인자 파서
# =========================

@dataclass
class ScriptArguments:
    model_name_or_path: str
    train_jsonl: str
    output_dir: str
    max_length: int = 1024
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    logging_steps: int = 50
    save_steps: int = 500
    seed: int = 42


def parse_args() -> ScriptArguments:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="기반 LLM 모델 (예: meta-llama/Meta-Llama-3-8B-Instruct)",
    )
    parser.add_argument(
        "--train_jsonl",
        type=str,
        required=True,
        help="SFT 학습용 JSONL 파일 경로",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="LoRA가 적용된 모델을 저장할 디렉터리",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )


    args = parser.parse_args()
    return ScriptArguments(**vars(args))

#jsonl path resolver
def resolve_jsonl_path(path: str):
    if path.startswith("s3://"):
        # s3://bucket/key 형태 파싱
        no_scheme = path[5:]  # "s3://"
        bucket, key = no_scheme.split("/", 1)

        local_path = "/tmp/train.jsonl"
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        s3 = boto3.client("s3")
        s3.download_file(bucket, key, local_path)
        print(f"[INFO] downloaded {path} -> {local_path}")
        return local_path
    return path


# =========================
# 3. 메인 로직
# =========================

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1) 토크나이저 / 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=True,
    )
    # Llama 계열은 패딩 토큰이 없을 수 있으므로 bos/eos 재사용
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # 2) LoRA 설정 및 적용
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # target_modules는 모델 구조에 따라 조정 필요 (예시는 Llama 스타일)
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3) Dataset 준비
    jsonl_path = resolve_jsonl_path(args.train_jsonl)
    train_dataset = JsonlSftDataset(
        jsonl_path=args.train_jsonl,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    # 4) Data collator (LM용)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # 5) TrainingArguments 설정
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        fp16=True,
        bf16=False,
        seed=args.seed,
        report_to="none",  # 필요 시 wandb 등으로 변경
        dataloader_num_workers=4,
    )

    # 6) Trainer 생성 및 학습
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    # 7) LoRA 가중치 저장
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("LoRA fine-tuning 완료. 저장 위치:", args.output_dir)


if __name__ == "__main__":
    main()
