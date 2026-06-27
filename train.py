"""
4-bit QLoRA fine-tuning of Qwen2.5-1.5B-Instruct for HTTP security classification.
Framework : Unsloth + TRL SFTTrainer
Loss      : full-sequence (prompt + completion); TRL 0.24 removed train_on_responses_only —
            completion_only_loss=True is only supported for prompt/completion-field datasets,
            not the ChatML text-field format we use here.
"""
import json
from pathlib import Path

from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data")
OUTPUT_DIR = Path("output")

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_NAME  = "unsloth/Qwen2.5-1.5B-Instruct"
MAX_SEQ_LEN = 512

# ── LoRA ──────────────────────────────────────────────────────────────────────
LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE   = 2
GRAD_ACCUM   = 4      # effective batch = 8
EPOCHS       = 5
LR           = 2e-4
WARMUP_RATIO = 0.1
SEED         = 42


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def make_dataset(records: list[dict], tokenizer) -> Dataset:
    """
    Convert each record to a ChatML-formatted string.
    The prompt (minus the trailing '\n\n输出:') becomes the user turn;
    the completion JSON becomes the assistant turn.
    """
    suffix = "\n\n输出:"

    def fmt(rec: dict) -> dict:
        user_content = rec["prompt"]
        if user_content.endswith(suffix):
            user_content = user_content[: -len(suffix)]
        messages = [
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": rec["completion"]},
        ]
        return {"text": tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )}

    return Dataset.from_list([fmt(r) for r in records])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load base model in 4-bit NF4
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,        # auto-detect: fp16 on RTX 4060
        load_in_4bit=True,
    )

    # Attach LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=LORA_TARGETS,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    # Build datasets
    train_ds = make_dataset(load_jsonl(DATA_DIR / "train.jsonl"), tokenizer)
    val_ds   = make_dataset(load_jsonl(DATA_DIR / "val.jsonl"),   tokenizer)

    print(f"Train: {len(train_ds)} samples  |  Val: {len(val_ds)} samples")

    # Configure SFTTrainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=SFTConfig(
            output_dir=str(OUTPUT_DIR),
            # Batching
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            # Schedule
            num_train_epochs=EPOCHS,
            learning_rate=LR,
            lr_scheduler_type="cosine",
            warmup_ratio=WARMUP_RATIO,
            # Precision (Unsloth detected bf16 on this GPU/stack)
            fp16=False,
            bf16=True,
            # Logging & checkpointing
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            # Misc
            seed=SEED,
            dataset_text_field="text",
            max_length=MAX_SEQ_LEN,
            report_to="none",
        ),
    )

    # Train
    trainer.train()

    # Save LoRA adapter
    adapter_dir = OUTPUT_DIR / "lora_adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"\nLoRA adapter saved → {adapter_dir}")


if __name__ == "__main__":
    main()
