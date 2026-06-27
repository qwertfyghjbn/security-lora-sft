"""
Baseline evaluation: Qwen2.5-1.5B-Instruct with NO LoRA adapter.

Runs the same prompt format and metrics as inference.py so the numbers are
directly comparable after fine-tuning.

Usage:
  python eval_baseline.py
  python eval_baseline.py --val-path data/val.jsonl   # default
  python eval_baseline.py --request "GET /user?id=1' OR 1=1-- HTTP/1.1\nHost: x"
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from unsloth import FastLanguageModel

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME     = "unsloth/Qwen2.5-1.5B-Instruct"
VAL_PATH       = Path("data/val.jsonl")
MAX_SEQ_LEN    = 512
MAX_NEW_TOKENS = 256

PROMPT_PREFIX = (
    "你是一个安全告警分类助手。请判断下面 HTTP 请求或解码后的请求日志的攻击类型，并严格输出 JSON。\n\n"
    "输出字段:\n"
    "- attack_type: normal, sql_injection, xss, path_traversal, ssti, command_injection\n"
    "- risk: low, medium, high\n"
    "- reason: 简短说明判断依据\n\n"
    "只输出 JSON，不要输出其他内容。\n\n"
    "HTTP 请求或日志:\n"
)

VALID_ATTACK_TYPES = frozenset({
    "normal", "sql_injection", "xss",
    "path_traversal", "ssti", "command_injection",
})
VALID_RISKS = frozenset({"low", "medium", "high"})

CATEGORY_ORDER = [
    "normal", "sql_injection", "xss",
    "path_traversal", "ssti", "command_injection",
]


# ── Model ─────────────────────────────────────────────────────────────────────

def load_base_model():
    """Load the base model without any adapter."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


# ── Inference ─────────────────────────────────────────────────────────────────

def classify(http_request: str, model, tokenizer) -> dict:
    messages = [{"role": "user", "content": PROMPT_PREFIX + http_request}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.1,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = output_ids[0][input_ids.shape[-1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return _parse_and_validate(raw)


def _try_parse_json(text: str) -> dict | None:
    text = text.strip()
    # Base model often wraps output in ```json ... ``` — strip that first
    m_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m_block:
        text = m_block.group(1).strip()
    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Greedy outer-object match — non-greedy fails when reason contains { or }
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


def _parse_and_validate(raw: str) -> dict:
    result = _try_parse_json(raw)
    if result is None:
        return {
            "attack_type": "normal",
            "risk": "low",
            "reason": "输出解析失败",
            "_parse_error": True,
            "_raw": raw,
        }
    if result.get("attack_type") not in VALID_ATTACK_TYPES:
        result["attack_type"] = "normal"
        result.setdefault("_warnings", []).append("invalid attack_type → defaulted to normal")
    if result.get("risk") not in VALID_RISKS:
        result["risk"] = "low"
        result.setdefault("_warnings", []).append("invalid risk → defaulted to low")
    return result


def _extract_request(prompt: str) -> str:
    """Pull the raw HTTP request out of a prompt string."""
    marker = "HTTP 请求或日志:\n"
    idx = prompt.find(marker)
    request = prompt[idx + len(marker):]
    if request.endswith("\n\n输出:"):
        request = request[: -len("\n\n输出:")]
    return request


# ── Evaluation ────────────────────────────────────────────────────────────────

def _build_confusion_row(per_class: dict, pred_label: str, true_label: str) -> None:
    per_class[true_label]["confusion"][pred_label] += 1


def evaluate(val_path: Path, model, tokenizer) -> None:
    records = [
        json.loads(l)
        for l in val_path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]

    correct_type = correct_risk = parse_errors = 0
    per_class: dict[str, dict] = {
        c: {"total": 0, "correct": 0, "confusion": defaultdict(int)}
        for c in VALID_ATTACK_TYPES
    }

    for rec in records:
        request  = _extract_request(rec["prompt"])
        expected = json.loads(rec["completion"])
        pred     = classify(request, model, tokenizer)

        exp_type = expected.get("attack_type", "normal")
        pred_type = pred.get("attack_type", "normal")

        per_class[exp_type]["total"] += 1
        per_class[exp_type]["confusion"][pred_type] += 1

        if pred.get("_parse_error"):
            parse_errors += 1
        if pred_type == exp_type:
            correct_type += 1
            per_class[exp_type]["correct"] += 1
        if pred.get("risk") == expected.get("risk"):
            correct_risk += 1

    n = len(records)
    w = 45

    print(f"\n{'─' * w}")
    print(f"  Baseline: {MODEL_NAME}")
    print(f"  Val set : {val_path}  ({n} samples)")
    print(f"{'─' * w}")
    print(f"  attack_type accuracy : {correct_type:3d}/{n}  ({correct_type/n:.1%})")
    print(f"  risk accuracy        : {correct_risk:3d}/{n}  ({correct_risk/n:.1%})")
    print(f"  JSON parse errors    : {parse_errors:3d}/{n}")

    print(f"\n  Per-class attack_type accuracy:")
    for cls in CATEGORY_ORDER:
        stats = per_class[cls]
        if stats["total"] == 0:
            continue
        acc = stats["correct"] / stats["total"]
        bar = "█" * stats["correct"] + "░" * (stats["total"] - stats["correct"])
        print(f"    {cls:22s}  {bar}  {stats['correct']}/{stats['total']}  ({acc:.0%})")

    # Confusion matrix (only rows/cols that actually appear)
    active = [c for c in CATEGORY_ORDER if per_class[c]["total"] > 0]
    print(f"\n  Confusion matrix (rows=true, cols=predicted):")
    header = f"    {'':22s}" + "".join(f"{c[:6]:>8}" for c in active)
    print(header)
    for true_cls in active:
        row = f"    {true_cls:22s}"
        for pred_cls in active:
            count = per_class[true_cls]["confusion"][pred_cls]
            row += f"{count:>8}" if count == 0 else f"\033[1m{count:>8}\033[0m"
        print(row)

    print(f"{'─' * w}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Baseline model evaluation")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--val-path", type=Path, default=VAL_PATH,
        help="Path to validation JSONL (default: data/val.jsonl)",
    )
    group.add_argument(
        "--request", type=str,
        help="Single HTTP request string to classify",
    )
    args = parser.parse_args()

    model, tokenizer = load_base_model()

    if args.request:
        result = classify(args.request, model, tokenizer)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        evaluate(args.val_path, model, tokenizer)


if __name__ == "__main__":
    main()
