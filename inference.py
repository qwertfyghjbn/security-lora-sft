"""
Inference and validation-set evaluation using the trained LoRA adapter.

Usage:
  # Evaluate on validation set
  python inference.py --eval

  # Classify a single request from stdin
  python inference.py --request "GET /user?id=1' OR '1'='1 HTTP/1.1\nHost: example.com"
"""
import argparse
import json
import re
from pathlib import Path

from unsloth import FastLanguageModel

# ── Paths ─────────────────────────────────────────────────────────────────────
ADAPTER_PATH = Path("output/lora_adapter")
VAL_PATH     = Path("data/val.jsonl")
MAX_SEQ_LEN  = 512
MAX_NEW_TOKENS = 256

# ── Prompt template (matches training format) ─────────────────────────────────
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


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(ADAPTER_PATH),
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



    return None


# ── Validation set evaluation ─────────────────────────────────────────────────

def evaluate(model, tokenizer) -> None:
    records = [json.loads(l) for l in VAL_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]

    suffix = "\n\n输出:"
    correct_type = correct_risk = parse_errors = 0
    per_class: dict[str, dict] = {}

    for rec in records:
        # Extract the raw HTTP request from the prompt
        prompt = rec["prompt"]
        idx = prompt.find("HTTP 请求或日志:\n")
        request_part = prompt[idx + len("HTTP 请求或日志:\n"):]
        if request_part.endswith(suffix):
            request_part = request_part[: -len(suffix)]

        expected = json.loads(rec["completion"])
        pred     = classify(request_part, model, tokenizer)

        exp_type = expected.get("attack_type", "")
        per_class.setdefault(exp_type, {"total": 0, "correct": 0})
        per_class[exp_type]["total"] += 1

        if pred.get("_parse_error"):
            parse_errors += 1

        if pred.get("attack_type") == exp_type:
            correct_type += 1
            per_class[exp_type]["correct"] += 1

        if pred.get("risk") == expected.get("risk"):
            correct_risk += 1

    n = len(records)
    print(f"\n{'─'*45}")
    print(f"Validation set  ({n} samples)")
    print(f"{'─'*45}")
    print(f"  attack_type accuracy : {correct_type}/{n}  ({correct_type/n:.1%})")
    print(f"  risk accuracy        : {correct_risk}/{n}  ({correct_risk/n:.1%})")
    print(f"  JSON parse errors    : {parse_errors}/{n}")
    print(f"\nPer-class attack_type accuracy:")
    for cls, stats in sorted(per_class.items()):
        acc = stats["correct"] / stats["total"]
        bar = "█" * stats["correct"] + "░" * (stats["total"] - stats["correct"])
        print(f"  {cls:22s}  {bar}  {stats['correct']}/{stats['total']}  ({acc:.0%})")
    print(f"{'─'*45}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Security classifier inference")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--eval",    action="store_true", help="Evaluate on data/val.jsonl")
    group.add_argument("--request", type=str,            help="Raw HTTP request string to classify")
    args = parser.parse_args()

    model, tokenizer = load_model()

    if args.eval:
        evaluate(model, tokenizer)
    else:
        result = classify(args.request, model, tokenizer)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
