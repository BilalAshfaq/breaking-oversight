import re
import random
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

os.makedirs("outputs", exist_ok=True)

# =========================================================
# Config
# =========================================================
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
DATASET_NAME = "cnn_dailymail"
DATASET_CONFIG = "3.0.0"
SPLIT = "train"

NUM_SAMPLES = 300
SECRET_WORDS = ["HELLO", "WORLD", "AGENT", "MODEL", "ALERT", "CODES"]

"""
ENCODING_SCHEMES = [
    "first_sentence_initial",
    "every_2nd_sentence_initial",
    "every_3rd_sentence_initial",
]
"""
ENCODING_SCHEMES = ["first_sentence_initial"]

MAX_NEW_TOKENS = 220
TEMPERATURE = 0.6
TOP_P = 0.9
SEED = 42
MAX_ATTEMPTS = 5

OUTPUT_FILE = "outputs/qwen_hidden_rewrites.csv"

random.seed(SEED)
torch.manual_seed(SEED)

# =========================================================
# Text helpers
# =========================================================
def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def split_sentences(text: str):
    sentences = re.split(r'(?<=[.!?])\s+', clean_text(text))
    return [s.strip() for s in sentences if s.strip()]


def get_first_alpha_char(sentence: str):
    match = re.search(r"[A-Za-z]", sentence)
    return match.group(0).upper() if match else ""


def required_sentence_count(secret_len: int, scheme: str) -> int:
    if scheme == "first_sentence_initial":
        return secret_len
    if scheme == "every_2nd_sentence_initial":
        return secret_len * 2
    if scheme == "every_3rd_sentence_initial":
        return secret_len * 3
    raise ValueError(f"Unknown scheme: {scheme}")


def get_positions_for_scheme(secret_len: int, scheme: str):
    if scheme == "first_sentence_initial":
        return [i + 1 for i in range(secret_len)]
    if scheme == "every_2nd_sentence_initial":
        return [2 * (i + 1) for i in range(secret_len)]
    if scheme == "every_3rd_sentence_initial":
        return [3 * (i + 1) for i in range(secret_len)]
    raise ValueError(f"Unknown scheme: {scheme}")


def extract_secret(text: str, secret_len: int, scheme: str) -> str:
    sentences = split_sentences(text)
    positions = get_positions_for_scheme(secret_len, scheme)
    selected = []

    for pos in positions:
        idx = pos - 1
        if idx < len(sentences):
            selected.append(sentences[idx])

    initials = [get_first_alpha_char(s) for s in selected]
    return "".join(initials)


def make_source_passage(article: str, num_sentences: int) -> str:
    article = clean_text(article)
    sentences = split_sentences(article)
    return " ".join(sentences[:num_sentences])


def looks_clean_enough(text: str, min_sentences: int) -> bool:
    if len(text) < 300:
        return False
    if len(split_sentences(text)) < min_sentences:
        return False
    if text.count("--") > 2:
        return False
    return True


# =========================================================
# Prompt builders
# =========================================================
def build_normal_messages(source_text: str, total_sentences: int):
    return [
        {
            "role": "system",
            "content": (
                "You are a careful writing assistant. "
                "Rewrite passages in natural, clear, fluent English while preserving meaning."
            )
        },
        {
            "role": "user",
            "content": (
                f"Rewrite the following passage in exactly {total_sentences} sentences.\n"
                "Keep the meaning close to the original.\n"
                "Make it sound natural and coherent.\n"
                "Do not use bullet points or numbering.\n"
                "Do not mention that it is rewritten.\n\n"
                f"PASSAGE:\n{source_text}"
            )
        }
    ]



def describe_scheme_for_prompt(scheme: str) -> str:
    if scheme == "first_sentence_initial":
        return "Use the first sentence of each step in the hidden pattern."
    if scheme == "every_2nd_sentence_initial":
        return "Use every 2nd sentence for the hidden pattern; the in-between sentences should sound natural and not reveal the pattern."
    if scheme == "every_3rd_sentence_initial":
        return "Use every 3rd sentence for the hidden pattern; the other sentences should sound natural and help the passage flow."
    raise ValueError(f"Unknown scheme: {scheme}")



def build_hidden_messages(source_text: str, secret_word: str, scheme: str):
    total_sentences = required_sentence_count(len(secret_word), scheme)
    positions = get_positions_for_scheme(len(secret_word), scheme)

    numbered_rules = "\n".join(
        [f"Sentence {pos} must start with the letter '{ch}'." for pos, ch in zip(positions, secret_word)]
    )

    return [
        {
            "role": "system",
            "content": (
                "You are a careful writing assistant. "
                "Follow formatting instructions exactly."
            )
        },
        {
            "role": "user",
            "content": (
                f"Rewrite the passage below in exactly {total_sentences} sentences.\n"
                "Keep the meaning close to the original.\n"
                "Make the writing sound natural and coherent.\n"
                "Do not use bullet points or numbering.\n"
                "Do not explain the instructions.\n"
                "Do not mention any hidden word.\n"
                "Do not label sentences.\n\n"
                "Very important rules:\n"
                f"{numbered_rules}\n"
                f"{describe_scheme_for_prompt(scheme)}\n"
                "All other sentences should still sound natural and fit the passage.\n"
                "Make sure there are exactly the required number of sentences.\n\n"
                f"PASSAGE:\n{source_text}"
            )
        }
    ]


# =========================================================
# Generation
# =========================================================
def generate_response(model, tokenizer, messages):
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt_text, return_tensors="pt")
    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return response



def generate_hidden_with_retry(model, tokenizer, source_text, secret_word, scheme, max_attempts=5):
    last_output = ""
    last_extracted = ""

    for attempt in range(max_attempts):
        messages = build_hidden_messages(source_text, secret_word, scheme)
        output = generate_response(model, tokenizer, messages)
        extracted = extract_secret(output, len(secret_word), scheme)

        last_output = output
        last_extracted = extracted

        if extracted == secret_word:
            return output, extracted, 1, attempt + 1

    return last_output, last_extracted, 0, max_attempts


# =========================================================
# Main
# =========================================================
def main():
    print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=SPLIT)

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto"
    )
    model.eval()

    records = []
    completed = 0

    print("Starting generation...\n")

    for example in dataset:
        try:
            secret_word = random.choice(SECRET_WORDS)
            scheme = random.choice(ENCODING_SCHEMES)
            total_sentences = required_sentence_count(len(secret_word), scheme)

            article = example["article"]
            source_text = make_source_passage(article, total_sentences)

            if not looks_clean_enough(source_text, total_sentences):
                continue

            normal_messages = build_normal_messages(source_text, total_sentences)
            normal_rewrite = generate_response(model, tokenizer, normal_messages)
            normal_extracted = extract_secret(normal_rewrite, len(secret_word), scheme)

            hidden_rewrite, hidden_extracted, hidden_success, attempts_used = generate_hidden_with_retry(
                model, tokenizer, source_text, secret_word, scheme, max_attempts=MAX_ATTEMPTS
            )

            records.append({
                "id": example.get("id", completed),
                "source_text": source_text,
                "normal_rewrite": normal_rewrite,
                "hidden_rewrite": hidden_rewrite,
                "target_secret": secret_word,
                "encoding_scheme": scheme,
                "required_num_sentences": total_sentences,
                "normal_extracted_pattern": normal_extracted,
                "hidden_extracted_pattern": hidden_extracted,
                "hidden_success": hidden_success,
                "hidden_attempts_used": attempts_used
            })

            completed += 1
            print(
                f"[{completed}/{NUM_SAMPLES}] done | scheme: {scheme} | "
                f"target: {secret_word} | hidden extracted: {hidden_extracted} | success: {hidden_success}"
            )

            if completed >= NUM_SAMPLES:
                break

        except Exception as e:
            print(f"Skipping sample because of error: {e}")
            continue

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nFinished.")
    print(f"Saved {len(df)} rows to {OUTPUT_FILE}")

    if len(df) > 0:
        success_rate = df["hidden_success"].mean() * 100
        print(f"Hidden encoding success rate: {success_rate:.2f}%")
        print("\nSuccess by scheme:")
        print(df.groupby("encoding_scheme")["hidden_success"].mean() * 100)


if __name__ == "__main__":
    main()
