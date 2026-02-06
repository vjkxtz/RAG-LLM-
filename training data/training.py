import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainingArguments
from unsloth import SFTTrainer

# ------------------
# CONFIG
# ------------------
MODEL_NAME = "llama3.1-rag:latest"
DATA_PATH = "sample.json"
OUTPUT_DIR = "outputs"
MERGED_DIR = "llama3-unsloth-merged"

MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True

# ------------------
# LOAD MODEL
# ------------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=LOAD_IN_4BIT,
)

# ------------------
# APPLY LORA
# ------------------
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=True,
)

# ------------------
# LOAD DATASET
# ------------------
dataset = load_dataset("json", data_files=DATA_PATH)

def format_chat(example):
    text = ""
    for msg in example["messages"]:
        role = msg["role"]
        content = msg["content"]
        text += f"<|{role}|>\n{content}\n"
    return {"text": text}

dataset = dataset.map(format_chat, remove_columns=dataset["train"].column_names)

# ------------------
# TRAIN
# ------------------
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=500,
        learning_rate=2e-4,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        output_dir=OUTPUT_DIR,
        save_steps=100,
        optim="adamw_8bit",
        report_to="none",
    ),
)

trainer.train()

# ------------------
# MERGE LORA → BASE MODEL
# ------------------
model.save_pretrained_merged(
    MERGED_DIR,
    tokenizer,
    save_method="merged_16bit",
)

print("✅ Training complete")
print(f"✅ Merged model saved to: {MERGED_DIR}")
print("➡️ Next step: convert to GGUF using llama.cpp")

