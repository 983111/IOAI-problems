# Qwen3-8B Fine-tuning on IOAI Problems

Fine-tuning Qwen3-8B Base on 145 AI Olympiad multiple-choice questions using Unsloth and QLoRA.

---

## Results

| Metric | Value |
|---|---|
| Model | Qwen3-8B Base |
| Fine-tuning method | QLoRA (LoRA rank 16) |
| Trainable parameters | 43,646,976 of 8,234,382,336 (0.53%) |
| Training examples | 145 |
| Test examples | 24 (held out, never seen during training) |
| Epochs | 3 |
| Training time | 258 seconds (~4 minutes) on T4 GPU |
| Peak VRAM | 9.658 GB |
| Final training loss | 1.4767 |
| Test accuracy | 95.8% (23/24 correct) |

### Training loss curve

| Step | Loss |
|---|---|
| 10 | 2.153 |
| 20 | 1.611 |
| 30 | 1.438 |
| 40 | 1.279 |
| 50 | 1.192 |

### Test results

| No. | ID | Truth | Predicted | Result |
|---|---|---|---|---|
| 1 | ioai_rl_001 | B | B | CORRECT |
| 2 | ioai_rl_002 | B | B | CORRECT |
| 3 | ioai_rl_003 | C | C | CORRECT |
| 4 | ioai_rl_004 | B | B | CORRECT |
| 5 | ioai_dl_015 | B | B | CORRECT |
| 6 | ioai_dl_016 | C | C | CORRECT |
| 7 | ioai_ml_020 | B | B | CORRECT |
| 8 | ioai_ml_021 | B | B | CORRECT |
| 9 | ioai_ml_022 | D | D | CORRECT |
| 10 | ioai_ml_023 | B | B | CORRECT |
| 11 | ioai_dl_017 | B | B | CORRECT |
| 12 | ioai_dl_018 | B | B | CORRECT |
| 13 | ioai_nlp_001 | B | B | CORRECT |
| 14 | ioai_nlp_002 | A | A | CORRECT |
| 15 | ioai_nlp_003 | A | A | CORRECT |
| 16 | ioai_ml_024 | B | B | CORRECT |
| 17 | ioai_ml_025 | B | B | CORRECT |
| 18 | ioai_dl_019 | B | B | CORRECT |
| 19 | ioai_ml_026 | B | B | CORRECT |
| 20 | ioai_dl_020 | B | B | CORRECT |
| 21 | ioai_ml_027 | B | B | CORRECT |
| 22 | ioai_ml_028 | B | B | CORRECT |
| 23 | ioai_ml_029 | D | B | WRONG |
| 24 | ioai_ml_030 | B | B | CORRECT |

### Wrong answer analysis (fine-tuned model)

**ioai_ml_029** — Which normalization technique is most appropriate when data has outliers and you want to bound it to a fixed range like [0, 1]?

- Expected: D (Robust scaling using median and IQR)
- Predicted: B (Min-Max scaling)
- The model correctly explained that Min-Max is sensitive to outliers and that Robust scaling is outlier-resistant, but then predicted B anyway. The question asks for bounding to [0, 1] which is a Min-Max property, and the model prioritized that over the outlier requirement. This is a genuine ambiguity in the question wording.

---

## Base model vs fine-tuned model comparison

Both models tested on the same 24 held-out questions.

| Metric | Base model | Fine-tuned model |
|---|---|---|
| Correct | 16/24 | 23/24 |
| Wrong | 8/24 | 1/24 |
| Accuracy | 66.7% | 95.8% |
| Improvement | — | +29.1 percentage points |

### Per-question comparison

| No. | ID | Truth | Base | Fine-tuned |
|---|---|---|---|---|
| 1 | ioai_rl_001 | B | UNKNOWN | B (CORRECT) |
| 2 | ioai_rl_002 | B | B | B (CORRECT) |
| 3 | ioai_rl_003 | C | C | C (CORRECT) |
| 4 | ioai_rl_004 | B | UNKNOWN | B (CORRECT) |
| 5 | ioai_dl_015 | B | B | B (CORRECT) |
| 6 | ioai_dl_016 | C | C | C (CORRECT) |
| 7 | ioai_ml_020 | B | UNKNOWN | B (CORRECT) |
| 8 | ioai_ml_021 | B | B | B (CORRECT) |
| 9 | ioai_ml_022 | D | D | D (CORRECT) |
| 10 | ioai_ml_023 | B | B | B (CORRECT) |
| 11 | ioai_dl_017 | B | B | B (CORRECT) |
| 12 | ioai_dl_018 | B | UNKNOWN | B (CORRECT) |
| 13 | ioai_nlp_001 | B | B | B (CORRECT) |
| 14 | ioai_nlp_002 | A | A | A (CORRECT) |
| 15 | ioai_nlp_003 | A | UNKNOWN | A (CORRECT) |
| 16 | ioai_ml_024 | B | UNKNOWN | B (CORRECT) |
| 17 | ioai_ml_025 | B | UNKNOWN | B (CORRECT) |
| 18 | ioai_dl_019 | B | B | B (CORRECT) |
| 19 | ioai_ml_026 | B | B | B (CORRECT) |
| 20 | ioai_dl_020 | B | B | B (CORRECT) |
| 21 | ioai_ml_027 | B | B | B (CORRECT) |
| 22 | ioai_ml_028 | B | B | B (CORRECT) |
| 23 | ioai_ml_029 | D | B | B (WRONG) |
| 24 | ioai_ml_030 | B | B | B (CORRECT) |

### Why the base model shows UNKNOWN for 7 questions

The base model answers in plain prose without stating "The correct answer is X". For example, for ioai_rl_001 it responded:

> "The term in square brackets represents the temporal difference (TD) error. In Q-learning, the TD error is the difference between..."

The reasoning is correct but the `extract_letter` function in the test notebook could not parse a letter from it. The fine-tuned model learned from training data to always start with "The correct answer is B." which makes extraction reliable.

This means the base model's true accuracy on reasoning is likely higher than 66.7% — several UNKNOWN responses contained the correct answer in the explanation. However for a production use case, structured output format matters, and the fine-tuned model handles this correctly on 23 out of 24 questions.

### Key takeaway

Fine-tuning on 145 examples for 3 epochs (258 seconds of training) improved structured answer accuracy from 66.7% to 95.8%. The improvement comes from two things: the model learned to format answers with a clear letter prefix, and it reinforced correct reasoning patterns for AI/ML topics.

---

## Project structure

```
.
├── README.md
├── train.jsonl          # 145 training examples in chat format
├── step1_train.ipynb    # Fine-tuning notebook (run this first)
└── step2_test.ipynb     # Evaluation notebook (run this after training)
```

---

## Dataset format

Every example in `train.jsonl` follows this format:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Question text with A/B/C/D options"
    },
    {
      "role": "assistant",
      "content": "The correct answer is B. Explanation text."
    }
  ]
}
```

Source: 169 problems from the IOAI problem set. First 145 used for training, last 24 held out for testing.

---

## Setup

### Requirements

- Google Colab with T4 GPU (free tier is sufficient)
- Google Drive (for saving adapter weights between sessions)
- The `train.jsonl` file

### Training (step1_train.ipynb)

1. Upload `step1_train.ipynb` to Google Colab
2. Set runtime: Runtime > Change runtime type > T4 GPU
3. Run all cells in order
4. Cell 3 will prompt you to upload `train.jsonl`
5. Cell 12 will save adapter weights to your Google Drive at `MyDrive/qwen3_ioai_lora`
6. Total time: roughly 20-30 minutes including model download

### Testing (step2_test.ipynb)

1. Open a new Colab tab and upload `step2_test.ipynb`
2. Set runtime to T4 GPU
3. Run all cells in order
4. Cell 2 loads the adapter from Google Drive
5. Cell 6 runs all 24 test questions and prints results
6. Cell 10 saves a CSV of results to Google Drive

---

## Configuration

| Parameter | Value |
|---|---|
| Base model | unsloth/Qwen3-8B-Base |
| Quantization | 4-bit (QLoRA) |
| LoRA rank | 16 |
| LoRA alpha | 16 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Batch size | 2 per device |
| Gradient accumulation | 4 steps (effective batch size = 8) |
| Learning rate | 2e-4 |
| LR scheduler | Linear |
| Optimizer | adamw_8bit |
| Epochs | 3 |
| Max sequence length | 2048 |
| Chat template | ChatML (manually set, Base model has none by default) |

---

## Known issues

**Tokenizer chat template error during dataset formatting**

Qwen3-8B Base does not have a chat template set in its tokenizer config. Add this before calling `dataset.map()`:

```python
QWEN_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
    "{% elif message['role'] == 'assistant' %}"
    "<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
tokenizer.chat_template = QWEN_CHAT_TEMPLATE
```

**CUDA out of memory**

Reduce `per_device_train_batch_size` to 1 and increase `gradient_accumulation_steps` to 8 in the TrainingArguments.

**Module not found after reconnecting**

Colab resets installed packages when a session ends. Always re-run the install cell at the top of each notebook in a new session.

**Adapter not found in Google Drive**

The folder is saved as `qwen3_ioai_lora` at the root of your Drive. Confirm Cell 12 in the training notebook completed without error.
