# Lab 21 — Evaluation Report

**Học viên**: Nguyễn Khôi Lâm — 2A202600607  
**Ngày nộp**: 2026-06-25  
**Submission option**: C (code-only)

## 1. Setup

- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit`
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`, 200 samples initially; 193 retained after deduplication, placeholder filtering, and removal of outputs shorter than 10 tokens
- **Split**: 173 train + 20 eval (90/10, seed 42)
- **Format**: Alpaca instruction/input/response
- **max_seq_length**: 1024 (p95 = 562, rounded up to the next power of two)
- **GPU**: Tesla T4, 15.6 GB reported VRAM (14.563 GB usable by Unsloth)
- **Training**: 3 epochs, effective batch size 8, cosine schedule, learning rate `2e-4`, 10% warmup, `adamw_8bit`
- **LoRA targets**: `q_proj`, `v_proj`; alpha/r = 2 for all ranks
- **Training cost**: approximately **$0.07** for 11.6 minutes at an estimated $0.35/hour
- **Environment**: Unsloth 2026.6.9, Transformers 5.5.0, TRL 0.15.2, PyTorch 2.11.0+cu128, CUDA 12.8

The cleaning stage removed seven unsuitable samples before the deterministic train/eval split. All metrics below come from the completed rerun of the corrected notebook.

## 2. Rank Experiment Results

| Rank | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|---:|---:|---:|---:|---:|---:|
| 8 | 1,843,200 | 3.77 min | 7.22 GB | 1.5402 | 4.6656 |
| 16 | 3,686,400 | 4.08 min | 6.62 GB | 1.4932 | 4.4515 |
| 64 | 14,745,600 | 3.77 min | 8.00 GB | 1.4321 | 4.1875 |
| Base | 0 | — | — | 1.9403 | 6.9610 |

All four perplexity values were computed on the same 20-example held-out split. Relative to the base model, perplexity decreased by 33.0% for rank 8, 36.1% for rank 16, and 39.8% for rank 64.

Rank 64 reduced perplexity by 0.4781 (10.2%) relative to rank 8, but required eight times as many trainable parameters. Rank 16 achieved a substantial improvement over rank 8 with one quarter of rank 64's adapter parameters. Peak VRAM did not increase monotonically between ranks 8 and 16 because allocator state and model reload timing affect this measurement; rank 64 still produced the highest peak.

## 3. Loss Curve Analysis

![Recorded r=16 training loss](results/loss_curve.png)

Evaluation during training was disabled on the T4 to reduce OOM risk, so the plot contains training loss only. It therefore cannot prove or disprove overfitting from train/eval divergence. The final held-out r=16 loss of 1.4932 indicates reasonable generalization, and all three adapters substantially outperformed the base loss of 1.9403. A rigorous overfitting claim would still require evaluation at multiple checkpoints.

## 4. Qualitative Comparison

The notebook captured five comparisons using independent base and adapter-wrapped model objects with deterministic decoding.

| Prompt | Base | Fine-tuned r=16 | Assessment |
|---|---|---|---|
| Explain machine learning to a beginner | Correct and concise explanation of learning from data | Understandable, but uses awkward Vietnamese and an unnecessary AI comparison | Base is clearer |
| Write Python for Fibonacci(n) | Provides a complete recursive implementation | Describes an iterative implementation | Fine-tuned approach is more efficient conceptually; code should still be executed |
| List five UI/UX principles | Starts well but mixes English and includes a vague “Dimensional Design” item | Uses Vietnamese consistently and emphasizes navigation and user experience | Fine-tuned output is better localized and more practical |
| Summarize LoRA vs QLoRA | Correctly expands both terms and mentions low-rank adaptation | Correctly expands both terms but describes them too generally as model-size reduction | Mixed; base is more technically precise |
| Distinguish prompting, RAG, and fine-tuning | Gives a broad distinction but incorrectly refers to NLU | Incorrectly expands RAG as “Reinforcement Augmented Generation” | Fine-tuned output regresses on factuality |

The qualitative results are mixed. Fine-tuning improved Vietnamese localization, structure, and practical presentation in some examples, especially UI/UX and Fibonacci. However, it introduced a serious factual error by expanding RAG as “Reinforcement Augmented Generation” instead of “Retrieval-Augmented Generation.” This is consistent with using a general Vietnamese Alpaca dataset rather than a curated technical dataset. The failed example is intentionally retained rather than cherry-picked away.

## 5. Conclusion về Rank Trade-off

Trên thí nghiệm này, rank 16 có ROI phù hợp nhất cho triển khai thực tế. So với rank 8, rank 16 tăng số tham số trainable từ 1.84 triệu lên 3.69 triệu nhưng giảm perplexity từ 4.6656 xuống 4.4515. Rank 64 tiếp tục giảm perplexity xuống 4.1875, tuy nhiên phải dùng 14.75 triệu tham số trainable, tức gấp bốn lần rank 16, trong khi mức cải thiện so với rank 16 chỉ khoảng 5.9%. Đây là dấu hiệu diminishing returns: tăng capacity của hai ma trận LoRA vẫn giúp tối ưu loss, nhưng lợi ích biên giảm so với chi phí adapter và VRAM. Cả ba adapter đều cải thiện rõ rệt so với perplexity 6.9610 của base model. Thời gian train gần như ngang nhau trong run ngắn này nên chưa phản ánh đầy đủ chi phí ở quy mô lớn; adapter size và serving cost mới là khác biệt đáng chú ý. Nếu deploy production, tôi chọn rank 16 và ưu tiên cải thiện chất lượng dataset, đặc biệt là factuality, trước khi tăng rank. Lỗi mở rộng sai thuật ngữ RAG cho thấy perplexity thấp hơn không đồng nghĩa câu trả lời luôn đúng.

## 6. What I Learned

- Perplexity must be paired with qualitative factuality checks; rank 64 scored best quantitatively, but the r=16 adapter still expanded RAG incorrectly.
- Rank controls adapter capacity, not knowledge freshness. Better curation and domain alignment can matter more than increasing trainable parameters.
- A fair before/after comparison requires separate base and adapter-wrapped model instances plus identical deterministic decoding settings.
