# Lab 21 — Evaluation Report

**Học viên**: Nguyễn Khôi Lâm — 2A202600607  
**Ngày nộp**: 2026-06-25  
**Submission option**: C (code-only)

## 1. Setup

- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit`
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`, 200 samples before cleaning and split; recorded run used 180 train + 20 eval
- **Format**: Alpaca instruction/input/response
- **max_seq_length**: 1024 (p95 = 562, rounded up to the next power of two)
- **GPU**: Tesla T4, 14.563 GB usable VRAM
- **Training**: 3 epochs, effective batch size 8, cosine schedule, learning rate `2e-4`, 10% warmup, `adamw_8bit`
- **LoRA targets**: `q_proj`, `v_proj`; alpha/r = 2 for all ranks
- **Training cost**: approximately **$0.07** for 12.2 minutes at an estimated $0.35/hour
- **Environment recorded by Colab**: Unsloth 2026.5.2, Transformers 5.5.0, TRL 0.15.2, PyTorch 2.10.0+cu128

The submission notebook adds the rubric-required cleaning pass: exact deduplication, placeholder filtering, and removal of responses shorter than 10 tokenizer tokens. Because this cleaning was added after the recorded GPU run, rerunning from the dataset cell may change the retained sample count and metrics slightly.

## 2. Rank Experiment Results

| Rank | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|---:|---:|---:|---:|---:|---:|
| 8 | 1,843,200 | 4.00 min | 7.22 GB | 1.5577 | 4.7479 |
| 16 | 3,686,400 | 4.26 min | 6.62 GB | 1.5161 | 4.5544 |
| 64 | 14,745,600 | 3.99 min | 8.00 GB | 1.4768 | 4.3790 |
| Base | 0 | — | — | Not captured in the original run | Not captured in the original run |

All adapter values above come directly from the completed Colab outputs. I do not invent a base-model number: the original run omitted it. The corrected stripped notebook now computes base loss and perplexity on the same eval split; that cell must be rerun on a GPU to populate the fourth number.

Rank 64 reduced perplexity by 0.369 (7.8%) relative to rank 8, but required eight times as many trainable parameters. Rank 16 achieved most of that improvement with one quarter of rank 64's adapter parameters. Peak VRAM did not increase monotonically between ranks 8 and 16 because allocator state and model reload timing affect this measurement; rank 64 still produced the highest peak.

## 3. Loss Curve Analysis

![Recorded r=16 training loss](results/loss_curve.png)

The recorded T4 run disabled evaluation during training to avoid OOM, so the plot contains training loss only. It therefore cannot prove or disprove overfitting from train/eval divergence. The held-out r=16 loss of 1.5161 confirms reasonable generalization at the end of training, but a rigorous overfitting claim would require eval loss at multiple checkpoints. The notebook saves each epoch, making that follow-up possible.

## 4. Qualitative Comparison

The original notebook captured five comparisons. The corrected notebook uses two independent model objects and deterministic decoding; this avoids accidentally treating an adapter-wrapped model as the base model.

| Prompt | Base | Fine-tuned r=16 | Assessment |
|---|---|---|---|
| Explain machine learning to a beginner | Broadly correct but wordier | More direct beginner-oriented definition | Small improvement |
| Write Python for Fibonacci(n) | Gives recursive/iterative framing | Adds explicit invalid-input handling | Improvement, pending code execution |
| List five UI/UX principles | Starts from user friendliness | More compact list emphasizing conversion/adaptation | Mixed; base is more user-centered |
| Summarize LoRA vs QLoRA | Correctly expands Low-Rank Adaptation | Incorrectly expands LoRA as “Layer-wise Adaptive Regularization Optimization” | Clear regression/factual hallucination |
| Distinguish prompting, RAG, and fine-tuning | Broad high-level distinction | Similar high-level distinction | Roughly unchanged in captured excerpt |

The qualitative results are mixed. Fine-tuning improved concision and formatting in some examples but introduced a serious factual error on LoRA. This is consistent with the dataset being general Vietnamese Alpaca data rather than a curated fine-tuning-domain dataset. The example is intentionally retained as a loss case rather than cherry-picked away.

## 5. Conclusion về Rank Trade-off

Trên thí nghiệm này, rank 16 có ROI phù hợp nhất cho triển khai thực tế. So với rank 8, rank 16 tăng số tham số trainable từ 1.84 triệu lên 3.69 triệu nhưng giảm perplexity từ 4.7479 xuống 4.5544. Rank 64 tiếp tục giảm perplexity xuống 4.3790, tuy nhiên phải dùng 14.75 triệu tham số trainable, tức gấp bốn lần rank 16, trong khi mức cải thiện perplexity chỉ khoảng 3.9%. Đây là dấu hiệu diminishing returns: tăng capacity của hai ma trận LoRA vẫn giúp tối ưu loss, nhưng lợi ích biên giảm nhanh. Thời gian train gần như ngang nhau trong run ngắn này nên chưa phản ánh đầy đủ chi phí ở quy mô lớn; adapter size, VRAM và serving cost mới là khác biệt đáng chú ý. Nếu deploy production, tôi chọn rank 16 và ưu tiên cải thiện chất lượng dataset, đặc biệt là factuality, trước khi tăng rank. Kết quả LoRA/QLoRA sai trong qualitative test cho thấy perplexity thấp hơn không đồng nghĩa câu trả lời luôn đúng.

## 6. What I Learned

- Perplexity must be paired with qualitative factuality checks; rank 64 scored best quantitatively, but even r=16 produced a confident domain-specific error.
- Rank controls adapter capacity, not knowledge freshness. Better curation and domain alignment can matter more than increasing trainable parameters.
- A fair before/after comparison requires separate base and adapter-wrapped model instances plus identical deterministic decoding settings.
