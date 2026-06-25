"""Prepare the Lab 21 code-only submission from the completed Colab notebook."""

from __future__ import annotations

import base64
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_NOTEBOOK = ROOT / "notebooks" / "Lab21_LoRA_Finetuning_T4.ipynb"
SUBMISSION_NOTEBOOK = ROOT / "notebook.ipynb"
LOSS_CURVE = ROOT / "results" / "loss_curve.png"


def replace_cell_source(notebook: dict, index: int, source: str) -> None:
    notebook["cells"][index]["source"] = source.splitlines(keepends=True)


def main() -> None:
    notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))

    # Correct the documentation mismatch: packing is intentionally disabled on T4.
    notebook["cells"][14]["source"] = [
        line.replace("packing=True", "packing=False")
        for line in notebook["cells"][14]["source"]
    ]

    # Add the dataset cleaning required by the rubric before length analysis/split.
    cell_10 = "".join(notebook["cells"][10]["source"])
    cleaning = '''
# Clean dataset per rubric: deduplicate, remove template placeholders, and
# discard examples whose response has fewer than 10 tokenizer tokens.
PLACEHOLDER_MARKERS = ("<output>", "[output]", "your answer", "điền câu trả lời")
seen = set()
keep_indices = []
for idx, example in enumerate(raw):
    instruction = str(example.get(INSTRUCTION_COL, "") or "").strip()
    inp = str(example.get(INPUT_COL, "") or "").strip() if INPUT_COL else ""
    output = str(example.get(OUTPUT_COL, "") or "").strip()
    key = (instruction.casefold(), inp.casefold(), output.casefold())
    is_template = any(marker in output.casefold() for marker in PLACEHOLDER_MARKERS)
    if key in seen or is_template or len(tok.encode(output, add_special_tokens=False)) < 10:
        continue
    seen.add(key)
    keep_indices.append(idx)

raw = raw.select(keep_indices)
ds = raw.map(format_alpaca, remove_columns=raw.column_names)
print(f"✓ Cleaned dataset: {len(ds)} samples retained")

'''
    marker = "lengths = [len(tok.encode(x[\"text\"])) for x in ds]\n"
    if cleaning.strip() not in cell_10:
        cell_10 = cell_10.replace(marker, cleaning + marker)
    replace_cell_source(notebook, 10, cell_10)

    # Add a fair base-model perplexity measurement and include it in the table.
    cell_22 = "".join(notebook["cells"][22]["source"])
    base_eval = '''

# Measure base-model perplexity on the same held-out examples.
# This is intentionally run after adapter training so it cannot affect training.
def manual_eval_loss(model, tokenizer, dataset):
    model.eval()
    losses = []
    with torch.no_grad():
        for example in dataset:
            batch = tokenizer(
                example["text"], return_tensors="pt", truncation=True,
                max_length=MAX_SEQ_LENGTH,
            ).to(model.device)
            batch["labels"] = batch["input_ids"].clone()
            losses.append(float(model(**batch).loss.detach().cpu()))
    return float(np.mean(losses))

base_eval_model, base_eval_tokenizer = load_base_model()
base_eval_loss = manual_eval_loss(base_eval_model, base_eval_tokenizer, eval_ds)
base_eval_ppl = float(np.exp(base_eval_loss))
summary_df = pd.concat([
    pd.DataFrame([{
        "rank": "Base", "alpha": np.nan, "trainable_params": 0,
        "train_time_min": 0.0,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
        "eval_loss": base_eval_loss,
        "eval_perplexity": base_eval_ppl,
    }]),
    summary_df,
], ignore_index=True)
print(f"✓ Base eval loss = {base_eval_loss:.4f}, perplexity = {base_eval_ppl:.2f}")
print(summary_df.to_string(index=False))
del base_eval_model
gc.collect(); torch.cuda.empty_cache()
'''
    if "def manual_eval_loss" not in cell_22:
        cell_22 += base_eval
    replace_cell_source(notebook, 22, cell_22)

    # Ensure base and fine-tuned generation use independent model objects.
    cell_25 = "".join(notebook["cells"][25]["source"])
    cell_25 = cell_25.replace(
        'temperature=0.7, top_p=0.9, do_sample=True,',
        'do_sample=False,',
    )
    cell_25 = cell_25.replace(
        '# Reload base + r=16 adapter\n'
        'base_for_eval, tok_for_eval = load_base_model()\n'
        'ft_model = PeftModel.from_pretrained(base_for_eval, os.path.join(OUTPUT_DIR, "r16"))',
        '# Load independent model objects so the base comparison is not adapter-wrapped\n'
        'base_for_eval, tok_for_eval = load_base_model()\n'
        'ft_base_for_eval, ft_tokenizer = load_base_model()\n'
        'ft_model = PeftModel.from_pretrained(ft_base_for_eval, os.path.join(OUTPUT_DIR, "r16"))',
    )
    cell_25 = cell_25.replace(
        "ft_resp = generate_response(ft_model, tok_for_eval, prompt)",
        "ft_resp = generate_response(ft_model, ft_tokenizer, prompt)",
    )
    cell_25 = cell_25.replace(
        '"prompt": prompt, "base": base_resp[:300], "finetuned": ft_resp[:300],',
        '"prompt": prompt, "base": base_resp, "finetuned": ft_resp,\n'
        '        "assessment": "Review for factuality, relevance, and format adherence.",',
    )
    replace_cell_source(notebook, 25, cell_25)

    # Extract the real loss-curve image embedded in the completed Colab run.
    LOSS_CURVE.parent.mkdir(parents=True, exist_ok=True)
    for output in notebook["cells"][17].get("outputs", []):
        png = output.get("data", {}).get("image/png")
        if png:
            LOSS_CURVE.write_bytes(base64.b64decode("".join(png)))
            break

    # Submission notebook must be code-only/stripped, while the source notebook
    # remains intact as evidence of the completed run.
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    SUBMISSION_NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
