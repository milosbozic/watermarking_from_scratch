# LLM Text Watermarking from Scratch

A from-scratch Python implementation of the green-red list watermark for LLM-generated text ([Kirchenbauer et al., 2023](https://arxiv.org/abs/2301.10226)), built on Hugging Face `transformers`. The notebook implements the watermark, detects it with a proper statistical test, measures its cost in text quality, and tests how well it survives editing and paraphrasing.

It accompanies a two-part blog series:

1. [How LLM Text Watermarking Works: Build One from Scratch in Python](LINK_TO_PART_1)
2. [Does an LLM Watermark Survive Editing? Testing 4 Schemes Against Paraphrasing](LINK_TO_PART_2)

## What's inside

- **Watermark generation:** a keyed hash of the context picks a green list at every step, and a `LogitsProcessor` adds a bias δ to green tokens.
- **Detection:** a one-sided z-test with p-values and a controlled false-positive rate, checked against human text from the Brown corpus.
- **Trade-offs:** watermark strength vs. text quality, and why the watermark lives in high-entropy positions.
- **Attacks:** random word noise, meaning-preserving synonym swaps, and a full LLM paraphrase, run through one evaluation harness.
- **Seeding schemes:** LeftHash, N-gram hash, Min-hash and SimHash, compared on robustness to edits and resistance to watermark stealing.

## Quick start

```bash
pip install "transformers>=5.0" torch nltk scipy pandas matplotlib
jupyter notebook watermarking.ipynb
```

Requires Python 3.10+. The default model is `openbmb/MiniCPM5-2B`; any Hugging Face causal language model should work. A GPU is recommended; on a CPU, set `QUICK_RUN = True` for a fast first pass. Generations are cached in `wm_cache/`, so re-running a cell doesn't regenerate text. Replace `SECRET_KEY` with your own value.

## Key findings

- At δ = 2, the watermark barely affects text quality, but this instruction-tuned model needs 150–200 tokens for reliable detection.
- Light editing weakens the watermark gradually; changing about 10% of the words brings a 200-token text down to the detection threshold.
- A full LLM paraphrase removes the watermark for all four seeding schemes.
- No seeding scheme wins on both robustness and security: schemes that survive edits best reuse their green lists most, which makes them easier to steal.

## References

- J. Kirchenbauer et al. [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226). ICML 2023.
- J. Kirchenbauer et al. [On the Reliability of Watermarks for Large Language Models](https://arxiv.org/abs/2306.04634). ICLR 2024.
- S. Dathathri et al. [Scalable watermarking for identifying large language model outputs](https://www.nature.com/articles/s41586-024-08025-4). *Nature*, 2024.
- N. Jovanović, R. Staab, M. Vechev. [Watermark Stealing in Large Language Models](https://arxiv.org/abs/2402.19361). ICML 2024.

## License

MIT
