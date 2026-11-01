# Retrieval confidence under a limited target-label budget

Code and derived results accompanying **100 Labels on Retrieval Confidence: Source Adaptation or Target-Only Learning?**

This research artifact compares adapting a source predictor with training a target-only predictor and estimating a target prior, using the same target-query annotation budgets. It contains saved outputs from a four-collection study: SciFact, NFCorpus, ArguAna, and SciDocs; BM25 plus POTION-B and POTION-R static retrievers; 20 paired fitting repetitions; 25/50/100-label budgets; and cutoffs 1/5/10.

## Result and review status

All four primary confidence intervals cross zero. The evidence establishes neither a winner nor equivalence. This is a research artifact released for inspection, not a claim of peer review, acceptance, or human-author approval. The manuscript's final declarations, independent author review, and external grammar/similarity checks remain pending. The draft manuscript is not included here.

The archived outputs contain 1,479 distinct held-out queries, 5,324,400 probability values, and 14,400 metric rows. Repeated predictions are not independent observations. FINAL_SCIENTIFIC_AUDIT.json records a saved-output audit performed on 1 November 2026, including recalculation of the metrics and primary intervals and reconstruction of the numeric fitted models. This audit is not a fresh end-to-end reproduction.

## Files

- Source scripts and pinned requirements describe the executed pipeline.
- `features/` contains derived feature arrays, ranked IDs, scores, and judged-hit labels.
- `results/` contains predictions, fitted parameters, fit identifiers, metrics, contrasts, and execution metadata.
- Protocols, provenance, model manifests, and data/model terms document the inputs and amendments.
- `Reproduce_Colab.ipynb` is provided but has not been executed in Colab.

Raw benchmark text, model weights, credentials, and the unfinished manuscript are excluded. Third-party dataset and model terms still apply; see Data_and_Model_Terms.md. No new license is granted for this research artifact; authors must approve the license for their own work. Third-party files retain their original terms.

## Reproduction

Use separate Python environments for `requirements-analysis.txt` and `requirements-embedding.txt`, as described in the execution and inference manifests. The original analysis used Python 3.14.3; embedding used Python 3.12. Numerical differences across platforms are possible.

For a fresh run from downloaded inputs, choose a new destination outside this repository:

```sh
python run_reproduction.py --output ../clean-reproduction --embedding-python /path/to/embedding/python
```

This helper and the notebook were prepared but not run as a clean remote reproduction. Full corpus inference can require substantial time and memory. Contextual MiniLM/BGE attempts did not finish and are excluded from the reported study.

For independent arithmetic verification after obtaining/extracting the original BEIR inputs:

```sh
python audit_saved_outputs.py . /path/to/BEIR/data
```

The independent audit uses NumPy and writes a fresh report to `qa/`. The original validator is also retained. Saved probabilities, pinned source hashes, and protocol history document the actual executed study; none of these materials constitutes human sign-off.
