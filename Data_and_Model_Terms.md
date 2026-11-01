# Input provenance and terms

All four official BEIR archive MD5 values matched, and SHA-256 values, file sizes and extracted-file hashes are recorded in `data_provenance.json`. SciFact/NFCorpus/ArguAna reused verified cached public archives; SciDocs was downloaded. All v3 features and predictions were computed for this protocol. Earlier project results were not imported as v3 measurements.

- SciFact: [original license](https://github.com/allenai/scifact/blob/master/LICENSE.md) distinguishes CC BY 4.0 claim/evidence annotations from S2ORC-derived abstracts under ODC-By 1.0.
- NFCorpus: [original project](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/) permits academic use, with underlying NutritionFacts.org content terms also relevant.
- ArguAna: [original project](https://webis.de/data/arguana-counterargs.html) and [dataset DOI](https://doi.org/10.5281/zenodo.3973257). The DOI was resolved from the original project, but Zenodo access failed; exact downstream redistribution terms were not independently confirmed here. The package excludes the corpus and query text.
- SciDocs: [original repository license](https://github.com/allenai/scidocs/blob/master/LICENSE) is CC BY 4.0. The BEIR task is an adaptation of the original benchmark; newer SciRepEval exists, so this is a historical benchmark choice.
- Model cards: [POTION-base-8M](https://huggingface.co/minishlab/potion-base-8M) and [POTION-retrieval-32M](https://huggingface.co/minishlab/potion-retrieval-32M) identify MIT licensing. Exact revisions, downloaded file hashes and sizes appear in `static_model_manifest.json`. Static token pooling is used through FastEmbed. Model2Vec's repository is cited in the manuscript. No model weights are redistributed.

BEIR's framework license does not override each dataset's terms. Derived IDs, rankings, features and probabilities are included for review, without full raw documents or query text. Public release still requires author review of rights and attribution. Pretrained model training used public datasets; contamination or semantic overlap has not been excluded. No medical decision making, human participant study, or private user corpus was involved.
