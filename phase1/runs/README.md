# Training Runtime Modules

- `conv_runall/`: modules from `CS781_90M_800M_RunAll_Kaggle.ipynb`.
- `encdec_300m/`: modules from `CS781_EncDec_50M_300M_Kaggle.ipynb`.

Each folder contains the model, preprocessing, training, generation, correctness checks and recovery/orchestration modules. The model and preprocessing files come from the later notebook cells that overwrite the initial support bundle. The source was extracted with Python AST literal parsing and base64/gzip decoding, without executing the notebook cells.

Use the original notebook as the entry point. These modules keep their original local imports and expect to run from the notebook's configured output directory. Importing them does not prepare the datasets, saved configuration, tokenizer, evaluator or checkpoints.

`../../docs/recovered_sources.json` lists the original filenames and hashes. The standalone Conv model in `../models/conv_hybrid.py` matches the Conv Run-All model. The standalone LegalEncDec model also matches its notebook copy byte-for-byte.
