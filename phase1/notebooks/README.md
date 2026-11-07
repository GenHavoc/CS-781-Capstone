# Kaggle Training Notebooks

The four notebooks here retain their original bytes. Their filenames and SHA-256 hashes are listed in `../../docs/recovered_sources.json`. They have no saved execution outputs or literal HF/GitHub credentials.

## Setup

For Conv, import `CS781_90M_800M_RunAll_Kaggle.ipynb` into Kaggle. For LegalEncDec, use `CS781_EncDec_50M_300M_Kaggle.ipynb`. Select GPU T4 ×2, enable Internet, and set a Hugging Face token in Kaggle Secrets (`HF` or `HF_TOKEN`, depending on the notebook). Check the configuration before choosing Run All.

The notebooks embed the original tokenizer and evaluator, verify their checksums, and write runtime code to their configured `/kaggle/working/` directory. Private Hugging Face recovery backups are enabled by default: check `REMOTE_BACKUP` and `BACKUP_REPO` before starting. Compatible checkpoints saved at optimizer-update boundaries can be used to resume an interrupted run.

## Earlier versions

`LegalPrefixConv_90M_10Hour_Training_Evaluation.ipynb` and `LegalPrefixConv_89M_Kaggle.ipynb` are earlier Conv versions. They belong to the same model family. The 89M-named notebook includes official-test submission packaging; the later Run-All versions focus on training and local evaluation.

The notebooks don't include trained weights or logs establishing a completed token budget. They were inspected and syntax-checked during repo preparation, without a GPU training run.
