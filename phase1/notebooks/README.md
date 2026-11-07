# Original Kaggle Notebooks

These four original chat attachments are preserved byte-for-byte. Their SHA-256 values appear in `../../docs/recovered_sources.json`. They contain no saved execution outputs or literal HF/GitHub credentials.

For the latest recovered Conv workflow, import `CS781_90M_800M_RunAll_Kaggle.ipynb` into Kaggle. For LegalEncDec, import `CS781_EncDec_50M_300M_Kaggle.ipynb`. Select GPU T4 ×2, enable Internet, and provide a Hugging Face token through Kaggle Secrets (`HF` or `HF_TOKEN`, as supported by the notebook). Review the configuration and then use Run All.

The notebooks embed and checksum-verify the original tokenizer/evaluator and write runtime code into their configured `/kaggle/working/` output directory. Their defaults enable private Hugging Face recovery backups; inspect `REMOTE_BACKUP` and `BACKUP_REPO` before running. Interrupted runs can resume from compatible optimizer-boundary checkpoints as described in each notebook.

`LegalPrefixConv_90M_10Hour_Training_Evaluation.ipynb` and `LegalPrefixConv_89M_Kaggle.ipynb` preserve earlier versions. These are iterations of the same model family, not evidence of distinct architectural submissions. The 89M-named notebook includes official-test submission packaging, whereas the later recovered Run-All notebooks focus on training and local evaluation.

None of these recovered notebooks supplies trained weights or proves a completed token budget. They have been inspected and syntax-checked here, but were not run on a GPU during repository recovery.
