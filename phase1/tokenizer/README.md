# Custom-Trained 50k BPE Tokenizer

This is the custom-trained byte-level BPE tokenizer for the project. The serialized tokenizer is [legal_bpe_50k_fixed.json](legal_bpe_50k_fixed.json).

## Configuration

| Setting | Value |
| --- | --- |
| Algorithm | Byte Pair Encoding (BPE) |
| Vocabulary size | 50,000, including the four special tokens |
| Token ID range | 0–49,999 |
| Learned merge rules | 49,740 |
| Pre-tokenizer | ByteLevel; `add_prefix_space=true`, `trim_offsets=true`, `use_regex=true` |
| Decoder | ByteLevel |
| Normalizer | None |
| Post-processor | None |

## Special tokens

| Token | ID |
| --- | ---: |
| `<pad>` | 0 |
| `<unk>` | 1 |
| `<bos>` | 2 |
| `<eos>` | 3 |

The JSON has no post-processor to insert BOS/EOS automatically. Add them in the data or generation pipeline when the model requires them.

## Loading

Install the notebook dependencies from `requirements-phase1.txt`, then load the tokenizer from the repo root:

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("phase1/tokenizer/legal_bpe_50k_fixed.json")
encoding = tokenizer.encode("The court considered the evidence.")
print(encoding.tokens)
print(encoding.ids)
print(tokenizer.decode(encoding.ids))
```

## File identity

The uploaded `legal_bpe_50k_fixed(1).json` matches the existing repo file byte-for-byte. It is stored under the canonical filename above so the notebook and model paths continue to work.

- File size: 3,603,550 bytes.
- SHA-256: `636cf69e301bdd6ed22f1d89cb9ce01edb4e104f9456e5b99c6fa30fcd882233`.
