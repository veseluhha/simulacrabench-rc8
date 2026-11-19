# SimulacraBench RC8

RC8 predicts distributions over missing survey answers. It fits a separate ensemble for each instrument and uses visible answers to estimate when a question was asked. The entry point is `main.predict(frame, schema)`. The source in this repository matches the Python files in TEST submission 1057396 from team `veseluhha`.

| TEST skill | UNICEF | World Bank | UNHCR |
| ---: | ---: | ---: | ---: |
| 0.4698 | 0.3197 | 0.2093 | 0.8805 |

## Assemble the submission

From the repository root, fetch the two checkpoints into `weights/` using the Hugging Face CLI. For TabPFN, request access on the [model page](https://huggingface.co/Prior-Labs/tabpfn_3_5) and authenticate with an account that has access.

```sh
hf download Prior-Labs/tabpfn_3_5 tabpfn-v3.5-fast-20260909.safetensors --revision 06bf2ba35c80a92a3b9abb436b99cf49e7a0365e --local-dir weights
hf download jingang/TabICL tabicl-classifier-v2-20260212.ckpt --revision 4dcd344ece2c00be9e831fdd35bed57b5ad83e19 --local-dir weights
shasum -a 256 -c SHA256SUMS
cut -c67- SHA256SUMS | zip -X -q rc8.zip -@
```

The manifest covers all 18 files used in the competition. The original ZIP had SHA256 `56fd9a040b298e8fa727b71a66d9c320a135c50d1eea192d73ff7b0e83d3bad1`. ZIP metadata can change the archive hash when packaging the same files.

## Run on a public sandbox

Use the [organizers' starter kit](https://github.com/SituatedEvals/public/tree/5c69e17b191e82fe867eea6daf06c8ba8feaf622) for input preparation and scoring. From this repository, run the following in an environment with the kit requirements installed.

```sh
git clone https://github.com/SituatedEvals/public.git ../simulacrabench-kit
git -C ../simulacrabench-kit checkout 5c69e17b191e82fe867eea6daf06c8ba8feaf622
cd ../simulacrabench-kit
python tools/check_submission_zip.py ../simulacrabench-rc8/rc8.zip
python make_sandbox.py --schema data/unicef.json --out _sandbox/unicef
python score.py --data _sandbox/unicef --schema data/unicef.json --phase 2 --submission ../simulacrabench-rc8
```

The full ensemble uses a GPU. CPU execution uses the fallback predictors. The final worker processed three instruments in one process with a shared 3,600-second budget. In phase 2, TRAIN and DEV responses are visible and the organizer masks TEST responses. Use the submission defaults with all `SB_*` overrides unset. The public sandbox contains synthetic respondents. The competition score above comes from the organizer's hidden data.

## Licensing

The RC8 source code is available under the [MIT License](LICENSE). The [TabPFN-3.5 Non-Commercial License](weights/LICENSE) and the [TabICL BSD 3-Clause License](weights/LICENSE-TabICL) apply to their respective checkpoints. See [NOTICE](NOTICE) for attribution.
