# pet-train

Training pipeline for Train-Pet-Pipeline (SFT + DPO + audio CNN).

## Recent

- **v2.2.5** — F024 lineage: collect git SHAs from sibling repos into ModelCard; F025 checkpoint_uri now points to the correct adapter sub-directory. Finding docs: [`F024`](../pet-infra/docs/ecosystem-validation/2026-04-25-findings/F024-collect-git-shas-from-sibling-repos.md) / [`F025`](../pet-infra/docs/ecosystem-validation/2026-04-25-findings/F025-checkpoint-uri-points-to-nonexistent-adapter-subdir.md)
- **v2.2.4** — F022 fix: LLaMA-Factory SFT/DPO wrapper now captures trainer metrics into `ModelCard.metrics` correctly. Finding doc: [`F022`](../pet-infra/docs/ecosystem-validation/2026-04-25-findings/F022-llamafactory-sft-wrapper-not-capturing-metrics.md)

## License

This project is licensed under the [Business Source License 1.1](LICENSE) (BSL 1.1).
On **2030-04-22** it converts automatically to the Apache License, Version 2.0.

> Note: BSL 1.1 is **source-available**, not OSI-approved open source.
> Production / commercial use requires a separate commercial license.

![License: BSL 1.1](https://img.shields.io/badge/license-BSL%201.1-blue.svg)
