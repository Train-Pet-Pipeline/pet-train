-include ../pet-infra/shared/Makefile.include

.PHONY: setup test lint clean train-sft train-dpo train-audio merge collect-logits

setup:
	pip install -e ".[dev]"
	cd vendor/LLaMA-Factory && pip install -e ".[torch,metrics]"

test:
	pytest tests/ -v

train-sft:
	bash scripts/train_sft.sh $(CONFIG)

train-dpo:
	bash scripts/train_dpo.sh $(CONFIG)

train-audio:
	bash scripts/train_audio.sh $(CONFIG)

merge:
	bash scripts/merge_lora.sh $(ADAPTER_PATH) $(OUTPUT_PATH)

collect-logits:
	bash scripts/collect_logits.sh
