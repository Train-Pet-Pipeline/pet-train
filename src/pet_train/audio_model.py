"""Audio CNN training CLI — PANNs MobileNetV2 transfer learning.

Entry point called by ``scripts/train_audio.sh``::

    python -m pet_train.audio_model \\
        --config configs/audio/mobilenetv2_transfer_v1.yaml \\
        --params params.yaml \\
        --output_dir outputs/audio/... \\
        --run_name experiment_name
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pytorch_lightning as pl
import torch
import torchaudio
import yaml
from torch.utils.data import DataLoader, Dataset, random_split

from pet_train.audio_model_arch import MobileNetV2AudioSet

logger = logging.getLogger(__name__)


class AudioDataset(Dataset):
    """Directory-based audio dataset.

    Expects ``data_dir/{class_name}/*.wav`` structure.
    Labels are mapped to indices by sorted class order from params.

    Args:
        data_dir: Root directory containing per-class subdirectories.
        classes: List of class name strings.
        sample_rate: Target sample rate.
        max_duration_sec: Maximum clip duration in seconds (longer clips are truncated).
    """

    def __init__(
        self,
        data_dir: Path,
        classes: list[str],
        sample_rate: int = 16000,
        max_duration_sec: float = 5.0,
    ) -> None:
        """Initialize audio dataset from directory structure."""
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_duration_sec)
        self.samples: list[tuple[Path, int]] = []

        for cls_name in classes:
            cls_dir = data_dir / cls_name
            if not cls_dir.exists():
                logger.warning("Class directory not found: %s", cls_dir)
                continue
            for audio_file in sorted(cls_dir.iterdir()):
                if audio_file.suffix.lower() in (".wav", ".mp3", ".flac", ".ogg"):
                    self.samples.append((audio_file, self.class_to_idx[cls_name]))

        logger.info(
            '{"event": "dataset_loaded", "total": %d, "classes": %s}',
            len(self.samples), json.dumps({c: 0 for c in classes}),
        )

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Load and return a single (waveform, label) pair."""
        path, label = self.samples[idx]
        waveform, sr = torchaudio.load(str(path))

        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Truncate or pad to fixed length
        if waveform.shape[1] > self.max_samples:
            waveform = waveform[:, :self.max_samples]
        elif waveform.shape[1] < self.max_samples:
            pad = self.max_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))

        return waveform.squeeze(0), label


class AudioLitModel(pl.LightningModule):
    """PyTorch Lightning wrapper for audio CNN training.

    Args:
        model: MobileNetV2AudioSet model instance.
        learning_rate: Optimizer learning rate.
        num_classes: Number of output classes.
    """

    def __init__(self, model: MobileNetV2AudioSet, learning_rate: float, num_classes: int) -> None:
        """Initialize lightning module."""
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Compute training loss."""
        waveform, labels = batch
        logits = self(waveform)
        loss = self.loss_fn(logits, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        """Compute validation metrics."""
        waveform, labels = batch
        logits = self(waveform)
        loss = self.loss_fn(logits, labels)
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return [optimizer], [scheduler]


def main() -> None:
    """CLI entry point for audio CNN training."""
    logging.basicConfig(
        level=logging.INFO,
        format='{"time":"%(asctime)s","level":"%(levelname)s","module":"%(name)s","msg":"%(message)s"}',
    )

    parser = argparse.ArgumentParser(description="Audio CNN training")
    parser.add_argument("--config", required=True, help="Audio experiment config YAML")
    parser.add_argument("--params", required=True, help="Global params.yaml")
    parser.add_argument("--output_dir", required=True, help="Output directory for checkpoints")
    parser.add_argument("--run_name", default="audio_run", help="Experiment name")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    with open(args.params) as f:
        params = yaml.safe_load(f)

    audio_params = params["audio"]
    train_cfg = config.get("training", {})
    model_cfg = config.get("model", {})

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build model
    num_classes = model_cfg.get("num_classes", len(audio_params["classes"]))
    model = MobileNetV2AudioSet(
        num_classes=num_classes,
        sample_rate=audio_params["sample_rate"],
    )

    # Load pretrained weights if available
    pretrained_path = model_cfg.get("pretrained_path")
    if pretrained_path and Path(pretrained_path).exists():
        checkpoint = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("model", checkpoint)
        # Replace classifier if num_classes differs
        filtered = {k: v for k, v in state_dict.items() if "classifier" not in k}
        model.load_state_dict(filtered, strict=False)
        logger.info("Loaded pretrained backbone from %s", pretrained_path)

    # Freeze backbone if configured
    if model_cfg.get("freeze_backbone", False):
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
        logger.info("Backbone frozen, only classifier is trainable")

    # Load dataset
    data_dir = Path(audio_params["data_dir"])
    dataset = AudioDataset(
        data_dir=data_dir,
        classes=audio_params["classes"],
        sample_rate=audio_params["sample_rate"],
    )

    if len(dataset) == 0:
        logger.error("No audio samples found in %s", data_dir)
        raise RuntimeError(f"No audio samples found in {data_dir}")

    # Split dataset
    n_total = len(dataset)
    n_train = max(1, int(n_total * audio_params["train_split"]))
    n_val = max(1, int(n_total * audio_params["val_split"]))
    n_test = n_total - n_train - n_val
    if n_test < 0:
        n_test = 0
        n_val = n_total - n_train

    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )
    logger.info("Dataset split: train=%d, val=%d, test=%d", n_train, n_val, n_test)

    batch_size = min(train_cfg.get("batch_size", 32), n_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=0)

    # Training
    lit_model = AudioLitModel(
        model=model,
        learning_rate=train_cfg.get("learning_rate", 1e-3),
        num_classes=num_classes,
    )

    callbacks = []
    if train_cfg.get("early_stopping_patience"):
        callbacks.append(pl.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=train_cfg["early_stopping_patience"],
            mode="min",
        ))
    callbacks.append(pl.callbacks.ModelCheckpoint(
        dirpath=str(output_dir),
        filename="best-{epoch}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    ))

    # Determine accelerator
    if torch.cuda.is_available():
        accelerator = "gpu"
    elif torch.backends.mps.is_available():
        accelerator = "mps"
    else:
        accelerator = "cpu"

    trainer = pl.Trainer(
        max_epochs=train_cfg.get("max_epochs", 50),
        accelerator=accelerator,
        callbacks=callbacks,
        default_root_dir=str(output_dir),
        enable_progress_bar=True,
        log_every_n_steps=1,
    )

    trainer.fit(lit_model, train_loader, val_loader)

    # Save final model
    final_path = output_dir / "final_model.pt"
    torch.save(model.state_dict(), final_path)
    logger.info("Final model saved to %s", final_path)

    # Save training report
    report = {
        "run_name": args.run_name,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "num_classes": num_classes,
        "classes": audio_params["classes"],
        "best_val_loss": trainer.callback_metrics.get("val_loss", float("inf")),
        "output_dir": str(output_dir),
    }
    # Convert tensor values
    report = {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in report.items()}

    report_path = output_dir / "training_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Training report saved to %s", report_path)


if __name__ == "__main__":
    main()
