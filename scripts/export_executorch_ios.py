#!/usr/bin/env python3
"""Export OpenVoice models to ExecuTorch for iOS with Core ML + fallback."""

import argparse
from pathlib import Path

import torch

from executorch.exir import EdgeCompileConfig, to_edge
from executorch.backends.apple.coreml import CoreMLPartitioner

from openvoice import utils
from openvoice.serialization import safe_torch_load
from openvoice.models import SynthesizerTrn


class TTSInferModule(torch.nn.Module):
    def __init__(self, model: SynthesizerTrn):
        super().__init__()
        self.model = model

    def forward(self, x, x_lengths, sid, noise_scale, length_scale, noise_scale_w, sdp_ratio):
        audio, _, _, _ = self.model.infer(
            x,
            x_lengths,
            sid=sid,
            noise_scale=noise_scale,
            length_scale=length_scale,
            noise_scale_w=noise_scale_w,
            sdp_ratio=sdp_ratio,
        )
        return audio


class VoiceConversionModule(torch.nn.Module):
    def __init__(self, model: SynthesizerTrn):
        super().__init__()
        self.model = model

    def forward(self, spec, spec_lengths, src_embedding, tgt_embedding, tau):
        audio, _, _ = self.model.voice_conversion(
            spec,
            spec_lengths,
            sid_src=src_embedding,
            sid_tgt=tgt_embedding,
            tau=tau,
        )
        return audio


class ReferenceEncoderModule(torch.nn.Module):
    def __init__(self, model: SynthesizerTrn):
        super().__init__()
        self.model = model

    def forward(self, spec):
        return self.model.ref_enc(spec).unsqueeze(-1)


def load_model(config_path: Path, checkpoint_path: Path, device: torch.device) -> SynthesizerTrn:
    hps = utils.get_hparams_from_file(str(config_path))
    model = SynthesizerTrn(
        len(getattr(hps, "symbols", [])),
        hps.data.filter_length // 2 + 1,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    ).to(device)
    model.eval()
    checkpoint_dict = safe_torch_load(str(checkpoint_path), map_location=device)
    model.load_state_dict(checkpoint_dict["model"], strict=False)
    return model


def export_module(
    module,
    example_inputs,
    output_path: Path,
    *,
    dynamic_shapes=None,
    export_fallback: bool = True,
):
    exported = torch.export.export(module, example_inputs, dynamic_shapes=dynamic_shapes)
    edge_program = to_edge(exported, compile_config=EdgeCompileConfig())
    coreml_partitioner = CoreMLPartitioner()
    delegated = edge_program.to_backend(coreml_partitioner)
    executorch_program = delegated.to_executorch()
    executorch_program.save(str(output_path))
    if export_fallback:
        fallback_path = output_path.with_name(output_path.stem + \"_portable.pte\")
        edge_program.to_executorch().save(str(fallback_path))


def export_all_models(
    config_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    device: torch.device,
    *,
    max_text_len: int,
    max_spec_len: int,
    batch_size: int,
):
    model = load_model(config_path, checkpoint_path, device)
    output_dir.mkdir(parents=True, exist_ok=True)

    tts_module = TTSInferModule(model)
    tts_inputs = (
        torch.zeros(batch_size, max_text_len, dtype=torch.long, device=device),
        torch.full((batch_size,), max_text_len, dtype=torch.long, device=device),
        torch.zeros(batch_size, dtype=torch.long, device=device),
        torch.tensor(0.667, device=device),
        torch.tensor(1.0, device=device),
        torch.tensor(0.6, device=device),
        torch.tensor(0.2, device=device),
    )
    export_module(
        tts_module,
        tts_inputs,
        output_dir / "openvoice_tts_infer_coreml.pte",
        dynamic_shapes={
            "x": {1: torch.export.Dim("text_len", min=1, max=max_text_len)},
            "x_lengths": {0: torch.export.Dim("batch", min=1, max=batch_size)},
            "sid": {0: torch.export.Dim("batch", min=1, max=batch_size)},
        },
    )

    vc_module = VoiceConversionModule(model)
    vc_inputs = (
        torch.zeros(batch_size, model.enc_q.spec_channels, max_spec_len, device=device),
        torch.full((batch_size,), max_spec_len, dtype=torch.long, device=device),
        torch.zeros(batch_size, model.enc_q.gin_channels, 1, device=device),
        torch.zeros(batch_size, model.enc_q.gin_channels, 1, device=device),
        torch.tensor(0.3, device=device),
    )
    export_module(
        vc_module,
        vc_inputs,
        output_dir / "openvoice_voice_conversion_coreml.pte",
        dynamic_shapes={
            "spec": {2: torch.export.Dim("spec_len", min=1, max=max_spec_len)},
            "spec_lengths": {0: torch.export.Dim("batch", min=1, max=batch_size)},
            "src_embedding": {0: torch.export.Dim("batch", min=1, max=batch_size)},
            "tgt_embedding": {0: torch.export.Dim("batch", min=1, max=batch_size)},
        },
    )

    if model.n_speakers == 0:
        ref_module = ReferenceEncoderModule(model)
        ref_inputs = (
            torch.zeros(batch_size, model.enc_q.spec_channels, max_spec_len, device=device),
        )
        export_module(
            ref_module,
            ref_inputs,
            output_dir / "openvoice_reference_encoder_coreml.pte",
            dynamic_shapes={
                "spec": {2: torch.export.Dim("spec_len", min=1, max=max_spec_len)},
            },
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export OpenVoice models to ExecuTorch using Core ML delegate with fallback.",
    )
    parser.add_argument("--config", required=True, type=Path, help="Path to model config JSON.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to model checkpoint.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for .pte outputs.")
    parser.add_argument("--device", default="cpu", help="Device for export (cpu or cuda).")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-text-len", type=int, default=256)
    parser.add_argument("--max-spec-len", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    export_all_models(
        args.config,
        args.checkpoint,
        args.output_dir,
        device,
        batch_size=args.batch_size,
        max_text_len=args.max_text_len,
        max_spec_len=args.max_spec_len,
    )


if __name__ == "__main__":
    main()
