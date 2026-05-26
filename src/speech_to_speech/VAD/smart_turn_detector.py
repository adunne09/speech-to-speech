from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SmartTurnResult:
    complete: bool
    probability: float


class SmartTurnDetector:
    """ONNX-backed wrapper for Pipecat Smart Turn endpoint detection."""

    def __init__(
        self,
        model_repo: str,
        model_filename: str,
        threshold: float,
    ) -> None:
        try:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            from transformers import WhisperFeatureExtractor
        except ImportError as e:
            raise ImportError(
                "Smart Turn requires onnxruntime and huggingface_hub. "
                "Install project dependencies again after updating, or install onnxruntime."
            ) from e

        self.threshold = threshold
        self.feature_extractor = WhisperFeatureExtractor(chunk_length=8)

        model_path = hf_hub_download(repo_id=model_repo, filename=model_filename)
        session_options = ort.SessionOptions()
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.inter_op_num_threads = 1
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(model_path, sess_options=session_options)
        logger.info("Smart Turn enabled: repo=%s file=%s threshold=%.2f", model_repo, model_filename, threshold)

    def predict(self, audio: np.ndarray) -> SmartTurnResult:
        audio = np.asarray(audio, dtype=np.float32)
        audio = self._last_eight_seconds(audio)

        inputs = self.feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="np",
            padding="max_length",
            max_length=8 * 16000,
            truncation=True,
            do_normalize=True,
        )
        input_features = np.expand_dims(inputs.input_features.squeeze(0).astype(np.float32), axis=0)
        outputs = self.session.run(None, {"input_features": input_features})
        probability = float(outputs[0][0].item())
        return SmartTurnResult(complete=probability > self.threshold, probability=probability)

    @staticmethod
    def _last_eight_seconds(audio: np.ndarray) -> np.ndarray:
        max_samples = 8 * 16000
        if audio.shape[0] > max_samples:
            return audio[-max_samples:]
        return audio
