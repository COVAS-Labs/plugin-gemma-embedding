"""
EmbeddingGemma Plugin for COVAS:NEXT
Provides offline embedding capabilities using the Google EmbeddingGemma model.
"""

from typing import override, Any, List
import os
import json
import threading
import numpy as np
import onnxruntime
from tokenizers import Tokenizer

from lib.PluginHelper import PluginHelper, EmbeddingModel
from lib.PluginSettingDefinitions import (
    PluginSettings,
    ModelProviderDefinition,
    SettingsGrid,
    ParagraphSetting,
    NumericalSetting,
    SelectOption,
    SelectSetting,
)
from lib.PluginBase import PluginBase, PluginManifest
from lib.Logger import log

DEFAULT_ONNX_THREADS = min(4, max(1, (os.cpu_count() or 1) // 2))
SETTINGS_SCHEMA_VERSION = 1


def _migrate_settings(settings: dict[str, Any], from_version: int) -> None:
    if from_version == 0:
        try:
            if int(settings.get("onnx_threads", DEFAULT_ONNX_THREADS)) > 4:
                settings["onnx_threads"] = 4
        except (TypeError, ValueError):
            pass


def _apply_pending_migrations(settings: dict[str, Any]) -> None:
    try:
        version = int(settings.get("settings_version", 0))
    except (TypeError, ValueError):
        version = 0

    while version < SETTINGS_SCHEMA_VERSION:
        _migrate_settings(settings, version)
        version += 1
        settings["settings_version"] = version


def _discover_model_files(model_dir: str) -> list[str]:
    preferred = [
        os.path.join("onnx", "model_quantized.onnx"),
        "model_quantized.onnx",
        os.path.join("onnx", "model_fp16.onnx"),
        "model_fp16.onnx",
    ]
    model_files: list[str] = []
    for root, _, files in os.walk(model_dir):
        for filename in files:
            if not filename.lower().endswith(".onnx") or "model" not in filename.lower():
                continue
            relative_path = os.path.relpath(os.path.join(root, filename), model_dir)
            model_files.append(relative_path)

    return sorted(set(model_files), key=lambda path: (preferred.index(path) if path in preferred else len(preferred), path))


class GemmaEmbeddingModel(EmbeddingModel):
    """Gemma Embedding model implementation."""
    
    def __init__(self, model_dir: str, model_file: str | None = None, onnx_threads: int = DEFAULT_ONNX_THREADS):
        super().__init__("gemma-embedding")
        self.model_dir = model_dir
        self.model_file = model_file
        self.onnx_threads = max(1, int(onnx_threads))
        self._session = None
        self._tokenizer = None
        self._max_length = None
        threading.Thread(target=self._warmup, name="gemma-embedding-warmup", daemon=True).start()

    def _warmup(self) -> None:
        try:
            self.create_embedding("warmup")
            log("info", "Gemma embedding warmup complete")
        except Exception as exc:
            log("warning", f"Gemma embedding warmup failed: {exc}")
    
    def _get_model(self):
        """Lazily initialize the model."""
        if self._session is None:
            if not os.path.exists(self.model_dir):
                raise ValueError(f"Model directory not found: {self.model_dir}")

            log('info', f"Loading Gemma Embedding model from {self.model_dir}")
            
            try:
                # Load tokenizer (lightweight; avoids transformers)
                tokenizer_path = os.path.join(self.model_dir, "tokenizer.json")
                if not os.path.exists(tokenizer_path):
                    for root, _, files in os.walk(self.model_dir):
                        if "tokenizer.json" in files:
                            tokenizer_path = os.path.join(root, "tokenizer.json")
                            break

                if not os.path.exists(tokenizer_path):
                    raise ValueError(f"tokenizer.json not found in {self.model_dir}")

                self._tokenizer = Tokenizer.from_file(tokenizer_path)

                # Best-effort max length from config.json (optional)
                config_path = os.path.join(self.model_dir, "config.json")
                if os.path.exists(config_path):
                    with open(config_path, "r", encoding="utf-8") as f:
                        config = json.load(f)
                    self._max_length = (
                        config.get("max_position_embeddings")
                        or config.get("max_sequence_length")
                        or config.get("max_length")
                    )

                model_files = _discover_model_files(self.model_dir)
                if self.model_file in model_files:
                    onnx_path = os.path.join(self.model_dir, self.model_file)
                elif model_files:
                    onnx_path = os.path.join(self.model_dir, model_files[0])
                else:
                    raise ValueError(f"ONNX model file not found in {self.model_dir}")

                session_options = onnxruntime.SessionOptions()
                session_options.intra_op_num_threads = self.onnx_threads
                session_options.inter_op_num_threads = 1
                session_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL

                log('info', f"Gemma Embedding using {self.onnx_threads} CPU threads")
                self._session = onnxruntime.InferenceSession(
                    onnx_path,
                    sess_options=session_options,
                )
                
            except Exception as e:
                log('error', f"Failed to initialize Gemma Embedding model: {e}")
                raise

        return self._session, self._tokenizer

    def _tokenize(self, tokenizer: Tokenizer, text: str) -> tuple[np.ndarray, np.ndarray]:
        encoding = tokenizer.encode(text)
        input_ids = np.asarray([encoding.ids], dtype=np.int64)
        attention_mask = np.asarray([encoding.attention_mask], dtype=np.int64)

        if self._max_length is not None and input_ids.shape[1] > int(self._max_length):
            input_ids = input_ids[:, : int(self._max_length)]
            attention_mask = attention_mask[:, : int(self._max_length)]

        return input_ids, attention_mask

    def _build_onnx_inputs(self, session: onnxruntime.InferenceSession, input_ids: np.ndarray, attention_mask: np.ndarray) -> dict[str, np.ndarray]:
        seq_len = input_ids.shape[1]
        feed: dict[str, np.ndarray] = {}

        for input_meta in session.get_inputs():
            name = input_meta.name
            lowered = name.lower()

            if "input_ids" in lowered or lowered == "ids":
                value = input_ids
            elif "attention_mask" in lowered or lowered == "mask":
                value = attention_mask
            elif "token_type_ids" in lowered or "segment_ids" in lowered:
                value = np.zeros((1, seq_len), dtype=np.int64)
            elif "position_ids" in lowered:
                value = np.arange(seq_len, dtype=np.int64).reshape(1, seq_len)
            else:
                raise ValueError(f"Unsupported ONNX input: {name}")

            # Cast to the model-declared integer type when possible
            declared_type = (input_meta.type or "").lower()
            if "int32" in declared_type:
                value = value.astype(np.int32)
            elif "int64" in declared_type:
                value = value.astype(np.int64)

            feed[name] = value

        return feed

    def create_embedding(self, input_text: str) -> tuple[str, List[float]]:
        """Create embedding for the given text."""
        try:
            session, tokenizer = self._get_model()
            
            prefixes = {
                "query": "task: search result | query: ",
                "document": "title: none | text: ",
            }
            
            # Using document prefix as default
            text = prefixes["document"] + input_text
            input_ids, attention_mask = self._tokenize(tokenizer, text)
            feed = self._build_onnx_inputs(session, input_ids, attention_mask)

            outputs = session.run(None, feed)
            sentence_embedding = outputs[-1] if len(outputs) > 1 else outputs[0]

            return (self.model_name, sentence_embedding[0].tolist())
            
        except Exception as e:
            log('error', f"Gemma Embedding failed: {e}")
            raise

class GemmaEmbeddingPlugin(PluginBase):
    """
    Plugin providing Gemma Embedding services.
    """

    settings_schema_version = SETTINGS_SCHEMA_VERSION
    
    def __init__(self, plugin_manifest: PluginManifest):
        super().__init__(plugin_manifest)
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
        model_files = _discover_model_files(model_dir)
        provider_settings = []
        if len(model_files) > 1:
            provider_settings.append(
                SettingsGrid(
                    key="model",
                    label="Model",
                    fields=[
                        SelectSetting(
                            key="model_file",
                            label="Model format",
                            type="select",
                            readonly=False,
                            placeholder=None,
                            default_value=model_files[0],
                            select_options=[
                                SelectOption(key=model_file, label=model_file, value=model_file, disabled=False)
                                for model_file in model_files
                            ],
                            multi_select=False,
                        )
                    ],
                )
            )
        provider_settings.append(
            SettingsGrid(
                key="performance",
                label="Performance",
                fields=[
                    NumericalSetting(
                        key="onnx_threads",
                        label="CPU Threads",
                        type="number",
                        readonly=False,
                        placeholder=str(DEFAULT_ONNX_THREADS),
                        default_value=DEFAULT_ONNX_THREADS,
                        min_value=1,
                        max_value=max(1, os.cpu_count() or 1),
                        step=1,
                    ),
                ],
            )
        )
        
        self.settings_config = PluginSettings(
            key="Gemma Embedding",
            label="Gemma Embedding",
            icon="memory",
            grids=[
                SettingsGrid(
                    key="general",
                    label="General",
                    fields=[
                        ParagraphSetting(
                            key="info_text",
                            label=None,
                            type="paragraph",
                            readonly=False,
                            placeholder=None,
                            content='To use Gemma Embedding, select it as your "Embedding provider" in "Advanced" → "Embedding Settings".'
                        ),
                    ]
                ),
            ]
        )
        
        self.model_providers = [
            ModelProviderDefinition(
                kind='embedding',
                id='gemma-embedding',
                label='EmbeddingGemma (Local)',
                settings_config=provider_settings
            )
        ]

    @override
    def migrate_settings(self, settings: dict[str, Any], from_version: int) -> None:
        _migrate_settings(settings, from_version)
    
    @override
    def create_model(self, provider_id: str, settings: dict[str, Any]) -> EmbeddingModel:
        """Create a model instance for the given provider."""
        
        if provider_id == 'gemma-embedding':
            _apply_pending_migrations(settings)
            plugin_dir = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.join(plugin_dir, "model")
            model_files = _discover_model_files(model_dir)
            model_file = str(settings.get("model_file", model_files[0] if model_files else ""))
            onnx_threads = int(settings.get("onnx_threads", DEFAULT_ONNX_THREADS))
            return GemmaEmbeddingModel(model_dir=model_dir, model_file=model_file, onnx_threads=onnx_threads)
        
        raise ValueError(f'Unknown Gemma provider: {provider_id}')

if __name__ == "__main__":
    # For testing purposes
    plugin_manifest = PluginManifest(
        name="Gemma Embedding Plugin",
        version="0.0.11",
        author="COVAS:NEXT",
        description="Gemma Embedding Plugin for COVAS:NEXT"
    )
    plugin = GemmaEmbeddingPlugin(plugin_manifest)
    log('info', "Gemma Embedding Plugin initialized successfully.")
