"""
EmbeddingGemma Plugin for COVAS:NEXT
Provides offline embedding capabilities using the Google EmbeddingGemma model.
"""

from typing import override, Any, List
import os
import numpy as np
import onnxruntime
from transformers import AutoTokenizer, PretrainedConfig

from lib.PluginHelper import PluginHelper, EmbeddingModel
from lib.PluginSettingDefinitions import (
    PluginSettings,
    ModelProviderDefinition,
    SettingsGrid,
    ParagraphSetting,
)
from lib.PluginBase import PluginBase, PluginManifest
from lib.Logger import log

class GemmaEmbeddingModel(EmbeddingModel):
    """Gemma Embedding model implementation."""
    
    def __init__(self, model_dir: str):
        super().__init__("gemma-embedding")
        self.model_dir = model_dir
        self._session = None
        self._tokenizer = None
        self._config = None
    
    def _get_model(self):
        """Lazily initialize the model."""
        if self._session is None:
            if not os.path.exists(self.model_dir):
                raise ValueError(f"Model directory not found: {self.model_dir}")

            log('info', f"Loading Gemma Embedding model from {self.model_dir}")
            
            try:
                # Load tokenizer and model config
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
                self._config = PretrainedConfig.from_pretrained(self.model_dir)

                # ONNX session
                # Look for the ONNX file
                onnx_path = os.path.join(self.model_dir, "onnx", "model_fp16.onnx")
                if not os.path.exists(onnx_path):
                     onnx_path = os.path.join(self.model_dir, "model_fp16.onnx")
                
                if not os.path.exists(onnx_path):
                     # Try finding any onnx file
                     for root, dirs, files in os.walk(self.model_dir):
                        for file in files:
                            if file.endswith(".onnx") and "model" in file:
                                onnx_path = os.path.join(root, file)
                                break
                        if os.path.exists(onnx_path):
                            break

                if not os.path.exists(onnx_path):
                    raise ValueError(f"ONNX model file not found in {self.model_dir}")

                self._session = onnxruntime.InferenceSession(onnx_path)
                
            except Exception as e:
                log('error', f"Failed to initialize Gemma Embedding model: {e}")
                raise

        return self._session, self._tokenizer

    def create_embedding(self, input_text: str) -> tuple[str, List[float]]:
        """Create embedding for the given text."""
        try:
            session, tokenizer = self._get_model()
            
            prefixes = {
                "query": "task: search result | query: ",
                "document": "title: none | text: ",
            }
            
            # Using document prefix as default
            inputs = tokenizer([prefixes["document"] + input_text], padding=True, return_tensors="np")

            _, sentence_embedding = session.run(None, inputs.data)
            
            return (self.model_name, sentence_embedding[0].tolist())
            
        except Exception as e:
            log('error', f"Gemma Embedding failed: {e}")
            raise

class GemmaEmbeddingPlugin(PluginBase):
    """
    Plugin providing Gemma Embedding services.
    """
    
    def __init__(self, plugin_manifest: PluginManifest):
        super().__init__(plugin_manifest)
        
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
                label='Gemma Embedding (Offline)',
                settings_config=[]
            )
        ]
    
    @override
    def create_model(self, provider_id: str, settings: dict[str, Any]) -> EmbeddingModel:
        """Create a model instance for the given provider."""
        
        if provider_id == 'gemma-embedding':
            plugin_dir = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.join(plugin_dir, "model")
            
            return GemmaEmbeddingModel(model_dir=model_dir)
        
        raise ValueError(f'Unknown Gemma provider: {provider_id}')

if __name__ == "__main__":
    # For testing purposes
    plugin_manifest = PluginManifest(
        name="Gemma Embedding Plugin",
        version="1.0.0",
        author="COVAS:NEXT",
        description="Gemma Embedding Plugin for COVAS:NEXT"
    )
    plugin = GemmaEmbeddingPlugin(plugin_manifest)
    log('info', "Gemma Embedding Plugin initialized successfully.")