# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.gguf_loader import GGUFModelLoader
from vllm.model_executor.model_loader.weight_utils import download_gguf
from vllm.transformers_utils.gguf_utils import discover_gguf_shards


class TestDiscoverGgufShards:
    """Test discover_gguf_shards utility."""

    def test_non_split_file(self, tmp_path):
        """Non-split GGUF returns a single-element list."""
        f = tmp_path / "model-Q4_K_M.gguf"
        f.touch()
        assert discover_gguf_shards(str(f)) == [str(f)]

    def test_split_shards(self, tmp_path):
        """Split GGUF discovers all shards in order."""
        for i in range(1, 4):
            (tmp_path / f"model-Q4_K_M-{i:05d}-of-00003.gguf").touch()

        result = discover_gguf_shards(
            str(tmp_path / "model-Q4_K_M-00001-of-00003.gguf")
        )
        assert result == [
            str(tmp_path / "model-Q4_K_M-00001-of-00003.gguf"),
            str(tmp_path / "model-Q4_K_M-00002-of-00003.gguf"),
            str(tmp_path / "model-Q4_K_M-00003-of-00003.gguf"),
        ]

    def test_split_shards_from_middle(self, tmp_path):
        """Passing any shard discovers all of them."""
        for i in range(1, 3):
            (tmp_path / f"model-Q2_K-{i:05d}-of-00002.gguf").touch()

        result = discover_gguf_shards(
            str(tmp_path / "model-Q2_K-00002-of-00002.gguf")
        )
        assert result == [
            str(tmp_path / "model-Q2_K-00001-of-00002.gguf"),
            str(tmp_path / "model-Q2_K-00002-of-00002.gguf"),
        ]

    def test_missing_shard(self, tmp_path):
        """Raises FileNotFoundError when a shard is missing."""
        (tmp_path / "model-Q4_K_M-00001-of-00003.gguf").touch()
        # shard 2 missing
        (tmp_path / "model-Q4_K_M-00003-of-00003.gguf").touch()

        with pytest.raises(FileNotFoundError, match="Missing GGUF shard"):
            discover_gguf_shards(
                str(tmp_path / "model-Q4_K_M-00001-of-00003.gguf")
            )


class TestGGUFDownload:
    """Test GGUF model downloading functionality."""

    @patch("vllm.model_executor.model_loader.weight_utils.download_weights_from_hf")
    def test_download_gguf_single_file(self, mock_download):
        """Test downloading a single GGUF file."""
        # Setup mock
        mock_folder = "/tmp/mock_cache"
        mock_download.return_value = mock_folder

        # Mock glob to return a single file
        with (
            patch("glob.glob") as mock_glob,
            patch("pathlib.Path.is_file", return_value=False),
        ):
            mock_glob.side_effect = lambda pattern, **kwargs: (
                [f"{mock_folder}/model-IQ1_S.gguf"] if "IQ1_S" in pattern else []
            )

            result = download_gguf("unsloth/Qwen3-0.6B-GGUF", "IQ1_S")

            # Verify download_weights_from_hf was called with correct patterns
            mock_download.assert_called_once_with(
                model_name_or_path="unsloth/Qwen3-0.6B-GGUF",
                cache_dir=None,
                allow_patterns=[
                    "*-IQ1_S.gguf",
                    "*-IQ1_S-*.gguf",
                    "*/*-IQ1_S.gguf",
                    "*/*-IQ1_S-*.gguf",
                ],
                revision=None,
                ignore_patterns=None,
            )

            # Non-split file returns single-element list
            assert result == [f"{mock_folder}/model-IQ1_S.gguf"]

    @patch("vllm.model_executor.model_loader.weight_utils.download_weights_from_hf")
    def test_download_gguf_sharded_files(self, mock_download):
        """Test downloading sharded GGUF files returns all shards."""
        with tempfile.TemporaryDirectory() as mock_folder:
            mock_download.return_value = mock_folder

            # Create actual shard files so discover_gguf_shards can find them
            shard1 = Path(mock_folder) / "model-Q2_K-00001-of-00002.gguf"
            shard2 = Path(mock_folder) / "model-Q2_K-00002-of-00002.gguf"
            shard1.touch()
            shard2.touch()

            with patch("glob.glob") as mock_glob:
                mock_glob.side_effect = lambda pattern, **kwargs: (
                    [str(shard1), str(shard2)]
                    if "Q2_K" in pattern
                    else []
                )

                result = download_gguf("unsloth/gpt-oss-120b-GGUF", "Q2_K")

                # Should return both shards in order
                assert result == [str(shard1), str(shard2)]

    @patch("vllm.model_executor.model_loader.weight_utils.download_weights_from_hf")
    def test_download_gguf_subdir(self, mock_download):
        """Test downloading GGUF files from subdirectory."""
        mock_folder = "/tmp/mock_cache"
        mock_download.return_value = mock_folder

        with (
            patch("glob.glob") as mock_glob,
            patch("pathlib.Path.is_file", return_value=False),
        ):
            mock_glob.side_effect = lambda pattern, **kwargs: (
                [f"{mock_folder}/Q2_K/model-Q2_K.gguf"]
                if "Q2_K" in pattern or "**/*.gguf" in pattern
                else []
            )

            result = download_gguf("unsloth/gpt-oss-120b-GGUF", "Q2_K")

            # Non-split file in subdir returns single-element list
            assert result == [f"{mock_folder}/Q2_K/model-Q2_K.gguf"]

    @patch("vllm.model_executor.model_loader.weight_utils.download_weights_from_hf")
    @patch("glob.glob", return_value=[])
    def test_download_gguf_no_files_found(self, mock_glob, mock_download):
        """Test error when no GGUF files are found."""
        mock_folder = "/tmp/mock_cache"
        mock_download.return_value = mock_folder

        with pytest.raises(ValueError, match="Downloaded GGUF files not found"):
            download_gguf("unsloth/Qwen3-0.6B-GGUF", "IQ1_S")


class TestGGUFModelLoader:
    """Test GGUFModelLoader class methods."""

    @patch("os.path.isfile", return_value=True)
    def test_prepare_weights_local_file(self, mock_isfile):
        """Test _prepare_weights with local file returns list."""
        load_config = LoadConfig(load_format="gguf")
        loader = GGUFModelLoader(load_config)

        model_config = MagicMock()
        model_config.model = "/path/to/model.gguf"

        with patch(
            "vllm.transformers_utils.gguf_utils.Path.is_file",
            return_value=False,
        ):
            result = loader._prepare_weights(model_config)
            # Non-split local file → single-element list
            assert result == ["/path/to/model.gguf"]
            mock_isfile.assert_called_once_with("/path/to/model.gguf")

    @patch("vllm.model_executor.model_loader.gguf_loader.hf_hub_download")
    @patch("os.path.isfile", return_value=False)
    def test_prepare_weights_https_url(self, mock_isfile, mock_hf_download):
        """Test _prepare_weights with HTTPS URL returns list."""
        load_config = LoadConfig(load_format="gguf")
        loader = GGUFModelLoader(load_config)

        mock_hf_download.return_value = "/downloaded/model.gguf"

        model_config = MagicMock()
        model_config.model = "https://huggingface.co/model.gguf"

        result = loader._prepare_weights(model_config)
        assert result == ["/downloaded/model.gguf"]
        mock_hf_download.assert_called_once_with(
            url="https://huggingface.co/model.gguf"
        )

    @patch("vllm.model_executor.model_loader.gguf_loader.hf_hub_download")
    @patch("os.path.isfile", return_value=False)
    def test_prepare_weights_repo_filename(self, mock_isfile, mock_hf_download):
        """Test _prepare_weights with repo_id/filename.gguf format returns list."""
        load_config = LoadConfig(load_format="gguf")
        loader = GGUFModelLoader(load_config)

        mock_hf_download.return_value = "/downloaded/model.gguf"

        model_config = MagicMock()
        model_config.model = "unsloth/Qwen3-0.6B-GGUF/model.gguf"

        with patch(
            "vllm.transformers_utils.gguf_utils.Path.is_file",
            return_value=False,
        ):
            result = loader._prepare_weights(model_config)
            assert result == ["/downloaded/model.gguf"]
            mock_hf_download.assert_called_once_with(
                repo_id="unsloth/Qwen3-0.6B-GGUF", filename="model.gguf"
            )

    @patch("vllm.config.model.get_hf_image_processor_config", return_value=None)
    @patch("vllm.transformers_utils.config.file_or_path_exists", return_value=True)
    @patch("vllm.config.model.get_config")
    @patch("vllm.config.model.is_gguf", return_value=True)
    @patch("vllm.model_executor.model_loader.gguf_loader.download_gguf")
    @patch("os.path.isfile", return_value=False)
    def test_prepare_weights_repo_quant_type(
        self,
        mock_isfile,
        mock_download_gguf,
        mock_is_gguf,
        mock_get_config,
        mock_file_exists,
        mock_get_image_config,
    ):
        """Test _prepare_weights with repo_id:quant_type format returns list."""
        mock_hf_config = MagicMock()
        mock_hf_config.architectures = ["Qwen3ForCausalLM"]

        class MockTextConfig:
            max_position_embeddings = 4096
            sliding_window = None
            model_type = "qwen3"
            num_attention_heads = 32

        mock_text_config = MockTextConfig()
        mock_hf_config.get_text_config.return_value = mock_text_config
        mock_hf_config.dtype = "bfloat16"
        mock_get_config.return_value = mock_hf_config

        load_config = LoadConfig(load_format="gguf")
        loader = GGUFModelLoader(load_config)

        # download_gguf now returns list[str]
        mock_download_gguf.return_value = ["/downloaded/model-IQ1_S.gguf"]

        model_config = ModelConfig(
            model="unsloth/Qwen3-0.6B-GGUF:IQ1_S", tokenizer="Qwen/Qwen3-0.6B"
        )
        result = loader._prepare_weights(model_config)
        assert result == ["/downloaded/model-IQ1_S.gguf"]
        mock_download_gguf.assert_called_once_with(
            "unsloth/Qwen3-0.6B-GGUF",
            "IQ1_S",
            cache_dir=None,
            revision=None,
            ignore_patterns=["original/**/*"],
        )

    @patch("vllm.config.model.get_hf_image_processor_config", return_value=None)
    @patch("vllm.config.model.get_config")
    @patch("vllm.config.model.is_gguf", return_value=False)
    @patch("vllm.transformers_utils.gguf_utils.check_gguf_file", return_value=False)
    @patch("os.path.isfile", return_value=False)
    def test_prepare_weights_invalid_format(
        self,
        mock_isfile,
        mock_check_gguf,
        mock_is_gguf,
        mock_get_config,
        mock_get_image_config,
    ):
        """Test _prepare_weights with invalid format."""
        mock_hf_config = MagicMock()
        mock_hf_config.architectures = ["Qwen3ForCausalLM"]

        class MockTextConfig:
            max_position_embeddings = 4096
            sliding_window = None
            model_type = "qwen3"
            num_attention_heads = 32

        mock_text_config = MockTextConfig()
        mock_hf_config.get_text_config.return_value = mock_text_config
        mock_hf_config.dtype = "bfloat16"
        mock_get_config.return_value = mock_hf_config

        load_config = LoadConfig(load_format="gguf")
        loader = GGUFModelLoader(load_config)

        # Create ModelConfig with a valid repo_id to avoid validation errors
        # Then test _prepare_weights with invalid format
        model_config = ModelConfig(model="unsloth/Qwen3-0.6B")
        # Manually set model to invalid format after creation
        model_config.model = "invalid-format"
        with pytest.raises(ValueError, match="Unrecognised GGUF reference"):
            loader._prepare_weights(model_config)
