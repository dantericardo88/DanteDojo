"""5 adversarial bypass vectors for BlockQuantizer — all must be blocked."""
import pytest
from dojo.stream.block_quantizer import BlockQuantizer, BlockQuantizerError


class TestAdversarialBypassVectors:
    def setup_method(self):
        self.quantizer = BlockQuantizer(precision_floor="4-bit", mock_mode=True)

    def test_vector1_direct_floor_violation(self):
        """Vector 1: Direct attempt to quantize below precision floor."""
        with pytest.raises(BlockQuantizerError, match="Precision floor violation"):
            self.quantizer.quantize_block("WEIGHT_DATA", "GPTQ", "2-bit")

    def test_vector2_unicode_homoglyph_precision(self):
        """Vector 2: Unicode homoglyph hyphen in precision string — must not bypass floor."""
        # U+2010 HYPHEN looks like ASCII hyphen but is not in _PRECISION_RANK
        with pytest.raises(BlockQuantizerError):
            self.quantizer.quantize_block("WEIGHT_DATA", "GPTQ", "4‐bit")

    def test_vector3_empty_block_bypasses_nothing(self):
        """Vector 3: Empty block — must fail before any floor check."""
        with pytest.raises(BlockQuantizerError, match="cannot be empty"):
            self.quantizer.quantize_block("", "GPTQ", "2-bit")

    def test_vector4_format_injection_blocked(self):
        """Vector 4: SQL-injection-style format string — blocked by format allowlist."""
        with pytest.raises(BlockQuantizerError, match="format must be one of"):
            self.quantizer.quantize_block("WEIGHT_DATA", "GPTQ; DROP TABLE models;--", "4-bit")

    def test_vector5_unknown_precision_below_floor(self):
        """Vector 5: Unknown precision string gets rank 0, below any floor — blocked."""
        with pytest.raises(BlockQuantizerError):
            self.quantizer.quantize_block("WEIGHT_DATA", "GPTQ", "1-bit")
