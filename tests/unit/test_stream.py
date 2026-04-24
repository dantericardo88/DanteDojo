"""Unit tests for dojo/stream/ layer streaming organ."""

import pytest

from dojo.core.evidence import EvidenceLogger
from dojo.stream import (
    BlockQuantizer,
    BlockQuantizerError,
    KVCacheStreamer,
    KVCacheStreamerError,
    MetaDeviceManager,
    MetaDeviceManagerError,
    PrefetchGovernor,
    PrefetchGovernorError,
    StreamBalancer,
    StreamBalancerError,
    WeightStreamer,
    WeightStreamerError,
)
from dojo.stream.evidence_sovereign import EvidenceSovereign


@pytest.fixture(autouse=True)
def clear_logger():
    EvidenceLogger.clear()
    yield
    EvidenceLogger.clear()


# ---------------------------------------------------------------------------
# EvidenceSovereign
# ---------------------------------------------------------------------------

class TestEvidenceSovereign:
    def test_seal_returns_hash(self):
        s = EvidenceSovereign()
        result = s.seal_operation({"op": "test"})
        assert "operation_hash" in result
        assert len(result["operation_hash"]) == 64

    def test_seal_deterministic(self):
        s = EvidenceSovereign()
        r1 = s.seal_operation({"op": "test"})
        r2 = s.seal_operation({"op": "test"})
        assert r1["operation_hash"] == r2["operation_hash"]

    def test_trigger_halt(self):
        s = EvidenceSovereign()
        result = s.trigger_halt("breach")
        assert result["status"] == "Halted"
        assert result["reason"] == "breach"
        assert s.state == "Halting"


# ---------------------------------------------------------------------------
# WeightStreamer
# ---------------------------------------------------------------------------

class TestWeightStreamer:
    def test_stream_layer_mock(self):
        ws = WeightStreamer(mock_mode=True)
        result = ws.stream_layer("layer_0", "disk")
        assert result.status == "LayerReady"
        assert result.layer_id == "layer_0"

    def test_single_layer_residency_enforced(self):
        ws = WeightStreamer(mock_mode=True)
        ws.stream_layer("layer_0")
        with pytest.raises(WeightStreamerError, match="residency"):
            ws.stream_layer("layer_1")

    def test_evict_clears_active(self):
        ws = WeightStreamer(mock_mode=True)
        ws.stream_layer("layer_0")
        evicted = ws.evict_current()
        assert evicted == "layer_0"
        assert ws.active_layer is None

    def test_evict_none_when_empty(self):
        ws = WeightStreamer(mock_mode=True)
        assert ws.evict_current() is None

    def test_invalid_source_tier(self):
        ws = WeightStreamer(mock_mode=True)
        with pytest.raises(WeightStreamerError):
            ws.stream_layer("layer_0", "invalid_tier")

    def test_empty_layer_id_raises(self):
        ws = WeightStreamer(mock_mode=True)
        with pytest.raises(WeightStreamerError):
            ws.stream_layer("")

    def test_load_model_mock(self):
        ws = WeightStreamer(mock_mode=True)
        result = ws.load_model("some/model")
        assert result["status"] == "MockModelLoaded"
        assert result["mock"] is True

    def test_stats_increment(self):
        ws = WeightStreamer(mock_mode=True)
        ws.stream_layer("layer_0")
        ws.evict_current()
        ws.stream_layer("layer_1")
        assert ws.get_stats()["layers_streamed"] == 2

    def test_emits_evidence_on_stream(self):
        ws = WeightStreamer(mock_mode=True)
        ws.stream_layer("layer_0")
        events = EvidenceLogger.get_events()
        assert any("stream.layer.loaded" in str(e) for e in events)

    def test_stream_sequence_after_evict(self):
        ws = WeightStreamer(mock_mode=True)
        ws.stream_layer("layer_0")
        ws.evict_current()
        result = ws.stream_layer("layer_0")
        assert result.status == "LayerReady"


# ---------------------------------------------------------------------------
# KVCacheStreamer
# ---------------------------------------------------------------------------

class TestKVCacheStreamer:
    def test_page_kv_entry_mock(self):
        kv = KVCacheStreamer(max_cache_size=1000, mock_mode=True)
        result = kv.page_kv_entry("seq_1", "value", 100)
        assert result.status == "KVReady"
        assert result.key == "seq_1"

    def test_sovereign_size_enforced(self):
        kv = KVCacheStreamer(max_cache_size=50, mock_mode=True)
        with pytest.raises(KVCacheStreamerError, match="Sovereign"):
            kv.page_kv_entry("big", "value", 100)

    def test_empty_key_raises(self):
        kv = KVCacheStreamer(mock_mode=True)
        with pytest.raises(KVCacheStreamerError):
            kv.page_kv_entry("", "v", 10)

    def test_zero_size_raises(self):
        kv = KVCacheStreamer(mock_mode=True)
        with pytest.raises(KVCacheStreamerError):
            kv.page_kv_entry("k", "v", 0)

    def test_negative_size_raises(self):
        kv = KVCacheStreamer(mock_mode=True)
        with pytest.raises(KVCacheStreamerError):
            kv.page_kv_entry("k", "v", -5)

    def test_stats_accumulate(self):
        kv = KVCacheStreamer(max_cache_size=1000, mock_mode=True)
        kv.page_kv_entry("a", "v", 10)
        kv.page_kv_entry("b", "v", 20)
        stats = kv.get_stats()
        assert stats["pages_stored"] == 2
        assert stats["current_size"] == 30

    def test_cache_to_cpu_mock(self):
        kv = KVCacheStreamer(mock_mode=True)
        kv.page_kv_entry("k1", "v", 10)
        moved = kv.cache_to_cpu("k1")
        assert "k1" in moved

    def test_cache_to_gpu_mock(self):
        kv = KVCacheStreamer(mock_mode=True)
        kv.page_kv_entry("k1", "v", 10)
        moved = kv.cache_to_gpu("k1")
        assert "k1" in moved

    def test_get_cache_devices(self):
        kv = KVCacheStreamer(mock_mode=True)
        kv.page_kv_entry("k1", "v", 10)
        devices = kv.get_cache_devices()
        assert "k1" in devices


# ---------------------------------------------------------------------------
# PrefetchGovernor
# ---------------------------------------------------------------------------

class TestPrefetchGovernor:
    def test_issue_prefetch_mock(self):
        pg = PrefetchGovernor(max_lookahead=3, mock_mode=True)
        result = pg.issue_prefetch(["layer_1", "layer_2"])
        assert result.status == "PrefetchIssued"
        assert len(result.layers_issued) == 2

    def test_lookahead_overrun_raises(self):
        pg = PrefetchGovernor(max_lookahead=2, mock_mode=True)
        with pytest.raises(PrefetchGovernorError, match="overrun"):
            pg.issue_prefetch(["a", "b", "c"])

    def test_empty_list_raises(self):
        pg = PrefetchGovernor(mock_mode=True)
        with pytest.raises(PrefetchGovernorError):
            pg.issue_prefetch([])

    def test_invalid_lookahead_raises(self):
        with pytest.raises(PrefetchGovernorError):
            PrefetchGovernor(max_lookahead=0)
        with pytest.raises(PrefetchGovernorError):
            PrefetchGovernor(max_lookahead=17)

    def test_consume_prefetch(self):
        pg = PrefetchGovernor(mock_mode=True)
        pg.issue_prefetch(["layer_1"])
        assert pg.consume_prefetch("layer_1") is True
        assert pg.consume_prefetch("layer_1") is False

    def test_stats(self):
        pg = PrefetchGovernor(max_lookahead=5, mock_mode=True)
        pg.issue_prefetch(["a", "b"])
        stats = pg.get_stats()
        assert stats["prefetches_issued"] == 2
        assert stats["pending_count"] == 2


# ---------------------------------------------------------------------------
# StreamBalancer
# ---------------------------------------------------------------------------

class TestStreamBalancer:
    def test_start_overlap_mock(self):
        sb = StreamBalancer(max_streams=2, mock_mode=True)
        result = sb.start_overlap("matmul", "load_weights")
        assert result.status == "OverlapComplete"

    def test_stream_exhaustion_raises(self):
        sb = StreamBalancer(max_streams=1, mock_mode=True)
        sb.start_overlap("op1", "mem1")
        with pytest.raises(StreamBalancerError, match="exhaustion"):
            sb.start_overlap("op2", "mem2")

    def test_empty_compute_op_raises(self):
        sb = StreamBalancer(mock_mode=True)
        with pytest.raises(StreamBalancerError):
            sb.start_overlap("", "mem")

    def test_empty_memory_op_raises(self):
        sb = StreamBalancer(mock_mode=True)
        with pytest.raises(StreamBalancerError):
            sb.start_overlap("op", "")

    def test_end_overlap_releases_slot(self):
        sb = StreamBalancer(max_streams=1, mock_mode=True)
        sb.start_overlap("op", "mem")
        sb.end_overlap()
        result = sb.start_overlap("op2", "mem2")
        assert result.status == "OverlapComplete"

    def test_stats(self):
        sb = StreamBalancer(max_streams=3, mock_mode=True)
        sb.start_overlap("op1", "mem1")
        sb.start_overlap("op2", "mem2")
        stats = sb.get_stats()
        assert stats["overlaps_completed"] == 2


# ---------------------------------------------------------------------------
# MetaDeviceManager
# ---------------------------------------------------------------------------

class TestMetaDeviceManager:
    def test_load_model_lazy_mock(self):
        mdm = MetaDeviceManager(mock_mode=True)
        result = mdm.load_model_lazy("some/model")
        assert result["status"] == "MockLazyLoaded"
        assert result["device"] == "meta"

    def test_allocate_full_model_always_raises(self):
        mdm = MetaDeviceManager(mock_mode=True)
        with pytest.raises(MetaDeviceManagerError, match="single-layer residency"):
            mdm.allocate_full_model()

    def test_forward_tensor_op_mock(self):
        mdm = MetaDeviceManager(mock_mode=True)
        result = mdm.forward_tensor_op("forward", None)
        assert result.status == "RoutedOp"
        assert result.op == "forward"

    def test_empty_op_raises(self):
        mdm = MetaDeviceManager(mock_mode=True)
        with pytest.raises(MetaDeviceManagerError):
            mdm.forward_tensor_op("", None)

    def test_stats(self):
        mdm = MetaDeviceManager(mock_mode=True)
        mdm.forward_tensor_op("forward", None)
        mdm.forward_tensor_op("attention", None)
        assert mdm.get_stats()["ops_routed"] == 2


# ---------------------------------------------------------------------------
# BlockQuantizer
# ---------------------------------------------------------------------------

class TestBlockQuantizer:
    def test_quantize_block_mock(self):
        bq = BlockQuantizer(precision_floor="4-bit", mock_mode=True)
        result = bq.quantize_block("raw_data", "GGUF", "8-bit")
        assert result.status == "QuantizedBlock"

    def test_precision_floor_enforced(self):
        bq = BlockQuantizer(precision_floor="8-bit", mock_mode=True)
        with pytest.raises(BlockQuantizerError, match="floor"):
            bq.quantize_block("raw", "GGUF", "4-bit")

    def test_invalid_precision_raises(self):
        bq = BlockQuantizer(mock_mode=True)
        with pytest.raises(BlockQuantizerError):
            bq.quantize_block("raw", "GGUF", "1-bit")

    def test_empty_block_raises(self):
        bq = BlockQuantizer(mock_mode=True)
        with pytest.raises(BlockQuantizerError):
            bq.quantize_block("", "GGUF", "8-bit")

    def test_invalid_floor_raises(self):
        with pytest.raises(BlockQuantizerError):
            BlockQuantizer(precision_floor="3-bit")

    def test_quantize_tensor_mock(self):
        bq = BlockQuantizer(mock_mode=True)
        result = bq.quantize_tensor("fake_tensor", "8-bit")
        assert result == "fake_tensor"

    def test_tensor_floor_violation(self):
        bq = BlockQuantizer(precision_floor="8-bit", mock_mode=True)
        with pytest.raises(BlockQuantizerError, match="floor"):
            bq.quantize_tensor("t", "4-bit")

    def test_stats_increment(self):
        bq = BlockQuantizer(mock_mode=True)
        bq.quantize_block("raw", "GGUF", "8-bit")
        bq.quantize_block("raw", "GPTQ", "4-bit")
        assert bq.get_stats()["blocks_quantized"] == 2
