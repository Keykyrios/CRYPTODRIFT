"""
Test suite for CryptoDrift.

Tests the critical components without requiring GPU or model loading.
Uses pure Python/numpy only — no external API calls.
"""

import json
import ast
import sys
import os
import tempfile

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


# ═══════════════════════════════════════════════════════════════════
# Test 1: JSON-RPC 2.0 Protocol
# ═══════════════════════════════════════════════════════════════════

def test_jsonrpc_request():
    from src.mcp.jsonrpc import JsonRpcRequest, JSONRPC_VERSION

    req = JsonRpcRequest(method="test_method", params={"key": "value"}, id=1)
    d = req.to_dict()

    assert d["jsonrpc"] == "2.0", f"Expected '2.0', got '{d['jsonrpc']}'"
    assert d["method"] == "test_method"
    assert d["params"] == {"key": "value"}
    assert d["id"] == 1
    assert not req.is_notification

    # JSON serialization roundtrip
    raw = req.to_json()
    assert isinstance(raw, str)
    parsed = json.loads(raw)
    assert parsed["jsonrpc"] == JSONRPC_VERSION
    print("  ✓ JsonRpcRequest creation and serialization")


def test_jsonrpc_notification():
    from src.mcp.jsonrpc import JsonRpcRequest

    notif = JsonRpcRequest(method="notifications/initialized", params={})
    assert notif.is_notification
    d = notif.to_dict()
    assert "id" not in d or d.get("id") is None
    print("  ✓ JsonRpcRequest notification (no id)")


def test_jsonrpc_response():
    from src.mcp.jsonrpc import JsonRpcResponse, JsonRpcError

    # Success response
    resp = JsonRpcResponse.success(id=1, result={"status": "ok"})
    assert not resp.is_error
    d = resp.to_dict()
    assert d["result"] == {"status": "ok"}
    assert d["id"] == 1

    # Error response
    err = JsonRpcError.method_not_found("test")
    err_resp = JsonRpcResponse.error_response(id=2, error=err)
    assert err_resp.is_error
    assert err_resp.error.code == -32601
    print("  ✓ JsonRpcResponse success and error")


def test_jsonrpc_response_mutual_exclusion():
    from src.mcp.jsonrpc import JsonRpcResponse, JsonRpcError, JsonRpcValidationError

    # Spec: response MUST contain result OR error, NOT both
    try:
        JsonRpcResponse(
            id=1,
            result="something",
            error=JsonRpcError(code=-1, message="err"),
        )
        assert False, "Should have raised"
    except JsonRpcValidationError:
        pass
    print("  ✓ JsonRpcResponse rejects both result and error")


def test_jsonrpc_parse_message():
    from src.mcp.jsonrpc import parse_message, JsonRpcRequest, JsonRpcResponse

    # Parse request
    raw = '{"jsonrpc":"2.0","method":"test","id":1}'
    msg = parse_message(raw)
    assert isinstance(msg, JsonRpcRequest)
    assert msg.method == "test"

    # Parse response
    raw = '{"jsonrpc":"2.0","result":"ok","id":1}'
    msg = parse_message(raw)
    assert isinstance(msg, JsonRpcResponse)
    assert msg.result == "ok"
    print("  ✓ parse_message correctly distinguishes requests and responses")


def test_jsonrpc_version_validation():
    from src.mcp.jsonrpc import parse_message, JsonRpcValidationError

    # Wrong version should fail
    try:
        parse_message('{"jsonrpc":"1.0","method":"test","id":1}')
        assert False, "Should have raised"
    except JsonRpcValidationError as e:
        assert "2.0" in str(e)
    print("  ✓ Version validation rejects non-2.0")


# ═══════════════════════════════════════════════════════════════════
# Test 2: MCP Messages
# ═══════════════════════════════════════════════════════════════════

def test_mcp_initialize_messages():
    from src.mcp.jsonrpc import IdGenerator
    from src.mcp.messages import (
        make_initialize_request,
        make_initialize_response,
        make_initialized_notification,
        MCP_PROTOCOL_VERSION,
    )

    id_gen = IdGenerator()

    # Initialize request
    req = make_initialize_request(id_gen)
    assert req.method == "initialize"
    assert req.id is not None
    assert req.params["protocolVersion"] == MCP_PROTOCOL_VERSION

    # Initialize response
    resp = make_initialize_response(req.id)
    assert resp.result["protocolVersion"] == MCP_PROTOCOL_VERSION
    assert "capabilities" in resp.result
    assert "serverInfo" in resp.result

    # Initialized notification (no id)
    notif = make_initialized_notification()
    assert notif.is_notification
    assert notif.method == "notifications/initialized"
    print("  ✓ MCP initialize handshake messages")


def test_mcp_tool_messages():
    from src.mcp.jsonrpc import IdGenerator
    from src.mcp.messages import (
        make_tools_list_request,
        make_tool_call_request,
        REFINE_CODE_TOOL,
    )

    id_gen = IdGenerator()

    # Tools list
    req = make_tools_list_request(id_gen)
    assert req.method == "tools/list"

    # Tool call
    call = make_tool_call_request(
        id_gen,
        tool_name="refine_code",
        arguments={"code": "print(1)", "iteration": 0, "strategy": "SF"},
    )
    assert call.method == "tools/call"
    assert call.params["name"] == "refine_code"
    assert call.params["arguments"]["strategy"] == "SF"

    # Verify REFINE_CODE_TOOL has valid JSON Schema
    schema = REFINE_CODE_TOOL["inputSchema"]
    assert schema["type"] == "object"
    assert "code" in schema["properties"]
    assert "iteration" in schema["properties"]
    assert "strategy" in schema["properties"]
    assert set(schema["required"]) == {"code", "iteration", "strategy"}
    print("  ✓ MCP tool call messages and schema")


def test_mcp_interceptor():
    from src.mcp.messages import MessageInterceptor

    interceptor = MessageInterceptor()
    msg1 = interceptor.intercept("client_to_server", '{"test":1}')
    msg2 = interceptor.intercept("server_to_client", '{"test":2}')

    assert len(interceptor.get_messages()) == 2
    assert len(interceptor.get_messages("client_to_server")) == 1
    assert msg1.size_bytes == len('{"test":1}'.encode("utf-8"))
    assert interceptor.total_bytes() > 0

    interceptor.clear()
    assert len(interceptor.get_messages()) == 0
    print("  ✓ MessageInterceptor capture and filtering")


# ═══════════════════════════════════════════════════════════════════
# Test 3: Vulnerability Detection
# ═══════════════════════════════════════════════════════════════════

def test_iv_reuse_detection():
    from src.vulns.rules.iv_reuse import IVReuseRule

    rule = IVReuseRule()

    # Should detect hardcoded IV
    code = '''
from Crypto.Cipher import AES
iv = b"0123456789abcdef"
cipher = AES.new(key, AES.MODE_CBC, iv=iv)
'''
    tree = ast.parse(code)
    rule.visit(tree)
    assert len(rule.findings) > 0, "Should detect hardcoded IV"
    assert any(f.vuln_type == "IV_REUSE" for f in rule.findings)
    print("  ✓ IV_REUSE detects hardcoded IV")


def test_weak_kdf_detection():
    from src.vulns.rules.weak_kdf import WeakKDFRule

    rule = WeakKDFRule()

    # Should detect low PBKDF2 iterations
    code = '''
import hashlib
key = hashlib.pbkdf2_hmac("sha256", password, salt, 10000)
'''
    tree = ast.parse(code)
    rule.visit(tree)
    assert len(rule.findings) > 0, "Should detect low PBKDF2 iterations"
    assert any("10000" in f.description for f in rule.findings)
    print("  ✓ WEAK_KDF detects low PBKDF2 iterations (10000 < 600000)")


def test_timing_vuln_detection():
    from src.vulns.rules.timing_vuln import TimingVulnRule

    rule = TimingVulnRule()

    # Should detect timing-unsafe comparison
    code = '''
if computed_mac == expected_mac:
    return True
'''
    tree = ast.parse(code)
    rule.visit(tree)
    assert len(rule.findings) > 0, "Should detect timing-unsafe compare"
    print("  ✓ TIMING_UNSAFE_COMPARE detects == on MAC values")


def test_ecb_mode_detection():
    from src.vulns.rules.ecb_mode import ECBModeRule

    rule = ECBModeRule()

    code = '''
from Crypto.Cipher import AES
cipher = AES.new(key, AES.MODE_ECB)
'''
    tree = ast.parse(code)
    rule.visit(tree)
    assert len(rule.findings) > 0, "Should detect ECB mode"
    print("  ✓ ECB_MODE detects AES.MODE_ECB")


def test_hardcoded_key_detection():
    from src.vulns.rules.hardcoded_key import HardcodedKeyRule

    rule = HardcodedKeyRule()

    code = '''
secret_key = "my-super-secret-key-12345"
'''
    tree = ast.parse(code)
    rule.visit(tree)
    assert len(rule.findings) > 0, "Should detect hardcoded key"
    print("  ✓ HARDCODED_KEY detects literal key assignment")


def test_weak_hash_detection():
    from src.vulns.rules.weak_hash import WeakHashRule

    rule = WeakHashRule()

    code = '''
import hashlib
h = hashlib.md5(data)
'''
    tree = ast.parse(code)
    rule.visit(tree)
    assert len(rule.findings) > 0, "Should detect MD5 usage"
    print("  ✓ WEAK_HASH detects hashlib.md5")


def test_predictable_rng_detection():
    from src.vulns.rules.predictable_rng import PredictableRNGRule

    rule = PredictableRNGRule()

    code = '''
import random
from Crypto.Cipher import AES
key = random.randbytes(32)
'''
    tree = ast.parse(code)
    rule.visit(tree)
    assert len(rule.findings) > 0, "Should detect random.randbytes in crypto context"
    print("  ✓ PREDICTABLE_RNG detects random module in crypto context")


def test_detector_orchestrator():
    from src.vulns.detector import CryptoVulnDetector

    detector = CryptoVulnDetector()

    # Code with multiple vulnerabilities
    bad_code = '''
import hashlib
import random
from Crypto.Cipher import AES

secret_key = b"hardcoded-key-value-here!"
iv = b"0123456789abcdef"
cipher = AES.new(secret_key, AES.MODE_ECB)
h = hashlib.md5(data)
nonce = random.randbytes(12)
'''
    report = detector.analyze(bad_code)
    assert report.total_count >= 3, f"Expected ≥3 vulns, got {report.total_count}"
    assert report.critical_count >= 1, "Should have at least 1 CRITICAL"

    # Clean code should have 0 vulns
    clean_code = '''
import os
from Crypto.Cipher import AES

def encrypt(key, data):
    nonce = os.urandom(12)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    return cipher.encrypt_and_digest(data)
'''
    clean_report = detector.analyze(clean_code)
    assert clean_report.total_count == 0, f"Clean code has {clean_report.total_count} vulns"
    print("  ✓ CryptoVulnDetector orchestrator (multi-rule, clean vs dirty)")


def test_detector_compare():
    from src.vulns.detector import CryptoVulnDetector

    detector = CryptoVulnDetector()

    before = '''
import os
from Crypto.Cipher import AES
nonce = os.urandom(12)
cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
'''
    after = '''
from Crypto.Cipher import AES
iv = b"static_iv_value!"
cipher = AES.new(key, AES.MODE_ECB)
'''
    delta = detector.compare(before, after)
    assert delta.net_delta > 0, "Should show increase in vulns"
    assert len(delta.introduced) > 0, "Should identify introduced vulns"
    print("  ✓ CryptoVulnDetector.compare detects degradation")


# ═══════════════════════════════════════════════════════════════════
# Test 4: Attention Entropy Math
# ═══════════════════════════════════════════════════════════════════

def test_attention_entropy():
    from src.attention.entropy import attention_entropy, normalized_entropy

    # Uniform distribution: max entropy
    uniform = np.ones((1, 8)) / 8  # 8 items, all equal
    h = attention_entropy(uniform)
    expected = np.log2(8)  # = 3.0
    assert abs(h[0] - expected) < 0.01, f"Expected ~3.0, got {h[0]}"

    # Normalized should be ~1.0
    hn = normalized_entropy(uniform)
    assert abs(hn[0] - 1.0) < 0.01, f"Expected ~1.0, got {hn[0]}"

    # Peaked distribution: low entropy
    peaked = np.array([[0.97, 0.01, 0.005, 0.005, 0.003, 0.003, 0.002, 0.002]])
    h_peaked = attention_entropy(peaked)
    assert h_peaked[0] < 1.0, f"Peaked entropy should be low, got {h_peaked[0]}"

    hn_peaked = normalized_entropy(peaked)
    assert hn_peaked[0] < 0.5, f"Peaked normalized should be < 0.5, got {hn_peaked[0]}"
    print("  ✓ Shannon entropy: uniform=max, peaked=low")


def test_per_head_entropy():
    from src.attention.entropy import per_head_entropy

    # Shape: (num_heads, seq_len, seq_len)
    # 4 heads, 8 tokens
    np.random.seed(42)
    weights = np.random.dirichlet(np.ones(8), size=(4, 8))
    # weights shape: (4, 8, 8) — each row sums to 1

    result = per_head_entropy(weights)
    assert result.shape == (4,), f"Expected (4,), got {result.shape}"
    assert all(result > 0), "Entropy should be positive"
    print("  ✓ per_head_entropy shape and positivity")


def test_entropy_collapse_detection():
    from src.attention.entropy import detect_entropy_collapse

    # Simulate entropy history: gradual then sudden drop
    history = [
        np.array([0.8, 0.7, 0.9, 0.6]),  # iter 0
        np.array([0.75, 0.68, 0.85, 0.58]),  # iter 1
        np.array([0.3, 0.65, 0.82, 0.55]),  # iter 2 — head 0 collapses!
    ]

    collapses = detect_entropy_collapse(history, threshold_drop=0.3, window=2)
    assert len(collapses) > 0, "Should detect collapse in head 0"
    assert collapses[0][1] == 0, "Collapsed head should be index 0"
    assert collapses[0][2] > 0.3, "Drop magnitude should exceed threshold"
    print("  ✓ Entropy collapse detection")


# ═══════════════════════════════════════════════════════════════════
# Test 5: Crypto Token Vocabulary
# ═══════════════════════════════════════════════════════════════════

def test_crypto_token_vocab():
    from src.attention.crypto_tokens import (
        ALL_CRYPTO_KEYWORDS,
        CRYPTO_TOKEN_VOCABULARY,
        CRITICAL_TOKEN_PAIRS,
    )

    assert len(ALL_CRYPTO_KEYWORDS) > 20, "Should have >20 crypto keywords"
    assert "AES" in ALL_CRYPTO_KEYWORDS
    assert "urandom" in ALL_CRYPTO_KEYWORDS
    assert len(CRITICAL_TOKEN_PAIRS) > 5, "Should have >5 critical pairs"

    # Verify no duplicates in pairs
    pair_set = set(CRITICAL_TOKEN_PAIRS)
    assert len(pair_set) == len(CRITICAL_TOKEN_PAIRS), "No duplicate pairs"
    print("  ✓ Crypto token vocabulary completeness")


# ═══════════════════════════════════════════════════════════════════
# Test 6: Correlation Engine
# ═══════════════════════════════════════════════════════════════════

def test_correlator():
    from src.correlation.correlator import DriftCorrelator, IterationData

    correlator = DriftCorrelator(
        entropy_low_absolute=0.3,
    )

    # Build iteration data simulating degradation
    data = [
        IterationData(
            iteration=0,
            entropy_matrix=np.full((32, 32), 0.7),  # Healthy
            net_vuln_delta=0,
        ),
        IterationData(
            iteration=1,
            entropy_matrix=np.full((32, 32), 0.5),  # Some drop
            net_vuln_delta=0,
        ),
    ]

    # Simulate: iteration 2 has a vulnerability, iteration 1 had low entropy in one head
    entropy_with_collapse = np.full((32, 32), 0.5)
    entropy_with_collapse[15, 7] = 0.1  # Head (15,7) collapses
    data[1].entropy_matrix = entropy_with_collapse

    data.append(IterationData(
        iteration=2,
        entropy_matrix=np.full((32, 32), 0.4),
        vuln_types_introduced=["IV_REUSE"],
        net_vuln_delta=1,
    ))

    result = correlator.correlate("test-session", data)
    assert result.total_vuln_events == 1
    # The collapsed head should be identified as predictive
    assert len(result.predictive_heads) > 0, "Should find predictive heads"

    found_collapsed = any(
        h.layer == 15 and h.head == 7
        for h in result.predictive_heads
    )
    assert found_collapsed, "Should identify head (15,7) as predictive"
    print("  ✓ DriftCorrelator identifies predictive heads")


# ═══════════════════════════════════════════════════════════════════
# Test 7: Early Warning System
# ═══════════════════════════════════════════════════════════════════

def test_early_warning():
    from src.correlation.early_warning import (
        EarlyWarningSystem,
        PredictiveSignature,
    )

    ews = EarlyWarningSystem()

    # Add a known signature
    sig = PredictiveSignature(
        name="test_sig",
        layer=10,
        head=5,
        signal_type="entropy_collapse",
        vuln_type="IV_REUSE",
        threshold=0.3,
        confidence=0.8,
    )
    ews.add_signature(sig)

    # Check with healthy entropy (no warning)
    healthy = np.full((32, 32), 0.7)
    warnings = ews.check(healthy, iteration=0)
    assert len(warnings) == 0, "No warnings for healthy state"

    # Check with collapsed entropy (should warn)
    collapsed = np.full((32, 32), 0.7)
    collapsed[10, 5] = 0.15  # Below threshold
    warnings = ews.check(collapsed, iteration=1)
    assert len(warnings) > 0, "Should warn on entropy collapse"
    assert warnings[0].predicted_vuln_type == "IV_REUSE"
    assert warnings[0].severity in ("CRITICAL", "HIGH")
    print("  ✓ EarlyWarningSystem signature-based detection")


# ═══════════════════════════════════════════════════════════════════
# Test 8: SQLite Storage
# ═══════════════════════════════════════════════════════════════════

def test_database():
    from src.storage.database import CryptoDriftDB

    # Use temp file
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        db = CryptoDriftDB(db_path)
        db.connect()

        # Insert session
        db.insert_session(
            session_id="test-123",
            start_time=1000.0,
            strategy="SF",
            corpus_sample="aes_gcm_basic",
        )

        # Retrieve session
        session = db.get_session("test-123")
        assert session is not None
        assert session["strategy"] == "SF"
        assert session["corpus_sample"] == "aes_gcm_basic"

        # Insert iteration
        iter_id = db.insert_iteration(
            session_id="test-123",
            iteration_num=0,
            code_before="before",
            code_after="after",
            prompt="test prompt",
            timestamp=1001.0,
        )
        assert iter_id > 0

        # Insert vuln finding
        db.insert_vuln_finding(
            iteration_id=iter_id,
            vuln_type="IV_REUSE",
            severity="CRITICAL",
            line=5,
            col=0,
            description="test finding",
        )

        # List sessions
        sessions = db.list_sessions()
        assert len(sessions) == 1

        # Get iterations
        iters = db.get_iterations("test-123")
        assert len(iters) == 1

        db.close()
        print("  ✓ SQLite CRUD operations")
    finally:
        os.unlink(db_path)


# ═══════════════════════════════════════════════════════════════════
# Test 9: Complexity Tracking
# ═══════════════════════════════════════════════════════════════════

def test_complexity():
    from src.vulns.complexity import compute_complexity, compute_complexity_delta

    code = '''
def encrypt(key, data):
    if not key:
        raise ValueError("no key")
    for block in data:
        if block:
            result = process(block)
    return result
'''
    metrics = compute_complexity(code)
    assert metrics.num_functions == 1
    assert metrics.lines_of_code > 0
    assert metrics.cyclomatic_complexity > 0
    print("  ✓ Complexity metrics computation")


def test_complexity_delta():
    from src.vulns.complexity import compute_complexity_delta

    before = "def f():\n    return 1\n"
    after = "def f():\n    if x:\n        return 1\n    else:\n        return 2\n"
    delta = compute_complexity_delta(before, after)
    assert delta["cc_delta"] > 0, "Adding branches should increase CC"
    print("  ✓ Complexity delta computation")


# ═══════════════════════════════════════════════════════════════════
# Test 10: Experiment Strategies
# ═══════════════════════════════════════════════════════════════════

def test_strategies():
    from src.experiment.strategies import get_strategy, STRATEGIES

    assert len(STRATEGIES) == 4
    for code in ["EF", "FF", "SF", "AI"]:
        s = get_strategy(code)
        assert s.code == code
        assert len(s.prompt) > 10, f"Strategy {code} prompt too short"

    # Verify SF is the paradoxical one
    sf = get_strategy("SF")
    assert "PARADOX" in sf.description.upper(), "SF should document the paradox"
    print("  ✓ All 4 strategies defined with prompts")


# ═══════════════════════════════════════════════════════════════════
# Test 11: Corpus
# ═══════════════════════════════════════════════════════════════════

def test_corpus():
    from src.experiment.corpus import get_corpus, ALL_SAMPLES

    samples = get_corpus()
    assert len(samples) >= 6, f"Expected ≥6 samples, got {len(samples)}"

    for sample in samples:
        # Verify each sample is valid Python
        try:
            ast.parse(sample.code)
        except SyntaxError as e:
            assert False, f"Sample {sample.name} has SyntaxError: {e}"

        # Verify expected_vulns == 0 (baseline should be clean)
        assert sample.expected_vulns == 0, (
            f"Sample {sample.name} should have 0 expected vulns"
        )
    print("  ✓ All corpus samples are valid Python with 0 expected vulns")


def test_corpus_no_vulns():
    """Verify our detector agrees the corpus is clean."""
    from src.experiment.corpus import get_corpus
    from src.vulns.detector import CryptoVulnDetector

    detector = CryptoVulnDetector()
    for sample in get_corpus():
        report = detector.analyze(sample.code)
        if report.total_count > 0:
            print(f"  ⚠ Sample {sample.name} has {report.total_count} vulns:")
            for f in report.findings:
                print(f"    - {f.vuln_type}: {f.description}")
        # Some baseline samples may trigger heuristic rules but
        # they should not have CRITICAL issues
        critical = [f for f in report.findings if f.severity.value == "CRITICAL"]
        assert len(critical) == 0, (
            f"Sample {sample.name} has CRITICAL vulns: "
            f"{[f.description for f in critical]}"
        )
    print("  ✓ No CRITICAL vulns in baseline corpus")


# ═══════════════════════════════════════════════════════════════════
# Test 12: MCP Session Management
# ═══════════════════════════════════════════════════════════════════

def test_session_manager():
    from src.mcp.session import SessionManager, IterationRecord

    mgr = SessionManager()
    session = mgr.create_session(
        strategy="SF",
        corpus_sample="test_sample",
        corpus_category="aes_gcm",
    )

    assert mgr.active_session is not None
    assert session.status == "running"
    assert session.current_iteration == 0

    # Add iteration
    record = IterationRecord(
        iteration_num=0,
        code_before="original",
        code_after="refined",
        prompt="improve security",
        strategy="SF",
    )
    session.add_iteration(record)
    assert session.current_iteration == 1
    assert session.current_code == "refined"
    assert session.initial_code == "original"

    # Complete
    session.complete()
    assert session.status == "completed"
    assert session.end_time is not None
    assert session.duration > 0
    print("  ✓ Session lifecycle management")


# ═══════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("CryptoDrift Test Suite")
    print("=" * 60)

    tests = [
        ("JSON-RPC 2.0", [
            test_jsonrpc_request,
            test_jsonrpc_notification,
            test_jsonrpc_response,
            test_jsonrpc_response_mutual_exclusion,
            test_jsonrpc_parse_message,
            test_jsonrpc_version_validation,
        ]),
        ("MCP Messages", [
            test_mcp_initialize_messages,
            test_mcp_tool_messages,
            test_mcp_interceptor,
        ]),
        ("Vulnerability Detection", [
            test_iv_reuse_detection,
            test_weak_kdf_detection,
            test_timing_vuln_detection,
            test_ecb_mode_detection,
            test_hardcoded_key_detection,
            test_weak_hash_detection,
            test_predictable_rng_detection,
            test_detector_orchestrator,
            test_detector_compare,
        ]),
        ("Attention Entropy", [
            test_attention_entropy,
            test_per_head_entropy,
            test_entropy_collapse_detection,
        ]),
        ("Crypto Tokens", [
            test_crypto_token_vocab,
        ]),
        ("Correlation Engine", [
            test_correlator,
        ]),
        ("Early Warning", [
            test_early_warning,
        ]),
        ("SQLite Storage", [
            test_database,
        ]),
        ("Complexity", [
            test_complexity,
            test_complexity_delta,
        ]),
        ("Strategies", [
            test_strategies,
        ]),
        ("Corpus", [
            test_corpus,
            test_corpus_no_vulns,
        ]),
        ("Session Management", [
            test_session_manager,
        ]),
    ]

    total_passed = 0
    total_failed = 0
    failures = []

    for group_name, test_fns in tests:
        print(f"\n── {group_name} ──")
        for fn in test_fns:
            try:
                fn()
                total_passed += 1
            except Exception as e:
                total_failed += 1
                failures.append((fn.__name__, str(e)))
                print(f"  ✗ {fn.__name__}: {e}")

    print("\n" + "=" * 60)
    print(f"Results: {total_passed} passed, {total_failed} failed")
    if failures:
        print("\nFailures:")
        for name, err in failures:
            print(f"  - {name}: {err}")
    print("=" * 60)

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
