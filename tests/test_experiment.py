"""
Tests for the Consistency Heatmap experiment.

Test spec:
1. run_experiment.py with mock API responses → results/variance.json created with entries per model.
2. variance.json → Contains keys for at least 3 models.
3. generate_heatmap.py → heatmap.png exists and file size > 10KB.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from consistency_heatmap_ import SEED_PROMPTS
from consistency_heatmap_ import compute_variance, normalized_levenshtein, _levenshtein_distance
from generate_heatmap import build_matrix, generate_heatmap, load_variance


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_results(tmp_path):
    """Temporary results directory."""
    return tmp_path / "results"


@pytest.fixture()
def sample_variance_data():
    """Minimal valid variance data for 3 models, 50 prompts."""
    import random
    rng = random.Random(42)
    data = {}
    for model in ["qwen", "mistral", "gpt"]:
        data[model] = {str(i): rng.uniform(0.05, 0.45) for i in range(50)}
    return data


@pytest.fixture()
def variance_json_file(tmp_results, sample_variance_data):
    """Write sample variance data to a temp file and return the path."""
    tmp_results.mkdir(parents=True, exist_ok=True)
    p = tmp_results / "variance.json"
    with open(p, "w") as f:
        json.dump(sample_variance_data, f)
    return p


# ---------------------------------------------------------------------------
# 1. Prompts
# ---------------------------------------------------------------------------

class TestSeedPrompts:
    def test_exactly_50_prompts(self):
        assert len(SEED_PROMPTS) == 50

    def test_all_strings(self):
        assert all(isinstance(p, str) for p in SEED_PROMPTS)

    def test_no_empty_prompts(self):
        assert all(len(p.strip()) > 0 for p in SEED_PROMPTS)

    def test_all_unique(self):
        assert len(set(SEED_PROMPTS)) == len(SEED_PROMPTS)


# ---------------------------------------------------------------------------
# 2. Variance / Levenshtein utilities
# ---------------------------------------------------------------------------

class TestLevenshtein:
    def test_identical_strings_zero(self):
        assert _levenshtein_distance("hello", "hello") == 0

    def test_empty_strings_zero(self):
        assert _levenshtein_distance("", "") == 0

    def test_one_empty(self):
        assert _levenshtein_distance("", "abc") == 3
        assert _levenshtein_distance("abc", "") == 3

    def test_single_substitution(self):
        assert _levenshtein_distance("cat", "bat") == 1

    def test_insertion(self):
        assert _levenshtein_distance("cat", "cats") == 1


class TestNormalizedLevenshtein:
    def test_identical_is_zero(self):
        assert normalized_levenshtein("hello", "hello") == 0.0

    def test_both_empty_is_zero(self):
        assert normalized_levenshtein("", "") == 0.0

    def test_completely_different(self):
        # "a" vs "b": distance=1, max_len=1 → 1.0
        assert normalized_levenshtein("a", "b") == 1.0

    def test_result_in_unit_interval(self):
        val = normalized_levenshtein("The sky is blue", "The ocean is vast and blue")
        assert 0.0 <= val <= 1.0


class TestComputeVariance:
    def test_identical_responses_zero_variance(self):
        responses = ["The answer is X."] * 10
        assert compute_variance(responses) == 0.0

    def test_single_response_zero_variance(self):
        assert compute_variance(["only one"]) == 0.0

    def test_empty_list_zero_variance(self):
        assert compute_variance([]) == 0.0

    def test_variance_in_unit_interval(self):
        responses = [f"Response number {i} is different." for i in range(10)]
        v = compute_variance(responses)
        assert 0.0 <= v <= 1.0

    def test_high_variance_higher_than_low(self):
        consistent = ["Photosynthesis converts sunlight to energy."] * 10
        inconsistent = [
            "The moon is large.", "Dogs are mammals.", "Water boils at 100C.",
            "DNA is a molecule.", "Paris is the capital.", "Light is fast.",
            "Trees grow tall.", "Math is universal.", "Stars burn bright.",
            "Time moves forward.",
        ]
        v_consistent = compute_variance(consistent)
        v_inconsistent = compute_variance(inconsistent)
        assert v_inconsistent > v_consistent


# ---------------------------------------------------------------------------
# 3. Experiment runner with mocked API
# ---------------------------------------------------------------------------

def _make_session_mock(response_factory=None):
    """Return a mock requests.Session that fakes API responses."""
    if response_factory is None:
        response_factory = lambda model, prompt, call_idx: f"Mock response {call_idx}"

    session = MagicMock()
    call_counter = {"n": 0}

    def fake_post(url, headers=None, json=None, timeout=None):
        idx = call_counter["n"]
        call_counter["n"] += 1
        model = json.get("model", "unknown") if json else "unknown"
        prompt = (json.get("messages") or [{}])[0].get("content", "") if json else ""
        content = response_factory(model, prompt, idx)
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": content}}]
        }
        return mock_resp

    session.post.side_effect = fake_post
    return session


class TestRunExperiment:
    def test_variance_json_created(self, tmp_results):
        """Test spec 1: run_experiment with mock API → variance.json created."""
        from run_experiment import run_experiment
        from consistency_heatmap_ import MODELS

        session = _make_session_mock()

        result = run_experiment(
            models={"qwen": "qwen/test", "mistral": "mistral/test", "gpt": "gpt/test"},
            prompts=SEED_PROMPTS[:5],  # use 5 prompts for speed
            num_runs=3,
            results_dir=tmp_results,
            session=session,
            verbose=False,
        )

        variance_path = tmp_results / "variance.json"
        assert variance_path.exists(), "variance.json must be created"

    def test_variance_json_entries_per_model(self, tmp_results):
        """Test spec 1: variance.json has entries for each model."""
        from run_experiment import run_experiment

        session = _make_session_mock()
        result = run_experiment(
            models={"qwen": "qwen/test", "mistral": "mistral/test", "gpt": "gpt/test"},
            prompts=SEED_PROMPTS[:5],
            num_runs=3,
            results_dir=tmp_results,
            session=session,
            verbose=False,
        )

        variance_path = tmp_results / "variance.json"
        with open(variance_path) as f:
            data = json.load(f)

        # Each model should have entries for each prompt
        for model in ["qwen", "mistral", "gpt"]:
            assert model in data, f"Model '{model}' missing from variance.json"
            assert len(data[model]) == 5, f"Expected 5 prompt entries for {model}"

    def test_raw_responses_json_created(self, tmp_results):
        from run_experiment import run_experiment

        session = _make_session_mock()
        run_experiment(
            models={"qwen": "qwen/test"},
            prompts=SEED_PROMPTS[:3],
            num_runs=2,
            results_dir=tmp_results,
            session=session,
            verbose=False,
        )

        raw_path = tmp_results / "raw_responses.json"
        assert raw_path.exists(), "raw_responses.json must be created"

    def test_variance_values_in_unit_interval(self, tmp_results):
        from run_experiment import run_experiment

        # Return varied responses to produce non-zero variance
        call_counter = {"n": 0}
        def varied(model, prompt, idx):
            call_counter["n"] += 1
            return f"Unique response {call_counter['n']}: {prompt[:20]}"

        session = _make_session_mock(response_factory=varied)
        result = run_experiment(
            models={"gpt": "gpt/test"},
            prompts=SEED_PROMPTS[:3],
            num_runs=4,
            results_dir=tmp_results,
            session=session,
            verbose=False,
        )

        for model, scores in result.items():
            for p_idx, v in scores.items():
                assert 0.0 <= v <= 1.0, f"Variance {v} out of [0,1] for model={model} prompt={p_idx}"

    def test_returns_dict_structure(self, tmp_results):
        from run_experiment import run_experiment

        session = _make_session_mock()
        result = run_experiment(
            models={"m1": "test/m1", "m2": "test/m2"},
            prompts=SEED_PROMPTS[:2],
            num_runs=2,
            results_dir=tmp_results,
            session=session,
            verbose=False,
        )

        assert isinstance(result, dict)
        assert set(result.keys()) == {"m1", "m2"}
        for model_scores in result.values():
            assert isinstance(model_scores, dict)
            assert len(model_scores) == 2

    def test_api_called_correct_number_of_times(self, tmp_results):
        from run_experiment import run_experiment

        session = _make_session_mock()
        n_prompts = 3
        n_runs = 4
        n_models = 2
        run_experiment(
            models={"a": "test/a", "b": "test/b"},
            prompts=SEED_PROMPTS[:n_prompts],
            num_runs=n_runs,
            results_dir=tmp_results,
            session=session,
            verbose=False,
        )

        expected_calls = n_models * n_prompts * n_runs
        assert session.post.call_count == expected_calls


# ---------------------------------------------------------------------------
# 4. Variance JSON — at least 3 model keys (Test spec 2)
# ---------------------------------------------------------------------------

class TestVarianceJsonStructure:
    def test_contains_at_least_3_model_keys(self, variance_json_file):
        """Test spec 2: variance.json contains keys for at least 3 models."""
        with open(variance_json_file) as f:
            data = json.load(f)
        assert len(data) >= 3, f"Expected ≥3 model keys, got {len(data)}: {list(data.keys())}"

    def test_model_keys_are_strings(self, variance_json_file):
        with open(variance_json_file) as f:
            data = json.load(f)
        for key in data.keys():
            assert isinstance(key, str)

    def test_all_variance_values_are_floats(self, variance_json_file):
        with open(variance_json_file) as f:
            data = json.load(f)
        for model, scores in data.items():
            for p_idx, v in scores.items():
                assert isinstance(v, (int, float)), f"Non-numeric variance for {model}[{p_idx}]"

    def test_all_variance_values_in_unit_interval(self, variance_json_file):
        with open(variance_json_file) as f:
            data = json.load(f)
        for model, scores in data.items():
            for p_idx, v in scores.items():
                assert 0.0 <= float(v) <= 1.0, f"Out-of-range variance {v} for {model}[{p_idx}]"

    def test_prompt_count_per_model(self, variance_json_file):
        with open(variance_json_file) as f:
            data = json.load(f)
        for model, scores in data.items():
            assert len(scores) == 50, f"Expected 50 prompt entries for {model}, got {len(scores)}"


# ---------------------------------------------------------------------------
# 5. Heatmap generation (Test spec 3)
# ---------------------------------------------------------------------------

class TestHeatmapGeneration:
    def test_heatmap_png_created(self, variance_json_file, tmp_results):
        """Test spec 3a: heatmap.png is created."""
        out_path = tmp_results / "heatmap.png"
        generate_heatmap(
            variance_path=variance_json_file,
            output_path=out_path,
        )
        assert out_path.exists(), "heatmap.png must be created"

    def test_heatmap_png_size_gt_10kb(self, variance_json_file, tmp_results):
        """Test spec 3b: heatmap.png file size > 10KB."""
        out_path = tmp_results / "heatmap.png"
        generate_heatmap(
            variance_path=variance_json_file,
            output_path=out_path,
        )
        file_size = out_path.stat().st_size
        assert file_size > 10 * 1024, f"heatmap.png too small: {file_size} bytes (expected >10240)"

    def test_heatmap_is_valid_png(self, variance_json_file, tmp_results):
        """Verify the PNG magic bytes."""
        out_path = tmp_results / "heatmap.png"
        generate_heatmap(
            variance_path=variance_json_file,
            output_path=out_path,
        )
        with open(out_path, "rb") as f:
            header = f.read(8)
        assert header == b"\x89PNG\r\n\x1a\n", "File is not a valid PNG"

    def test_build_matrix_shape(self, sample_variance_data):
        matrix, model_names = build_matrix(sample_variance_data, num_prompts=50)
        assert matrix.shape == (3, 50)
        assert len(model_names) == 3

    def test_build_matrix_values(self, sample_variance_data):
        import numpy as np
        matrix, model_names = build_matrix(sample_variance_data, num_prompts=50)
        assert np.all(matrix >= 0.0)
        assert np.all(matrix <= 1.0)


# ---------------------------------------------------------------------------
# 6. Demo / integration
# ---------------------------------------------------------------------------

class TestDemoMockMode:
    def test_demo_runs_without_api_key(self, tmp_path, monkeypatch):
        """demo.py mock mode should complete and create output files."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "")
        monkeypatch.setenv("OUTPUTS_DIR", str(tmp_path / "demo_out"))
        monkeypatch.setenv("NUM_RUNS", "2")

        from demo import main
        main(live=False)

        out_dir = tmp_path / "demo_out"
        assert (out_dir / "variance.json").exists()
        assert (out_dir / "heatmap.png").exists()
        assert (out_dir / "report.md").exists()

    def test_demo_variance_has_3_models(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "")
        monkeypatch.setenv("OUTPUTS_DIR", str(tmp_path / "demo_out2"))
        monkeypatch.setenv("NUM_RUNS", "2")

        from demo import main
        main(live=False)

        variance_path = tmp_path / "demo_out2" / "variance.json"
        with open(variance_path) as f:
            data = json.load(f)
        assert len(data) >= 3

    def test_demo_creates_html_report(self, tmp_path, monkeypatch):
        """demo.py should generate an HTML report."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "")
        monkeypatch.setenv("OUTPUTS_DIR", str(tmp_path / "demo_html"))
        monkeypatch.setenv("NUM_RUNS", "2")

        from demo import main
        main(live=False)

        html_path = tmp_path / "demo_html" / "report.html"
        assert html_path.exists(), "report.html must be created"
        content = html_path.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "Consistency Heatmap" in content

    def test_demo_creates_csv(self, tmp_path, monkeypatch):
        """demo.py should generate a results.csv file."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "")
        monkeypatch.setenv("OUTPUTS_DIR", str(tmp_path / "demo_csv"))
        monkeypatch.setenv("NUM_RUNS", "2")

        from demo import main
        main(live=False)

        csv_path = tmp_path / "demo_csv" / "results.csv"
        assert csv_path.exists(), "results.csv must be created"
        lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
        assert lines[0].startswith("model,prompt_idx,prompt_text"), "CSV must have header"
        assert len(lines) > 1, "CSV must have data rows"

    def test_demo_creates_experiment_meta(self, tmp_path, monkeypatch):
        """demo.py should generate experiment_meta.json with enriched metadata."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "")
        monkeypatch.setenv("OUTPUTS_DIR", str(tmp_path / "demo_meta"))
        monkeypatch.setenv("NUM_RUNS", "2")

        from demo import main
        main(live=False)

        meta_path = tmp_path / "demo_meta" / "experiment_meta.json"
        assert meta_path.exists(), "experiment_meta.json must be created"
        with open(meta_path) as f:
            meta = json.load(f)
        assert "generated_at" in meta
        assert "model_stats" in meta
        assert "semantic_variance" in meta
        assert "cross_model_divergence" in meta
        assert meta["mode"] == "mock"


# ---------------------------------------------------------------------------
# 7. Semantic metrics
# ---------------------------------------------------------------------------

class TestSemanticMetrics:
    def test_jaccard_identical(self):
        from consistency_heatmap_ import jaccard_similarity
        assert jaccard_similarity("the cat sat", "the cat sat") == 1.0

    def test_jaccard_no_overlap(self):
        from consistency_heatmap_ import jaccard_similarity
        score = jaccard_similarity("apple banana", "orange grape")
        assert score == 0.0

    def test_jaccard_partial_overlap(self):
        from consistency_heatmap_ import jaccard_similarity
        score = jaccard_similarity("the cat sat on the mat", "the dog sat on the floor")
        assert 0.0 < score < 1.0

    def test_jaccard_both_empty(self):
        from consistency_heatmap_ import jaccard_similarity
        assert jaccard_similarity("", "") == 1.0

    def test_semantic_variance_identical(self):
        from consistency_heatmap_ import semantic_variance
        responses = ["the cat sat on the mat"] * 5
        assert semantic_variance(responses) == 0.0

    def test_semantic_variance_single(self):
        from consistency_heatmap_ import semantic_variance
        assert semantic_variance(["only one response"]) == 0.0

    def test_semantic_variance_in_unit_interval(self):
        from consistency_heatmap_ import semantic_variance
        responses = [f"response number {i} talks about topic {i*2}" for i in range(8)]
        v = semantic_variance(responses)
        assert 0.0 <= v <= 1.0

    def test_cross_model_divergence_keys(self):
        from consistency_heatmap_ import cross_model_divergence
        raw = {
            "modelA": {"0": ["The sky is blue.", "Clouds are white."],
                        "1": ["Water is wet."]},
            "modelB": {"0": ["Photosynthesis converts light.", "Plants absorb CO2."],
                        "1": ["Ice is solid water."]},
        }
        div = cross_model_divergence(raw)
        assert "0" in div
        assert "1" in div
        assert all(0.0 <= v <= 1.0 for v in div.values())

    def test_cross_model_divergence_empty(self):
        from consistency_heatmap_ import cross_model_divergence
        assert cross_model_divergence({}) == {}

    def test_compute_semantic_variance_data(self):
        from consistency_heatmap_ import compute_semantic_variance_data
        raw = {
            "m1": {"0": ["hello world", "hello earth", "hi world"],
                   "1": ["foo bar", "foo bar", "foo bar"]},
        }
        result = compute_semantic_variance_data(raw)
        assert "m1" in result
        assert "0" in result["m1"]
        assert "1" in result["m1"]
        # identical responses → 0
        assert result["m1"]["1"] == 0.0
        # varied responses → > 0
        assert result["m1"]["0"] >= 0.0


# ---------------------------------------------------------------------------
# 8. Statistics utilities
# ---------------------------------------------------------------------------

class TestStats:
    def test_empty_list(self):
        from consistency_heatmap_ import compute_stats
        s = compute_stats([])
        assert s["n"] == 0
        assert s["mean"] == 0.0

    def test_single_value(self):
        from consistency_heatmap_ import compute_stats
        s = compute_stats([0.5])
        assert s["n"] == 1
        assert s["mean"] == 0.5
        assert s["std"] == 0.0

    def test_known_values(self):
        from consistency_heatmap_ import compute_stats
        s = compute_stats([0.1, 0.2, 0.3, 0.4, 0.5])
        assert abs(s["mean"] - 0.3) < 1e-9
        assert s["min"] == 0.1
        assert s["max"] == 0.5
        assert s["median"] == 0.3

    def test_ci_bounds_valid(self):
        from consistency_heatmap_ import compute_stats
        import random
        rng = random.Random(7)
        vals = [rng.uniform(0, 1) for _ in range(30)]
        s = compute_stats(vals)
        assert 0.0 <= s["ci_95_low"] <= s["mean"] <= s["ci_95_high"] <= 1.0

    def test_std_positive_for_varied(self):
        from consistency_heatmap_ import compute_stats
        s = compute_stats([0.1, 0.5, 0.9])
        assert s["std"] > 0.0

    def test_verdict_thresholds(self):
        from consistency_heatmap_ import verdict
        assert verdict(0.05) == "Very consistent"
        assert verdict(0.15) == "Consistent"
        assert verdict(0.25) == "Moderate variance"
        assert verdict(0.4) == "High variance"


# ---------------------------------------------------------------------------
# 9. Export utilities
# ---------------------------------------------------------------------------

class TestExportUtils:
    def test_csv_created(self, tmp_results, sample_variance_data):
        from consistency_heatmap_ import export_variance_csv, SEED_PROMPTS
        out = tmp_results / "results.csv"
        tmp_results.mkdir(parents=True, exist_ok=True)
        export_variance_csv(sample_variance_data, SEED_PROMPTS, out)
        assert out.exists()

    def test_csv_has_header(self, tmp_results, sample_variance_data):
        from consistency_heatmap_ import export_variance_csv, SEED_PROMPTS
        out = tmp_results / "results.csv"
        tmp_results.mkdir(parents=True, exist_ok=True)
        export_variance_csv(sample_variance_data, SEED_PROMPTS, out)
        lines = out.read_text().strip().splitlines()
        assert lines[0] == "model,prompt_idx,prompt_text,levenshtein_variance"

    def test_csv_row_count(self, tmp_results, sample_variance_data):
        from consistency_heatmap_ import export_variance_csv, SEED_PROMPTS
        out = tmp_results / "results.csv"
        tmp_results.mkdir(parents=True, exist_ok=True)
        export_variance_csv(sample_variance_data, SEED_PROMPTS, out)
        lines = out.read_text().strip().splitlines()
        # header + 3 models × 50 prompts = 151 lines
        assert len(lines) == 1 + 3 * 50

    def test_csv_with_semantic_column(self, tmp_results, sample_variance_data):
        from consistency_heatmap_ import export_variance_csv, SEED_PROMPTS
        out = tmp_results / "results_sem.csv"
        tmp_results.mkdir(parents=True, exist_ok=True)
        # Use same data for semantic (just for testing shape)
        export_variance_csv(sample_variance_data, SEED_PROMPTS, out, semantic_data=sample_variance_data)
        lines = out.read_text().strip().splitlines()
        assert "semantic_variance" in lines[0]

    def test_meta_json_created(self, tmp_results, sample_variance_data):
        from consistency_heatmap_ import save_experiment_meta
        tmp_results.mkdir(parents=True, exist_ok=True)
        out = tmp_results / "meta.json"
        save_experiment_meta(
            out,
            mode="mock",
            models={"qwen": "qwen/qwen3-72b"},
            num_runs=2,
            num_prompts=50,
            variance_data=sample_variance_data,
        )
        assert out.exists()
        with open(out) as f:
            meta = json.load(f)
        assert meta["mode"] == "mock"
        assert "generated_at" in meta


# ---------------------------------------------------------------------------
# 10. HTML report
# ---------------------------------------------------------------------------

class TestHtmlReport:
    def test_html_created(self, tmp_results, sample_variance_data):
        from consistency_heatmap_ import generate_html_report, SEED_PROMPTS
        tmp_results.mkdir(parents=True, exist_ok=True)
        out = tmp_results / "report.html"
        generate_html_report(sample_variance_data, SEED_PROMPTS, out)
        assert out.exists()

    def test_html_valid_doctype(self, tmp_results, sample_variance_data):
        from consistency_heatmap_ import generate_html_report, SEED_PROMPTS
        tmp_results.mkdir(parents=True, exist_ok=True)
        out = tmp_results / "report.html"
        generate_html_report(sample_variance_data, SEED_PROMPTS, out)
        content = out.read_text(encoding="utf-8")
        assert content.startswith("<!DOCTYPE html>")

    def test_html_contains_model_names(self, tmp_results, sample_variance_data):
        from consistency_heatmap_ import generate_html_report, SEED_PROMPTS
        tmp_results.mkdir(parents=True, exist_ok=True)
        out = tmp_results / "report.html"
        generate_html_report(sample_variance_data, SEED_PROMPTS, out)
        content = out.read_text(encoding="utf-8")
        for model in sample_variance_data:
            assert model in content, f"Model '{model}' missing from HTML report"

    def test_html_size_reasonable(self, tmp_results, sample_variance_data):
        from consistency_heatmap_ import generate_html_report, SEED_PROMPTS
        tmp_results.mkdir(parents=True, exist_ok=True)
        out = tmp_results / "report.html"
        generate_html_report(sample_variance_data, SEED_PROMPTS, out)
        assert out.stat().st_size > 5 * 1024, "HTML report should be >5 KB"

    def test_html_with_semantic_data(self, tmp_results, sample_variance_data):
        from consistency_heatmap_ import generate_html_report, SEED_PROMPTS
        tmp_results.mkdir(parents=True, exist_ok=True)
        out = tmp_results / "report_sem.html"
        generate_html_report(
            sample_variance_data, SEED_PROMPTS, out,
            semantic_data=sample_variance_data,
        )
        content = out.read_text(encoding="utf-8")
        assert "Semantic" in content


# ---------------------------------------------------------------------------
# 11. API client — model IDs and retry behaviour
# ---------------------------------------------------------------------------

class TestApiClient:
    def test_default_model_ids_updated(self):
        """Ensure outdated model IDs are no longer the defaults."""
        from consistency_heatmap_ import MODELS
        old_ids = {
            "qwen/qwen-2.5-72b-instruct",
            "mistralai/mistral-7b-instruct",
            "openai/gpt-4o-mini",
        }
        for name, model_id in MODELS.items():
            assert model_id not in old_ids, (
                f"Model '{name}' still uses outdated ID '{model_id}'"
            )

    def test_models_dict_has_three_keys(self):
        from consistency_heatmap_ import MODELS
        assert len(MODELS) == 3
        assert set(MODELS.keys()) == {"qwen", "mistral", "gpt"}

    def test_retry_on_server_error(self, tmp_path):
        """query_model should retry on 500-level errors and eventually raise."""
        import requests as req
        from unittest.mock import MagicMock, patch
        from consistency_heatmap_ import query_model

        call_count = {"n": 0}

        def flaky_post(*args, **kwargs):
            call_count["n"] += 1
            mock_resp = MagicMock()
            mock_resp.status_code = 503
            mock_resp.raise_for_status.side_effect = req.exceptions.HTTPError(
                response=mock_resp
            )
            return mock_resp

        session = MagicMock()
        session.post.side_effect = flaky_post

        with patch("consistency_heatmap_.api_client.time.sleep"):  # skip actual sleeps
            with pytest.raises(RuntimeError, match="Failed to query model"):
                query_model("test/model", "hello", session=session)

        assert call_count["n"] == int(os.getenv("MAX_RETRIES", "3"))

    def test_auth_error_not_retried(self):
        """401 errors should fail immediately without retrying."""
        import requests as req
        from unittest.mock import MagicMock, patch
        from consistency_heatmap_ import query_model

        call_count = {"n": 0}

        def auth_fail(*args, **kwargs):
            call_count["n"] += 1
            mock_resp = MagicMock()
            mock_resp.status_code = 401
            mock_resp.raise_for_status.side_effect = req.exceptions.HTTPError(
                response=mock_resp
            )
            return mock_resp

        session = MagicMock()
        session.post.side_effect = auth_fail

        with patch("consistency_heatmap_.api_client.time.sleep"):
            with pytest.raises(RuntimeError, match="Authentication error"):
                query_model("test/model", "hello", session=session)

        assert call_count["n"] == 1, "Auth errors should not be retried"


# ---------------------------------------------------------------------------
# 12. run_experiment — dry-run and richer JSON
# ---------------------------------------------------------------------------

class TestRunExperimentEnhanced:
    def test_experiment_meta_json_created(self, tmp_results):
        """run_experiment should create experiment_meta.json alongside variance.json."""
        from run_experiment import run_experiment

        session = _make_session_mock()
        run_experiment(
            models={"qwen": "qwen/test", "mistral": "mistral/test", "gpt": "gpt/test"},
            prompts=SEED_PROMPTS[:3],
            num_runs=2,
            results_dir=tmp_results,
            session=session,
            verbose=False,
        )
        meta_path = tmp_results / "experiment_meta.json"
        assert meta_path.exists(), "experiment_meta.json must be created"

    def test_experiment_meta_has_semantic_variance(self, tmp_results):
        from run_experiment import run_experiment

        session = _make_session_mock()
        run_experiment(
            models={"qwen": "qwen/test"},
            prompts=SEED_PROMPTS[:3],
            num_runs=2,
            results_dir=tmp_results,
            session=session,
            verbose=False,
        )
        with open(tmp_results / "experiment_meta.json") as f:
            meta = json.load(f)
        assert "semantic_variance" in meta
        assert "model_stats" in meta
        assert "cross_model_divergence" in meta

    def test_export_csv_flag(self, tmp_results):
        """--export-csv flag should create results.csv."""
        from run_experiment import run_experiment

        session = _make_session_mock()
        run_experiment(
            models={"qwen": "qwen/test"},
            prompts=SEED_PROMPTS[:3],
            num_runs=2,
            results_dir=tmp_results,
            session=session,
            verbose=False,
            export_csv=True,
        )
        csv_path = tmp_results / "results.csv"
        assert csv_path.exists(), "results.csv must be created when export_csv=True"
