import math

import pandas as pd
import pytest
import torch

from wibench.aggregator import PandasAggregator
from wibench.algorithms.base import BaseAlgorithmWrapper
from wibench.attacks.base import BaseAttack
from wibench.config import DumpType, PipeLineConfig
from wibench.context import Context
from wibench.metrics.base import PostEmbedMetric, PostExtractMetric, PostPipelineMetric
from wibench.pipeline import AggregateMetricsStage, PostPipelineStage, StageRunner
from wibench.pipeline_type import PipelineType


class SkipTestAlgo(BaseAlgorithmWrapper):
    """Fails to embed images smaller than 16 px (like rivagan on small images)."""

    name = "skiptest_algo"

    def __init__(self):
        super().__init__({})

    def embed(self, image, watermark_data):
        if image.shape[-1] < 16:
            raise ValueError("image too small")
        return image * 0.5

    def extract(self, attacked_object, watermark_data):
        return 1.0


class SkipTestIdentityAttack(BaseAttack):
    name = "skiptest_identity"

    def __call__(self, object):
        return object


class SkipTestFailAttack(BaseAttack):
    name = "skiptest_fail_attack"

    def __call__(self, object):
        raise RuntimeError("attack failed")


class SkipTestOkMetric(PostEmbedMetric):
    name = "skiptest_ok"

    def __call__(self, img1, img2, watermark_data):
        return 1.0


class SkipTestFailMetric(PostEmbedMetric):
    name = "skiptest_fail"

    def __call__(self, img1, img2, watermark_data):
        raise RuntimeError("metric failed")


class SkipTestExtractMetric(PostExtractMetric):
    name = "skiptest_result"

    def __call__(self, img1, img2, watermark_data, extraction_result):
        return float(extraction_result)


class SkipTestPPMetric(PostPipelineMetric):
    """Counts objects it was updated with (all must be non-None)."""

    name = "skiptest_pp"

    def __init__(self):
        self.count = 0

    def update(self, object1, object2):
        assert object1 is not None and object2 is not None
        self.count += 1

    def reset(self):
        self.count = 0

    def __call__(self):
        return self.count


class SkipTestPPFailMetric(PostPipelineMetric):
    name = "skiptest_pp_fail"

    def update(self, object1, object2):
        raise RuntimeError("post pipeline metric failed")

    def reset(self):
        pass


STAGES = [
    "embed",
    "post_embed_metrics",
    "attack",
    "post_attack_metrics",
    "extract",
    "post_extract_metrics",
    "aggregate",
]
METRICS = {
    "post_embed_metrics": [("skiptest_ok", None), ("skiptest_fail", None)],
    "post_attack_metrics": [("skiptest_ok", None)],
    "post_extract_metrics": [("skiptest_result", None)],
}
POST_STAGES = [
    "post_pipeline_embed_metrics",
    "post_pipeline_attack_metrics",
    "post_pipeline_aggregate",
]
POST_METRICS = {
    "post_pipeline_embed_metrics": [("skiptest_pp", None)],
    "post_pipeline_attack_metrics": [("skiptest_pp", None), ("skiptest_pp_fail", None)],
}
ATTACKS = [("skiptest_identity", None), ("skiptest_fail_attack", None)]


def make_runner(result_path, skip_errors=True, stages=STAGES, metrics=METRICS):
    config = PipeLineConfig(
        result_path=result_path,
        aggregators=[{"CSV": {}}],
        skip_errors=skip_errors,
        seed=42,
    )
    return StageRunner(
        stages,
        ("skiptest_algo", None),
        ATTACKS,
        metrics,
        config,
        PipelineType.IMAGE,
    )


def make_context(object_id, size):
    return Context(
        object_id=object_id,
        run_id="test_run",
        dataset="test_ds",
        original_object={"image": torch.rand(3, size, size)},
        object_data_field="image",
    )


def test_skip_errors_writes_none_and_continues(tmp_path):
    runner = make_runner(tmp_path)
    good, bad = make_context("good", 32), make_context("bad", 8)
    runner.run(good)
    runner.run(bad)

    # successful image: failed attack/metric branches are None, the rest computed
    assert good.marked_object is not None
    assert good.marked_object_metrics["skiptest_ok"] == 1.0
    assert good.marked_object_metrics["skiptest_fail"] is None
    identity = good.attacked_object_metrics["skiptest_identity"]
    assert identity["attack_time"] > 0
    assert identity["skiptest_ok"] == 1.0
    assert identity["skiptest_result"] == 1.0
    assert good.extraction_result["skiptest_identity"] == 1.0
    failed = good.attacked_object_metrics["skiptest_fail_attack"]
    assert good.attacked_objects["skiptest_fail_attack"] is None
    assert good.extraction_result["skiptest_fail_attack"] is None
    assert all(failed[key] is None for key in failed)

    # failed embed: every downstream field is None
    assert bad.marked_object is None
    assert all(value is None for value in bad.marked_object_metrics.values())
    for attack_metrics in bad.attacked_object_metrics.values():
        assert all(value is None for value in attack_metrics.values())
    assert all(value is None for value in bad.extraction_result.values())

    # records have identical key sets -> consistent table columns
    for stage in runner.stages:
        if isinstance(stage, AggregateMetricsStage):
            stage.flush()
    table = pd.read_csv(tmp_path / "metrics_table.csv")
    assert len(table) == 2
    bad_row = table[table["object_id"] == "bad"].iloc[0]
    assert math.isnan(bad_row["embed_time"])
    assert math.isnan(bad_row["skiptest_fail_attack_skiptest_result"])
    good_row = table[table["object_id"] == "good"].iloc[0]
    assert good_row["skiptest_identity_skiptest_result"] == 1.0
    assert math.isnan(good_row["skiptest_fail_attack_attack_time"])


def test_raise_on_error(tmp_path):
    runner = make_runner(tmp_path, skip_errors=False)
    with pytest.raises(ValueError, match="image too small"):
        runner.run(make_context("bad", 8))


def test_post_pipeline_skip_errors(tmp_path):
    runner = make_runner(tmp_path)
    context_dir = tmp_path / "context_0"
    context_dir.mkdir()
    for context in (make_context("good", 32), make_context("bad", 8)):
        runner.run(context)
        context.dump(context_dir, DumpType.serialized)

    post_runner = make_runner(tmp_path, stages=POST_STAGES, metrics=POST_METRICS)
    post_context = Context(object_id="0", run_id="test_run", dataset="test_ds", original_object={})
    for stage in post_runner.post_pipeline_stages:
        if isinstance(stage, PostPipelineStage):
            stage.set_context_dir(context_dir)
        stage.process_object(post_context)

    # only the successfully embedded image contributes
    assert post_context.marked_object_metrics["skiptest_pp"] == 1
    identity = post_context.attacked_object_metrics["skiptest_identity"]
    # both metrics coexist for one attack; only the failed one is None
    assert identity["skiptest_pp"] == 1
    assert identity["skiptest_pp_fail"] is None
    # failed attack has no attacked objects to aggregate over
    assert post_context.attacked_object_metrics["skiptest_fail_attack"]["skiptest_pp"] == 0
    table = pd.read_csv(tmp_path / "post_pipeline_metrics_table.csv")
    assert len(table) == 1
    assert math.isnan(table["skiptest_identity_skiptest_pp_fail"].iloc[0])


def test_safe_append_csv_aligns_columns(tmp_path):
    from wibench.config import PandasAggregatorConfig

    aggregator = PandasAggregator(PandasAggregatorConfig(kind="CSV"), tmp_path)
    path = tmp_path / "aligned.csv"
    aggregator.safe_append_csv(pd.DataFrame([{"a": 1, "b": 2}]), path)
    # batch missing a column and bringing a new one: nothing is lost
    aggregator.safe_append_csv(pd.DataFrame([{"a": 3, "c": 4}]), path)

    table = pd.read_csv(path)
    assert list(table.columns) == ["a", "b", "c"]
    assert table["a"].tolist() == [1, 3]
    assert table["b"].iloc[0] == 2 and math.isnan(table["b"].iloc[1])
    assert math.isnan(table["c"].iloc[0]) and table["c"].iloc[1] == 4
