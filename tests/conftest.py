from __future__ import annotations

import copy
import os
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # makes `tests.fixtures` importable

from creditrisk.config import load_config  # noqa: E402
from tests.fixtures.synthetic import make_loan_level  # noqa: E402


@pytest.fixture(scope="session")
def cfg() -> dict:
    c = copy.deepcopy(load_config(ROOT / "configs" / "pipeline.yaml"))
    c["data"]["vintages_dev"] = [2010]
    c["data"]["vintages_oot"] = [2015]
    c["xgboost"]["n_trials"] = 2
    c["xgboost"]["max_rounds"] = 200
    c["xgboost"]["early_stopping_rounds"] = 20
    return c


@pytest.fixture(scope="session")
def loan_level():
    return make_loan_level()


@pytest.fixture(scope="session")
def fitted_models(cfg, loan_level):
    from creditrisk.challenger.xgb import XGBChallenger
    from creditrisk.features.catalog import ALL_FEATURES
    from creditrisk.scorecard.model import Scorecard

    tr, va = loan_level.iloc[:4000], loan_level.iloc[4000:]
    y_tr, y_va = tr["target"].to_numpy(), va["target"].to_numpy()
    sc = Scorecard(cfg["scorecard"], cfg["bands"], ALL_FEATURES).fit(tr, y_tr, va, y_va)
    xg = XGBChallenger(cfg["xgboost"], cfg["scorecard"], cfg["bands"], ALL_FEATURES).fit(
        tr, y_tr, va, y_va
    )
    return {"scorecard": sc, "xgboost": xg, "valid": va}


def _java_available() -> bool:
    return bool(os.environ.get("JAVA_HOME")) or shutil.which("java") is not None


@pytest.fixture(scope="session")
def spark():
    if not _java_available():
        pytest.skip("Java runtime not available - Spark tests run in Docker / CI")
    pytest.importorskip("pyspark")
    os.environ.setdefault("SPARK_MASTER", "local[2]")
    os.environ["SPARK_SHUFFLE_PARTITIONS"] = "4"
    from creditrisk.features.spark import get_spark

    session = get_spark("creditrisk-tests")
    yield session
    session.stop()


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory) -> Path:
    from tests.fixtures.synthetic import write_raw_files

    return write_raw_files(tmp_path_factory.mktemp("raw"), [2010, 2015])
