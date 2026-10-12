import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import fraud_model as fm  # noqa: E402

TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


def synthetic(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    t = rng.choice(TYPES, n)
    amount = rng.exponential(5000, n)
    old = amount * rng.uniform(1, 3, n)
    fraud = (np.isin(t, ["TRANSFER", "CASH_OUT"]) & (rng.random(n) < 0.2)).astype(int)
    new = np.where(fraud == 1, 0.0, old - amount)
    old = np.where(fraud == 1, amount, old)
    return pd.DataFrame({
        "type": t, "amount": amount, "oldbalanceOrg": old, "newbalanceOrig": new,
        "oldbalanceDest": 0.0, "newbalanceDest": 0.0, "isFraud": fraud,
    })


@pytest.fixture(scope="module")
def trained():
    df = synthetic()
    x_train, x_test, y_train, y_test = fm.split(df)
    p = fm.build_pipeline().fit(x_train, y_train)
    return p, x_test, y_test


def test_split_is_reproducible():
    df = synthetic()
    a = fm.split(df)[1].index
    b = fm.split(df)[1].index
    assert list(a) == list(b)


def test_pipeline_learns_signal(trained):
    p, x_test, y_test = trained
    assert (p.predict(x_test) == y_test).mean() > 0.8


def test_known_types_come_from_model(trained):
    assert fm.known_types(trained[0]) == TYPES


def test_unseen_type_no_longer_crashes(trained):
    # The original app offered "DEPOSIT", which the model never saw, and crashed with a ValueError.
    assert fm.predict_one(trained[0], "DEPOSIT", 100.0, 100.0, 0.0, 0.0, 0.0) in (0, 1)


def test_predict_one_flags_obvious_fraud(trained):
    assert fm.predict_one(trained[0], "TRANSFER", 9000.0, 9000.0, 0.0, 0.0, 0.0) == 1


def test_committed_model_loads_and_predicts():
    joblib = pytest.importorskip("joblib")
    import sklearn
    if sklearn.__version__ != "1.6.1":
        pytest.skip("committed pickle needs scikit-learn 1.6.1")
    p = joblib.load(os.path.join(ROOT, "fraud_detection_pipeline.pkl"))
    assert fm.known_types(p) == TYPES
    assert fm.predict_one(p, "CASH_IN", 1000.0, 1000.0, 0.0, 0.0, 0.0) in (0, 1)


def test_streamlit_app_runs_and_predicts():
    pytest.importorskip("streamlit")
    import sklearn
    if sklearn.__version__ != "1.6.1":
        pytest.skip("committed pickle needs scikit-learn 1.6.1")
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(ROOT, "fraud_detection.py"), default_timeout=30).run()
    assert not at.exception
    assert list(at.selectbox[0].options) == TYPES
    at.button[0].click().run()
    assert not at.exception
    assert at.subheader and "Prediction" in at.subheader[0].value
