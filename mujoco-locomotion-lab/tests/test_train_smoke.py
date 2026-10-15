import json

from locolab.evaluate import evaluate
from locolab.train import train


def test_train_eval_roundtrip(tmp_path):
    run_dir = train(
        env_id="InvertedPendulum-v5",
        epochs=2,
        steps_per_epoch=200,
        seed=0,
        out_dir=str(tmp_path),
    )
    assert (run_dir / "policy.pt").exists()
    assert (run_dir / "metrics.csv").exists()
    cfg = json.loads((run_dir / "config.json").read_text())
    assert cfg["env_id"] == "InvertedPendulum-v5"

    result = evaluate(run_dir, episodes=3)
    assert result["episodes"] == 3
    assert result["mean_return"] > 0
    assert (run_dir / "eval.json").exists()
