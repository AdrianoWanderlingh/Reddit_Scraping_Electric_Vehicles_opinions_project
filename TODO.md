# TODO
- Optional ONNX Runtime backend for CPU speed.
- Calibrate thresholds in `config/labeling.yaml` after batching improvements.
- Add unit tests to compare `score_all` vs per-subject scoring (within tolerance).
- Add CI job that runs `python scripts/quick_test.py` on a tiny parquet fixture (≤3 rows).
- Provide a setup step to pre-download models and eliminate first-run latency.
- Consider a lightweight HTTP microservice to keep models warm across repeated CLI invocations.
