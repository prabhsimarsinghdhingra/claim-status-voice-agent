### Benchmark - `gemini` brain vs. mock insurer

- Calls: **3**  |  Call success: **100%**  |  Field-level accuracy: **100%**
- Decision latency: p50 **1246.7 ms**, p95 **2781.7 ms**  |  Mean turns/call: **8.3**
- LLM API calls: **26**  |  Values dropped by the grounding check: **0**

| Scenario | Runs | Passed | Fields | Mean turns | Mean decision ms |
|---|---|---|---|---|---|
| ivr_denied_medical_necessity | 1 | 1/1 | 7/7 | 8.0 | 1323.2 |
| ivr_flaky_input_retry | 1 | 1/1 | 7/7 | 9.0 | 1776.4 |
| rep_denied | 1 | 1/1 | 7/7 | 8.0 | 1425.0 |
