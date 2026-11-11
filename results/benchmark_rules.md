### Benchmark - `rules` brain vs. mock insurer

- Calls: **160**  |  Call success: **100%**  |  Field-level accuracy: **100%**
- Decision latency: p50 **0.0 ms**, p95 **0.0 ms**  |  Mean turns/call: **8.1**

| Scenario | Runs | Passed | Fields | Mean turns | Mean decision ms |
|---|---|---|---|---|---|
| ivr_paid | 20 | 20/20 | 6/6 | 8.0 | 0.0 |
| ivr_denied_medical_necessity | 20 | 20/20 | 7/7 | 8.0 | 0.0 |
| ivr_pending | 20 | 20/20 | 5/5 | 8.0 | 0.0 |
| ivr_denied_missing_info | 20 | 20/20 | 7/7 | 8.0 | 0.0 |
| ivr_claim_not_found | 20 | 20/20 | 2/2 | 8.0 | 0.0 |
| ivr_flaky_input_retry | 20 | 20/20 | 7/7 | 9.0 | 0.0 |
| rep_denied | 20 | 20/20 | 7/7 | 8.0 | 0.0 |
| rep_paid | 20 | 20/20 | 6/6 | 8.0 | 0.0 |
