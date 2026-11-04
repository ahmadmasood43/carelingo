# Recorded runs

Hosted model outputs are not reproducible, so these recorded runs are the evidence behind
[the report](../docs/report.md). Every case is synthetic. Scoring them needs no API key or GPU.

| Files | What they hold |
|---|---|
| `narrative-test-pipeline.jsonl` | Final system on the 48 held-out dictations, with narrative findings |
| `narrative-test-14b.jsonl` | Ministral 14B reference drafts, extracted facts and narrative findings |
| `verify-injected-test.jsonl` | Fact verification of the held-out injected errors and controls |
| `narrative-think-injected-test.jsonl` | Narrative check on injected narratives and controls |
| `pipeline-tts-extract-large.jsonl` | End to end from synthetic speech, Large 4 extractor |
| `pipeline-typed-extract-large.jsonl` | Typed control for that run, same extractor |
| `asr-tts-pinned-{en,pl}.jsonl` | Transcripts of the synthetic dictations, language set |
| `asr-tts-{en,pl}.jsonl` | The same with language detection, kept to show its failure |
| `voices-{en,pl}.jsonl` | Which voice read each synthetic dictation |
| `asr-fleurs-{en,pl}.jsonl` | FLEURS transcripts (CC BY 4.0) |
| `asr-primock57-doctor.jsonl` | PriMock57 doctor utterances (CC BY 4.0, Babylon Health), cut and cleaned as described in the report |
| `baseline-dev-*.jsonl`, `verify-dev-*.jsonl`, `extract-dev-*.jsonl`, `narrative-think-dev-baselines.jsonl` | Development split runs behind the routing comparison |

The final-system pipeline records hold each outcome (note, tier, discrepancies, checklist) but
not the extracted facts. The Ministral 14B records hold the facts, so their verification can be
computed again.
