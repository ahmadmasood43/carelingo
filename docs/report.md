# CareLingo benchmark report

This is an independent engineering prototype. It has not been clinically validated and is
not a medical device. Every case is synthetic and every resident is fictional. Nothing here
was reviewed by a nurse, a clinician or a native speaker, so the results show how the
software behaves on this test set, not how it would behave on a ward.

## Summary

CareLingo turns a dictation in English or Polish into a German nursing note, checks that
the note preserves the dictated facts, and shows the nurse a checklist in their own
language.

1. **Typed dictation, final system:** 0 of 48 held-out notes had a clinical error (95% CI
   0–7%). The system also flagged 14 of those 48 correct notes (29%), which is its main
   weakness.
2. **The verifier finds real translation errors.** On drafts from a smaller model it caught
   18 of 23 notes with errors using facts alone (78%), and 22 of 23 with the narrative check
   (96%). It flagged all 324 injected fact errors.
3. **It does not protect against speech recognition errors.** From synthetic speech, 20 of
   48 notes had a clinical error (42%), and 6 of those 20 were marked ready with nothing
   flagged. The verifier compares the note with what was heard, not with what was said.
4. **Polish speakers get more false alarms:** 38% of clean notes against 21% for English.
5. **A strong first draft beat small-model-first routing on the development split.** Routing
   saved tokens, but it left more errors and put more notes in front of the nurse.

## 1. The question

A nurse who dictates in their first language still has to sign a German note they may not
fully read. Can a system show them, in their own language, what the note records, and warn
them when the translation changed a fact?

The approach assumes the dangerous errors are specific: a changed dose, a lost negation, a
wrong side, a claim nobody made. Those can be checked directly, so the verifier compares
typed facts instead of asking a model whether a translation looks right.

## 2. The system

| Step | Implementation | Runs |
|---|---|---|
| Speech recognition | faster-whisper `large-v3-turbo`, int8, language set from the user's profile | local GPU |
| German draft | `mistral-large-4`, `reasoning_effort=low` | Mistral API |
| Fact extraction | `ministral-14b-2512`, strict JSON schema, same prompt for dictation and draft | Mistral API |
| Verification | Rules: align typed facts, compare attributes, check numbers and dates | local, no model |
| Routing | Critical discrepancy: blocked. Major: check. None: ready | local, no model |
| Narrative check | `qwen3:8b` with reasoning lists statements the dictation does not support | local GPU, asynchronous |
| Nurse checklist | Fixed templates and a term table in English or Polish, no model | local, no model |

The checklist lists every extracted source fact, not only the problems, because some errors
are only visible to a person who remembers what they said. If an extraction fails, the note
is marked unverifiable and blocked, so a failed check can't pass as a clean one.

## 3. Data and labels

- **Cases:** 36 synthetic cases, each with a German reference note, typed gold facts, and
  English and Polish dictations. 12 are for development and 24 are held out, giving 48
  held-out dictations. The test split is pinned by a SHA-256 hash in the tests.
- **Authorship:** the cases were written by an LLM (Claude), a different family from the
  Mistral and Qwen models under test. They imitate spoken dictation (self-corrections, mixed
  German terms, implied units) but are probably cleaner and more explicit than real speech.
- **Error labels:** written by a single annotator, the same LLM, working to the project
  author's protocol. No clinician or native speaker reviewed them. A shared blind spot
  between the case author and the annotator would not show up in these numbers.
- **Blind labelling:** held-out labels were written before tiers or flags were read, with
  one exception: speech cases c013–c018 were labelled after seeing the flags.

A *clinical error* is an unsupported claim, a changed status, a lost hedge, a changed site, a
lost "as needed", or an omitted fact. Speech labels add a changed drug, value or room.

## 4. Protocol

- All tuning happened on the development split. Evaluation code was frozen at commit
  `df9eef1` before the first held-out run, and nothing found afterwards was fixed and
  re-scored. `git diff df9eef1 -- data/cases` is empty.
- Later changes do not alter the frozen behaviour: reporting commands, new optional pipeline
  flags with unchanged defaults, the `fact_omitted` label class, and a checklist display fix
  for capitalisation. The tests recompute this report's headline numbers from `results/`.
- Proportions use 95% Wilson intervals. With n = 48 these intervals are wide, and small
  differences between systems should not be over-read.
- Hosted model outputs are not reproducible. At temperature 0 with a fixed seed, only 7 of 24
  live Large 4 drafts matched an earlier run word for word. The recorded outputs are
  therefore committed in `results/`, and all scoring runs on them.

## 5. Results

### 5.1 Final system on typed dictation (held out)

| | All | English | Polish |
|---|---|---|---|
| Notes with a clinical error | 0/48 (CI 0–7%) | 0/24 | 0/24 |
| Clean notes flagged, facts | 14/48 (29%, CI 18–43%) | 5/24 (21%) | 9/24 (38%) |
| Clean notes flagged, with narrative check | 15/48 (31%, CI 20–45%) | 6/24 (25%) | 9/24 (38%) |
| Notes blocked | 3/48 (6%, CI 2–17%) | 2/24 | 1/24 |

All three blocked notes were false alarms. With no errors to find, this run says nothing
about recall, so sections 5.2 and 5.3 measure that separately.

### 5.2 Verifier recall on real errors (held out)

Drafts from `ministral-14b-2512` contain real mistakes, so they test the verifier on errors
nobody planted. Examples: Braden 12 (high risk) became "keine Risikogefährdung", weakness in
the left arm became "linksseitige Hemiparese", and a skin tear became "Dekubitusgrad I".

| | All | English | Polish |
|---|---|---|---|
| Notes with a clinical error | 23/48 (48%) | 9/24 | 14/24 |
| Caught, facts | 18/23 (78%, CI 58–90%) | 5/9 | 13/14 |
| Caught, with narrative check | 22/23 (96%, CI 79–99%) | 8/9 | 14/14 |
| Clean notes flagged, facts | 11/25 (44%, CI 27–63%) | 5/15 | 6/10 |
| Clean notes flagged, with narrative check | 16/25 (64%, CI 45–80%) | 9/15 | 7/10 |

All five notes missed by the fact check contained unsupported claims, mostly blanket
statements such as "Keine weiteren Auffälligkeiten", which typed facts cannot represent. Of
the four the narrative check then flagged, two were flagged for the right reason, one for an
unrelated reason, and one with garbled output.

### 5.3 Injected errors (held out)

Single, known errors were inserted into the gold notes, then checked against both source
languages. Injected errors are cleaner than real ones, so read these rates as an upper bound.

| Injected error | Flagged, facts | Marked critical |
|---|---|---|
| Value changed | 46/46 | 46/46 |
| Negation removed | 26/26 | 25/26 |
| Status flipped | 30/30 | 28/30 |
| Drug swapped | 24/24 | 24/24 |
| Room changed | 48/48 | 48/48 |
| Laterality flipped | 8/8 | 7/8 |
| Sentence omitted | 46/46 | 24/46 |
| Finding invented | 48/48 | 2/48 |
| Care action invented | 48/48 | 2/48 |
| **All fact errors** | **324/324 (CI 99–100%)** | |
| Narrative invented | 37/48 (77%) | 10/48 |
| Controls, no error (false alarms) | 21/96 (22%, CI 15–31%) | 2/96 |

With the narrative check, invented narratives rise to 48/48 and control false alarms to 24/96
(25%, CI 17–35%). On the development split, controls were flagged 12.5% of the time, so the
held-out rate shows how much tuning flattered that figure.

Error injection also found a real bug during development: a look-alike drug swap at the same
dose was hidden by a numeric evidence rule. The labelled set had not revealed it.

### 5.4 Cost and latency

| | Final system, typed (held out) |
|---|---|
| Cost per note | $0.00201 |
| Latency p50 / p95 / max | 10.8 s / 39.3 s / 81.8 s |
| Narrative check, asynchronous, p50 / max | 20.7 s / 56.1 s |

Costs use Mistral list prices read on 2026-11-03. The narrative check has no token cost, but
hardware and power are not included. Latency varied a lot: the same configuration had a p50
of 29 s on the development split, because of API variance and low rate limits on this
account. Treat these as one account's measurements, not capacity estimates. Speech
recognition added a real-time factor of 0.05–0.09 on a TITAN X (Pascal).

### 5.5 Routing policies (development split, n = 24)

This compares recorded steps and was decided before the held-out run.

| Policy | Clinical errors | Needs attention | Cost/note | Latency p50 |
|---|---|---|---|---|
| 14B, unverified | 6 | 0 | $0.00006 | 3.2 s |
| 14B, verified | 6 | 17 | $0.00058 | 14.4 s |
| 14B, redraft with Large 4 on critical | 5 | 16 | $0.00070 | 14.4 s |
| 14B, redraft with Large 4 on critical or major | 2 | 9 | $0.00149 | 29.6 s |
| Large 4 low, unverified | 1 | 0 | $0.00158 | 7.5 s |
| **Large 4 low, verified (chosen)** | **1** | **6** | **$0.00208** | **21.7 s** |

The small-model-first design had an acceptance target: close to Large 4 quality at no more
than half its cost. **It failed that target** (5 errors against 1). Most small-model errors
are not critical, so they don't trigger a redraft, and the main critical error was shared by
both models.

The saved tokens are also worth little compared with nurse time. Assuming a flagged note
takes 20–30 s of a nurse's time at about €35/h, each one costs €0.20–0.30, against a token
saving of about $0.0015 per note. That is an assumption, not a measurement.

### 5.6 Speech

**Recognition.** Word error rate is lowercased with punctuation removed. Numbers are not
normalised, which overstates it on number-heavy text.

| Data | Speakers | WER |
|---|---|---|
| FLEURS, 50 utterances each | real native speakers, read general text | en 6.4%, pl 6.2% |
| PriMock57, 432 utterances from 12 consultations | real clinicians, conversational English | 11.8% |
| Synthetic dictations, 48 | Piper voices with 20 dB pink noise | en 22.7%, pl 15.4% |

Synthetic English by voice: native 13.2%, L1 Mandarin 13.0%, L1 Arabic 26.4%, L1 Hindi
31.0%, L1 Spanish 34.1%. The non-native voices come from one low-quality Piper model, so this
mixes accent with synthesis artefacts and says little about real accents. Polish voices
ranged from 12.3% to 19.4%.

On PriMock57 a rough word-list check found 165 of 171 medical words recognised, and the real
misses were drug names (furosemide became "fruizamide"). A crude check on 12 consultations
supports only one conclusion: drug names are the weak spot.

Automatic language detection failed once in 24 English dictations and produced an Arabic
hallucination of note length. The design now sets the language from the user's profile.

**End to end from synthetic speech.** These 48 notes were labelled against what was said, not
what was heard. Ministral 14B was out of capacity at Mistral for over 90 minutes, so this run
used `mistral-large-4` as extractor. The typed control below uses the same extractor.

| | Speech | Typed control |
|---|---|---|
| Notes with a clinical error | 20/48 (42%, CI 29–56%) | 0/48 (CI 0–7%) |
| of which English / Polish | 18/24 / 2/24 | 0/24 / 0/24 |
| Errors marked ready, nothing flagged | 6/20 (30%, CI 15–52%) | none |
| Notes with only a misspelt drug | 3/48, all Polish | 0/48 |
| Clean notes flagged | 12/25 (48%, CI 30–67%) | 13/48 (27%, CI 17–41%) |
| Notes blocked | 12/48 | 2/48 |
| Resident name differs from the dictated name | 25/48 | 0/48 |
| Cost per note | $0.0142 | $0.0107 |
| Latency p50 / max | 29.2 s / 71.1 s | 22.6 s / 89.7 s |

Of 38 labelled errors, 37 came from recognition and one from the draft. The six unflagged
errors reached the nurse as ready:

| Case | Said | In the note |
|---|---|---|
| c013 en | Nitrendipin | Nitroglycerin |
| c015 en | did not fall | fact lost |
| c017 en | room number | room lost |
| c022 en | Pantoprazol, room number | Benazepril, room lost |
| c025 en | 3 cm wound | 30 cm wound |
| c035 en | Paracetamol 500 mg | Paracetamol 100 mg |

In four of the six, the wrong fact is visible on the checklist, so a nurse could spot it. In
the other two, the only sign is something missing from the checklist.

Of the 14 flagged error notes, about 9 were flagged for a reason related to the error (a
judgement by the same annotator). The rest were flagged by chance, which should not be counted as protection.
The verifier catches a recognition error only when the draft model and the extractor happen
to resolve the garbled text differently. When both make the same guess, the error passes.

**Extractor size.** `mistral-large-4` as extractor flagged 27% of clean typed notes, against
29% for Ministral 14B, at about five times the cost and twice the latency. These are separate
runs, but they give no reason to pay for the larger model, so the final system keeps 14B.
False alarms appear to come from extraction variance rather than model size.

## 6. Error analysis

**False alarms come from extraction, not from the drafts.** The causes seen were route fields
misused or empty, Polish terms left untranslated, spelling variants that didn't match an
alias, ambiguity between findings and care actions, findings invented from Polish text, and
malformed JSON (1 of 48 pipeline notes, 6 of 234 injected notes).

**Critical does not mean certain.** Critical flags had no false alarms in 104 clean checks on
the development split. On held-out data they blocked 3 of 48 correct notes and flagged 2 of
96 controls, so the development result did not generalise. The routing is still sound: it
stops a note for review, it doesn't assert an error.

**Negation in the source is a known weak point.** "Not more confused than usual" was
extracted as confusion being present (c029: both languages with Ministral 14B, Polish with
Large 4). The checklist then asks the nurse to confirm a fact they never stated, so the checklist can
expose extraction errors, but it can also carry them.

**The language gap is consistent.** Polish had more false alarms in every measurement: 38%
against 21% on pipeline notes, and 31% against 19% on injected controls with the narrative
check. Both the extractor and the 8B narrative model handle Polish worse. In a product, this
would mean Polish-speaking staff doing more review work for the same quality.

**Small-model drafts carry the risks verification handles worst.** Blanket claims and invented
plans are exactly what typed facts can't see, so the strong first draft avoids much of the
problem before verification starts.

## 7. Limitations

- **No clinical validation.** No clinician or native speaker reviewed cases, labels, the German
  notes or the checklist wording. Nothing here shows safety or usability for nurses.
- **Synthetic data, LLM labels.** The held-out set is small (48 dictations), and its author and
  annotator are the same model. "0 of 48" has an upper bound of 7% even if every label is right.
- **Vocabulary coverage is optimistic.** The controlled vocabulary and aliases were written by
  someone who knew the test concepts, so real dictations will contain more unknown terms.
- **The same extractor reads both sides,** so a shared blind spot can hide an error.
- **No real in-domain speech.** Synthetic speech is not evidence about accented, spontaneous
  speech on a noisy ward. PriMock57 is English consultation speech, not nursing dictation.
  No Polish or non-native speaker has been tested with real audio.
- **The speech run is a variant.** It used Large 4 as extractor, not the final configuration,
  so its cost and latency don't describe the final system.
- **The checklist is untested with people.** Whether nurses read it, or trust it too much, is
  unknown. A checklist that is usually right may train people to stop reading it.
- **Not a production system.** There is no authentication, audit trail, EHR integration or
  monitoring. The draft model was a public preview when tested, and hosted outputs are not
  reproducible. Only synthetic data was used, so GDPR processing of real health data, and
  whether any feature would count as a medical device, have not been assessed.

## 8. What would come next

The largest measured risk is recognition, and the fixes are outside the translation verifier.
None of these are built, and the expected effects are not measured:

1. Match the resident's name and room against the facility's resident list. This targets the
   5 room errors and most of the 25 name mismatches.
2. Check drug names against the resident's medication plan. This targets Nitroglycerin,
   Benazepril and similar substitutions.
3. Give Whisper the resident's drug names as a prompt.
4. Show the nurse the transcript, not only the extracted facts.

After that: test with real nurses and real recordings, reduce extraction variance, and improve
Polish with a stronger multilingual extractor and narrative model.

## 9. Reproducing the numbers

All scoring runs offline on `results/`, with no API key or GPU. The tests assert the main
figures, so `uv run pytest` fails if they drift.

```bash
uv run carelingo report \
  --pipeline results/narrative-test-pipeline.jsonl \
  --narrative results/narrative-test-pipeline.jsonl \
  --reference results/narrative-test-14b.jsonl \
  --reference-narrative results/narrative-test-14b.jsonl \
  --labels data/annotations/test-pipeline-large-low.yaml data/annotations/test-ministral-14b.yaml
```

```bash
uv run carelingo score-injection results/verify-injected-test.jsonl
```

```bash
uv run carelingo score-injection results/narrative-think-injected-test.jsonl --narrative
```

```bash
uv run carelingo score-speech \
  --pipeline results/pipeline-tts-extract-large.jsonl \
  --labels data/annotations/test-tts-pipeline.yaml
```

```bash
uv run carelingo score-speech \
  --pipeline results/pipeline-typed-extract-large.jsonl \
  --labels data/annotations/test-typed-large-extractor.yaml
```

```bash
uv run carelingo score-asr results/asr-*.jsonl --voices results/voices-*.jsonl
```

```bash
uv run carelingo simulate-routing \
  --baselines results/baseline-dev-*.jsonl \
  --verifications results/verify-dev-ministral-14b-2512.jsonl results/verify-dev-large-low.jsonl \
  --extractions results/extract-dev-ministral-14b-2512.jsonl \
  --annotations data/annotations/e001-dev.yaml data/annotations/e008-dev-large-low.yaml \
  --narrative results/narrative-think-dev-baselines.jsonl
```

Not everything is covered by a command: the PriMock57 medical-word check, the count of flagged
notes with a related reason, and the checklist visibility of the six unflagged errors were
one-off checks. `results/README.md` describes each run file, and
`scripts/primock_segments.py` rebuilds the PriMock57 utterances.

## Data and model attributions

- **PriMock57** (Babylon Health), CC BY 4.0. Doctor channel of 12 consultations, cut into
  utterances, with fillers removed from references. Paper: Papadopoulos Korfiatis et al.,
  ACL 2022.
- **FLEURS** (Google), CC BY 4.0. First 50 test utterances each of `en_us` and `pl_pl`.
- **Piper voices:** `en_GB-alba-medium` (CC BY 4.0), `en_US-l2arctic-medium` (CC BY-NC 4.0, so
  non-commercial only), and `pl_PL` gosia, darkman and mc_speech (CC0). Generated audio is not
  committed.
- **Software and models:** Piper (`piper-tts`, GPL-3.0, an optional dependency group), Whisper
  `large-v3-turbo` through faster-whisper, Qwen3 8B through Ollama, and Mistral's hosted models.
