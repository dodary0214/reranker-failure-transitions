# Adjudication prompts (verbatim)

The same prompt text was used for the canonical judge (GPT-4o) and for the
secondary robustness judge (GPT-5); only the model id and decoding parameters
differ (see the table at the end). Hashes of the exact strings are given so a
re-run can be checked against them.

The adjudication is two-staged. Stage 1 assigns an acceptability class without
showing the gold passage. Stage 2 is entered only for stage-1 outcomes that
require separating a gold-equivalent answer from a merely related passage, and
it is the only stage that sees the gold text. The three-way prompt assigns the
answers / valid-alternative / lexical-trap distinction.

## Stage 1 — system

```text
You judge whether a text chunk is SUFFICIENT BY ITSELF to answer a question. Be strict. Return JSON: {"label": "substitutable_yes|substitutable_no", "directly_answers": true|false, "reason": "one sentence"}
```

## Stage 1 — user template

```text
Question: "{question}"

Chunk:
"{chunk}"

Note: this chunk was previously classified as "does not directly state the answer". The question here is different: using ONLY the information in this chunk, could one actually WRITE an answer to the question?

STRICT criteria for "substitutable_yes":
- The core content the question asks for (the specific number / method / reason / entity) must be EXPLICITLY present in the chunk.
- Merely discussing the topic, or implying the answer, is NOT enough.

EXAMPLE (substitutable_yes): Question: "what standard speech transcription pipeline was used?" Chunk explicitly describes the two-pass transcription process by a large commercial vendor used to transcribe the data. → the chunk alone suffices to write the answer.
EXAMPLE (substitutable_no): Question: "What are the tasks used in the multi-task learning setup?" Chunk explains what multi-task learning is and why it helps, but never names the tasks. → topic discussion only, not substitutable.

Also set "directly_answers": true if you believe this chunk actually DOES directly state the answer (contradicting its previous classification).

Return JSON: {"label": "substitutable_yes|substitutable_no", "directly_answers": true|false, "reason": "one sentence"}
```

## Stage 2 — system

```text
You judge the MARGINAL contribution of a chunk when it is presented TOGETHER WITH a gold evidence passage that already answers the question. Return JSON: {"label": "complementary|contextual|uncertain", "reason": "one sentence"}
```

## Stage 2 — user template

```text
Question: "{question}"

Gold evidence passage (this DOES answer the question):
"{gold}"

Chunk:
"{chunk}"

IMPORTANT: do NOT judge whether the chunk answers the question by itself — that was already decided (it does not). Judge ONLY whether, when the chunk is presented together with the gold passage, it substantively improves the final answer.
- "complementary": adds substantive detail, evidence, numbers, settings, or conditions that strengthen or extend the gold answer.
- "contextual": topically related but contributes nothing to the answer itself (background, general narrative, adjacent topic).
- "uncertain": genuinely borderline — do not force a label.

EXAMPLE (complementary): Question: "What are the best within-language data augmentation methods?" Gold lists the methods (frequency/time masking, additive noise, speed/volume perturbation). Chunk reports that after speed perturbation the training data tripled, another 3-fold augmentation with additive noise made the train set 9x, and that all techniques were complementary in combination. → adds concrete settings/numbers that strengthen the gold answer.
EXAMPLE (contextual): Question: "What are the tasks used in the multi-task learning setup?" Gold lists the tasks. Chunk gives a general textbook explanation of what MTL is. → topically related background, contributes nothing to the answer itself.

Return JSON: {"label": "complementary|contextual|uncertain", "reason": "one sentence"}
```

## Three-way — system

```text
You classify a text chunk into one of three categories relative to a question. Be precise: synonyms and paraphrases of the answer count as "answers". Return JSON: {"label": "answers|valid_alt|lexical_trap", "reason": "one sentence"}
```

## Three-way — user template

```text
Question: "{question}"
Known answer: "{answer}"

Chunk to classify:
"{chunk_text}"

Classify this chunk:
- "answers": The chunk provides the specific information that answers the question (exact match, synonym, or paraphrase all count). It may also be a valid alternative answer not captured by the known answer.
- "valid_alt": The chunk does NOT answer the question, but provides clearly related and useful information (e.g. background, methodology details, related findings).
- "lexical_trap": The chunk does NOT answer the question AND provides little relevant information. It shares vocabulary/terminology with the question but is not substantively helpful.
```

## Prompt hashes (sha256 of the exact strings above)

```json
{
  "threeway_system": "8705b154f30258a58d842af0e3abbfbffbac59c08e5e951320afd92fbc9f530f",
  "threeway_user_template": "addead564302c713549caba5758ab605af843f3a86af4ee04f1fd722dd7a3bef",
  "stage1_system": "3070fb5e1aad49bc51eb16c4b4bf8536763792b17fe40d524814634adb681f9b",
  "stage1_user_template": "1f35f43666b4f9eaaf681d2cec0d609993536942640b45965a4934ffcc494a32",
  "stage2_system": "5a83ae9d949c05cddc96fbfc022eebea158c72bd8a887c4cd98120fe4bd5ca54",
  "stage2_user_template": "9bd1abd0bd7e6a195935cf4d48dbd717839a3b0f79dfa769c84a0fd3777aa88b",
  "examples": "783fa7edc96b7395f53d2bd34910ed79219e83e0018a49d5efaaa3b41563caf5"
}
```

## Judge configurations

| | canonical | secondary (robustness only) |
|---|---|---|
| model | `gpt-4o` (resolves to gpt-4o-2024-08-06) | `gpt-5-2025-08-07` |
| prompts | the text above | identical, unchanged |
| decoding | `{'temperature': 0, 'max_tokens': 250, 'response_format': {'type': 'json_object'}}` | `{"max_completion_tokens": 2000, "response_format": {"type": "json_object"}, "reasoning_effort": "minimal", "seed": 42}` |
| role | produces every label in `data/adjudication/canonical_gpt4o_5state.jsonl` and every number in the paper | produces `secondary_gpt5_5state.jsonl`; used **only** to measure label-choice sensitivity |

The two decoding blocks differ because the `gpt-5-2025-08-07` endpoint rejects
both `max_tokens` and `temperature=0`. `seed=42` is passed but the endpoint
gives no determinism guarantee, so **the secondary pass is not claimed to be
deterministic**; a second independent pass was run for exactly this reason and
is shipped as `judge_run="pass2"`.

The secondary judge never overwrites a canonical label, and no majority vote is
formed anywhere in this study. See `data/adjudication/README.md`.
