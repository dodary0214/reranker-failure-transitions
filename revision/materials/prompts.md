# Prompts added in the revision

## Semantic adjudication (Mistral-Small-3.2 sensitivity judge)
The Mistral judge used the original three-stage cascade with the **same prompts, control flow, label mapping and
retry policy** as the submitted GPT-4o/GPT-5 adjudication (`../../materials/adjudication_prompts.md`); only the
transport differs (local vLLM, OpenAI-compatible endpoint). Decoding: temperature 0, max_tokens 250, JSON object.
The judge sees the question, candidate passage, reference answer and gold passage only.

## Downstream answer generation (meta-llama/Llama-3.3-70B-Instruct)
Decoding: `{"temperature": 0.0, "max_tokens": 256}`. User message (single turn, no system message):

```
Answer the question using only the provided evidence passage.
Do not rely on external knowledge.
If the passage does not provide enough information to answer the question, return `INSUFFICIENT`.
Otherwise, provide a concise answer.

Question: {question}

Evidence passage:
{passage}

Answer:
```

Abstention: an answer is an abstention when, ignoring case, whitespace, backticks, periods and quotes, it equals
`INSUFFICIENT`. An answer for which is_insufficient() is true is not sent to the judge: correctness = INCORRECT, faithfulness = SUPPORTED (it asserts nothing), abstained = true.

## Downstream answer-quality evaluation (primary Qwen/Qwen3.5-27B; secondary google/gemma-3-27b-it)
Both evaluators receive the byte-identical prompt (Gemma's chat template has no system role and
prepends the system text to the user turn). Decoding: `{"temperature": 0.0, "max_tokens": 512, "response_format": {"type": "json_object"}, "chat_template_kwargs": {"enable_thinking": false}}` (the thinking switch
applies to Qwen only); up to 3 re-asks on transport error, invalid JSON or out-of-schema values.
Utility: `{"CORRECT": 1.0, "PARTIAL": 0.5, "INCORRECT": 0.0}`.

System message:
```
You evaluate an answer to a question about a scientific paper. Return JSON only: {"correctness": "CORRECT|PARTIAL|INCORRECT", "faithfulness": "SUPPORTED|UNSUPPORTED", "rationale": "one sentence"}
```

User message:
```
Question: {question}

Reference answer(s):
- {reference answer}

Reference evidence from the paper:
[1] {reference evidence}

Passage the answer was generated from:
{passage}

Generated answer:
{answer}

Judge two things.
correctness — compare the generated answer with the reference answer(s) and evidence:
- "CORRECT": conveys the key information the question asks for, consistent with the references (paraphrase is fine).
- "PARTIAL": conveys some of the key information but is incomplete, vague, or mixes correct and incorrect content.
- "INCORRECT": wrong, contradicts the references, or does not answer the question.
faithfulness — compare the generated answer only with the passage it was generated from:
- "SUPPORTED": every factual claim in the answer is stated in or directly entailed by that passage.
- "UNSUPPORTED": the answer contains a claim not supported by that passage.

Return JSON: {"correctness": "CORRECT|PARTIAL|INCORRECT", "faithfulness": "SUPPORTED|UNSUPPORTED", "rationale": "one sentence"}
```

Prompt hashes (sha256): `{"generator_user_template": "fe5ceafb39f8b1fd3ba2253fb312ca50258a4add99f6bb28c549f4f108ba8504", "judge_system": "637a44d5cec9e84c80c3cadcbe3ff627c382a6fc6e917799f795ac058d4818bd", "judge_user_template": "9764e3c480f1069922fefffdfb44cedfa6aa45c40acae3da4c5488efa93c8e1d"}`
