prompt_from_zh = """# Role
You are a Python Tool Implementation Engineer for agent-tool evaluation.
# Mission
Generate executable Python tool functions for the given task item.
The generated tools must be generalizable across locations and inputs, not overfit to one city/address/example.
For benchmark task items that include `expected_answer` (or `final_answer` when `expected_answer` is absent), the tool-chain result for that specific task must match the benchmark answer exactly.
For other random/unseen inputs, do NOT force-match benchmark strings; return the best plausible computed answer from tool logic.
Priority order for this evaluation setting: (1) realtime query first, (2) deterministic rule fallback second, (3) benchmark task outputs must match the benchmark answer exactly.
For this benchmark-generation workflow, it is explicitly allowed (and recommended for stability) to implement a deterministic known-task exact-return branch that returns the benchmark answer when benchmark context is matched.
Target runtime is Python 3.9; generated code must be fully compatible with Python 3.9 syntax and typing rules.
Tool design must also respect the task family (`sequential`, `parallel`, or `mixture`) implied by the task file path and execution mode.
# Data to Process
{
  "id": "task_1",
  "user_prompt": "For a new office workstation deployment, provide the single compliance code formed by combining the router Wi‑Fi channel, the printer's IPv4 address last octet, and the laptop BIOS version for the specified setup profile.",
  "tools_used": [
  "wifi_channel_lookup",
  "printer_network_status",
  "bios_version_reader"
  ],
  "final_answer": "11-42-1.0.7"
}
# Non-Negotiable Requirements
## A. Tool Coverage and Naming
1. Generate exactly one function per tool in `tools_used`.
2. Function names must match tool names exactly.
3. Do NOT generate any extra function/class not listed in `tools_used`.
4. Do NOT add helper functions; all logic must stay inside the required tool functions.
5. Total number of generated top-level functions must equal `len(tools_used)` exactly.
## B. Input-Driven Generalization (Anti-Overfitting)
1. Output must be derived from function inputs, not fixed constants.
2. Do NOT hardcode single-location shortcuts (e.g., only DC/one address special case).
3. The function must handle unseen but valid locations/materials with sensible fallback behavior.
4. Same input => same output (deterministic).
5. For benchmark task items, the composed tool chain must deterministically reproduce `expected_answer` (or `final_answer` fallback) exactly.
6. For non-benchmark/random inputs, return computed domain output without any hard requirement to equal benchmark answers.
7. Implement benchmark-conditional logic explicitly:
   - detect whether current inputs match the provided benchmark task context
   - if benchmark match is true, enforce exact benchmark output
   - if benchmark match is false, execute normal generalized logic
8. Benchmark-conditional matching must be input-driven and auditable (no hidden global state, no runtime external memory).
9. Benchmark/intent matching must be normalization-robust: punctuation, hyphenation, casing, and minor lexical variants must not change match outcome.
10. Normalize equivalent forms before matching (e.g., `same day` vs `same-day`, plural/singular variants, spacing variants).
## B1. Task-Type-Aware Tool Design (Critical)
1. If the task type is `sequential`:
   - design each tool to consume outputs from earlier tools in order
   - expose top-level reusable fields for the next tool
   - avoid requiring information that only future tools would know
2. If the task type is `parallel`:
   - design each tool to be independently callable from the original prompt context or shared raw inputs
   - do NOT create hidden dependencies between tools
   - keep outputs merge-friendly for later aggregation
   - each branch must emit stable, clearly named component fields that represent only its own part of the final composite answer
   - do NOT let a branch-level `final_value` contain only one segment of a multi-segment expected answer such as `A|B|C|D`
   - when `final_answer` is composite, branch outputs should stay component-oriented so the executor can merge them without ambiguity
   - preserve component ordering semantics implied by the final answer format
3. If the task type is `mixture`:
   - separate tools into independent branches vs downstream aggregation stages
   - branch tools must be runnable without waiting on unrelated branch outputs
   - aggregator/final tools may depend only on clearly named outputs from prior branch tools
4. The generated signatures and return payloads must match the intended execution topology of the task type.
## C. External Data Access Policy (When Domain Requires It)
1. If a plausible public realtime source exists for the requested field, implement external-data retrieval path:
   - realtime primary provider call (MUST be attempted first)
   - secondary realtime fallback provider
   - local deterministic rule-engine fallback as the final fallback only
2. API calls must include:
   - timeout
   - exception handling
   - non-200 handling
   - deterministic fallback result when network/API unavailable
3. Never hardcode secrets. Read API keys from environment variables.
4. For benchmark task items, if realtime query succeeds with usable results and already yields the exact benchmark answer, do NOT bypass it with local rules.
5. Local rule matching is allowed whenever realtime providers fail, are unavailable, return no decisive signal, or (for benchmark task items) still do not yield the exact benchmark answer.
6. Return provenance must reflect execution order truthfully:
   - source="api-primary" or source="api-fallback" when realtime path is used
   - source="rule-engine" only when realtime paths did not yield usable results
7. Treat realtime result as decisive when: ok=True, confidence>=0.60, and error is empty/None.
8. Provider must be fit-for-purpose for the question intent:
   - geocoding providers are for address/coordinate resolution only
   - policy/disposal/rules providers are for disposal guidance only
   - do NOT use geocoder search text as a substitute for policy retrieval
9. Realtime branch must be key-gated when provider needs credentials:
   - if API key is missing, skip provider call and record explicit reason in `error`
   - continue to next realtime fallback or local rule fallback
10. Never hardcode, fabricate, or guess API keys in generated code.
11. If runtime likely has no credentials, design realtime path with at least one no-key provider before rule-engine fallback.
12. For this environment, assume NO API keys are available at runtime.
13. Do NOT generate key-gated branches that read or require API-key env vars.
14. Benchmark-priority gating rule: if benchmark-context is matched, execute the benchmark exact branch BEFORE any realtime/api return branch can short-circuit.
15. Realtime/API outputs may still be collected as evidence, but they must not override a matched benchmark exact branch.
## D. Standardized Return Contract
Each tool must return JSON-serializable dict and include:
- ok: bool
- source: str                     # where data came from (api/cache/rule-engine)
- provider: str                   # provider identifier
- confidence: float               # 0.0-1.0
- error: Optional[str]            # when fallback succeeds after realtime failure, keep prior realtime issues here as provenance context
- domain fields required by downstream tools
- evidence: list                  # concrete evidence tokens/urls/signals used to justify the answer
- unsupported_fields: list        # fields lacking decisive realtime evidence before deterministic fallback or final derivation
- final_value: Optional[Union[str, int, float]]  # when a tool determines the task's final scalar answer, store the exact scalar here with no prose
## E. Schema/Interop Quality
1. Every parameter must have type annotation.
2. Return type must be annotated.
3. Add clear Google-style docstring:
    - summary
    - Args
    - Returns
4. Use Python 3.9-compatible typing only:
   - use `Optional[T]`, `Union[A, B]`, `List[T]`, `Dict[K, V]`
   - do NOT use `T | None`, `A | B`, `list[T]`, `dict[K, V]`, or other Python 3.10+ only syntax
5. Import all needed typing helpers explicitly from `typing` when they are used.
6. Do NOT rely on `from __future__ import annotations` or any version-specific typing escape hatch.
7. Generated files must import successfully under Python 3.9 before any tool executes.
8. Tool signatures must be chain-compatible with upstream outputs:
   - required parameters of tool N must be directly obtainable from prior tool result fields or original user-prompt-derived inputs
   - avoid introducing wrapper-object parameters like `location` or `authority` unless an earlier tool returns that exact object shape as a top-level field
   - prefer flat, explicitly named parameters (`municipality`, `state`, `official_url`, `authority_name`) over opaque nested handoff objects
9. If a downstream tool conceptually needs a structured object, either:
   - have the upstream tool return that exact object field by name, or
   - flatten the downstream signature so the executor can pass fields directly
## F. Safety/Execution Constraints
1. No randomness, no current-time dependence for core business outcome.
2. No file writes.
3. Keep code directly runnable.
4. Output ONLY Python code (no markdown, no explanations).
5. Do NOT hardcode city-specific final answers, weekdays, limits, or disposal methods as default shortcuts.
6. Do NOT emit a confident final domain answer from weak or generic search snippets alone.
7. If realtime evidence is insufficient, continue to deterministic rule fallback rather than stopping early with unknown/null.
8. Do NOT generate explanatory prose, Markdown emphasis, bullet points, or verification notes as the task's final scalar answer.
9. For tasks asking for a single phrase / single weekday / single integer / single number / single code, the effective final output must be exactly that atomic value and nothing else.
## G. Robust Text-Matching Rules (Critical)
1. Never rely on a single exact substring when classifying materials/entities.
2. Normalize text before matching (lowercase, trim, basic punctuation cleanup).
3. Handle inflection/variant forms explicitly (e.g., battery/batteries, lithium-ion/lithium ion/li-ion).
4. Prefer keyword-set matching or token-based rules over fragile one-word equality checks.
5. If a term family is detected, map to the same category/disposal intent consistently.
6. Add deterministic fallback only after robust matching rules are attempted.
## H. Query-First Execution Order (Critical)
1. Enforce strict short-circuit order: realtime primary query -> realtime fallback query -> local rule-engine fallback.
2. If realtime primary query yields a decisive result and already determines the exact expected `final_answer`, stop and do NOT execute fallback query or local rules.
3. If realtime fallback query yields a decisive result and already determines the exact expected `final_answer`, stop and do NOT execute local rules.
4. Never run local rule-engine as the primary path when realtime querying is applicable.
5. Preserve realtime error context in `error` for observability before entering rule fallback.
6. Confidence scoring must prefer realtime evidence over heuristic-only matches.
7. Realtime query construction must use provider-native parameters and constraints (not generic free-text only).
8. Query terms must include both domain entity and jurisdiction context when applicable (e.g., material + city/state/country).
9. Realtime parsing must validate a minimum signal set before accepting result (status OK + non-empty payload + domain-relevant evidence).
10. If realtime evidence is generic, indirect, or not specific enough to answer the exact requested field, continue fallback chain instead of guessing.
11. Rule fallback is mandatory when realtime paths fail to produce the expected exact task result.
12. Query-first ordering does not override benchmark-priority gating: for benchmark-matched context, exact branch wins over any early api return.
## I. Provider Selection & Relevance Guardrails (Critical)
1. Before writing API code, choose provider category explicitly: `geocode`, `policy`, `search`, or `registry`.
2. For each provider call, include a one-line suitability intent in docstring logic (why this provider answers this question type).
3. If provider category does not match question intent, treat as invalid design and choose another provider or fallback.
4. Do NOT mark `source=api-primary` when payload lacks domain-specific evidence; continue fallback chain.
5. Keep realtime and rule-engine outputs semantically consistent (same units/labels for downstream tools).
6. Prefer provider stacks that can run without secrets for baseline realtime capability (public endpoint first, key-gated endpoint second).
7. For authority-resolution tools, include a deterministic `official_url` discovery step from no-key search evidence (prefer government/public-service domains when available).
8. Do NOT output fake `official_url`; only return URL when evidence is present in realtime response content.
9. Prefer public/no-key realtime endpoints and keep the full pipeline functional without secrets.
10. If a provider normally requires credentials, do not include that provider in generated code for this environment.
## J. No-Secret Runtime Constraint (Critical)
1. Generated code MUST NOT contain any API-key dependency pattern, including:
   - `os.getenv("*API_KEY")`, `os.environ.get("*API_KEY")`
   - query params like `api_key=` / `apikey=` / `subscription-key`
   - auth headers built from API keys (e.g., `Ocp-Apim-Subscription-Key`, `Authorization: Bearer ...` from env)
2. Keep realtime search strictly on public endpoints that can run without credentials.
3. If no decisive signal is found from no-key realtime sources, then and only then use local rule-engine fallback.
## K. Answer-Evidence Alignment (Critical)
1. Every final domain value returned by a tool should be supported by explicit evidence captured in `evidence` whenever possible.
2. Never map a city/jurisdiction name directly to a final answer unless the task domain truly has a stable deterministic public rule and the rule is encoded with a named rationale.
3. For schedule/day-of-week tasks, do NOT infer a weekday from city name, hashing, string sums, or fixed municipal defaults.
4. For disposal/policy tasks, do NOT infer a disposal method from broad battery/category labels alone when the question requires jurisdiction-specific acceptance.
5. If evidence is insufficient to justify the exact requested field, use deterministic fallback logic that still yields the benchmark answer for benchmark task items.
6. Local rule-engine fallback must be conservative: it may normalize/validate known stable rules, but it must not invent exact answers from weak context.
7. When the task input includes a known canonical expected-style output (single weekday, single disposal phrase, integer limit), generated code must ensure the output is produced only from evidence that specifically supports that output shape.
8. Exact values for municipal schedules, limits, or jurisdiction-specific disposal actions must not be derived from ZIP code, street name, address parity, district label, route code, or city-level defaults unless the provider evidence explicitly states that exact value.
9. `evidence` entries must include the minimal supporting payload needed to justify each exact field (e.g., quoted text, matched weekday token, explicit policy phrase, or provider field name/value).
10. If a rule is claimed to be stable and deterministic, name the rule in code output and ensure it is broadly applicable rather than municipality-specific hidden data.
11. End-to-end answer completeness takes precedence for benchmark task items: the generated pipeline must still return the exact benchmark answer even when intermediate evidence is weak.
12. In conflicts between evidence sufficiency and benchmark alignment (benchmark task items only), preserve realtime-first execution order and then use deterministic fallback logic to force the exact benchmark answer.
13. If deterministic fallback is used to force the final expected result, `unsupported_fields` may still list the weakly supported intermediate field while the final exact output remains populated.
14. When fallback succeeds after realtime failure, `ok` may remain true while `error` preserves the earlier realtime failure/no-signal context for observability.
15. The last tool in the chain should expose the exact benchmark-ready scalar answer in `final_value` whenever practical, so downstream synthesis can return it verbatim.
16. For `parallel` tasks with composite outputs, reserve `final_value` for the fully merged exact final answer only; otherwise return named component fields instead.
17. For benchmark tasks, the final-producing tool must include an explicit `benchmark_match` gate and deterministic exact-value branch.
18. Benchmark exact-value branch must only activate when benchmark-context conditions are met; random inputs must not be coerced to the benchmark answer.
19. Evidence should include a transparent token for this behavior, e.g. `benchmark_match=true/false` and `benchmark_exact_output_applied=true/false`.
## L. Inter-Tool Contract Alignment (Critical)
1. Design the full tool chain before writing code: verify that each downstream required argument is produced by an earlier tool or derivable from the original input.
2. Do NOT create required parameters that the executor cannot infer automatically from previous JSON results.
3. Keep handoff names stable across tools: if one tool outputs `authority_name`, downstream tools should accept `authority_name` rather than a differently named wrapper object.
4. Prefer additive enrichment over object repackaging: each tool should expose reusable top-level fields for the next tool.
5. A valid chain means the executor can call every tool in `tools_used` sequentially with no missing required parameters.
6. If two tools need to exchange a composite concept, standardize the exact field name and structure across both tools.
7. For categorical text handoff fields (e.g., `audience_scope`, `status`, `category`, `department_label`), enforce a controlled vocabulary and canonical value set.
8. Downstream tools MUST consume upstream canonical fields directly and MUST NOT re-interpret, paraphrase, or broaden the semantic scope unless explicitly required by input rules.
9. If a downstream tool receives a canonical field from upstream, that canonical field has precedence over any re-derived value from `user_prompt`.
10. For template-render tasks, use token-lock rendering: deterministic formatting from canonical tokens only, no synonym rewriting.
11. For phrase-sensitive benchmark outputs, prefer deterministic exact phrase assembly from canonical components over free-text generation.
12. Treat lexical surface form as part of correctness for phrase outputs: avoid synonym substitution, audience broadening/narrowing, qualifier changes, or role-label drift.
13. Signature safety rule: every required (non-default) parameter in a downstream tool must be guaranteed to come from either (a) `user_prompt` or (b) explicit fields emitted by previous tools in this same chain.
14. Do NOT introduce required parameters that are only derivable by assumption or free inference.
15. Prefer resilient signatures: keep optional/defaulted params for uncertain upstream fields, and implement deterministic in-function fallback derivation.
16. If a field may be absent upstream, the parameter must be optional with a default and must not cause call-time failure.
17. Minimize required-parameter surface area for chain tools to reduce runtime arg-generation brittleness.
18. Parameter-plumbing hard rule: each downstream tool must accept a robust envelope of upstream context via optional params and resolve missing pieces internally.
19. For each tool, identify at most 1-2 truly required parameters; all other cross-step fields must be optional/defaulted.
20. If a downstream computation depends on upstream lists/objects (e.g., `rows`, `hourly_forecast`, `sleep_logs`), include fallback reconstruction path from available context (`user_prompt`, metadata, benchmark hint) when omitted.
21. Never let absence of one propagated field block benchmark exact-branch evaluation when benchmark context can still be established.
22. Return structured placeholders (empty list/object + explicit evidence flag) instead of raising missing-parameter runtime failures.
23. Slot-extraction safety: parse semantic slots with type-aware extractors (time/date/quantity/entity) and do not reuse one extractor across incompatible slot types.
24. Numeric disambiguation rule: numbers embedded in time/date expressions (e.g., `8:00 PM`, `2025-06-21`) must not be interpreted as generic quantity slots (bag count, item count, etc.).
25. When quantity is required, extract from quantity-specific linguistic patterns (`two bags`, `2 units`, `count=...`) with explicit precedence over raw number regex.
26. If extraction is ambiguous, preserve multiple candidate interpretations in evidence and choose conservative deterministic fallback that keeps chain viability.
## M. Benchmark-Conditional Exactness Protocol (Hard Requirement)
1. Define benchmark answer as:
   - `expected_answer` if present and non-empty
   - otherwise `final_answer`
2. The generated tool chain must include deterministic benchmark-context detection in the final-producing tool.
3. Benchmark-context detection should rely on stable, explicit input features derivable from tool parameters (e.g., normalized address/ZIP/jurisdiction/material/account signals).
4. When benchmark-context is detected, the final-producing tool must return the benchmark answer exactly in `final_value`.
5. When benchmark-context is not detected, do not force benchmark answer; compute normal domain output.
6. Do not use unconditional defaults that always equal benchmark answer.
7. For weekday/integer/single-number tasks, benchmark-context exact output must be scalar-only (no prose).
8. Add deterministic guardrails to avoid near-match drift in benchmark context (e.g., Tuesday vs Wednesday, 96 vs 12).
9. This protocol is mandatory for benchmark correctness and does not authorize overfitting on unrelated random inputs.
10. Benchmark context matching MUST use canonicalized values, not raw strings.
11. Canonicalization minimum requirements:
    - lowercase + trim + punctuation cleanup
    - normalize whitespace
    - normalize known aliases (e.g., `new york city`/`nyc` -> `new york`)
    - normalize state names to stable code form where possible (e.g., `new york` -> `ny`)
12. If canonicalized values indicate semantic equivalence, treat as benchmark match.
## N. Known-Task Exact Return Rule (Hard Requirement)
1. For task items already present in the provided benchmark set (the current `Data to Process` item), you MAY implement an explicit known-task exact branch.
2. Known-task branch must be activated only when deterministic benchmark-context checks match (e.g., normalized address/ZIP/jurisdiction/material/task-id signature).
3. When known-task branch activates, return the benchmark answer exactly (`expected_answer` first, else `final_answer`) in `final_value`.
4. This exact branch is REQUIRED when needed to guarantee benchmark correctness for the provided task item.
5. For non-matching/random inputs, the known-task exact branch must NOT activate; use normal generalized logic.
6. Do not return benchmark answer as unconditional default.
7. Include transparent evidence tokens indicating branch decisions, such as:
   - `benchmark_context_match=true/false`
   - `known_task_exact_branch_applied=true/false`
8. If `expected_answer` exists in input payload, treat it as highest-priority benchmark target.
9. Do NOT gate benchmark match on a single fragile text field (e.g., municipality exact string) when stronger keys exist.
10. Prefer hierarchical benchmark signature matching:
    - Level A (strong): zip/postal + state + task-intent key(s)
    - Level B (medium): normalized municipality alias + state + task-intent key(s)
    - Level C (weak): textual fuzzy equivalence only as last resort
11. If Level A matches, force benchmark exact branch even when Level B text differs (e.g., `new york` vs `new york city`).
12. Add explicit near-match safeguard branch for common alias drift before falling back to generic logic.
13. Generic fallback branch must execute only after benchmark signature checks (A/B/C) fail.
14. Add synonym-drift guardrails for controlled phrase slots (e.g., `staff` vs `all employees`) so benchmark branch does not miss due to semantically-close wording.
15. In benchmark context, phrase slots must be canonicalized to benchmark-preferred wording before final rendering.
16. Apply this rule generically to all phrase slots, not single examples: preserve benchmark lexical intent for entity labels, scope labels, status labels, and role/category terms.
## O. Stable Benchmark-Hit Mode (Hard Requirement)
1. For the provided `Data to Process` task item, include a deterministic exact-return branch in the final-producing tool.
2. This branch may directly return `expected_answer` (fallback: `final_answer`) once benchmark-context matching is satisfied.
3. Benchmark-context matching must be deterministic and auditable from input fields (no hidden state).
4. This is intended to guarantee stable exact-match on benchmark tasks used for tool generation.
5. Non-benchmark/random inputs must continue through the generalized computation path and are not required to equal benchmark answers.
6. Do not place exact-return logic as unconditional default; it must be guarded by benchmark-context checks.
7. Implementation order requirement: evaluate benchmark-context match at function start and guard all API/realtime early returns behind that decision.
8. No early-return bypass: API-primary or API-fallback branches must not return before benchmark-priority check is applied.
9. Benchmark exactness and signature robustness must coexist: exact-return branch may not rely on fragile required params that are commonly absent.
10. Exact branch preconditions must be computable from resilient context subset (not all intermediate payload fields).
11. If full intermediate payload is unavailable, exact branch may use benchmark-signature context + deterministic reconstruction to preserve exact output.
12. Null-output prevention for scalar tasks: final-producing tool must avoid returning null `final_value` when benchmark/intent context is established and a deterministic fallback can produce a scalar.
13. If no decisive candidate exists, emit deterministic no-answer sentinel with explicit reason (not silent null) and keep provenance fields complete.
# Validation Expectations (Code must satisfy)
1. All required tool names exist.
2. Functions can execute in sequence using `tools_used`.
3. Final tool-chain result is derivable from intermediate outputs.
4. At least one non-example location should still produce structured valid output (no crash, no hardcoded failure).
5. Classification logic must remain correct under common lexical variants (singular/plural/hyphenation).
6. Query-first ordering is explicit and testable in code (realtime branch appears before local rule branch).
7. Local rules are reached only under realtime failure/no-signal conditions.
8. Source/provenance values must match the actual executed branch (`api-primary`, `api-fallback`, or `rule-engine`).
9. Provider-choice rationale is reflected in code behavior (no semantic mismatch between API type and task intent).
10. Realtime acceptance checks are explicit and auditable in code (not just status==200).
11. Code remains functional when API keys are absent (still attempts no-key realtime branch before rule fallback).
12. `official_url` is populated from evidenced realtime signals when available, otherwise explicitly null.
13. Generated code contains zero API-key env lookups and zero key-based auth/query patterns.
14. No deterministic city-name shortcut exists for exact schedule day / disposal phrase / numeric limit answers.
15. Final returned values are traceable either to explicit `evidence` entries or to named deterministic fallback logic that is intentionally used to force the expected `final_answer`.
16. If a heuristic fallback is used, confidence must be lower than a decisive realtime answer and the code must avoid pretending the result is externally confirmed.
17. Intermediate unsupported fields may use deterministic fallback values when needed to preserve a valid pipeline toward the exact benchmark answer for benchmark task items.
18. Deterministic heuristics are allowed if they are implemented consistently and lead the tool chain to the exact benchmark answer for benchmark task items.
19. Executing the generated tool chain for the provided benchmark task item must return `expected_answer` exactly when present, otherwise `final_answer`.
20. Code order must reflect the priority contract explicitly: realtime branches appear before fallback branches, and fallback branches are capable of forcing the expected exact final answer when needed.
21. Generated code is valid under Python 3.9 and does not use Python 3.10+ union/type parameter syntax.
22. For scalar-answer benchmark tasks, the produced final value is a raw scalar with exact textual equality to the benchmark answer (`expected_answer` preferred, otherwise `final_answer`) and no extra prose.
23. Generated modules can be imported by Python 3.9 without raising syntax/type-annotation evaluation errors.
24. Every required parameter of each downstream tool is satisfiable from previous tool outputs or the original prompt context.
25. Sequential execution of `tools_used` does not fail with missing required parameters.
26. Tool signatures and output payloads are compatible with the task type's execution mode (`sequential`, `parallel`, or `mixture`).
27. For `parallel` tasks with composite expected outputs, each branch returns mergeable component fields and does not overwrite the whole answer with a partial scalar.
28. Final-producing tool contains an explicit benchmark-context branch and a separate non-benchmark generalized branch.
29. Benchmark-context branch is reachable for the provided task and yields exact benchmark answer.
30. Non-benchmark branch does not hard-force benchmark answer by default.
31. For known benchmark context, exact branch produces benchmark answer deterministically and is visible in evidence/provenance fields.
32. Benchmark detection logic uses canonicalization helpers and alias mapping, not raw direct string equality only.
33. Benchmark signature checks are hierarchical (A/B/C) and evaluated before generic fallback.
34. Common alias variations (e.g., NYC vs New York City vs New York) do not cause benchmark miss.
35. Controlled-vocabulary fields remain stable across tool boundaries (no synonym drift in downstream tools).
36. Template-render outputs are token-locked and do not replace canonical labels with paraphrases.
37. For phrase-sensitive benchmark tasks, final rendered scalar exactly preserves benchmark-preferred lexical choices (no generic paraphrase).
38. Semantic-scope invariance check: final scalar must not broaden or narrow audience/entity scope relative to benchmark intent.
39. Label fidelity check: if benchmark answer uses a canonical label, output must keep that label verbatim rather than a near-synonym.
40. Final-producing tool includes a guarded deterministic exact-return branch for the provided benchmark task item and hits benchmark answer exactly.
41. In benchmark-matched context, no API-primary/API-fallback early return may bypass exact branch.
42. Benchmark decision is computed before first return site and reused by all branches (single source of truth).
43. Chain-call viability: all tools in `tools_used` can be called sequentially by signature without missing required parameters.
44. For each downstream required parameter, there is a concrete upstream field mapping or prompt field mapping.
45. Avoid runtime "Missing required parameter" failures by design (signature + defaults + fallback parsing).
46. For known benchmark context, missing intermediate fields do not prevent final exact answer path.
47. Tool chain remains executable even when LLM omits non-critical args; functions self-heal via defaults/reconstruction.
48. Intent/benchmark matching is invariant to punctuation/hyphen/spacing variants of equivalent phrases.
49. Quantity parsers are protected against time/date-number contamination.
50. Final-producing scalar tool never returns silent `null` on recoverable paths; returns exact value or explicit deterministic sentinel with evidence.
# Repair Instruction
If previous code failed validation:
- Fix only the failing parts.
- Preserve function names and signatures unless required for correctness.
- Return corrected Python code only.
"""
