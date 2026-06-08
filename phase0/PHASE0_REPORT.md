# Phase-0 IFBench inspection (reuse of cached Chunk-13 responses)

INSPECTION ONLY -- no GEPA, no LLM calls. Feedback + per-constraint pass/fail recomputed deterministically from cached seed-program responses (Qwen2.5-7B-Instruct-Turbo @ temp 0.6, seed 0, 150 D_feedback). **No GO/NO-GO call here -- that is the operator's decision.**

## Consistency check

- recomputed score == cached record score (per-record): 149/150
- cached scores == difficulty_table scores (multiset, order-independent): True  -- the frozen table IS these cached scores
- recomputed == cached (multiset): False (differs by exactly the 1 flip below)
- recompute-vs-cache mismatches (1): a known non-deterministic verifier (langdetect) flips across runs:
  - idx=96 id=36653 cached=0.2 -> recomputed=0.0; nondeterministic constraint(s): ['language:response_language']
- Note: `records.json` file order != `difficulty_table.scores` order, so an index-aligned table compare is meaningless; the multiset check above is the correct one.
- Nondeterminism surface: 6/150 examples include the langdetect verifier `language:response_language` (ids [14243, 31282, 36653, 41135, 43345, 8946]). Across two runs of this script the recompute matched the cache 150/150 and 149/150; the single flip was id=36653 (score 0.2 vs 0.0). That one verifier is not run-stable; all others are deterministic.

## (A) Base-score distribution

(Computed from the frozen cached Chunk-13 scores -- the canonical difficulty-table basis -- so these are stable run-to-run.)

- n = 150
- floor (score <= 0.01): 18 (12.0%)
- middle (0.01 < score < 0.99): 120 (80.0%)
- ceiling (score >= 0.99): 12 (8.0%)
- strict all-or-nothing rate (all constraints satisfied, score==1): 12 (8.0%)
- mean=0.4467  std=0.2784  quartiles(Q1,Q2,Q3)=(0.250, 0.367, 0.667)
- fraction within [0.4,0.6]: 16.7%
- distinct score values (11): 0.0:18, 0.2:7, 0.25:23, 0.3333:27, 0.4:7, 0.5:14, 0.6:4, 0.6667:24, 0.75:11, 0.8:3, 1.0:12
- n_constraints distribution: 3:70, 4:57, 5:23  (range 3-5)
- **kill condition triggered: none**
  - bimodal test (extremes>=80% AND middle<=20%): False (extremes=20.0%, middle=80.0%)
  - unimodal-near-0.5 test ([0.4,0.6]>=80% AND std<0.10): False ([0.4,0.6]=16.7%, std=0.2784)

Histogram: `phase0/ifbench_base_score_hist.png`

## (B) Feedback parseability (free proxy?)

- Structured per-constraint pass/fail is available DIRECTLY from the verifiers (`score_and_feedback` returns `violations`/`satisfied` lists; `_verify_one` is a pure boolean check). No LLM, no string parsing needed.
- The rendered feedback STRING also names every failed constraint as `  - instruction_id (k=v, ...)` under a `Violated (N):` header, so it is regex-parseable too.
- Round-trip parse (regex over feedback string == verifier violation ids) matches for ALL 150 examples: True
- Caveat: the per-constraint labels are deterministic EXCEPT the langdetect verifier `language:response_language`, which can flip across runs (1/150 examples differed from the cached scoring). The proxy is otherwise exact and LLM-free.

### Worked parse examples

- idx=0 score=0.0000  roundtrip_match=True
  - parsed failed ids: ['length_constraints:number_paragraphs', 'punctuation:no_comma', 'change_case:english_capital']
  - verifier violations: ['length_constraints:number_paragraphs', 'punctuation:no_comma', 'change_case:english_capital']
  - feedback_text:
    ```
    Score 0.0000 (0/3 constraints satisfied).
    Satisfied (0): none.
    Violated (3):
      - length_constraints:number_paragraphs (num_paragraphs=7)
      - punctuation:no_comma
      - change_case:english_capital
    ```
- idx=1 score=0.3333  roundtrip_match=True
  - parsed failed ids: ['detectable_format:bigram_wrapping', 'length_constraints:number_sentences']
  - verifier violations: ['detectable_format:bigram_wrapping', 'length_constraints:number_sentences']
  - feedback_text:
    ```
    Score 0.3333 (1/3 constraints satisfied).
    Satisfied (1): detectable_content:number_placeholders (num_placeholders=7).
    Violated (2):
      - detectable_format:bigram_wrapping
      - length_constraints:number_sentences (num_sentences=26, relation='at least')
    ```
- idx=3 score=0.4000  roundtrip_match=True
  - parsed failed ids: ['letters:letter_counting2', 'count:count_increment_word', 'first_word:first_word_answer']
  - verifier violations: ['letters:letter_counting2', 'count:count_increment_word', 'first_word:first_word_answer']
  - feedback_text:
    ```
    Score 0.4000 (2/5 constraints satisfied).
    Satisfied (2): length_constraints:number_words (num_words=532, relation='at least'), detectable_content:postscript (postscript_marker='P.P.S').
    Violated (3):
      - letters:letter_counting2 (let_frequency=12, let_relation='less than', letter='r')
      - count:count_increment_word (keyword1='relative', keyword2='noise')
      - first_word:first_word_answer (first_word='condition')
    ```
- idx=5 score=0.6667  roundtrip_match=True
  - parsed failed ids: ['keywords:start_end']
  - verifier violations: ['keywords:start_end']
  - feedback_text:
    ```
    Score 0.6667 (2/3 constraints satisfied).
    Satisfied (2): detectable_format:multiple_sections (num_sections=3, section_spliter='Sec.'), detectable_format:title.
    Violated (1):
      - keywords:start_end
    ```

## Artifacts
- `phase0/ifbench_inspection.json` (150 results)
- `phase0/smoke20_feedback.txt` (20 full raw feedbacks)
- `phase0/ifbench_base_score_hist.png`