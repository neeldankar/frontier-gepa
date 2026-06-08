# Phase-0 IFBench constraint-type landscape

Descriptive tabulation of `phase0/ifbench_inspection.json` (n=150). INSPECTION ONLY: no GEPA, no LLM, no re-scoring. Type descriptions are read from each verifier's `check_following()` in `src/ifbench_verifiers/instructions.py`. No taxonomy, no grouping, no GO/NO-GO -- tables and descriptions only.

- distinct constraint types present: 54 (the full catalog)
- types appearing in any violations list: 50
- types appearing only as satisfied (never violated): 4

## (1) Violation-type frequency table

Per type that appears in any example's violations: how many of the 150 examples fail on it, the satisfied-example count for contrast, and the base-score spread (mean/min/max) of the examples that fail on it. Sorted by fail count. Also in `type_frequency.csv`.

| instruction_id | n_fail | n_sat | mean_score(fail) | min | max | what check_following() checks |
| --- | --- | --- | --- | --- | --- | --- |
| length_constraints:nth_paragraph_first_word | 14 | 0 | 0.3488 | 0.0 | 0.75 | must have exactly N paragraphs (split on blank lines) AND the nth paragraph's first word equals a given word |
| count:counting_composition | 13 | 0 | 0.2731 | 0.0 | 0.5 | must have exactly 3 '***'-split paragraphs, each with exactly n_sent sentences, each sentence exactly n_words words |
| detectable_format:bigram_wrapping | 13 | 4 | 0.2641 | 0.0 | 0.6667 | words taken in non-overlapping pairs must be wrapped so each pair starts with '<<' and ends with '>>' |
| copy:repeat_phrase | 12 | 0 | 0.2667 | 0.0 | 0.75 | the given phrase must appear exactly small_n times, each occurrence with exactly one word changed |
| detectable_format:sentence_hyphens | 12 | 0 | 0.3264 | 0.0 | 0.75 | all sentences must be joined by hyphens with no spaces between them |
| keywords:keyword_specific_position | 12 | 0 | 0.2292 | 0.0 | 0.6667 | the m-th word of the n-th sentence must equal the given keyword |
| count:lowercase_counting | 10 | 3 | 0.3367 | 0.0 | 0.6667 | the number of all-lowercase words must be <= N |
| detectable_format:square_brackets | 10 | 2 | 0.3083 | 0.0 | 0.6667 | EVERY word in the response must be enclosed in [square brackets] |
| keywords:start_end | 10 | 3 | 0.3583 | 0.0 | 0.6667 | the response's first and last token (nltk) must be the same word |
| keywords:word_once | 10 | 1 | 0.3467 | 0.0 | 0.8 | a given keyword must occur exactly once (regex count == 1) |
| paragraphs:paragraphs2 | 10 | 4 | 0.215 | 0.0 | 0.4 | must have exactly 2 paragraphs separated by a blank line (\n\n) |
| count:count_increment_word | 9 | 2 | 0.4722 | 0.25 | 0.75 | keyword1 must occur exactly once AND keyword2 exactly twice |
| count:count_unique | 9 | 0 | 0.3593 | 0.0 | 0.8 | every tokenized word in the response must be unique (no repeated tokens at all) |
| detectable_format:number_highlighted_sections | 9 | 7 | 0.3815 | 0.0 | 0.75 | must contain >= N markdown-highlighted spans (text wrapped in *...* or **...**) |
| detectable_format:number_bullet_lists | 8 | 2 | 0.2875 | 0.0 | 0.75 | number of markdown bullet lines (lines starting '*' or '-') must equal exactly N |
| first_word:first_word_answer | 8 | 6 | 0.2792 | 0.0 | 0.8 | the first word of the whole response must equal the given word |
| keywords:no_adjacent_consecutive | 8 | 6 | 0.3833 | 0.0 | 0.75 | no two adjacent words may start with consecutive letters of the alphabet |
| startend:quotation | 8 | 5 | 0.1562 | 0.0 | 0.5 | the response must be wrapped in double-quote characters at both start and end |
| change_case:english_lowercase | 7 | 8 | 0.169 | 0.0 | 0.6 | the entire response must be lowercase AND langdetect-detected as English (langdetect; non-deterministic) |
| detectable_content:number_placeholders | 7 | 3 | 0.2548 | 0.0 | 0.75 | must contain >= N bracketed [placeholders] (regex on [...]) |
| first_word:first_word_sent | 7 | 3 | 0.4524 | 0.25 | 0.6667 | the first word of EVERY sentence must equal the given word |
| last_word:last_word_sent | 7 | 3 | 0.3381 | 0.0 | 0.6667 | the last word of EVERY sentence (punctuation stripped) must equal the given word |
| change_case:capital_word_frequency | 6 | 8 | 0.1444 | 0.0 | 0.3333 | count of ALL-CAPS words must be < N ('less than') or >= N (else), nltk tokenized |
| change_case:english_capital | 6 | 3 | 0.2778 | 0.0 | 0.75 | the entire response must be uppercase AND langdetect-detected as English (langdetect; non-deterministic) |
| keywords:letter_frequency | 6 | 9 | 0.15 | 0.0 | 0.25 | a given single letter must occur < N or >= N times across the whole response (case-insensitive char count) |
| last_word:last_word_answer | 6 | 7 | 0.2722 | 0.0 | 0.6 | the last word of the whole response (punctuation stripped) must equal the given word |
| length_constraints:number_words | 6 | 9 | 0.2583 | 0.0 | 0.5 | word count must be < N ('less than') or >= N ('at least') |
| startend:end_checker | 6 | 4 | 0.3722 | 0.0 | 0.75 | response (stripped of quotes/case) must end with the given phrase |
| copy:copying_multiple | 5 | 0 | 0.3167 | 0.0 | 0.6667 | the response must be the given prompt repeated exactly N times, separated by '******' |
| length_constraints:number_sentences | 5 | 8 | 0.3867 | 0.2 | 0.6667 | sentence count must be < N ('less than') or >= N ('at least') |
| punctuation:punctuation_dot | 5 | 4 | 0.3233 | 0.0 | 0.6667 | the response must contain no '.' (dot) characters |
| keywords:frequency | 4 | 12 | 0.3333 | 0.25 | 0.5 | a given keyword must occur < N times ('less than') or >= N times ('at least'), counted by regex |
| keywords:word_count_different_numbers | 4 | 3 | 0.2083 | 0.0 | 0.3333 | a given keyword must occur < N or >= N times (same logic as keywords:frequency) |
| length_constraints:number_paragraphs | 4 | 3 | 0.15 | 0.0 | 0.6 | number of paragraphs split on the '***' divider must equal exactly N |
| letters:letter_counting | 4 | 5 | 0.4458 | 0.2 | 0.75 | the total count of alphabetic letters must be >= N ('at least') or < N ('less than') |
| new:copy_span_idx | 4 | 0 | 0.45 | 0.2 | 0.6667 | the response must equal the substring prompt[n_start:n_end] verbatim (stripped, case-insensitive) |
| paragraphs:paragraphs | 4 | 6 | 0.1833 | 0.0 | 0.3333 | must have exactly 2 paragraphs separated by the '***' divider |
| detectable_content:postscript | 3 | 6 | 0.0833 | 0.0 | 0.25 | must contain a postscript section starting with the given marker (e.g. 'P.S.' / 'P.P.S') |
| detectable_format:json_format | 3 | 0 | 0.3833 | 0.25 | 0.5 | the whole response (stripped of ``` fences) must parse as valid JSON |
| keywords:exclude_word_harder | 3 | 5 | 0.3611 | 0.25 | 0.5 | the given keyword must not appear as a space-delimited token (' keyword ') |
| letters:letter_counting2 | 3 | 3 | 0.2167 | 0.0 | 0.4 | a given single letter must occur < N or >= N times (case-insensitive char count; same class as keywords:letter_frequency) |
| copy:copy | 2 | 0 | 0.375 | 0.25 | 0.5 | the response must equal the given prompt text verbatim (stripped, case-insensitive) |
| copy:copying_simple | 2 | 0 | 0.4667 | 0.3333 | 0.6 | the response must equal the given prompt verbatim (stripped, case-insensitive) |
| keywords:existence | 2 | 7 | 0.2667 | 0.2 | 0.3333 | response must contain ALL of the given keywords (case-insensitive regex search) |
| keywords:palindrome | 2 | 8 | 0.2917 | 0.25 | 0.3333 | the response must contain at least one palindrome word |
| language:response_language | 2 | 4 | 0.3333 | 0.0 | 0.6667 | langdetect-detected language of the whole response must equal the given code (langdetect; non-deterministic across runs) |
| combination:repeat_prompt | 1 | 1 | 0.25 | 0.25 | 0.25 | response must start with the given prompt text verbatim (then answer) |
| combination:two_responses | 1 | 1 | 0.3333 | 0.3333 | 0.3333 | must contain exactly two distinct responses separated by '******' |
| detectable_format:title | 1 | 14 | 0.0 | 0.0 | 0.0 | must contain a non-empty title wrapped in << >> |
| punctuation:no_comma | 1 | 13 | 0.0 | 0.0 | 0.0 | the response must contain no commas |

## (3) Full type universe

All 54 distinct types present, with violated/satisfied example counts and whether each was ever violated.

| instruction_id | n_violated | n_satisfied | appears_as | what check_following() checks |
| --- | --- | --- | --- | --- |
| detectable_format:bigram_wrapping | 13 | 4 | mixed | words taken in non-overlapping pairs must be wrapped so each pair starts with '<<' and ends with '>>' |
| detectable_format:number_highlighted_sections | 9 | 7 | mixed | must contain >= N markdown-highlighted spans (text wrapped in *...* or **...**) |
| keywords:frequency | 4 | 12 | mixed | a given keyword must occur < N times ('less than') or >= N times ('at least'), counted by regex |
| change_case:english_lowercase | 7 | 8 | mixed | the entire response must be lowercase AND langdetect-detected as English (langdetect; non-deterministic) |
| detectable_format:title | 1 | 14 | mixed | must contain a non-empty title wrapped in << >> |
| keywords:letter_frequency | 6 | 9 | mixed | a given single letter must occur < N or >= N times across the whole response (case-insensitive char count) |
| length_constraints:number_words | 6 | 9 | mixed | word count must be < N ('less than') or >= N ('at least') |
| change_case:capital_word_frequency | 6 | 8 | mixed | count of ALL-CAPS words must be < N ('less than') or >= N (else), nltk tokenized |
| first_word:first_word_answer | 8 | 6 | mixed | the first word of the whole response must equal the given word |
| keywords:no_adjacent_consecutive | 8 | 6 | mixed | no two adjacent words may start with consecutive letters of the alphabet |
| length_constraints:nth_paragraph_first_word | 14 | 0 | violated-only | must have exactly N paragraphs (split on blank lines) AND the nth paragraph's first word equals a given word |
| paragraphs:paragraphs2 | 10 | 4 | mixed | must have exactly 2 paragraphs separated by a blank line (\n\n) |
| punctuation:no_comma | 1 | 13 | mixed | the response must contain no commas |
| count:counting_composition | 13 | 0 | violated-only | must have exactly 3 '***'-split paragraphs, each with exactly n_sent sentences, each sentence exactly n_words words |
| count:lowercase_counting | 10 | 3 | mixed | the number of all-lowercase words must be <= N |
| keywords:start_end | 10 | 3 | mixed | the response's first and last token (nltk) must be the same word |
| last_word:last_word_answer | 6 | 7 | mixed | the last word of the whole response (punctuation stripped) must equal the given word |
| length_constraints:number_sentences | 5 | 8 | mixed | sentence count must be < N ('less than') or >= N ('at least') |
| punctuation:punctuation_exclamation | 0 | 13 | satisfied-only | the response must contain no '!' characters |
| startend:quotation | 8 | 5 | mixed | the response must be wrapped in double-quote characters at both start and end |
| copy:repeat_phrase | 12 | 0 | violated-only | the given phrase must appear exactly small_n times, each occurrence with exactly one word changed |
| detectable_format:sentence_hyphens | 12 | 0 | violated-only | all sentences must be joined by hyphens with no spaces between them |
| detectable_format:square_brackets | 10 | 2 | mixed | EVERY word in the response must be enclosed in [square brackets] |
| keywords:forbidden_words | 0 | 12 | satisfied-only | NONE of the given forbidden words may appear (whole-word, case-insensitive) |
| keywords:keyword_specific_position | 12 | 0 | violated-only | the m-th word of the n-th sentence must equal the given keyword |
| count:count_increment_word | 9 | 2 | mixed | keyword1 must occur exactly once AND keyword2 exactly twice |
| keywords:word_once | 10 | 1 | mixed | a given keyword must occur exactly once (regex count == 1) |
| detectable_content:number_placeholders | 7 | 3 | mixed | must contain >= N bracketed [placeholders] (regex on [...]) |
| detectable_format:number_bullet_lists | 8 | 2 | mixed | number of markdown bullet lines (lines starting '*' or '-') must equal exactly N |
| first_word:first_word_sent | 7 | 3 | mixed | the first word of EVERY sentence must equal the given word |
| keywords:palindrome | 2 | 8 | mixed | the response must contain at least one palindrome word |
| last_word:last_word_sent | 7 | 3 | mixed | the last word of EVERY sentence (punctuation stripped) must equal the given word |
| paragraphs:paragraphs | 4 | 6 | mixed | must have exactly 2 paragraphs separated by the '***' divider |
| startend:end_checker | 6 | 4 | mixed | response (stripped of quotes/case) must end with the given phrase |
| change_case:english_capital | 6 | 3 | mixed | the entire response must be uppercase AND langdetect-detected as English (langdetect; non-deterministic) |
| count:count_unique | 9 | 0 | violated-only | every tokenized word in the response must be unique (no repeated tokens at all) |
| detectable_content:postscript | 3 | 6 | mixed | must contain a postscript section starting with the given marker (e.g. 'P.S.' / 'P.P.S') |
| keywords:existence | 2 | 7 | mixed | response must contain ALL of the given keywords (case-insensitive regex search) |
| letters:letter_counting | 4 | 5 | mixed | the total count of alphabetic letters must be >= N ('at least') or < N ('less than') |
| punctuation:punctuation_dot | 5 | 4 | mixed | the response must contain no '.' (dot) characters |
| detectable_format:multiple_sections | 0 | 8 | satisfied-only | must contain >= N sections delimited by '<spliter> <number>' (e.g. 'SECTION 1') |
| keywords:exclude_word_harder | 3 | 5 | mixed | the given keyword must not appear as a space-delimited token (' keyword ') |
| keywords:word_count_different_numbers | 4 | 3 | mixed | a given keyword must occur < N or >= N times (same logic as keywords:frequency) |
| length_constraints:number_paragraphs | 4 | 3 | mixed | number of paragraphs split on the '***' divider must equal exactly N |
| language:response_language | 2 | 4 | mixed | langdetect-detected language of the whole response must equal the given code (langdetect; non-deterministic across runs) |
| letters:letter_counting2 | 3 | 3 | mixed | a given single letter must occur < N or >= N times (case-insensitive char count; same class as keywords:letter_frequency) |
| copy:copying_multiple | 5 | 0 | violated-only | the response must be the given prompt repeated exactly N times, separated by '******' |
| new:copy_span_idx | 4 | 0 | violated-only | the response must equal the substring prompt[n_start:n_end] verbatim (stripped, case-insensitive) |
| detectable_format:json_format | 3 | 0 | violated-only | the whole response (stripped of ``` fences) must parse as valid JSON |
| combination:repeat_prompt | 1 | 1 | mixed | response must start with the given prompt text verbatim (then answer) |
| combination:two_responses | 1 | 1 | mixed | must contain exactly two distinct responses separated by '******' |
| copy:copy | 2 | 0 | violated-only | the response must equal the given prompt text verbatim (stripped, case-insensitive) |
| copy:copying_simple | 2 | 0 | violated-only | the response must equal the given prompt verbatim (stripped, case-insensitive) |
| detectable_format:constrained_response | 0 | 1 | satisfied-only | response must contain one of a fixed set of allowed answer options |

Satisfied-only (never violated, 4): detectable_format:constrained_response, detectable_format:multiple_sections, keywords:forbidden_words, punctuation:punctuation_exclamation

Violated-only (never satisfied, 11): copy:copy, copy:copying_multiple, copy:copying_simple, copy:repeat_phrase, count:count_unique, count:counting_composition, detectable_format:json_format, detectable_format:sentence_hyphens, keywords:keyword_specific_position, length_constraints:nth_paragraph_first_word, new:copy_span_idx

## (4) Per-example structure

Distribution of violation COUNT per example (how many constraints each example fails):

| n_violations | n_examples |
| --- | --- |
| 0 | 12 |
| 1 | 38 |
| 2 | 45 |
| 3 | 37 |
| 4 | 15 |
| 5 | 3 |

Distribution of DISTINCT violation types per example:

| n_distinct_violation_types | n_examples |
| --- | --- |
| 0 | 12 |
| 1 | 38 |
| 2 | 45 |
| 3 | 37 |
| 4 | 15 |
| 5 | 3 |

(Note: per-example violation count and distinct-type count are identical here -> within an example the violated constraints are all distinct types: True.)

## (5) score x n_constraints cross-tab

Rows = fractional base score; columns = number of constraints in the example; cells = example count. Confirms which fractions come from 3/4/5-constraint examples.

| score | cc=3 | cc=4 | cc=5 | row_total |
| --- | --- | --- | --- | --- |
| 0.0 | 7 | 9 | 3 | 19 |
| 0.2 | 0 | 0 | 6 | 6 |
| 0.25 | 0 | 23 | 0 | 23 |
| 0.3333 | 27 | 0 | 0 | 27 |
| 0.4 | 0 | 0 | 7 | 7 |
| 0.5 | 0 | 14 | 0 | 14 |
| 0.6 | 0 | 0 | 4 | 4 |
| 0.6667 | 24 | 0 | 0 | 24 |
| 0.75 | 0 | 11 | 0 | 11 |
| 0.8 | 0 | 0 | 3 | 3 |
| 1.0 | 12 | 0 | 0 | 12 |
| **total** | 70 | 57 | 23 | 150 |
