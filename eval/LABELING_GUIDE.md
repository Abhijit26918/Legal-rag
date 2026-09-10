# Guide: Building the 75-100 Question Eval Set

A practical, no-jargon guide to reading these cases and turning them into
question/answer pairs. Keep this open while you work.

## 1. What data you're actually working with

For each case, you mainly care about four things:

| Field | What it is |
|---|---|
| `case_name` | e.g. "Maropulos v. County of Los Angeles" |
| `date_filed` | When the court decided it |
| `precedential_status` | `Published` or `Unpublished` — see glossary below, this matters |
| `plain_text` | The actual full text of the opinion — this is what you read |

You don't need to touch the CSV files directly — ask me for any case by
name or ID and I'll pull the text for you, the way I did with Maropulos.

## 2. Glossary — the ~15 terms that cover almost everything

**Who's who in the case:**
- **Plaintiff** — the person suing (in these cases, usually the person claiming police/officials did something wrong to them)
- **Defendant** — the person being sued (usually the officer/official)
- **Appellant** — whoever *lost* below and is asking the appeals court to fix it
- **Appellee** — whoever *won* below and is defending that win

**The core legal shield these cases fight over:**
- **Qualified immunity** — a legal shield that protects officials from being sued, UNLESS both of these are true: (1) they actually violated someone's constitutional right, AND (2) that right was "clearly established" at the time (meaning: existing case law had already made it obvious this specific conduct was illegal). If either part fails, the officer is protected.
- **"Clearly established"** — not just "this seems obviously wrong in hindsight." Courts want an earlier case with similar-enough facts that a reasonable officer *should have known* their conduct was illegal.

**Which constitutional amendment applies** (a common source of confusion — pick based on who the person was):
- **Fourth Amendment** — applies to force used during an arrest/seizure (most of your questions will be this)
- **Eighth Amendment** — applies to force against someone already *convicted* and imprisoned
- **Fourteenth Amendment** — applies to pretrial detainees (arrested, not yet convicted) and due-process claims generally

**Reading the court's own structure:**
- **Holding** — what the court actually decided (this is your "answer")
- **Dicta** — side commentary that isn't the actual ruling (don't cite this as the holding)
- **Procedural history** — what happened in the lower court before this appeal (background, not the holding)
- **Per curiam** — "by the court" — an opinion issued by the panel collectively, no single judge's name attached

**Outcomes you'll see at the end of an opinion:**
- **Affirmed** — the lower court's decision stands
- **Reversed** — the lower court was wrong, decision overturned
- **Vacated** — the lower court's decision is wiped out (often paired with "remanded")
- **Remanded** — sent back to the lower court to redo something
- **Dismissed** — the appeal itself wasn't even properly before the court (see Maropulos — this doesn't mean either side "won" on the merits)

**Precedent status (matters for how confidently you can use a case):**
- **Published** — carries full precedential weight; other courts must follow it. Prefer these for your questions.
- **Unpublished** — persuasive only, not binding. Still fine to use, just note it.

**Treatment (only relevant for trap questions later):**
- **Overruled / Superseded / Abrogated** — a later case says this one is no longer good law
- **Distinguished** — a later court says "this case doesn't apply here because the facts are different" (doesn't kill the precedent, just narrows it)

## 3. How to read one opinion, in order

Don't read top to bottom like a novel. Do this instead:

1. **Skim the caption** (the first ~20 lines) — just note: who sued whom, which court, what date, published or not. Ignore the list of attorney names.
2. **Skip to right after the word "OPINION"** — courts almost always state the case in one paragraph right there: who did what to whom, and what's being appealed.
3. **Scan for numbered markers** like `[1]`, `[2]`, `[3]` or `1.` `2.` `3.` — these mark distinct legal points the court is making. Each one is a candidate for its own question.
4. **Find the last paragraph** — look for words like "Accordingly," "Therefore," "For the foregoing reasons," or a standalone line like `AFFIRMED`, `REVERSED`, `DISMISSED`. That's the actual outcome.
5. **Work backward from that outcome** to the paragraph(s) that explain *why* — that reasoning is your answer's real content, not just the one-word outcome.

You very rarely need to read the whole thing. A short opinion (like Maropulos, ~1 page of substance) you can process in a few minutes once you know this pattern.

## 4. The Q&A format to fill in for each one

```json
{
  "question": "...",
  "answer": "...",
  "supporting_cases": [
    {"case_name": "...", "opinion_id": "...", "passage": "the exact quoted sentence(s)"}
  ],
  "type": "direct"
}
```

`type` is one of: `direct` (holding lookup), `multi_hop`, `trap`, `no_answer` — the last two we'll do together separately, since I can pre-source good candidates for those.

## 5. Suggested order of attack

1. Do 2-3 more together with me first (like Maropulos) until it feels automatic.
2. Then I'll hand you a batch of 8-10 short, published, single-issue candidates at a time — work through a batch, take a break, repeat.
3. Save trap questions and multi-hop questions for once you're comfortable with the basic direct-lookup ones — I'll pre-source real candidates for those.
4. Don't aim for 75-100 in one sitting. This is meant to be done in batches over multiple sessions.
