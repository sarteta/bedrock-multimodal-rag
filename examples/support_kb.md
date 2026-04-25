# Use case — customer support over a product knowledge base

## Customer

SaaS company with 800 help-center articles + 50 internal runbooks.
Support team currently spends 30-40% of tickets searching for the
right doc. Many tickets get escalated when a doc exists but the
agent didn't find it.

## What we built

```
Article ingestion (markdown) → Titan Text v2 embeddings → pgvector
       ↓
Support agent's chat sidebar → query → hybrid retrieval → top-5
       ↓
Claude on Bedrock generates draft answer with explicit citations
to the retrieved articles
       ↓
Agent reviews, copies into ticket reply, sends.
```

Key engineering choices:

1. **Hybrid retrieval, not pure semantic.** Pure semantic missed
   articles where the customer used a different term than the docs
   (e.g. customer says "payment failed", docs say "transaction
   declined"). BM25 caught these. Hybrid lifted Recall@10 from 0.62
   to 0.84 on a 200-question golden set.

2. **Citations enforced via tool-use.** Claude must call a
   `cite_source(doc_id, quote)` tool for each claim. Answers without
   tool calls get rejected by a post-check. Stops the "the LLM made
   it up" failure mode dead.

3. **Confidence-gated escalation.** If the eval/judge score on a
   draft answer is < 0.6 (faithfulness), the system surfaces it as
   "uncertain — please review carefully" instead of as an auto-draft.

## Results after 3 months

- Mean ticket resolution time: -28% (agents find the right doc
  faster, fewer escalations)
- Agents using the tool reported 60% of drafts went out
  with only minor edits

## What didn't work

- **Single-step "summarize the doc and reply"**: too lossy, agents
  found the LLM dropped key caveats. Reverted to "draft then human-
  edits" flow.
- **Embedding the runbooks the same way as articles**: the runbooks
  use jargon and abbreviated steps that semantic embeddings poorly
  capture. Indexed them separately with BM25-only and matched on
  exact runbook IDs that the article links pointed to.

## Reproducing this for your KB

The repo's `examples/eval_retrieval.py` script ingests a folder of
markdown and runs a small golden set against it. Drop your help
center exports in there, write 30-50 (query, expected article)
pairs, and you have a baseline. Iterating from there is the easy part.
