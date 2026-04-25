# Use case — legal contract analysis with image + text

## Customer

Mid-size law firm. Inbound contracts arrive as a mix of PDFs,
scanned documents (images), and the occasional emailed Word doc.
Associates spend hours per contract on first-pass analysis.

## What we built

For each contract:
1. PDF/image is OCR'd if needed. Pages with figures/diagrams are
   kept as images (don't lose the visual context).
2. Each page (or section) is embedded with **Titan Multimodal
   Embeddings** so a query like "what are the indemnity caps" can
   match both the relevant text section AND the diagram showing
   liability limits.
3. Claude on Bedrock answers the analyst's question, citing
   page+section, with image attachments inline when the answer
   references a figure.

## Why multimodal embeddings matter here

Contracts increasingly include diagrams: payment waterfalls,
organizational charts in M&A, equipment specs in vendor agreements.
Pure text embeddings throw this signal away. Multimodal embeddings
let a query like "show me the payment waterfall" actually surface
the diagram, not just paragraphs that mention the word "waterfall".

In the eval set we built (60 questions across 12 contracts), pure
text retrieval got Recall@5 = 0.71, multimodal got 0.89.

## Compliance considerations

- Bedrock supports a BAA / data-processing agreement under standard
  AWS terms. Simpler legal review than approving a third-party SaaS.
- Set the Bedrock model invocation to use a private VPC endpoint
  if your firm's policy requires it.
- We log every (query, retrieved-doc-ids, model-output) tuple to
  CloudWatch for audit. Cost ~$0.50/month per million queries —
  trivial.

## What we couldn't solve cleanly

- **Tables that span pages.** OCR of tables is still rough; multi-
  page tables get fragmented in retrieval. Workaround: dedicated
  table-extraction step (we used AWS Textract with the Tables API)
  feeding into a separate "tables" sub-index.
- **Languages other than English.** Titan embeddings work for
  English much better than for other languages (in our case Spanish
  contract clauses). For non-English we found that translating to
  English first via Bedrock then embedding hit acceptable Recall.
