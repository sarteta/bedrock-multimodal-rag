# Use case — real estate listings with photo retrieval

## Customer

Real estate agency, 4000 active listings. Agents constantly field
client requests like "show me 2-bedroom apartments with a tiled
patio in Belgrano under 250k". Pure text search fails on the patio
detail because the listing description doesn't mention "tiled" —
but the photos clearly show it.

## What we built

- Each listing's title + description is embedded with Titan Text v2.
- Each listing photo (typically 8-15 per listing) is embedded with
  Titan Multimodal v1.
- Both vector spaces are stored in pgvector with a `listing_id`
  foreign key linking photos back to listings.
- A query embeds the user's text, runs hybrid retrieval against
  BOTH text+photo embeddings, and aggregates back to listing-level.
- Top listings get ranked with a small Claude on Bedrock pass that
  scores how well each matches the original query (catches the
  "the photo matches but the listing is in a different
  neighborhood" failure mode).

## What surprised us

Naive semantic on photos alone was about 60% as good as text-only
search. Agents thought it would be much better; turns out photos
in real estate are usually generic-looking ("nice living room")
and embeddings don't carry neighborhood/price info.

The win came from FUSING photo and text: photos catch the visual
details ("tiled patio", "south-facing balcony"), text catches the
factual ones (neighborhood, price, bedroom count). Recall@5 on a
golden set of 80 client-style queries:
- text only: 0.61
- photo only: 0.34
- fused: 0.86

## Cost / scale notes

4000 listings × 12 photos avg × $0.0006 per multimodal embedding =
~$29 one-time backfill, then ~$0.012 per new listing.

Query cost is dominated by the rerank step (~$0.005 per query)
because the embedding+vector lookups are essentially free at this
scale. The agency runs ~500 client queries/day = $2.50/day = $75/mo.

For a $7M GMV/year agency, this is a rounding error. Hard problem
the agents previously solved with their memory + a lot of swearing.
