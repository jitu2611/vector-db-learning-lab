# Chapter 002 — Embedding Models

**Status:** In progress

An embedding model converts text into a numerical vector whose position represents approximate meaning.

```text
"How do I deploy to production?"
→ [0.12, -0.43, ..., 0.08]

"How can I release changes live?"
→ [0.10, -0.39, ..., 0.11]
```

These sentences use different words but express related intent, so their vectors should be near each other.

## Why embeddings matter

Exact keyword matching cannot reliably handle synonyms or ambiguous words.

```text
"deploy to production" ≈ "release changes live"

"bank" + "money" → financial bank
"bank" + "river" → river bank
```

A transformer embedding model uses surrounding words to create a context-aware representation, then pools its contextual token vectors into one vector for the sentence or document chunk.

```text
text → tokens → token IDs → transformer → contextual token vectors
     → pooling → final sentence embedding
```

## How models learn semantic similarity

Embedding models are commonly trained with related and unrelated examples.

```text
Positive pair (move vectors closer)
"reset my password"
"recover my login credentials"

Negative pair (move vectors farther apart)
"reset my password"
"deploy an application"
```

This general approach is called **contrastive learning**.

## Similarity and normalization

Vector databases compare embeddings with cosine similarity, dot product, or L2 distance.

For two normalized vectors:

```text
high cosine similarity / high dot product → similar direction → related meaning
low cosine similarity                    → different direction → less related meaning
```

A normalized vector has length `1`. For example:

```text
[3, 4]
length = √(3² + 4²) = 5
normalized = [3/5, 4/5] = [0.6, 0.8]
```

Normalization makes comparison focus on vector direction rather than raw vector size. For normalized vectors, cosine similarity and dot product produce the same ranking.

## Review questions

1. If two sentences have similar meaning, should their embeddings be near or far apart?
   - **Near/closer together.**
2. Why is semantic embedding search better than exact word matching?
   - It can handle different wording with the same meaning and uses context to reduce ambiguity.
3. For a positive training pair, does contrastive learning increase or decrease vector distance?
   - **Decrease it.**
4. If cosine similarity is close to `1`, what does that mean?
   - The normalized vectors point in almost the same direction and are likely semantically related.
5. After normalization, does cosine similarity mainly compare vector size or direction?
   - **Direction.**

## Practical implementation — next

We will build a local script that:

1. loads `all-MiniLM-L6-v2`;
2. embeds similar and unrelated sentences;
3. prints embedding dimensions and vector norms;
4. calculates cosine similarity ourselves;
5. compares similarity scores and records failure cases.

> Chapter 001 introduced the full tokenization → transformer → pooling → vector database pipeline. This chapter examines the embedding model and its output in detail.
