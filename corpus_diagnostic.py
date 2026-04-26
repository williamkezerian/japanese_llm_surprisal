"""
CORPUS STRUCTURE DIAGNOSTIC
=============================
Prints a comprehensive overview of how the UD Japanese GSD corpus
is annotated, so we can write accurate extraction scripts.

Run: python corpus_diagnostic.py
"""
import os
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_FILE = os.path.join(SCRIPT_DIR, "ja_gsd-ud-train.conllu")


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_conllu(filepath):
    sentences, tokens, sent_id, text = [], [], "", ""
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("# sent_id"):
                sent_id = line.split("=", 1)[-1].strip()
            elif line.startswith("# text"):
                text = line.split("=", 1)[-1].strip()
            elif line == "":
                if tokens:
                    sentences.append((sent_id, text, tokens))
                sent_id, text, tokens = "", "", []
            elif not line.startswith("#"):
                p = line.split("\t")
                if len(p) >= 10 and "-" not in p[0] and "." not in p[0]:
                    tokens.append({
                        "idx":    int(p[0]),
                        "surface": p[1],
                        "lemma":  p[2],
                        "upos":   p[3],
                        "xpos":   p[4],
                        "feats":  p[5],
                        "head":   int(p[6]) if p[6] != "_" else 0,
                        "deprel": p[7],
                        "deps":   p[8],
                        "misc":   p[9],
                    })
    if tokens:
        sentences.append((sent_id, text, tokens))
    return sentences


def sep(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

print(f"Loading {TRAIN_FILE} ...")
sentences = parse_conllu(TRAIN_FILE)
all_tokens = [t for _, _, toks in sentences for t in toks]
print(f"Loaded {len(sentences)} sentences, {len(all_tokens)} tokens.\n")


# ---------------------------------------------------------------------------
# 1. Column overview — sample raw rows
# ---------------------------------------------------------------------------
sep("1. RAW ROW SAMPLES (first 3 sentences, all tokens)")
for sent_id, text, tokens in sentences[:3]:
    print(f"\n  # sent_id = {sent_id}")
    print(f"  # text    = {text}")
    print(f"  {'idx':<4} {'surface':<12} {'lemma':<12} {'upos':<8} {'xpos':<30} {'head':<5} {'deprel':<15}")
    print(f"  {'-'*90}")
    for t in tokens:
        print(f"  {t['idx']:<4} {t['surface']:<12} {t['lemma']:<12} {t['upos']:<8} {t['xpos']:<30} {t['head']:<5} {t['deprel']:<15}")


# ---------------------------------------------------------------------------
# 2. UPOS distribution
# ---------------------------------------------------------------------------
sep("2. UPOS TAG DISTRIBUTION")
upos_counts = Counter(t["upos"] for t in all_tokens)
for tag, count in upos_counts.most_common():
    print(f"  {tag:<12} {count:>7}")


# ---------------------------------------------------------------------------
# 3. XPOS distribution (top 40)
# ---------------------------------------------------------------------------
sep("3. XPOS TAG DISTRIBUTION (top 40)")
xpos_counts = Counter(t["xpos"] for t in all_tokens)
for tag, count in xpos_counts.most_common(40):
    print(f"  {tag:<40} {count:>6}")


# ---------------------------------------------------------------------------
# 4. Deprel distribution
# ---------------------------------------------------------------------------
sep("4. DEPENDENCY RELATION DISTRIBUTION")
deprel_counts = Counter(t["deprel"] for t in all_tokens)
for rel, count in deprel_counts.most_common():
    print(f"  {rel:<20} {count:>7}")


# ---------------------------------------------------------------------------
# 5. How is する annotated?
# ---------------------------------------------------------------------------
sep("5. する — ALL FORMS (surface, lemma, upos, xpos)")
suru_variants = Counter()
suru_deprels  = Counter()
suru_xpos     = Counter()
for t in all_tokens:
    if "する" in t["surface"] or t["lemma"] in ("する", "為る"):
        suru_variants[f"surface={t['surface']:<10} lemma={t['lemma']:<10} upos={t['upos']:<6} xpos={t['xpos']}"] += 1
        suru_deprels[t["deprel"]] += 1
        suru_xpos[t["xpos"]] += 1

print("\n  Surface/lemma/upos/xpos combinations (top 30):")
for k, v in suru_variants.most_common(30):
    print(f"    {v:>4}x  {k}")
print("\n  Deprel distribution for する tokens:")
for k, v in suru_deprels.most_common():
    print(f"    {k:<20} {v}")
print("\n  XPOS distribution for する tokens:")
for k, v in suru_xpos.most_common():
    print(f"    {k:<40} {v}")


# ---------------------------------------------------------------------------
# 6. Verbal nouns — what xpos marks them?
# ---------------------------------------------------------------------------
sep("6. XPOS TAGS CONTAINING サ変 (verbal nouns)")
sa_hen = [(t["surface"], t["lemma"], t["upos"], t["xpos"], t["deprel"])
          for t in all_tokens if "サ変" in t["xpos"]]
print(f"\n  Total サ変 tokens: {len(sa_hen)}")
xpos_dist = Counter(t[3] for t in sa_hen)
print("\n  XPOS breakdown:")
for k, v in xpos_dist.most_common():
    print(f"    {k:<40} {v}")
deprel_dist = Counter(t[4] for t in sa_hen)
print("\n  Deprel breakdown for サ変 tokens:")
for k, v in deprel_dist.most_common():
    print(f"    {k:<20} {v}")
print("\n  Sample tokens (first 30):")
for surface, lemma, upos, xpos, deprel in sa_hen[:30]:
    print(f"    surface={surface:<12} lemma={lemma:<12} upos={upos:<6} xpos={xpos:<35} deprel={deprel}")


# ---------------------------------------------------------------------------
# 7. What are the dependents of する?
# ---------------------------------------------------------------------------
sep("7. DEPENDENTS OF する — deprel and upos of children")
idx_to_token = {}
sent_token_map = {}
for sent_id, text, tokens in sentences:
    for t in tokens:
        sent_token_map[(sent_id, t["idx"])] = t

child_patterns = Counter()
sample_suru_sents = []

for sent_id, text, tokens in sentences:
    tidx = {t["idx"]: t for t in tokens}
    for t in tokens:
        if t["lemma"] in ("する", "為る") and t["upos"] == "VERB":
            children = [c for c in tokens if c["head"] == t["idx"]]
            for c in children:
                child_patterns[f"deprel={c['deprel']:<15} upos={c['upos']:<8} xpos={c['xpos']}"] += 1
            if children and len(sample_suru_sents) < 10:
                sample_suru_sents.append((sent_id, text, t, children))

print(f"\n  Child deprel/upos/xpos patterns (top 30):")
for k, v in child_patterns.most_common(30):
    print(f"    {v:>4}x  {k}")

print(f"\n  Sample sentences with する and its children (10 examples):")
for sent_id, text, suru_tok, children in sample_suru_sents:
    print(f"\n  [{sent_id}] {text}")
    print(f"    する: surface={suru_tok['surface']:<8} lemma={suru_tok['lemma']:<8} xpos={suru_tok['xpos']}")
    for c in children:
        print(f"      child: surface={c['surface']:<12} lemma={c['lemma']:<12} upos={c['upos']:<6} deprel={c['deprel']:<15} xpos={c['xpos']}")


# ---------------------------------------------------------------------------
# 8. Specific pattern: NOUN immediately before する
# ---------------------------------------------------------------------------
sep("8. NOUN + する SEQUENCES (surface adjacency)")
adjacent_patterns = []
for sent_id, text, tokens in sentences:
    for i in range(1, len(tokens)):
        prev = tokens[i-1]
        curr = tokens[i]
        if curr["lemma"] in ("する", "為る") and curr["upos"] == "VERB":
            if prev["upos"] in ("NOUN", "PROPN"):
                adjacent_patterns.append((sent_id, text, prev, curr))

print(f"\n  Total NOUN immediately before する: {len(adjacent_patterns)}")
print(f"\n  Sample (first 20):")
for sent_id, text, noun, suru in adjacent_patterns[:20]:
    print(f"  [{sent_id}]")
    print(f"    text  : {text}")
    print(f"    noun  : surface={noun['surface']:<12} lemma={noun['lemma']:<12} xpos={noun['xpos']:<35} deprel={noun['deprel']} → head={noun['head']}")
    print(f"    suru  : surface={suru['surface']:<12} lemma={suru['lemma']:<12} xpos={suru['xpos']:<35} deprel={suru['deprel']} → head={suru['head']}")
    print()


# ---------------------------------------------------------------------------
# 9. Specific pattern: NOUN + を + する sequences
# ---------------------------------------------------------------------------
sep("9. NOUN + を + する SEQUENCES")
wo_suru_patterns = []
for sent_id, text, tokens in sentences:
    for i in range(2, len(tokens)):
        t0 = tokens[i-2]
        t1 = tokens[i-1]
        t2 = tokens[i]
        if (t2["lemma"] in ("する", "為る") and t2["upos"] == "VERB"
                and t1["surface"] == "を"
                and t0["upos"] in ("NOUN", "PROPN")):
            wo_suru_patterns.append((sent_id, text, t0, t1, t2))

print(f"\n  Total NOUN+を+する sequences: {len(wo_suru_patterns)}")
print(f"\n  Sample (first 20):")
for sent_id, text, noun, wo, suru in wo_suru_patterns[:20]:
    print(f"  [{sent_id}]")
    print(f"    text  : {text}")
    print(f"    noun  : surface={noun['surface']:<12} lemma={noun['lemma']:<12} xpos={noun['xpos']:<35} deprel={noun['deprel']} → head={noun['head']}")
    print(f"    を    : deprel={wo['deprel']:<15} → head={wo['head']}")
    print(f"    suru  : surface={suru['surface']:<12} xpos={suru['xpos']:<35} deprel={suru['deprel']} → head={suru['head']}")
    print()


# ---------------------------------------------------------------------------
# 10. Case particle surface forms and their deprels
# ---------------------------------------------------------------------------
sep("10. CASE PARTICLE INVENTORY (surface, deprel, frequency)")
particles = Counter()
for t in all_tokens:
    if t["upos"] == "ADP" or "助詞" in t["xpos"]:
        particles[f"surface={t['surface']:<8} deprel={t['deprel']:<15} xpos={t['xpos']}"] += 1
for k, v in particles.most_common(30):
    print(f"  {v:>6}x  {k}")


print("\n\nDiagnostic complete.")