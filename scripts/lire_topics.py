"""Affiche le CONTENU BRUT des topics du dernier run BERTrend.

Sert le garde-fou anti-habillage : on juge un clustering sur les documents
reels, jamais sur les labels generes. A lancer apres un entrainement :

    cd /workspace/BERTrend && /workspace/.venv/bin/python scripts/lire_topics.py

Options :
    --topics N    nombre de topics affiches (defaut 6)
    --docs N      documents montres par topic (defaut 5)
    --largeur N   troncature d'un document en caracteres (defaut 200)
"""

import argparse
import glob
import os
import textwrap
from collections import Counter

import pandas as pd

from bertrend import BERTREND_BASE_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", type=int, default=6)
    ap.add_argument("--docs", type=int, default=5)
    ap.add_argument("--largeur", type=int, default=200)
    args = ap.parse_args()

    base = BERTREND_BASE_DIR or os.path.join(os.path.dirname(__file__), "..", ".bertrend")
    motif = os.path.join(str(base), "**", "doc_info_df.pkl")
    tous = glob.glob(motif, recursive=True)
    if not tous:
        print("Aucun modele trouve sous", base)
        print("As-tu bien lance un entrainement ?")
        return

    # Les runs successifs s'empilent au meme endroit : on ne garde que le
    # dernier (fichiers ecrits dans les 15 min autour du plus recent).
    recent = max(os.path.getmtime(f) for f in tous)
    fichiers = sorted(f for f in tous if os.path.getmtime(f) > recent - 900)
    print(len(fichiers), "fenetres du dernier run (sur", len(tous), "au total)")

    lignes = []
    resume = []
    for f in fichiers:
        periode = os.path.basename(os.path.dirname(f))
        df = pd.read_pickle(f)
        compte = Counter(df["Topic"].tolist())
        n_topics = len([t for t in compte if t != -1])
        resume.append((periode, n_topics, compte.get(-1, 0), len(df)))
        for topic, g in df.groupby("Topic"):
            if topic == -1:
                continue
            lignes.append((len(g), periode, topic, g["Paragraph"].tolist()))

    print()
    print("=" * 78)
    print(" VUE D'ENSEMBLE PAR FENETRE")
    print("=" * 78)
    print(" periode      topics   bruit   docs")
    for periode, n_topics, bruit, total in resume:
        print("  %-12s %5d %7d %6d" % (periode, n_topics, bruit, total))

    lignes.sort(key=lambda x: -x[0])
    print()
    print("=" * 78)
    print(" LES", args.topics, "PLUS GROS TOPICS - CONTENU BRUT")
    print(" Question a se poser : ces documents parlent-ils du MEME sujet ?")
    print("=" * 78)

    for taille, periode, topic, docs in lignes[: args.topics]:
        print()
        print("###", periode, "| topic", topic, "|", taille, "documents")
        print("-" * 78)
        for d in docs[: args.docs]:
            texte = " ".join(str(d).split())
            print(" *", textwrap.shorten(texte, args.largeur, placeholder=" [...]"))


if __name__ == "__main__":
    main()
