# Déployer BERTrend sur RunPod (GPU loué à l'heure)

Guide pour faire tourner BERTrend avec GPU sur RunPod, embedder un dataset puis rapatrier les résultats en local pour itérer sur les paramètres sans frais.

## Prérequis

- Compte RunPod : https://www.runpod.io/
- Crédit ajouté (~$5 suffisent largement pour un run complet)
- Docker Hub (optionnel, si tu veux pré-pousser ton image)

## 1. Démarrer un Pod GPU

Sur la console RunPod :

1. **Deploy** → **GPU Pods**
2. **GPU type** : RTX 4090 ou L4 (~$0.30-0.50/h). Pour Solon-large, T4 (16GB VRAM) suffit aussi mais sera plus lent.
3. **Template** : « PyTorch 2.x » (vient avec CUDA, Python, Docker pré-installés)
4. **Container disk** : 30 GB (suffit pour les images + le modèle HF + les embeddings cachés)
5. **Volume disk** : 0 GB (rien à persister entre sessions, on télécharge à la fin)
6. **Expose HTTP ports** : ajoute `8084` (Weak Signals) et `6464` (embedding server). RunPod te donnera des URLs publiques type `https://<pod-id>-8084.proxy.runpod.net`.
7. **Deploy On-Demand**

Une fois le Pod « Running » : clique **Connect** → **Start Web Terminal** (ou utilise SSH).

## 2. Cloner le repo et configurer

Dans le terminal du Pod :

```bash
cd /workspace
git clone https://github.com/HamoudeFakhoury01/BERTrend.git
cd BERTrend
```

Crée le `.env` avec ta clé OpenAI (si tu veux les features LLM) :

```bash
cat > .env <<'EOF'
OPENAI_API_KEY=sk-proj-...   # mets ta vraie clé ici
OPENAI_DEFAULT_MODEL=gpt-4o-mini
EOF
```

## 3. Build et démarrage avec GPU

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

Le `--build` force la recompilation de l'image avec ton code local (donc tous tes patches). Compte 10-15 min pour le premier build (téléchargement des couches PyTorch + dépendances).

Vérifie que les 2 services sont healthy :

```bash
docker compose ps
# attendu : bertrend (Up healthy) + bertrend-embedding-server (Up healthy)
```

Si `embedding-server` reste en `health: starting` longtemps, c'est normal — il télécharge le modèle Solon-large depuis HuggingFace au premier démarrage (~2.8 GB).

Suivre les logs :

```bash
docker compose logs -f embedding_server   # voir le chargement du modèle
docker compose logs -f bertrend           # voir Streamlit démarrer
```

## 4. Utiliser BERTrend via l'URL publique

Sur la console RunPod, dans la liste des ports exposés du Pod, copie l'URL pour le port 8084 (genre `https://abc123-8084.proxy.runpod.net`).

Ouvre cette URL dans ton navigateur — tu vois la démo Weak Signals comme en local.

Workflow :

1. **Upload ton fichier Excel** (le full 89818 lignes)
2. **Sélectionne les colonnes** texte + timestamp, Submit
3. (Ferme la popup du bug si elle revient — voir notes en bas)
4. Règle `Minimum Characters` à 50 (filtre le bruit)
5. Clique **Embed Documents** — environ **15-30 min sur GPU T4/L4**
6. Une fois le succès affiché, **clique « Save State »** dans la sidebar (CRITIQUE — sinon tu perds tout en éteignant le Pod)
7. Tu peux directement entraîner BERTopic + voir les weak signals sur le Pod, ou rapatrier les embeddings et le faire en local

## 5. Récupérer les embeddings et arrêter le Pod

Depuis ta machine locale (pas le Pod) :

```bash
# Récupère le cache (embeddings.npy + state.pkl)
runpodctl receive <pod-id>:/workspace/BERTrend/.bertrend/cache ./.bertrend/cache_from_runpod
```

Ou via l'interface RunPod : **Files** → naviguer vers `/workspace/BERTrend/.bertrend/cache/` → Download.

Une fois les fichiers en local, copie-les dans ton `.bertrend/cache/` local. Lance ton BERTrend local, clique **Restore Previous Run** dans la sidebar — tu repars sur les embeddings calculés sur le GPU.

**Stoppe le Pod** sur RunPod (`Stop` puis `Terminate` si tu n'en as plus besoin) pour ne plus payer.

## 6. Notes / troubleshooting

### Le modèle d'embedding met 2-3 min à charger au premier démarrage
Normal — Solon-large fait 2.8 GB et est téléchargé depuis HuggingFace. Ensuite il est en cache.

### `health: starting` qui ne passe pas à `healthy` après 5 min
Le healthcheck de l'embedding-server tape sur `/health` qui n'est dispo qu'une fois le modèle chargé. Patience.

### La popup « Column Selection » revient à chaque clic
Bug connu de la démo BERTrend (le mapping de colonnes n'est pas persisté entre reruns Streamlit). Ferme avec le X — **ne clique pas Submit avec les valeurs par défaut** (`id` / `id`), ça corromprait ton dataframe. Le calcul tourne quand même sur les bonnes données.

### Erreur 401 « Could not validate credentials »
Vérifie que `BERTREND_SECRET_KEY` est bien défini dans `docker-compose.yml` env du service `embedding_server` (sinon les 2 workers uvicorn génèrent des clés différentes au démarrage et 50% des requêtes échouent).

### Erreur GPU « no NVIDIA driver detected »
Vérifie que le Pod RunPod a bien un GPU attribué (`nvidia-smi` dans le terminal). Si oui, vérifie que `nvidia-container-toolkit` est dispo (normalement inclus dans le template PyTorch).

### Le Pod a planté en cours de run
Avec nos patches (`embedding_client.py`), le cache incrémental sauve chaque batch dans `.bertrend/cache/embeddings_partial/batch_NNNN.npy`. Au redémarrage, les batches déjà OK sont skip et on reprend là où ça s'était arrêté.

## 7. Coûts estimés

| Phase | Durée | GPU | Coût |
|---|---|---|---|
| Build initial (téléchargement images) | 10-15 min | idle | ~$0.10 |
| Embedding 89k docs | 15-30 min | T4/L4 | ~$0.15-0.25 |
| Exploration / params | 1-2h | idle (CPU suffit) | ~$0.30-0.60 |
| **Total run complet** | ~2-3h | | **~$0.55-1** |

Pour itérer plusieurs jours sans embedder à nouveau : **rapatrie le cache et bosse en local** — coût $0.

## 8. Pour aller plus loin

Si tu finis par utiliser BERTrend en continu, l'étape suivante c'est OVH :
- VPS GPU dédié avec L4 ou A2 : ~150-200€/mois
- Ou VPS CPU 16GB pour iteration only (les embeddings une fois calculés ne nécessitent plus de GPU) : ~30-50€/mois
- Aligné avec la stack souveraine FR de CityMood.
