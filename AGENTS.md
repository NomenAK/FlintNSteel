# AGENTS.md

FlintNSteel : moteur d'inférence pour Recursive Flint sur une RTX 3090 sous Windows. Commencer par
[README.md](README.md), puis [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) et [docs/ROADMAP.md](docs/ROADMAP.md).

## Règles

- Un modèle (Flint), un GPU (sm_86), un OS (Windows). Pas d'abstraction multi-modèle ni multi-backend.
- Pas de chiffre sans mesure. Chaque chiffre dans la doc indique la machine, le fichier, le commit et les options.
  Une estimation est marquée comme telle.
- Pas de changement de défaut sans A/B en paires alternées (5 paires au minimum, médianes, paires gagnées).
  Voir [docs/BENCHMARKING.md](docs/BENCHMARKING.md).
- Un kernel n'entre qu'avec son test de parité contre ggml, sur des blocs réels du GGUF.
- Une option qui change les bits de sortie est opt-in tant que sa qualité n'est pas mesurée contre Q8_0.
- Le code transcrit de ggml, Strata ou rec garde sa licence et indique en tête le fichier et la révision d'origine
  ([docs/SOURCES.md](docs/SOURCES.md)).
- Toute décision d'architecture est consignée dans [docs/DECISIONS.md](docs/DECISIONS.md).

## Build (prévu)

MSVC BuildTools 2022 (`vcvars64.bat`), CUDA 13.4, Ninja, `-DCMAKE_CUDA_ARCHITECTURES=86-real`. Les modèles vont
dans `E:\models\flint\` (NVMe Gen5) et ne sont jamais ajoutés au dépôt.
