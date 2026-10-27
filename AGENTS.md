# AGENTS.md

FlintNSteel : moteur d'inférence pour Recursive Flint sur une RTX 3090 sous Windows. Commencer par
[README.md](README.md), puis [docs/ENGINE.md](docs/ENGINE.md) et [docs/BUILD.md](docs/BUILD.md).

## Règles

- Un modèle (Flint), un GPU (sm_86), un OS (Windows). Pas d'abstraction multi-modèle ni multi-backend.
- Pas de chiffre sans mesure. Chaque chiffre dans la doc indique la machine, le fichier, le commit et les options.
  Une estimation est marquée comme telle.
- Pas de changement de défaut sans A/B en paires alternées (5 paires au minimum, médianes, paires gagnées).
  Voir [docs/BENCHMARKING.md](docs/BENCHMARKING.md).
- Un kernel n'entre qu'avec son test de parité contre ggml, sur des blocs réels du GGUF.
- Une option qui change les bits de sortie est opt-in tant que sa qualité n'est pas mesurée contre Q8_0.
- Le code transcrit de ggml, Strata, rec, XGrammar ou exllamav3 garde sa licence et indique en tête le fichier et la
  révision d'origine ([THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)).
- Toute décision d'architecture est consignée dans le journal de décisions local (`perso/docs/DECISIONS.md`, hors dépôt).

## Build

MSVC 2022, CUDA 13.4, Ninja, `-DCMAKE_CUDA_ARCHITECTURES=86-real` : voir [docs/BUILD.md](docs/BUILD.md). Les modèles
(`*.gguf`, `*.safetensors`) ne sont jamais ajoutés au dépôt.
