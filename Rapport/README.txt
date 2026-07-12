RAPPORT — Classification de tumeurs cérébrales (IRM)
====================================================

COMPILE
  pdflatex main.tex        (deux passes pour la table des matières)
  # ou:  latexmk -pdf main.tex

STRUCTURE ACADÉMIQUE (M1 IA / Data Science)
  Page de garde
  Résumé
  Introduction Générale
  Chapitre 1  — État de l'art
  Chapitre 2  — Jeu de données et prétraitement
  Chapitre 3  — Extraction des caractéristiques
  Chapitre 4  — Optimisation des descripteurs (Phase 1)
  Chapitre 5  — Construction des ensembles (Phase 2)
  Chapitre 6  — Benchmark des modèles ML
  Chapitre 7  — Validation statistique
  Chapitre 8  — Analyse et discussion
  Chapitre 9  — Ouverture Deep Learning / VLMs
  Conclusion Générale
  Bibliographie
  Annexes A–E

FICHIERS
  main.tex                  préambule + assemblage
  results.tex               >>> métriques à remplir après benchmark <<<
  sections/00_resume.tex
  sections/01_introduction.tex   (Introduction Générale — texte fourni)
  chapters/ch01_*.tex … ch09_*.tex
  sections/10_conclusion.tex
  sections/11_bibliographie.tex
  annexes/annexes.tex

WORKFLOW
  Envoyer le texte section par section ; chaque bloc remplace les
  marqueurs « À compléter » dans le fichier LaTeX correspondant.

ANCIEN CONTENU (référence, non compilé)
  sections/01_problem.tex, 02_dataset.tex, 03_strategy.tex,
  04_descriptors.tex, 05_hyperparams.tex, 06_fusion_benchmark.tex,
  07_statistical_validation.tex, 08_results_discussion.tex,
  09_conclusion.tex

MÉTRIQUES
  Ouvrir results.tex et remplacer les \TODO{...} par les valeurs finales.
