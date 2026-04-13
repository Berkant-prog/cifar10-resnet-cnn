# Classification d'images CIFAR-10 avec des CNNs

Projet de deep learning comparant quatre architectures de réseaux de neurones convolutifs (CNN) sur le dataset CIFAR-10, avec une progression pédagogique allant d'un modèle baseline jusqu'à une ResNet custom avec scheduler avancé.

---

## Présentation du projet

Ce projet explore la classification supervisée multiclasse sur images en entraînant et comparant quatre modèles de complexité croissante. Chaque modèle introduit une nouvelle technique pour améliorer les performances et réduire l'overfitting.

**Dataset :** [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) — 60 000 images RGB 32×32 px réparties en 10 classes équilibrées.  
**Framework :** TensorFlow / Keras  
**Langage :** Python 3

---

## Structure du projet

```
projet_CNN_cifar10/
│
├── projet_CNN_cifar10_v4.ipynb   # Notebook principal
│
├── cnn_base.keras                # Modèle 1 sauvegardé
├── cnn_dropout.keras             # Modèle 2 sauvegardé
├── cnn_augmented.keras           # Modèle 3 sauvegardé
├── resnet_ultimate.keras         # Modèle 4 sauvegardé
├── best_resnet_ultimate.keras    # Meilleur checkpoint du modèle 4
│
├── history_cnn_base.pkl          # Historiques d'entraînement
├── history_cnn_dropout.pkl
├── history_cnn_augmented.pkl
├── history_cnn_deep.pkl
├── all_histories.pkl             # Tous les historiques agrégés
├── all_results.pkl               # Tous les résultats agrégés
│
└── *.png                         # Graphiques générés
    ├── exemples_dataset.png
    ├── distribution_classes.png
    ├── data_augmentation_exemples.png
    ├── comparaison_modeles.png
    ├── tradeoff_performance_temps.png
    ├── accuracy_par_classe.png
    ├── matrice_confusion.png
    ├── exemples_predictions.png
    └── synthese_comparative.png
```

---

## Dataset : CIFAR-10

| Propriété | Valeur |
|---|---|
| Nombre de classes | 10 |
| Images totales | 60 000 |
| Résolution | 32 × 32 pixels (RGB) |
| Répartition | 50 000 train / 10 000 test |
| Équilibre | 6 000 images par classe |
| Chargement | `keras.datasets.cifar10` (automatique) |

**Classes :** Avion · Voiture · Oiseau · Chat · Cerf · Chien · Grenouille · Cheval · Bateau · Camion

### Prétraitement

| Étape | Traitement | Justification |
|---|---|---|
| Normalisation | `/ 255.0` → [0, 1] | Stabilise la convergence |
| Aplatissement labels | `.flatten()` | CIFAR-10 retourne (N, 1), on veut (N,) |
| Split | 85% train / 15% val | Estimation fiable de la généralisation |

---

## Les 4 modèles

### Modèle 1 — CNN Baseline

Architecture minimale pour établir une référence de performance. Deux blocs Conv+Pool suivis d'un classifier Dense. Aucune régularisation.

```
Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Flatten → Dense(128) → Dense(10)
```

- **Optimizer :** Adam (lr=1e-3)
- **Epochs :** 20
- **Usage :** référence, mesure de l'overfitting de base

---

### Modèle 2 — CNN + Dropout

Même architecture que le Baseline, avec l'ajout de couches **Dropout** après chaque bloc convolutif (rate=0.25) et avant la sortie (rate=0.5) pour forcer la redondance des représentations et réduire l'overfitting.

```
Conv2D(32) → MaxPool → Dropout(0.25)
→ Conv2D(64) → MaxPool → Dropout(0.25)
→ Flatten → Dense(128) → Dropout(0.5) → Dense(10)
```

- **Optimizer :** Adam (lr=1e-3)
- **Epochs :** 20

---

### Modèle 3 — CNN + Data Augmentation (ImageDataGenerator)

Trois blocs convolutifs avec **BatchNormalization** et augmentation de données via `ImageDataGenerator`. L'augmentation est appliquée **uniquement sur le training set**, jamais sur la validation ni le test.

```
Conv2D(32) → BN → MaxPool → Dropout(0.15)
→ Conv2D(64) → BN → MaxPool → Dropout(0.15)
→ Conv2D(128) → BN → Dropout(0.2)
→ Flatten → Dropout(0.5) → Dense(256) → Dense(10)
```

**Configuration `ImageDataGenerator` (calibrée pour 32×32 px) :**

| Paramètre | Valeur | Pourquoi pas plus ? |
|---|---|---|
| `rotation_range` | 15° | 40° détruit l'info sur de si petites images |
| `width_shift_range` | 0.1 | Décalage modéré pour 32px |
| `height_shift_range` | 0.1 | Idem |
| `shear_range` | 0.1 | Idem |
| `zoom_range` | 0.1 | Idem |
| `horizontal_flip` | True | Les objets CIFAR-10 sont symétriques |
| `fill_mode` | `'nearest'` | Remplissage par pixel voisin |


- **Optimizer :** Adam (lr=1e-4)
- **Epochs :** 50 + EarlyStopping (patience=8)

---

### Modèle 4 — ResNet Ultimate (modèle état de l'art)

Le modèle le plus avancé, combinant trois techniques complémentaires :

#### Architecture — ResNet custom

Blocs **résiduels** avec *skip connections* : le gradient peut remonter directement, permettant d'entraîner des réseaux profonds sans dégradation des performances.

```
Input (32×32×3)
  │
  └─► Stem : Conv2D(32) → BN → ReLU
        │
        ├─► Stage 1 : 2× ResBlock(32)            → 32×32
        ├─► Stage 2 : 2× ResBlock(64,  stride=2) → 16×16
        ├─► Stage 3 : 2× ResBlock(128, stride=2) →  8×8
        └─► Stage 4 : 2× ResBlock(256, stride=2) →  4×4
              │
              └─► GlobalAvgPool → Dropout(0.4) → Dense(10, softmax)
```

Chaque `ResBlock` suit le schéma :
```
x ──► Conv → BN → ReLU → Conv → BN ──► Add → ReLU
│                                        ▲
└───────────── skip connection ───────────┘
```

#### Optimiseur — AdamW

Adam avec **weight decay découplé** (`weight_decay=1e-4`). Contrairement à L2 classique, le decay est appliqué directement sur les poids sans interagir avec le gradient, ce qui donne une meilleure régularisation.

#### Scheduler — Warmup + Cosine Decay

Planning proactif du learning rate en deux phases :

1. **Warmup** (5 époques) : montée linéaire de 0 → 1e-3 pour stabiliser le début de l'entraînement
2. **Cosine Decay** : descente douce suivant une courbe cosinus de 1e-3 → 1e-6

```
LR
1e-3 ─────╮
          │╲
          │ ╲
          │  ╲_____
1e-6 ─────────────── epochs
     warmup  cosine
```

- **Epochs max :** 80 + EarlyStopping sur `val_accuracy` (patience=15)
- **Checkpoint :** sauvegarde automatique du meilleur modèle

---

## Installation

```bash
pip install tensorflow matplotlib numpy scikit-learn seaborn Pillow tqdm
```

Le dataset CIFAR-10 est téléchargé automatiquement à la première exécution via :
```python
keras.datasets.cifar10.load_data()
```

---

## Résultats 

| Modèle | Test Accuracy (approx.) |
|---|---|
| CNN Base | ~66% |
| CNN + Dropout | ~71% |
| CNN + Data Aug. | 78% |
| ResNet Ultimate | 91% |

---

## Points pédagogiques clés

- **L'overfitting** se lit dans l'écart entre la courbe train et la courbe val — plus le gap est grand, plus le modèle mémorise au lieu de généraliser.
- **L'augmentation de données doit être proportionnelle à la résolution** : des rotations de 40° sur des images 32×32 détruisent l'information (résultats <20%).
- **La validation et le test ne doivent jamais être augmentés** — règle fondamentale respectée dans tous les modèles.
- **Les skip connections** permettent aux réseaux profonds d'entraîner efficacement sans vanishing gradient.
- **Le Cosine Decay avec Warmup** est plus régulier que `ReduceLROnPlateau` et adapté aux longs entraînements.

---

## Limites et pistes d'amélioration

**Limites actuelles :**
- Confusion fréquente entre classes proches (chat/chien, voiture/camion)
- Résolution 32×32 limitée — les transformations d'augmentation doivent rester légères
- Classification uniquement — pas de localisation d'objet
---

## Dépendances

| Package | Usage |
|---|---|
| `tensorflow` | Construction et entraînement des modèles |
| `numpy` | Manipulation des tableaux |
| `matplotlib` | Visualisations et courbes d'apprentissage |
| `seaborn` | Matrice de confusion |
| `scikit-learn` | `train_test_split`, `confusion_matrix` |
| `Pillow` | Traitement d'images |
| `pickle` | Sauvegarde des historiques |
