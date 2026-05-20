# TP — Segmentation de lignes manuscrites avec U-Net (TensorFlow / Keras)
### Énoncé

## Contexte et objectif

La segmentation de lignes est une brique fondamentale des pipelines HTR (*Handwritten Text Recognition*). Avant de reconnaître du texte manuscrit, il faut isoler chaque ligne d'écriture dans la page. Ce TP vous guide de la préparation des données jusqu'à l'évaluation d'un U-Net entraîné sur **RIMES**, un corpus de lettres manuscrites en français.

**Approche retenue** : U-Net sur RIMES (Zenodo, version complète), avec une validation de généralisation sur READ 2016 en bonus.

> **Convention TF/Keras** : contrairement à PyTorch qui utilise le format `(B, C, H, W)`, Keras adopte le format **channels-last** `(B, H, W, C)` pour tous les tenseurs image. Toute l'architecture et le pipeline de données doivent respecter cette convention.

---

## Partie 0 — Mise en place de l'environnement

Commencez par installer les dépendances nécessaires : **TensorFlow** (≥ 2.12, avec support GPU si disponible), **Albumentations** pour les augmentations géométriques avancées, **Pillow** et **lxml** pour la manipulation d'images et de fichiers XML, ainsi que **scikit-image** pour le post-traitement morphologique.

Vérifiez ensuite que TensorFlow détecte bien votre GPU avant de continuer.

Organisez votre projet selon la structure suivante :

```
tp_segmentation/
├── data/
│   ├── rimes_raw/
│   │   └── DVD1/         # images JPEG et XMLs à plat
│   └── rimes_masks/      # masques générés (étape 1)
├── utils.py
├── dataset.py
├── model.py
├── train.py
└── evaluate.py
```

---

## Partie 1 — Données RIMES : de l'annotation XML au masque binaire

### 1.1 Téléchargement des données

Le corpus RIMES complet est disponible sur Zenodo :  
[zenodo.org/records/10812725](https://zenodo.org/records/10812725)

Téléchargez l'archive `Images_Courriers.zip`. Elle contient trois dossiers (`DVD1`, `DVD2`, `DVD3`) représentant chacun un sous-ensemble du corpus. Pour ce TP, on travaille sur **un seul dossier**, par exemple `DVD1`. Décompressez-le dans `data/rimes_raw/DVD1/`.

Chaque page est représentée par une paire de fichiers **à plat** dans ce dossier : une image JPEG et un fichier XML du même nom. Par exemple `1_L.jpg` (la lettre n°1) et `1_L.xml` (ses annotations). Les suffixes `L`, `Q` et `F` désignent respectivement les lettres, questionnaires et fax.

### 1.2 Comprendre les annotations RIMES (Zenodo)

Le XML de chaque page décrit des **blocs de texte** (adresse expéditeur, corps de la lettre, signature…) et non des lignes individuelles. Chaque balise `<box>` donne les coordonnées en coins opposés du bloc — `top_left_x`, `top_left_y`, `bottom_right_x`, `bottom_right_y` — ainsi que son contenu transcrit dans un champ `<text>`, où les sauts de ligne sont représentés par `\n` :

```xml
<box top_left_x="76" top_left_y="91"
     bottom_right_x="1250" bottom_right_y="579">
  <type>Coordonnées Expéditeur</type>
  <text>Maxime Granier\n13 Grand rue\n57370 Dames et Quatre Vents</text>
</box>
```

Il n'existe donc pas de bounding-box de ligne explicite. Pour générer des masques de lignes, on **estime** leur position en divisant la hauteur de chaque bloc par le nombre de lignes qu'il contient, déduit du nombre de `\n` dans le texte :

```
nb_lignes     = text.count("\n") + 1
hauteur_ligne = (bottom_right_y - top_left_y) / nb_lignes
```

Chaque ligne estimée se voit attribuer une zone verticale d'égale hauteur, à laquelle on applique une dilatation verticale de 10 % pour couvrir ascendantes et descendantes.

### 1.3 Génération des masques binaires

Dans `utils.py`, implémentez deux fonctions :

**`xml_to_mask_rimes(xml_path, img_path, out_path)`** — Parse le fichier XML de la page, ouvre l'image pour en connaître les dimensions, et construit un masque NumPy initialisé à zéro. Pour chaque balise `<box>`, récupérez les quatre coordonnées de coin et le contenu du champ `<text>`. Calculez le nombre de lignes estimées (`nb_lines = text.count("\n") + 1`), puis pour chacune d'elles calculez sa zone verticale par division uniforme de la hauteur du bloc, et remplissez la zone correspondante dans le masque avec une dilatation verticale de 10 %. Sauvegardez le résultat en PNG avec des valeurs 0 ou 255.

**`batch_generate_masks_rimes(data_dir, mask_dir)`** — Itère sur tous les fichiers XML du dossier `data_dir`. Pour chacun, cherche l'image JPEG de même nom dans le même dossier (structure à plat) et appelle `xml_to_mask_rimes`. Les masques sont sauvegardés dans `mask_dir`.

> **Question 1** : Ouvrez quelques masques générés avec Matplotlib en les superposant à leurs images d'origine. L'estimation par division uniforme du bloc est-elle visuellement correcte sur le corps de la lettre ? Observez-vous des incohérences sur les blocs courts (adresse, signature) ? Proposez un critère pour filtrer ces blocs peu fiables lors de la génération des masques.

---

## Partie 2 — Pipeline de données avec `tf.data`

### 2.1 Stratégie de découpage en patches

Les pages RIMES font environ 1700 × 2200 pixels. Les redimensionner uniformément écraserait les détails fins de l'écriture. Préférez une stratégie de **découpage en patches de 512 × 512 pixels** : à l'entraînement, un patch aléatoire est extrait de chaque page ; en validation, un patch central est découpé de manière déterministe.

### 2.2 Augmentations avec Albumentations

`tf.image` ne fournit pas toutes les transformations nécessaires — en particulier `ElasticTransform`, indispensable pour l'écriture manuscrite. Utilisez la bibliothèque **Albumentations**, compatible via NumPy, pour construire deux pipelines d'augmentation :

- **Entraînement** : crop aléatoire 512×512, flip horizontal (probabilité faible), variation de luminosité/contraste, bruit gaussien, transformation élastique.
- **Validation** : crop central 512×512 uniquement.

Dans `dataset.py`, écrivez une fonction `load_and_augment(img_path, mask_path, training)` qui charge une image JPEG et son masque PNG avec PIL, applique le pipeline d'augmentation approprié, normalise l'image (µ = 0.5, σ = 0.25), puis retourne les deux tableaux NumPy au format channels-last `(H, W, 1)` en `float32`.

> **Question 2** : Pourquoi utilise-t-on `ElasticTransform` plutôt qu'une simple rotation pour des manuscrits ? Quel phénomène physique cette transformation simule-t-elle ?

### 2.3 Construction du `tf.data.Dataset`

`tf.data` n'accepte pas directement les fonctions Python qui effectuent des I/O disque — il faut encapsuler votre fonction de chargement via `tf.py_function`. Écrivez une fonction `make_tf_dataset(img_paths, mask_paths, batch_size, training)` qui :

1. Construit un `Dataset` à partir des listes de chemins avec `from_tensor_slices`.
2. Applique la fonction de chargement en parallèle avec `map` et `AUTOTUNE`.
3. Mélange les exemples à l'entraînement avec `shuffle`.
4. Regroupe en batchs et active le préchargement avec `prefetch(AUTOTUNE)`.

N'oubliez pas d'appeler `set_shape` après `tf.py_function` pour que Keras connaisse statiquement les dimensions des tenseurs.

Pour le split, pointez `data_dir` vers le dossier DVD choisi (`data/rimes_raw/DVD1`), listez les masques générés dans `rimes_masks/`, reconstituez les noms de fichiers JPEG correspondants (même stem, extension `.jpg` dans `data_dir`), puis divisez le corpus en 80 % d'entraînement et 20 % de validation avec `train_test_split`. Instanciez ensuite vos deux datasets.

---

## Partie 3 — Architecture U-Net en Keras

### 3.1 Rappel de l'architecture

Le U-Net est composé de deux chemins symétriques :

```
Encodeur (contracting path)       Décodeur (expanding path)
─────────────────────────────     ──────────────────────────
Conv3×3 → BN → ReLU (×2)         Conv2DTranspose 2×2
MaxPool2×2                    ←── Skip connection ──→ Conv3×3 (×2)
...                               ...
                  Bottleneck
                  Conv3×3 (×2)
```

L'**encodeur** extrait des caractéristiques de plus en plus abstraites en réduisant la résolution spatiale par max-pooling. Le **décodeur** remonte progressivement en résolution par transposition de convolution. Les **skip connections** transmettent les cartes de caractéristiques de chaque niveau de l'encodeur vers le niveau correspondant du décodeur, préservant les détails spatiaux fins (bords de lignes) qui seraient autrement perdus.

### 3.2 Implémentation avec la Functional API Keras

Dans `model.py`, implémentez les trois briques suivantes :

**`double_conv(x, filters)`** — Applique deux Conv2D 3×3 consécutives avec `padding="same"`, chacune suivie d'une BatchNormalization et d'une activation ReLU. Le `use_bias=False` est recommandé en présence de BatchNorm.

**`encoder_block(x, filters)`** — Appelle `double_conv`, puis applique un MaxPooling 2×2. Retourne le tenseur avant pooling (le *skip*) et le tenseur après pooling.

**`decoder_block(x, skip, filters)`** — Applique une Conv2DTranspose 2×2 avec stride 2 pour doubler la résolution spatiale. Concatène le résultat avec le skip correspondant (attention aux dimensions si la page a une taille impaire), puis applique `double_conv`.

Assemblez ensuite le modèle complet avec `build_unet(input_shape, base_features=64)` via la Functional API : empilez 4 niveaux d'encodeur, un bottleneck, puis 4 niveaux de décodeur. La couche de sortie est une Conv2D 1×1 avec activation **sigmoid** (le modèle Keras retourne directement des probabilités, contrairement aux logits de PyTorch).

> **Question 3** : Affichez le résumé du modèle avec `model.summary()`. Combien de paramètres entraînables obtenez-vous pour `base_features=64` ? Refaites le calcul pour `base_features=32`. Quel compromis capacité/vitesse cela implique-t-il pour un entraînement sur RIMES ?

---

## Partie 4 — Fonction de perte

### 4.1 Problème du déséquilibre de classes

Sur une page manuscrite typique, environ 85 % des pixels appartiennent au fond et seulement 15 % à des lignes de texte. Une **Binary Cross-Entropy (BCE) seule** convergera vers une solution triviale qui prédit presque toujours le fond, atteignant ainsi une haute accuracy sans apprendre à détecter les lignes.

### 4.2 Dice Loss

La **Dice Loss** est conçue pour pallier ce déséquilibre : elle mesure le recouvrement entre la prédiction et la vérité terrain, indépendamment de la proportion de chaque classe. Elle vaut 0 pour une prédiction parfaite et 1 pour une prédiction nulle.

Dans `utils.py` ou `train.py`, implémentez `dice_loss(y_true, y_pred)` en utilisant `tf.reduce_sum` sur les axes spatiaux et de batch. Rappel de la formule :

$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \cdot \sum(\hat{y} \cdot y) + \varepsilon}{\sum \hat{y} + \sum y + \varepsilon}$$

où ε est un petit terme de stabilité numérique (par exemple 1e-6).

### 4.3 Perte combinée BCE + Dice

Implémentez `combined_loss(y_true, y_pred, alpha=0.5)` qui pondère les deux termes : la BCE apporte une supervision dense pixel à pixel tandis que la Dice Loss force le modèle à bien couvrir les zones de texte.

> **Question 4** : Faites varier `alpha` entre 0.3 et 0.7 sur quelques époques. Quel effet observez-vous sur la courbe de loss en début d'entraînement ? Quel terme domine et pourquoi ?

---

## Partie 5 — Compilation et entraînement

### 5.1 Métrique personnalisée IoU

L'IoU (*Intersection over Union*) est la métrique de référence en segmentation. Keras ne fournit pas d'IoU binaire cumulatif avec seuillage — implémentez-le comme une sous-classe de `keras.metrics.Metric`.

La classe doit maintenir deux accumulateurs (`intersection` et `union`) mis à jour à chaque batch, et calculer leur ratio dans `result()`. Pensez à appliquer un seuil (0.5) aux prédictions avant le calcul.

### 5.2 Compilation du modèle

Compilez le modèle avec :
- **Optimiseur** : AdamW avec weight decay de 1e-4 et un scheduler `CosineDecay` — équivalent du `CosineAnnealingLR` de PyTorch — sur 30 époques.
- **Perte** : votre `combined_loss`.
- **Métriques** : votre `IoUMetric`.

### 5.3 Callbacks et lancement de l'entraînement

Configurez deux callbacks avant d'appeler `model.fit()` :
- **`ModelCheckpoint`** : sauvegarde automatique du meilleur modèle (selon `val_loss`) au format `.keras`.
- **`TensorBoard`** : journalisation des courbes de loss et d'IoU pour les visualiser avec `tensorboard --logdir logs/`.

Lancez l'entraînement sur 30 époques et commentez l'évolution des courbes de loss et d'IoU sur les ensembles d'entraînement et de validation.

---

## Partie 6 — Métriques d'évaluation

Dans `evaluate.py`, implémentez une fonction `evaluate(model, dataset, threshold=0.5)` qui itère sur le dataset de validation et calcule pour chaque image :

- La **Pixel Accuracy** : proportion de pixels correctement classés.
- L'**IoU** : intersection sur union pour la classe "ligne".

Seuillage : binarisez les probabilités de sortie du modèle à 0.5 avant calcul. Affichez les moyennes sur l'ensemble de la validation.

> **Question 5** : L'IoU mesure-t-il bien la qualité de séparation entre les lignes ? Imaginez un cas où deux lignes adjacentes fusionnent dans le masque prédit : comment l'IoU réagit-il par rapport à la Pixel Accuracy ? Lequel des deux révèle mieux ce type d'erreur ?

---

## Partie 7 — Visualisation des prédictions

Dans `evaluate.py`, écrivez une fonction `visualize_prediction(model, dataset, idx, threshold)` qui affiche côte à côte, pour un exemple donné du dataset de validation :

1. L'image d'entrée (en niveaux de gris).
2. Le masque de vérité terrain.
3. Le masque prédit binarisé.

Utilisez `matplotlib.pyplot` avec une colormap en niveaux de gris. Examinez plusieurs exemples et identifiez les types d'erreurs les plus fréquentes (fusions de lignes, zones manquées, bords mal définis).

---

## Partie 8 — Bonus : généralisation sur READ 2016

Chargez quelques pages de READ 2016 (manuscrits historiques) et appliquez votre modèle entraîné sur RIMES **sans ré-entraînement**.

Comme les pages complètes dépassent largement la taille d'entrée du modèle (512×512), utilisez une stratégie de **fenêtre glissante avec chevauchement** (*sliding window*) : découpez la page en patches avec un stride inférieur à la taille du patch (par exemple stride = 256), inférez sur chaque patch, puis reconstituez la carte de probabilités complète en moyennant les prédictions sur les zones de chevauchement.

> **Question Bonus** : Comparez visuellement les prédictions sur RIMES et sur READ 2016. Quels artefacts apparaissent sur les manuscrits historiques ? Pourquoi le *domain shift* entre manuscrits modernes et historiques pose-t-il problème, et quelles stratégies permettraient d'y remédier ?

---

## Mémento — Principales fonctions et classes utilisées

### Manipulation de données (PIL, lxml, NumPy)

| Fonction / Méthode | Rôle |
|---|---|
| `Image.open(path).convert("L")` | Charge une image en niveaux de gris (mode L = 8 bits) |
| `np.zeros((H, W), dtype=np.uint8)` | Crée un tableau vide pour le masque |
| `mask[y0:y1, x0:x1] = 255` | Remplit une zone rectangulaire dans le masque |
| `Image.fromarray(arr).save(path)` | Sauvegarde un tableau NumPy en PNG |
| `etree.parse(xml_path)` | Parse un fichier XML avec lxml |
| `root.iter("box")` | Itère sur toutes les balises `<box>` du XML RIMES |
| `element.get("top_left_x", 0)` | Récupère un attribut de coordonnée (défaut 0) |
| `box.find("text")` | Accède au champ `<text>` d'une balise `<box>` |
| `text.count("\n")` | Compte les sauts de ligne pour estimer le nombre de lignes |

### Augmentations (Albumentations)

| Fonction / Classe | Rôle |
|---|---|
| `A.Compose([...])` | Chaîne de transformations appliquées conjointement à l'image et au masque |
| `A.RandomCrop(h, w)` | Découpe un patch aléatoire |
| `A.CenterCrop(h, w)` | Découpe un patch centré (déterministe) |
| `A.HorizontalFlip(p=0.3)` | Flip horizontal avec probabilité p |
| `A.RandomBrightnessContrast(p)` | Variation aléatoire de luminosité et contraste |
| `A.GaussNoise(p)` | Ajout de bruit gaussien |
| `A.ElasticTransform(p, alpha, sigma)` | Déformation élastique simulant les variations naturelles de l'écriture |
| `aug(image=img, mask=mask)` | Applique le pipeline en synchronisant image et masque |

### Pipeline `tf.data`

| Fonction / Méthode | Rôle |
|---|---|
| `tf.data.Dataset.from_tensor_slices((imgs, masks))` | Crée un dataset à partir de listes de chemins |
| `dataset.shuffle(buffer_size)` | Mélange les exemples (buffer_size ≈ taille du dataset) |
| `dataset.map(fn, num_parallel_calls=AUTOTUNE)` | Applique une transformation en parallèle |
| `tf.py_function(func, inp, Tout)` | Encapsule une fonction Python dans le graphe TF |
| `tensor.set_shape([...])` | Définit statiquement la forme après `py_function` |
| `dataset.batch(batch_size)` | Regroupe les exemples en batchs |
| `dataset.prefetch(AUTOTUNE)` | Précharge le prochain batch pendant l'entraînement GPU |

### Construction du modèle (Keras Functional API)

| Fonction / Classe | Rôle |
|---|---|
| `keras.Input(shape=(...))` | Point d'entrée du modèle |
| `layers.Conv2D(filters, kernel, padding, use_bias)` | Convolution 2D |
| `layers.Conv2DTranspose(filters, kernel, strides)` | Convolution transposée (upsampling appris) |
| `layers.MaxPooling2D(2)` | Sous-échantillonnage par maximum (facteur 2) |
| `layers.BatchNormalization()` | Normalisation par batch |
| `layers.Activation("relu")` | Activation ReLU |
| `layers.Concatenate()([x1, x2])` | Concaténation de tenseurs (skip connection) |
| `layers.Resizing(h, w)` | Recadrage spatial si dimensions impaires |
| `keras.Model(inputs, outputs)` | Instancie le modèle à partir du graphe |
| `model.summary()` | Affiche l'architecture et le nombre de paramètres |

### Fonctions de perte et métriques

| Fonction / Classe | Rôle |
|---|---|
| `keras.losses.BinaryCrossentropy()` | BCE pour sortie sigmoid (probabilités) |
| `tf.reduce_sum(tensor, axis=[...])` | Somme sur des axes spécifiés |
| `tf.reduce_mean(tensor)` | Moyenne globale |
| `keras.metrics.Metric` | Classe de base pour les métriques personnalisées |
| `self.add_weight(name, initializer)` | Déclare un accumulateur dans une métrique |
| `self.intersection.assign_add(val)` | Incrémente un accumulateur |

### Compilation et entraînement

| Fonction / Classe | Rôle |
|---|---|
| `keras.optimizers.AdamW(lr, weight_decay)` | Optimiseur Adam avec décroissance de poids |
| `keras.optimizers.schedules.CosineDecay(lr0, steps)` | Scheduler cosinus décroissant |
| `model.compile(optimizer, loss, metrics)` | Configure l'entraînement |
| `model.fit(train_ds, validation_data, epochs, callbacks)` | Lance l'entraînement |
| `keras.callbacks.ModelCheckpoint(path, monitor, save_best_only)` | Sauvegarde le meilleur checkpoint |
| `keras.callbacks.TensorBoard(log_dir)` | Journalisation pour TensorBoard |
| `model.predict(batch, verbose=0)` | Inférence batch (retourne des probabilités) |
| `model.save("model.keras")` | Sauvegarde complète du modèle |
| `keras.models.load_model("model.keras")` | Rechargement du modèle sauvegardé |

---

## Pour aller plus loin

- **IAM** : entraînez le même pipeline sur IAM (manuscrit anglais) et comparez les performances avec RIMES. Analysez les différences liées à la langue et au style d'écriture.
- **DVD2 et DVD3** : étendez le corpus en incluant les deux autres dossiers. Observez l'impact sur l'IoU de validation.
- **DeepLab v3+** : `keras_cv` expose `DeepLabV3Plus` avec un backbone ResNet50 pré-entraîné. Swappez le U-Net contre cette architecture et évaluez le gain sur les pages à interlignes serrés.
- **Baseline prédiction** : au lieu d'un masque binaire, prédire la *ligne de base* comme une carte de chaleur (heatmap) — approche adoptée par les systèmes HTR modernes comme DAN.
- **Post-traitement morphologique** : appliquez une fermeture morphologique (`skimage.morphology.closing`) sur les masques prédits pour combler les trous dans les lignes. Mesurez l'impact sur l'IoU.
- **Boucle d'entraînement personnalisée** : si vous souhaitez davantage de contrôle (gradient clipping, accumulation de gradients), remplacez `model.fit()` par une boucle `tf.GradientTape` manuelle.
