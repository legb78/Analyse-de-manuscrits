# Tutoriel — Analyse de manuscrits médiévaux avec YOLO-gen

**Référence :** Torres Aguilar, S. (2025). *From Codicology to Code: A Comparative Study of Transformer and YOLO-based Detectors for Layout Analysis in Historical Documents*. arXiv:2506.20326
**Modèle :** `magistermilitum/YOLO_manuscripts` (Hugging Face, licence MIT)

---

## I. Contexte et objectif

L'**analyse de mise en page** (*Document Layout Analysis*, DLA) consiste à localiser et classifier les régions structurelles d'un document — blocs de texte, enluminures, lettrines, marginalia — avant toute reconnaissance de caractères.

Dans ce TP, vous implémenterez un pipeline Python qui applique **YOLO-gen 11x-OBB** sur un folio du *Roman de la Rose* (BnF, Français 25526). Ce modèle a été entraîné sur trois corpus de manuscrits occidentaux (XII–XVIIe s.) : e-NDP, CATMuS et HORAE.

---

## II. Architecture du pipeline

Le script est organisé en quatre fonctions appelées séquentiellement depuis `main()` :

```
load_model()  →  diagnose_thresholds()  →  visualize()  →  report_and_export()
  YOLO            (results, float)          None              None
```

### Pourquoi des boîtes orientées (OBB) ?

Les modèles YOLO classiques produisent des **boîtes droites** (AABB) alignées sur les axes. Les manuscrits médiévaux contiennent des annotations obliques et des éléments non rectangulaires : les **OBB** sont des quadrilatères librement orientés, décrits par **quatre points** (8 coordonnées) au lieu de deux coins opposés (4 coordonnées).

> **Point clé :** l'accès aux détections OBB se fait via `result.obb` (et non `result.boxes`). Les coordonnées sont fournies par `box.xyxyxyxy`, un tenseur de forme `(1, 8)` à remodeler en `(4, 2)`.

---

## III. Signatures des fonctions à implémenter

### `load_model() -> YOLO`

**Rôle :** télécharge `best.pt` depuis Hugging Face et retourne un objet `YOLO` prêt à l'inférence.

**Contrainte :** `YOLO()` attend un chemin local vers un fichier `.pt`, pas un identifiant HF. Il faut donc d'abord télécharger le fichier avec `hf_hub_download()`, puis passer le chemin retourné à `YOLO()`.

**Gestion du token :** résoudre dans cet ordre : (1) variable `HF_TOKEN` dans le script, (2) variable d'environnement `HF_TOKEN`, (3) cache `huggingface-cli login`. En cas d'échec (403), afficher un message d'aide et quitter avec `sys.exit()`.

| Paramètre | Type | Description |
|-----------|------|-------------|
| — | — | Lit les constantes globales `HF_MODEL_ID` et `HF_TOKEN` |
| *retour* | `YOLO` | Modèle chargé avec ses poids |

---

### `diagnose_thresholds(model, image_path) -> tuple[results, float]`

**Rôle :** teste la détection à plusieurs seuils décroissants (`0.50 → 0.25 → 0.10 → 0.05 → 0.01`) et retourne les résultats au premier seuil ayant produit au moins une détection et inférieur ou égal à `CONF_FINAL`. Imprime un tableau de diagnostic.

**Paramètres d'inférence conseillés :** `iou=0.45`, `imgsz=1280`, `verbose=False`.

| Paramètre | Type | Description |
|-----------|------|-------------|
| `model` | `YOLO` | Modèle retourné par `load_model()` |
| `image_path` | `str` | Chemin vers l'image à analyser |
| *retour* | `tuple` | `(results, conf_retenu)` |

---

### `visualize(results, img_pil, conf, save_path) -> None`

**Rôle :** dessine les boîtes OBB (polygones orientés) sur l'image et sauvegarde la figure.

**Point technique :** chaque détection se dessine avec `matplotlib.patches.Polygon` (pas `Rectangle`). Les 8 coordonnées brutes sont à remodeler en `(4, 2)`. Superposer deux polygones : un rempli (`alpha=0.15`) et un contour (`facecolor="none"`).

**Cas sans détection :** afficher un message centré sur l'image et sauvegarder quand même.

| Paramètre | Type | Description |
|-----------|------|-------------|
| `results` | liste ultralytics | Retour de `model.predict()` |
| `img_pil` | `PIL.Image` | Image originale ouverte en RGB |
| `conf` | `float` | Seuil retenu (pour le titre) |
| `save_path` | `Path` | Chemin de sauvegarde de la figure |

---

### `report_and_export(results, conf, save_path) -> None`

**Rôle :** affiche un rapport console groupé par classe (effectif, confiance moyenne, maximum) et exporte les détections en JSON.

**Structure JSON attendue :**

```json
{
  "model":          "magistermilitum/YOLO_manuscripts",
  "paper":          "arXiv:2506.20326",
  "conf_threshold": 0.10,
  "regions": [
    { "class": "Text", "confidence": 0.823,
      "polygon": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] }
  ]
}
```

| Paramètre | Type | Description |
|-----------|------|-------------|
| `results` | liste ultralytics | Retour de `model.predict()` |
| `conf` | `float` | Seuil retenu |
| `save_path` | `Path` | Chemin du fichier JSON de sortie |

---

## IV. Mémento des modules et fonctions

### `huggingface_hub`

| Fonction | Signature | Description |
|----------|-----------|-------------|
| `hf_hub_download` | `hf_hub_download(repo_id, filename, token=None)` | Télécharge un fichier depuis HF, retourne son chemin local dans le cache. `token` obligatoire si le dépôt est restreint. |

### `ultralytics` — classe `YOLO`

| Symbole | Signature | Description |
|---------|-----------|-------------|
| `YOLO()` | `YOLO(model: str)` | Charge un modèle depuis un fichier `.pt` local. |
| `.predict()` | `model.predict(source, conf, iou, imgsz, verbose)` | Lance l'inférence. Retourne une liste de `Results`. |
| `.names` | `dict[int, str]` | Dictionnaire `{id: nom_de_classe}`. |
| `result.obb` | `OBBs \| None` | Conteneur des détections OBB. `None` si aucune. Itérable : `for box in result.obb`. |
| `box.cls` | `Tensor (1,)` | Indice de classe. Extraire : `int(box.cls[0])`. |
| `box.conf` | `Tensor (1,)` | Score de confiance. Extraire : `float(box.conf[0])`. |
| `box.xyxyxyxy` | `Tensor (1, 8)` | Coordonnées des 4 coins. Remodeler : `box.xyxyxyxy[0].cpu().numpy().reshape(4, 2)`. |

### `PIL.Image`

| Fonction | Signature | Description |
|----------|-----------|-------------|
| `Image.open()` | `Image.open(fp: str)` | Ouvre une image depuis un chemin. |
| `.convert()` | `img.convert(mode: str)` | Convertit le mode couleur. Utiliser `"RGB"`. |

### `matplotlib.pyplot` et `matplotlib.patches`

| Symbole | Signature | Description |
|---------|-----------|-------------|
| `plt.subplots()` | `plt.subplots(nrows, ncols, figsize)` | Crée une figure et ses axes. |
| `ax.imshow()` | `ax.imshow(X)` | Affiche une image dans les axes. |
| `ax.add_patch()` | `ax.add_patch(patch)` | Ajoute un patch à la figure. |
| `ax.text()` | `ax.text(x, y, s, fontsize, color, bbox)` | Place une étiquette. `bbox` : dict de style du fond. |
| `ax.legend()` | `ax.legend(handles, loc, fontsize, framealpha)` | Ajoute une légende. `handles` : liste de `Patch`. |
| `Polygon` | `Polygon(xy, closed, linewidth, edgecolor, facecolor, alpha)` | Polygone défini par un tableau `(N, 2)`. Depuis `matplotlib.patches`. |
| `Patch` | `Patch(color, label)` | Carré de légende coloré. Depuis `matplotlib.patches`. |
| `fig.savefig()` | `fig.savefig(fname, dpi, bbox_inches)` | Sauvegarde la figure. Utiliser `bbox_inches="tight"`. |

### `numpy`

| Fonction | Signature | Description |
|----------|-----------|-------------|
| `np.array()` | `np.array(object)` | Convertit en tableau NumPy. |
| `.reshape()` | `arr.reshape(shape)` | Remodèle un tableau. Ex : `.reshape(4, 2)`. |
| `.argmin()` | `arr.argmin(axis)` | Indice du minimum. Ex : `pts[:, 1].argmin()` → coin le plus haut. |
| `np.mean()` | `np.mean(a)` | Calcule la moyenne. |

### Bibliothèque standard

| Symbole | Description |
|---------|-------------|
| `Path(path_str)` | Chemin multiplateforme (`pathlib`). `.mkdir(exist_ok=True)` crée le répertoire si absent. |
| `json.dump(obj, fp, ensure_ascii, indent)` | Sérialise en JSON. `ensure_ascii=False` préserve les accents. |
| `defaultdict(default_factory)` | Dict à initialisation automatique. Ex : `defaultdict(list)`. |
| `os.environ.get(key)` | Lit une variable d'environnement sans exception si absente. |
| `sys.exit(msg)` | Arrête le programme en affichant `msg`. |

---

## V. Instructions de mise en œuvre

### Installation

```bash
pip install ultralytics huggingface_hub pillow matplotlib
```

### Authentification Hugging Face

1. Créer un compte sur **huggingface.co**.
2. Générer un token (lecture seule) : **huggingface.co/settings/tokens**.
3. Lancer `huggingface-cli login` et coller le token.

### Structure de fichiers attendue

```
projet/
├── yologen_obb.py
├── Le_Roman_de_la_Rose.jpeg
└── yologen_output/
    ├── yologen_detections.jpg
    └── yologen_detections.json
```

### Squelette de départ

```python
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

IMAGE_PATH  = "Le_Roman_de_la_Rose.jpeg"
OUTPUT_DIR  = Path("yologen_output")
HF_MODEL_ID = "magistermilitum/YOLO_manuscripts"
HF_TOKEN    = None   # ou "hf_..."
CONF_FINAL  = 0.10

def load_model() -> YOLO:
    """À implémenter."""
    ...

def diagnose_thresholds(model: YOLO, image_path: str):
    """À implémenter."""
    ...

def visualize(results, img_pil: Image.Image, conf: float, save_path: Path) -> None:
    """À implémenter."""
    ...

def report_and_export(results, conf: float, save_path: Path) -> None:
    """À implémenter."""
    ...

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    model         = load_model()
    results, conf = diagnose_thresholds(model, IMAGE_PATH)
    img_pil       = Image.open(IMAGE_PATH).convert("RGB")
    visualize(results, img_pil, conf, OUTPUT_DIR / "yologen_detections.jpg")
    report_and_export(results, conf, OUTPUT_DIR / "yologen_detections.json")

if __name__ == "__main__":
    main()
```

### Pièges courants

> **`result.obb` vaut `None` ou est vide.** Toujours tester ce cas avant d'itérer, sous peine d'exception.

> **Fichier `best.pt` corrompu.** Si le diagnostic affiche `0` à tous les seuils, vérifiez que le fichier fait bien ~118 Mo. Un téléchargement interrompu produit un fichier invalide.
