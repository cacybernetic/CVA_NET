# glob

Le module `glob` en Python est utilisé pour rechercher des fichiers
et des répertoires dont les noms correspondent à un motif spécifié.
Il est particulièrement utile pour filtrer des fichiers par extension
ou pour trouver des fichiers qui respectent un certain format de nommage.
Voici un guide détaillé sur son utilisation.

## Introduction

### 1. Importation du Module

Avant de pouvoir utiliser `glob`, vous devez l'importer dans votre script Python :

```python
import glob
```

### 2. Utilisation de `glob.glob()`

La méthode principale de `glob` est `glob.glob()`, qui retourne une liste
de chemins de fichiers correspondant à un motif donné.

Syntaxe :

```python
glob.glob(pattern, recursive=False)
```

- **pattern** : Le motif à utiliser pour rechercher les fichiers. Vous pouvez
utiliser des caractères génériques comme `*` (qui correspond à zéro
ou plusieurs caractères) et `?` (qui correspond à un seul caractère).
- **recursive** : Si défini sur `True`, permet une recherche récursive
dans les sous-répertoires.

Exemples :

1. Lister tous les fichiers `.py` dans le répertoire courant

```python
import glob

# Liste tous les fichiers .py dans le répertoire courant
files = glob.glob("*.py")
print(files)
```

```
['main.py', 'code.py', '__init__.py']
```

2. Recherche récursive de fichiers `.md`

```python
import glob

# Liste tous les fichiers .py dans le répertoire courant et ses sous-répertoires
files = glob.glob("**/*.md", recursive=True)
print(files)
```

```
['CONTRIBUTING.md', 'README.md', 'env/lib/python3.10/site-packages/numpy/random/LICENSE.md', 'env/lib/python3.10/site-packages/seaborn-0.13.2.dist-info/LICENSE.md', 'env/lib/python3.10/site-packages/lazy_loader-0.4.dist-info/LICENSE.md', 'env/lib/python3.10/site-packages/torchgen/packaged/autograd/README.md', 'env/lib/python3.10/site-packages/scipy/fft/_pocketfft/LICENSE.md', 'env/lib/python3.10/site-packages/idna-3.10.dist-info/LICENSE.md', 'docs/argparse.md', 'docs/enum.md', 'docs/glob.md']
```

### 3. Utilisation de `glob.iglob()`

La méthode `glob.iglob()` fonctionne de manière similaire à `glob.glob()`,
mais retourne un itérateur au lieu d'une liste. Cela peut être plus efficace
en mémoire, surtout si vous traitez un grand nombre de fichiers.

Exemple :

```python
import glob

# Itérer sur tous les fichiers .md
# dans le répertoire courant et ses sous-répertoires :
for file in glob.iglob("**/*.md", recursive=True):
    print(file)
```

```
CONTRIBUTING.md
README.md
env/lib/python3.10/site-packages/numpy/random/LICENSE.md
env/lib/python3.10/site-packages/seaborn-0.13.2.dist-info/LICENSE.md
env/lib/python3.10/site-packages/lazy_loader-0.4.dist-info/LICENSE.md
env/lib/python3.10/site-packages/torchgen/packaged/autograd/README.md
env/lib/python3.10/site-packages/scipy/fft/_pocketfft/LICENSE.md
env/lib/python3.10/site-packages/idna-3.10.dist-info/LICENSE.md
docs/argparse.md
docs/enum.md
docs/glob.md
```


### 4. Caractères Génériques

Voici quelques caractères génériques que vous pouvez utiliser avec `glob` :

- **`*`** : Correspond à zéro ou plusieurs caractères.
- **`?`** : Correspond à exactement un caractère.
- **`[...]`** : Correspond à un ensemble de caractères. Par exemple, `[abc]`
correspond à 'a', 'b' ou 'c'.

Exemple avec Caractères Génériques :

```python
import glob

# Liste tous les fichiers qui commencent par 'data'
# et se terminent par '.csv'
files = glob.glob("data*.csv")
print(files)

# Liste tous les fichiers qui ont exactement
# trois caractères avant l'extension .txt
files = glob.glob("???*.txt")
print(files)

# Liste tous les fichiers qui contiennent
# soit 'report' soit 'summary'
files = glob.glob("*[report|summary]*")
print(files)
```

Le module `glob` est un outil puissant et flexible pour rechercher
des fichiers en Python. Il permet d'utiliser des motifs simples
pour filtrer et trouver rapidement des fichiers correspondant à vos critères,
ce qui est particulièrement utile pour la gestion de fichiers et l'analyse
de données. En utilisant des méthodes comme `glob()` et `iglob()`,
vous pouvez facilement intégrer la recherche de fichiers
dans vos scripts Python.

### Reférence

- [1] https://techbeamers.com/python-glob/
- [2] https://www.boardinfinity.com/blog/glob-in-python-2/
- [3] https://towardsdatascience.com/the-python-glob-module-47d82f4cbd2d?gi=40d4a82590f4


## Sélectionner en fonction de plusieurs extensions

Pour sélectionner des fichiers en fonction de plusieurs extensions
avec le module `glob`, vous pouvez utiliser une approche similaire
à celle décrite dans les résultats de recherche. Voici un exemple détaillé
qui montre comment procéder :

### Exemple de Code

```python
import glob

# Définir les extensions à sélectionner
extensions = ['*.txt', '*.csv', '*.md']

# Créer une liste pour stocker les fichiers sélectionnés
selected_files = []

# Utiliser glob pour sélectionner des fichiers avec plusieurs extensions
for ext in extensions:
    selected_files.extend(glob.glob(ext))

# Afficher les fichiers sélectionnés
print("Fichiers sélectionnés :", selected_files)
```

#### Explication du Code

1. **Importation du Module** : Le module `glob` est importé
pour permettre la recherche de fichiers.
  
2. **Définition des Extensions** : Une liste `extensions` est créée 
pour contenir les motifs correspondant aux différentes extensions de fichiers
que vous souhaitez sélectionner.

3. **Création d'une Liste** : Une liste vide `selected_files` est initialisée
pour stocker les fichiers trouvés.

4. **Boucle sur les Extensions** :
   - Pour chaque extension dans la liste `extensions`, la méthode
   `glob.glob(ext)` est appelée.
   - Les résultats sont ajoutés à la liste `selected_files` à l'aide
   de la méthode `extend()`.

5. **Affichage des Résultats** : Les fichiers sélectionnés sont affichés.

#### Remarques

- **Répertoire Courant** : Assurez-vous que le script est exécuté
dans le même répertoire où se trouvent les fichiers avec les extensions
spécifiées, ou ajustez le chemin d'accès dans les motifs (par exemple,
`path/to/files/*.txt`).
  
- **Aucune Correspondance** : Si vous obtenez une liste vide
(`[]`), cela signifie qu'aucun fichier correspondant aux motifs spécifiés
n'a été trouvé dans le répertoire courant.

### Exemple d'Utilisation avec Chemins Relatifs

Si vos fichiers se trouvent dans un sous-répertoire, vous pouvez spécifier
le chemin relatif :

```python
import glob

# Définir les extensions à sélectionner
extensions = ['path/to/files/*.txt', 'path/to/files/*.csv',
              'path/to/files/*.md']

# Créer une liste pour stocker les fichiers sélectionnés
selected_files = []

# Utiliser glob pour sélectionner des fichiers avec plusieurs extensions
for ext in extensions:
    selected_files.extend(glob.glob(ext))

# Afficher les fichiers sélectionnés
print("Fichiers sélectionnés :", selected_files)
```

L'utilisation de `glob` pour sélectionner des fichiers avec plusieurs
extensions est simple et efficace. En suivant cette approche,
vous pouvez facilement filtrer et travailler avec différents types
de fichiers dans vos scripts Python.
