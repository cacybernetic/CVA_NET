# argparse
Le module `argparse` en Python est utilisé pour créer des interfaces en ligne de commande. Il permet de définir des arguments que votre programme peut accepter, de les analyser et de gérer les options fournies par l'utilisateur. Voici une explication détaillée de son fonctionnement, accompagnée d'exemples.

## Introduction

### 1. Création d'un Analyseur

Pour commencer à utiliser `argparse`, vous devez créer un objet `ArgumentParser`. Cet objet contiendra toutes les informations nécessaires pour interpréter les arguments de la ligne de commande.

Exemple :

```python
import argparse

# Créer un analyseur
parser = argparse.ArgumentParser(description='Un exemple simple d\'utilisation d\'argparse.')
```

### 2. Ajouter des Arguments

Vous pouvez ajouter des arguments à l'analyseur en utilisant la méthode `add_argument()`. Cette méthode prend plusieurs paramètres :

- **name or flags** : Le nom de l'argument, qui peut être positionnel ou optionnel.
- **type** : Le type de l'argument (par exemple, `int`, `float`, `str`).
- **help** : Une description de l'argument, qui sera affichée lorsque l'utilisateur demande de l'aide.
- **required** : Si cet argument est obligatoire.
- **default** : La valeur par défaut si l'argument n'est pas fourni.

Exemple :

```python
parser.add_argument('number', type=int, help='Un nombre entier à traiter.')
parser.add_argument('--verbose', action='store_true', help='Affiche des informations détaillées.')
```

### 3. Analyser les Arguments

Après avoir défini tous les arguments, vous devez appeler `parse_args()` pour analyser les arguments fournis par l'utilisateur dans la ligne de commande.

Exemple :

```python
args = parser.parse_args()
```

### 4. Utiliser les Arguments

Les arguments analysés sont accessibles via les attributs de l'objet retourné par `parse_args()`. Vous pouvez utiliser ces attributs dans votre programme.

#### Exemple Complet :

Voici un exemple complet qui montre comment utiliser `argparse` pour accepter un nombre entier et une option verbose :

```python
import argparse

# Créer un analyseur
parser = argparse.ArgumentParser(description='Un exemple simple d\'utilisation d\'argparse.')

# Ajouter des arguments
parser.add_argument('number', type=int, help='Un nombre entier à traiter.')
parser.add_argument('--verbose', action='store_true', help='Affiche des informations détaillées.')

# Analyser les arguments
args = parser.parse_args()

# Utiliser les arguments
if args.verbose:
    print(f"Vous avez fourni le nombre : {args.number}")
else:
    print(args.number)
```

#### Exécution du Programme

Vous pouvez exécuter ce programme depuis la ligne de commande comme suit :

```bash
$ python mon_programme.py 5 --verbose
Vous avez fourni le nombre : 5
```

Ou sans l'option verbose :

```bash
$ python mon_programme.py 5
5
```

### 5. Gestion des Erreurs

`argparse` gère automatiquement les erreurs liées aux arguments. Par exemple, si vous fournissez une valeur non entière pour un argument qui attend un entier, le programme affichera un message d'erreur approprié.

Exemple d'Erreur :

```bash
$ python mon_programme.py cinq
usage: mon_programme.py [-h] number
mon_programme.py: error: argument number: invalid int value: 'cinq'
```

### 6. Affichage de l'Aide

`argparse` génère automatiquement une aide pour votre programme. L'utilisateur peut afficher cette aide en utilisant le drapeau `-h` ou `--help`.

Exemple d'Aide :

```bash
$ python mon_programme.py -h
usage: mon_programme.py [-h] number

Un exemple simple d'utilisation d'argparse.

positional arguments:
  number           Un nombre entier à traiter.

optional arguments:
  -h, --help       show this help message and exit
  --verbose        Affiche des informations détaillées.
```


Le module `argparse` est un outil puissant et flexible pour gérer
les arguments de la ligne de commande dans vos programmes Python.
Il permet de définir facilement des options et des paramètres tout
en offrant une gestion automatique des erreurs et une documentation intégrée.
En utilisant `argparse`, vous pouvez rendre vos scripts plus interactifs
et conviviaux pour les utilisateurs.

### References

- [1] https://docs.python.org/fr/3/library/argparse.html
- [2] https://docs.python.org/fr/3.8/library/argparse.html
- [3] https://docs.python.org/fr/3/howto/argparse.html
- [4] https://docs.python.org/fr/3.5/howto/argparse.html
- [5] https://docs.python.org/ja/3.11/howto/argparse.html
- [6] https://stackoverflow.com/questions/7427101/simple-argparse-example-wanted-1-argument-3-results
- [7] https://www.cherryservers.com/blog/how-to-use-python-argparse
- [8] https://realpython.com/command-line-interfaces-python-argparse/


## Configurer des valeurs enumérées

Pour configurer `argparse` avec des valeurs d'énumération en Python, vous pouvez utiliser le module `enum` pour définir vos énumérations et les intégrer facilement dans votre analyseur d'arguments. Voici un guide détaillé sur la façon de procéder, accompagné d'exemples.

### 1. Importer les Modules Nécessaires

Vous devez importer les modules `argparse` et `enum`.

```python
import argparse
from enum import Enum
```

### 2. Définir une Énumération

Créez une classe d'énumération en utilisant `enum.Enum`. Chaque membre de l'énumération doit avoir un nom unique.

#### Exemple

```python
class Color(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"
```

### 3. Créer un Analyseur d'Arguments

Utilisez `argparse.ArgumentParser` pour créer un analyseur d'arguments.

Exemple :

```python
parser = argparse.ArgumentParser(description="Choisissez une couleur.")
```

### 4. Ajouter un Argument avec des Valeurs d'Énumération

Lorsque vous ajoutez un argument, vous pouvez utiliser le type de l'énumération comme type de conversion et spécifier les choix possibles.

Exemple :

```python
parser.add_argument(
    'color',
    type=Color,
    choices=list(Color),
    help='Choisissez une couleur parmi : {}'.format(', '.join([color.value for color in Color]))
)
```

### 5. Analyser les Arguments

Utilisez `parse_args()` pour analyser les arguments fournis par l'utilisateur.

#### Exemple Complet

Voici un exemple complet qui montre comment configurer `argparse` avec des valeurs d'énumération :

```python
import argparse
from enum import Enum

# Définir l'énumération
class Color(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"

# Créer l'analyseur d'arguments
parser = argparse.ArgumentParser(description="Choisissez une couleur.")

# Ajouter l'argument avec des valeurs d'énumération
parser.add_argument(
    'color',
    type=Color,
    choices=list(Color),
    help='Choisissez une couleur parmi : {}'.format(', '.join([color.value for color in Color]))
)

# Analyser les arguments
args = parser.parse_args()

# Utiliser l'argument analysé
print(f"Vous avez choisi la couleur : {args.color.value}")
```

### 6. Exécution du Programme

Vous pouvez exécuter ce script depuis la ligne de commande. Voici quelques exemples :

1. Choisir une couleur valide

```bash
$ python mon_script.py red
Vous avez choisi la couleur : red
```

2. Essayer une couleur invalide

Si vous essayez de passer une couleur qui n'est pas dans l'énumération, `argparse` affichera un message d'erreur :

```bash
$ python mon_script.py yellow
usage: mon_script.py [-h] {red,green,blue}
mon_script.py: error: argument color: invalid choice: 'yellow' (choose from 'red', 'green', 'blue')
```

L'utilisation de `argparse` avec des valeurs d'énumération permet de valider
facilement les entrées utilisateur tout en rendant votre code plus lisible.
En définissant des énumérations, vous pouvez gérer des choix prédéfinis
de manière élégante et efficace. Ce guide vous montre comment configurer
cela étape par étape, ce qui peut être très utile pour créer des scripts
Python interactifs et robustes.

### References

- [1] https://bugs.python.org/issue25061
- [2] https://terashift.co.nz/fun-python-enums/
- [3] https://github.com/python/cpython/issues/86667
- [4] https://docs.python.org/fr/3/howto/argparse.html
- [5] https://www.reddit.com/r/Python/comments/oxi1v2/using_argparseaction_for_effortless_loglevel_and/
- [6] https://stackoverflow.com/questions/43968006/support-for-enum-arguments-in-argparse/60750535
- [7] https://docs.python.org/uk/3/library/argparse.html

## Affichage de version
Pour afficher la version de votre programme
en utilisant `argparse`, vous pouvez ajouter un argument avec l'action
`'version'`. Voici un exemple simple de code Python qui montre comment
procéder :

### Exemple de Code

```python
import argparse

# Définir la version de votre programme
__version__ = '1.0.0'

def main():
    # Créer un analyseur d'arguments
    parser = argparse.ArgumentParser(
        prog='Mon Programme',  # Nom du programme
        description='Un super programme très utile'  # Description du programme
    )

    # Ajouter un argument pour afficher la version
    parser.add_argument(
        '-V', '--version',  # Options pour afficher la version
        action='version',  # Action pour afficher la version
        version=f'%(prog)s {__version__}'  # Formatage de la version
    )

    # Analyser les arguments
    args = parser.parse_args()

if __name__ == "__main__":
    main()
```

### Explication

- **`argparse.ArgumentParser()`** : Crée un analyseur d'arguments.
- **`prog`** : Spécifie le nom du programme affiché dans les messages d'aide.
- **`description`** : Spécifie une brève description du programme.
- **`add_argument('-V', '--version')`** : Ajoute un argument pour afficher
la version.
  - **`action='version'`** : Indique que l'argument doit afficher la version.
  - **`version=f'%(prog)s {__version__}'`** : Formatage de la version,
  incluant le nom du programme (`%(prog)s`) et la version (`__version__`).

### Utilisation

Pour afficher la version de votre programme, exécutez-le avec l'option
`--version` ou `-V` :

```bash
python mon_programme.py --version
```

Cela affichera quelque chose comme :

```
Mon Programme 1.0.0
```

En utilisant `argparse`, vous pouvez facilement ajouter une option
pour afficher la version de votre programme. Cela est particulièrement
utile pour fournir des informations sur votre application aux utilisateurs.

### Références
- [1] https://stackoverflow.com/questions/15405636/pythons-argparse-to-show-programs-version-with-prog-and-version-string-formatt
- [2] https://docs.python.org/fr/3.13/library/argparse.html
- [3] https://docs.python.org/3/library/argparse.html
- [4] https://docs.python.org/fr/3.8/library/argparse.html
- [5] https://www.developpez.net/forums/d2087583/autres-langages/python/general-python/utilisation-librairie-argparse/
- [6] https://docs.python.org/fr/3/howto/argparse.html
- [7] https://docs.python.org/fr/3.5/howto/argparse.html
- [8] https://python.readthedocs.io/fr/hack-in-language/howto/argparse.html

