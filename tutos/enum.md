# Enum
Les énumérations (ou `Enum`) en Python, introduites dans la version 3.4,
permettent de définir des ensembles de valeurs constantes qui sont liées
à des noms symboliques. Cela rend le code plus lisible et maintenable.
Voici une explication détaillée sur le fonctionnement avancé des énumérations
en Python, accompagnée d'exemples.

## Introduction

### 1. Création d'une Énumération

Pour créer une énumération, vous devez importer la classe `Enum`
du module `enum` et définir une nouvelle classe qui hérite de `Enum`.

Exemple de Base :

```python
from enum import Enum

class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3
```

### 2. Accès aux Valeurs et Noms

Vous pouvez accéder aux noms et aux valeurs des membres d'une énumération.

Exemple :

```python
print(Color.RED)          # Output: Color.RED
print(Color.RED.name)     # Output: 'RED'
print(Color.RED.value)    # Output: 1
```

### 3. Comparaison des Éléments

Les membres d'une énumération peuvent être comparés entre eux.

### Exemple

```python
if Color.RED == Color.GREEN:
    print("Les couleurs sont identiques.")
else:
    print("Les couleurs sont différentes.")  # Output: Les couleurs sont différentes.
```

### 4. Itération sur les Éléments

Vous pouvez itérer sur les membres d'une énumération.

Exemple :

```python
for color in Color:
    print(color)
```

```
Color.RED
Color.GREEN
Color.BLUE
```

### 5. Méthodes Personnalisées dans les Énumérations

Vous pouvez ajouter des méthodes à une classe d'énumération pour étendre
son comportement.

Exemple :

```python
class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3

    def describe(self):
        return f"{self.name} is represented by the value {self.value}"

print(Color.RED.describe())  # Output: RED is represented by the value 1
```

### 6. Énumérations avec des Valeurs Non Entières

Les énumérations ne sont pas limitées aux valeurs entières. Vous pouvez
utiliser n'importe quel type immuable comme valeur.

Exemple avec des Chaînes de Caractères :

```python
class Status(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"

print(Status.PENDING.value)  # Output: 'pending'
```

### 7. Énumérations avec des Comportements Avancés

Vous pouvez utiliser des fonctionnalités avancées comme l'héritage
et les membres auto-assignés.

Exemple d'Héritage :

```python
from enum import Enum, auto

class BaseColor(Enum):
    RED = auto()
    GREEN = auto()

class ExtendedColor(BaseColor):
    BLUE = auto()
    YELLOW = auto()

for color in ExtendedColor:
    print(color)
```


```
BaseColor.RED
BaseColor.GREEN
ExtendedColor.BLUE
ExtendedColor.YELLOW
```

### 8. Comparaison entre Énumérations Différentes

Les membres d'énumérations différentes ne peuvent pas être
comparés directement.

Exemple :

```python
class Animal(Enum):
    DOG = 1
    CAT = 2

class Plant(Enum):
    TREE = 1
    FLOWER = 2

# Cela lèvera une erreur si décommenté :
# if Animal.DOG == Plant.TREE:
#     print("C'est un animal et une plante.")
```


Les énumérations en Python offrent un moyen puissant et flexible de gérer
des ensembles de valeurs constantes. Elles améliorent la lisibilité du code
et permettent de créer des structures plus robustes en encapsulant les valeurs
associées à des noms significatifs. En utilisant les fonctionnalités avancées
des énumérations, vous pouvez créer des classes d'énumération qui répondent
à vos besoins spécifiques tout en maintenant un code propre et maintenable.

## Enum et les list

Pour utiliser `enum.Enum` avec une liste en Python, vous pouvez suivre
ces étapes détaillées. L'exemple ci-dessous montre comment définir
une énumération, créer une liste contenant des membres de cette énumération,
et afficher les valeurs et noms des membres.

### Étapes pour Utiliser `enum.Enum` avec une Liste

1. **Définir l'Énumération** : Créez une classe d'énumération en héritant 
de `Enum`.
2. **Créer une Liste** : Définissez une liste contenant des membres
de l'énumération.
3. **Afficher les Éléments de la Liste** : Itérez sur la liste
pour afficher les noms et valeurs des membres.

### Exemple Complet

Voici un exemple complet qui illustre ces étapes :

```python
from enum import Enum

# 1. Définir l'énumération
class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3

# 2. Créer une liste contenant des membres de l'énumération
color_list = [Color.RED, Color.GREEN, Color.BLUE]

# 3. Afficher la liste des couleurs
print("Liste des couleurs :")
for color in color_list:
    print(color.name, color.value)
```

Résultat de l'Exécution :

Lorsque vous exécutez ce code, vous obtiendrez la sortie suivante :

```
Liste des couleurs :
RED 1
GREEN 2
BLUE 3
```

- **Définition de l'Énumération** : La classe `Color` est définie comme
une énumération avec trois membres : `RED`, `GREEN`, et `BLUE`,
chacun associé à une valeur entière.

- **Création de la Liste** : `color_list` est une liste qui contient
les membres de l'énumération.

- **Affichage des Éléments** : Le code itère sur `color_list` et imprime
le nom et la valeur de chaque couleur.

Encore un autre exemple :

```python
from enum import Enum

# 1. Définir l'énumération
class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3

values_list = list(map(lambda x: x.value, Color))
print(values_list)
```

```
[1, 2, 3]
```


Utiliser `enum.Enum` avec une liste en Python est simple et efficace.
Cela permet de regrouper des constantes sous des noms significatifs
tout en facilitant leur utilisation dans des structures comme des listes.
Ce modèle améliore la lisibilité et la maintenabilité du code.

