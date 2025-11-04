# Décorateur de fonction
## Introduction
Les décorateurs de fonctions en Python sont un outil puissant
et élégant pour modifier ou étendre le comportement de fonctions
ou de méthodes. Voici tout ce que vous devez savoir à leur sujet :

### Qu'est-ce qu'un Décorateur ?

Un décorateur est une fonction qui prend une autre fonction comme argument,
ajoute une fonctionnalité ou modifie son comportement, et retourne
une nouvelle fonction. Il utilise la syntaxe `@nom_du_decorateur`
pour appliquer la modification.

### Syntaxe

```python
@mon_decorateur
def ma_fonction():
    # Corps de la fonction
    pass
```

Ceci est équivalent à :

```python
def ma_fonction():
    # Corps de la fonction
    pass

ma_fonction = mon_decorateur(ma_fonction)
```

### Explication Détaillée

1. **Fonction Décoratrice** :
   Une fonction décoratrice prend une fonction en argument et retourne
   une autre fonction, généralement une version modifiée de la fonction
   originale.

2. **Fonction Décorée** :
   La fonction décorée est la fonction sur laquelle le décorateur est appliqué.

3. **Syntaxe `@`** :
   L'opérateur `@` est un raccourci syntaxique pour appliquer
   un décorateur à une fonction.

### Exemple Simple

```python
def mon_decorateur(func):
    def wrapper():
        print("Avant l'appel de la fonction.")
        func()
        print("Après l'appel de la fonction.")
    return wrapper

@mon_decorateur
def dis_bonjour():
    print("Bonjour !")

dis_bonjour()
```

**Sortie:**

```
Avant l'appel de la fonction.
Bonjour !
Après l'appel de la fonction.
```

### Décorateurs avec Arguments

Si vous voulez passer des arguments à votre décorateur,
vous devez créer une fonction qui prend ces arguments et retourne
une fonction décoratrice.

```python
def decorateur_avec_args(arg1, arg2):
    def mon_decorateur(func):
        def wrapper(*args, **kwargs):
            print(f"Arguments du décorateur : {arg1}, {arg2}")
            return func(*args, **kwargs)
        return wrapper
    return mon_decorateur

@decorateur_avec_args("valeur1", "valeur2")
def ma_fonction(nom):
    print(f"Bonjour, {nom}!")

ma_fonction("Alice")
```

**Sortie:**

```
Arguments du décorateur : valeur1, valeur2
Bonjour, Alice!
```

### Préserver les Métadonnées de la Fonction

Lorsque vous décorez une fonction, vous perdez ses métadonnées
(nom, docstring, etc.). Pour les préserver, utilisez `functools.wraps`.

```python
import functools

def mon_decorateur(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """C'est le wrapper !"""
        print("Avant l'appel de la fonction.")
        result = func(*args, **kwargs)
        print("Après l'appel de la fonction.")
        return result
    return wrapper

@mon_decorateur
def ma_fonction():
    """C'est ma fonction !"""
    print("Ma fonction est appelée.")

print(ma_fonction.__name__)  # ma_fonction
print(ma_fonction.__doc__)   # C'est ma fonction !
```

### Cas d'Utilisation Courants

1. **Logging** : Enregistrer des informations sur les appels de fonction.
2. **Contrôle d'Accès** : Vérifier si un utilisateur a les permissions
nécessaires pour exécuter une fonction.
3. **Gestion du Temps** : Mesurer le temps d'exécution d'une fonction.
4. **Caching** : Mémoriser les résultats d'une fonction
pour éviter de recalculer des valeurs.

### Décorateurs de Classes et de Méthodes

Les décorateurs peuvent également être utilisés pour décorer des classes
et des méthodes.

#### Décorateur de Classe

```python
def singleton(cls):
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

@singleton
class MaClasse:
    pass

a = MaClasse()
b = MaClasse()
print(a is b)  # True (a et b sont la même instance)
```

#### Décorateur de Méthode

```python
class MaClasse:
    def __init__(self, x):
        self.x = x

    @property
    def valeur(self):
        return self.x

obj = MaClasse(10)
print(obj.valeur)  # 10
```

Les décorateurs de fonctions sont un outil puissant pour étendre
et modifier le comportement des fonctions en Python. Ils permettent
de rendre le code plus propre, plus modulaire et plus réutilisable.
Avec les bonnes pratiques (comme l'utilisation de `functools.wraps`),
ils peuvent améliorer la qualité et la maintenabilité de votre code.

