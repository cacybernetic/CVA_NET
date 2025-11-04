# Décorateur de classes

## Introduction
Les décorateurs de classes en Python sont des fonctions qui modifient
ou étendent le comportement d'une classe. Ils sont similaires aux décorateurs
de fonctions, mais ils s'appliquent à une classe entière plutôt
qu'à une seule fonction. Voici tout ce que vous devez savoir
sur les décorateurs de classes :

### Qu'est-ce qu'un Décorateur de Classe ?

Un décorateur de classe est une fonction qui prend une classe
comme argument et retourne une nouvelle classe ou modifie la classe existante.
Cela permet d'ajouter des fonctionnalités, de modifier des attributs
ou d'effectuer des vérifications.

### Syntaxe

La syntaxe pour appliquer un décorateur à une classe est similaire
à celle des fonctions, utilisant le symbole `@`.

```python
@mon_decorateur
class MaClasse:
    pass
```

### Exemple Simple

Voici un exemple simple d'un décorateur de classe qui ajoute un attribut
à la classe :

```python
def mon_decorateur(cls):
    cls.nouvel_attribut = "Ceci est un nouvel attribut"
    return cls

@mon_decorateur
class MaClasse:
    pass

# Utilisation
obj = MaClasse()
print(obj.nouvel_attribut)  # Affiche : Ceci est un nouvel attribut
```

### Fonctionnement Interne

Un décorateur de classe prend généralement la forme suivante :

```python
def mon_decorateur(cls):
    # Modifications ou ajouts à la classe
    return cls
```

### Décorateurs avec Arguments

Si vous souhaitez passer des arguments à votre décorateur de classe,
vous devez créer une fonction qui retourne le décorateur.

```python
def decorateur_avec_args(arg1):
    def mon_decorateur(cls):
        cls.arg1 = arg1
        return cls
    return mon_decorateur

@decorateur_avec_args("valeur")
class MaClasse:
    pass

# Utilisation
print(MaClasse.arg1)  # Affiche : valeur
```

### Cas d'Utilisation Courants

1. **Ajout d'Attributs** : Ajouter des attributs ou des méthodes à une classe.
2. **Modification du Comportement** : Modifier le comportement d'une méthode
dans la classe.
3. **Validation** : Effectuer des vérifications lors de la création
d'une instance.
4. **Enregistrement** : Enregistrer les classes pour une utilisation
ultérieure (comme dans les frameworks).

### Exemple Avancé

Voici un exemple plus avancé où nous utilisons un décorateur
pour enregistrer les classes créées :

```python
classes_enregistrees = []

def enregistrer_classe(cls):
    classes_enregistrees.append(cls)
    return cls

@enregistrer_classe
class ClasseA:
    pass

@enregistrer_classe
class ClasseB:
    pass

# Utilisation
print(classes_enregistrees)  # Affiche : [<class '__main__.ClasseA'>, <class '__main__.ClasseB'>]
```

### Décorateurs et Héritage

Les décorateurs de classes peuvent également être utilisés avec l'héritage.
Lorsque vous décorez une sous-classe, le décorateur s'applique
à la sous-classe, mais vous pouvez également appeler le décorateur 
sur la super-classe.

```python
def decorateur(cls):
    cls.attribut = "Valeur"
    return cls

@decorateur
class Base:
    pass

class SousClasse(Base):
    pass

# Utilisation
print(SousClasse.attribut)  # Affiche : Valeur (héritée de Base)
```

Les décorateurs de classes en Python sont un outil puissant pour modifier
et étendre le comportement des classes. Ils permettent d'ajouter facilement
des fonctionnalités, de valider des données et d'améliorer l'organisation
du code. En utilisant les bonnes pratiques, comme la préservation
des métadonnées et l'utilisation de décorateurs avec arguments,
vous pouvez créer des classes plus robustes et modulaires.

