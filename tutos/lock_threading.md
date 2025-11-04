# Lock threading
## Introduction
Les clauses de verrouillage (lock) en Python, principalement fournies
par le module `threading`, sont essentielles pour gérer l'accès concurrent
aux ressources partagées dans un environnement multithread.
Voici une explication détaillée sur leur fonctionnement,
leur utilisation et les meilleures pratiques.

### 1. Qu'est-ce qu'un Lock ?

Un **Lock** est un objet de synchronisation qui permet de s'assurer
qu'une seule thread à la fois peut accéder à une ressource partagée.
Cela évite les conditions de course où plusieurs threads tentent de modifier
une ressource simultanément, ce qui peut entraîner des résultats imprévisibles.

#### Exemple de Création d'un Lock

```python
import threading

# Créer un objet Lock
lock = threading.Lock()
```

### 2. Utilisation des Locks

Pour utiliser un lock, vous devez l'acquérir avant d'accéder à la ressource
protégée et le relâcher après avoir terminé l'opération.
Cela peut être fait manuellement ou en utilisant une instruction `with`
qui gère automatiquement l'acquisition et la libération du lock.

#### Exemple d'Utilisation Manuelle

```python
import threading

# Ressource partagée
balance = 0
lock = threading.Lock()

def deposit(amount):
    global balance
    lock.acquire()  # Acquérir le lock
    try:
        balance += amount
    finally:
        lock.release()  # Toujours libérer le lock

# Exemple d'utilisation
deposit(100)
```

#### Exemple avec `with`

Utiliser `with` est recommandé car cela garantit que le lock sera toujours
libéré, même si une exception se produit.

```python
def deposit(amount):
    global balance
    with lock:  # Acquérir le lock automatiquement
        balance += amount  # Opération protégée par le lock
```

### 3. Problèmes de Deadlock

Un problème courant lors de l'utilisation des locks est le **deadlock**,
qui se produit lorsque deux ou plusieurs threads attendent indéfiniment
que l'autre libère un lock. Cela peut arriver si un thread essaie d'acquérir
un lock qu'il détient déjà (dans le cas des locks non réentrants)
ou si plusieurs locks sont acquis dans un ordre différent
par différents threads.

#### Exemple de Deadlock

```python
class BankAccount:
    def __init__(self):
        self.balance = 0
        self.lock = threading.Lock()

    def deposit(self, amount):
        with self.lock:
            self.balance += amount
            self._update_balance(amount) 
            #: Peut causer un deadlock si _update_balance essaie
            #: d'acquérir le même lock

    def _update_balance(self, amount):
        with self.lock:  # Deadlock ici si appelé depuis deposit()
            self.balance += amount
```

#### Prévention des Deadlocks

Pour éviter les deadlocks :
- Évitez d'acquérir plusieurs locks à la fois.
- Utilisez des locks réentrants (`RLock`) si vous devez acquérir
le même lock plusieurs fois dans le même thread.

### 4. RLocks (Reentrant Locks)

Un **RLock** permet à un thread d'acquérir le même lock plusieurs fois
sans se bloquer. Chaque appel à `acquire()` doit être équilibré
par un appel à `release()`.

#### Exemple d'Utilisation de RLock

```python
rlock = threading.RLock()

def safe_function():
    with rlock:  # Acquérir le RLock
        # Code qui nécessite un accès exclusif
        with rlock:  # Peut acquérir à nouveau sans deadlock
            # Code supplémentaire qui nécessite également un accès exclusif
```

Les locks sont essentiels pour garantir la sécurité des threads
lors de l'accès à des ressources partagées en Python.
En utilisant correctement les locks et en évitant les situations de deadlock,
vous pouvez créer des applications multithread robustes et fiables.
L'utilisation de `with` pour gérer les locks est une bonne pratique
qui simplifie le code et réduit les risques d'erreurs.

### Reference
- [1] https://realpython.com/python-thread-lock/
- [2] https://learn.microsoft.com/vi-vn/dotnet/csharp/language-reference/statements/lock
- [3] https://docs.python.org/fr/3.14/library/threading.html
- [4] https://docs.oracle.com/javase/specs/jls/se7/html/jls-17.html
- [5] https://www.reddit.com/r/learnpython/comments/8vs286/threading_condition_vs_threading_lock/
- [6] https://learn.microsoft.com/en-us/dotnet/api/system.threading.lock?view=net-9.0
- [7] https://stackoverflow.com/questions/53656737/does-python-threading-lock-lock-everything-that-needs-locking
- [8] https://tqdm.github.io/docs/tqdm/

