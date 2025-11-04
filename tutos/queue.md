# Les Queues

## Introduction
Le module `queue` en Python est utilisé pour gérer des files d'attente
de manière sécurisée, surtout dans des contextes de programmation multithread.
Voici un aperçu de son utilisation, de ses types et de ses principales
fonctionnalités.

### Types de Queue

1. **Queue** : Une file d'attente standard qui fonctionne
selon le principe FIFO (First In First Out).

```python
import queue
q = queue.Queue(maxsize=0)  # maxsize=0 signifie pas de limite
```

2. **LifoQueue** : Une file d'attente qui fonctionne
selon le principe LIFO (Last In First Out), semblable à une pile.

```python
q = queue.LifoQueue(maxsize=0)
```

3. **PriorityQueue** : Une file d'attente qui permet d'ajouter
des éléments avec une priorité, les éléments avec une priorité
plus élevée sont récupérés avant ceux avec une priorité plus basse.

```python
q = queue.PriorityQueue(maxsize=0)
```

### Méthodes Principales

- `qsize()` : Retourne la taille approximative de la file.
- `empty()` : Retourne `True` si la file est vide, sinon `False`.
- `full()` : Retourne `True` si la file est pleine, sinon `False`.
- `put(item[, block[, timeout]])` : Ajoute un élément à la file.
Si `block` est `True`, il attend qu'une place se libère si la file est pleine.
- `get([block[, timeout]])` : Retire et renvoie un élément de la file.
Si `block` est `True`, il attend qu'un élément soit disponible
si la file est vide.
- `task_done()` : Indique que la tâche associée à un élément retiré
a été terminée.
- `join()` : Bloque jusqu'à ce que toutes les tâches soient terminées.

Exemple d'Utilisation :

Voici un exemple simple qui montre comment utiliser
une queue avec des threads :

```python
import threading
import queue
import time
import random

# Fonction de travail pour les threads
def worker(q):
    while True:
        item = q.get()
        if item is None:  # Condition pour arrêter le thread
            break
        print(f'Traitement de {item}')
        time.sleep(random.uniform(0.1, 1))  # Simuler un travail
        q.task_done()

# Créer une queue et des threads
q = queue.Queue()
num_worker_threads = 3
threads = []

# Démarrer les threads
for _ in range(num_worker_threads):
    t = threading.Thread(target=worker, args=(q,))
    t.start()
    threads.append(t)

# Ajouter des éléments à la queue
for item in range(10):
    q.put(item)

# Attendre que toutes les tâches soient terminées
q.join()

# Arrêter les threads
for _ in range(num_worker_threads):
    q.put(None)  # Envoyer un signal pour arrêter les threads

for t in threads:
    t.join()  # Attendre que tous les threads se terminent

print("Tous les travaux sont terminés.")
```

Le module `queue` en Python est essentiel pour gérer les files d'attente
dans des applications multithread, permettant une communication
sûre entre les threads. En utilisant ses différentes classes et méthodes,
vous pouvez facilement mettre en place des systèmes efficaces pour traiter
des tâches en parallèle tout en maintenant une synchronisation appropriée.

### Références
[1] https://fraoustin.fr/old/python_queue.html
[2] https://www.askpython.com/python-modules/python-queue
[3] https://docs.python.org/fr/3/library/queue.html
[4] https://www.simplilearn.com/tutorials/python-tutorial/queue-in-python
[5] https://docs.python.org/fr/3.7/library/queue.html

