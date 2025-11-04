# Programmation asynchrone

## Introduction
L'utilisation des fonctions asynchrones en Python
permet d'exécuter des tâches de manière non bloquante,
ce qui est particulièrement utile pour les opérations
d'E/S (entrées/sorties) comme les requêtes réseau
ou la lecture/écriture de fichiers. Voici un aperçu
de la programmation asynchrone en Python,
y compris comment cela fonctionne et un exemple pratique.

### Qu'est-ce que la Programmation Asynchrone ?

La programmation asynchrone permet à un programme
d'exécuter plusieurs tâches simultanément sans attendre
que chaque tâche soit terminée avant de commencer la suivante.
Cela est particulièrement utile pour les tâches longues
ou celles qui impliquent des délais d'attente, comme les requêtes HTTP.

### Concepts Clés

1. **Coroutines** : Une coroutine est une fonction définie
avec le mot-clé `async`. Elle peut être suspendue et reprise,
permettant ainsi à d'autres coroutines
de s'exécuter pendant qu'elle attend un résultat.
   
2. **`await`** : Utilisé pour appeler une coroutine et attendre
son résultat sans bloquer le fil d'exécution.

3. **Boucle d'événements** : C'est le cœur de la programmation asynchrone.
Elle gère l'exécution des coroutines et permet de passer le contrôle
entre elles.

4. **Tâches** : Les tâches (`asyncio.Task`) sont des objets qui représentent
une coroutine en cours d'exécution. Elles permettent de gérer l'exécution
et le suivi des coroutines.

### Exemple de Code Asynchrone

Voici un exemple simple utilisant `asyncio` pour illustrer comment créer
et exécuter des fonctions asynchrones :

```python
import asyncio
import time

async def say_after(delay, message):
    await asyncio.sleep(delay)  # Simule une opération asynchrone
    print(message)

async def main():
    print(f"Started at {time.strftime('%X')}")
    
    # Créer des tâches asynchrones
    task1 = asyncio.create_task(say_after(1, 'Hello'))
    task2 = asyncio.create_task(say_after(2, 'World'))

    # Attendre que les deux tâches soient complètes
    await task1
    await task2
    
    print(f"Finished at {time.strftime('%X')}")

# Exécuter la boucle d'événements
asyncio.run(main())
```

- **Définition des Coroutines** : La fonction `say_after` est définie
comme une coroutine avec `async def`. Elle utilise
`await asyncio.sleep(delay)` pour simuler un délai.
  
- **Création de Tâches** : Dans la fonction `main`, deux tâches sont créées
avec `asyncio.create_task()`, ce qui permet aux deux messages d'être imprimés
en parallèle.

- **Attente des Tâches** : `await task1` et `await task2`
assurent que le programme attend que chaque tâche soit terminée
avant de continuer.

- **Exécution** : `asyncio.run(main())` démarre l'exécution de la boucle
d'événements et lance la coroutine principale.


### Avantages de l'Utilisation des Fonctions Asynchrones

- **Efficacité** : Permet d'effectuer plusieurs opérations simultanément
sans bloquer l'exécution.
- **Réactivité** : Améliore la réactivité des applications, notamment
dans les interfaces utilisateur ou les serveurs web.
- **Gestion Simplifiée des E/S** : Facilite la gestion des opérations
d'E/S qui peuvent prendre du temps, comme les requêtes réseau ou l'accès
aux fichiers.

Les fonctions asynchrones en Python, principalement gérées
par le module `asyncio`, offrent un moyen puissant et efficace
de gérer la concurrence. En utilisant des coroutines
et une boucle d'événements, vous pouvez écrire du code qui reste réactif
tout en exécutant des tâches potentiellement longues.
Cela est particulièrement bénéfique dans les applications modernes
où l'efficacité et la réactivité sont cruciales.

### Références
- [1] https://python.doctor/page-programmation-asynchrone-python-thread-threading
- [2] https://www.espresso-jobs.com/conseils-carriere/ti/guide-de-la-programmation-asynchrone-dans-python
- [3] https://www.editions-eni.fr/livre/python-3-traitement-de-donnees-et-techniques-de-programmation-2e-edition-9782409044441/programmation-asynchrone-initiation
- [4] https://zestedesavoir.com/articles/pdf/1568/decouvrons-la-programmation-asynchrone-en-python.pdf
- [5] https://docs.djangoproject.com/fr/3.0/topics/async/
- [6] https://www.youtube.com/watch?v=A4oQEv3Ny1M
- [7] https://tahe.developpez.com/tutoriels-cours/ecmascript6/?page=programmation-evenementielle-et-fonctions-asynchrones

