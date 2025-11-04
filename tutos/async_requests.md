# Requête HTTP asynchrone

## Introduction
Pour effectuer des requêtes HTTP asynchrones en Python,
la bibliothèque `requests` n'est pas directement compatible
avec les fonctionnalités asynchrones de `asyncio`.
Cependant, il existe plusieurs approches pour contourner cette limitation.
Voici un résumé des méthodes disponibles et un exemple pratique.

### Approches pour Utiliser `requests` avec Asynchrone

1. **Utiliser `asyncio.to_thread()`** :
Cette méthode permet d'exécuter des appels bloquants dans un thread séparé,
ce qui permet à l'événement principal de continuer à s'exécuter.
Cela simule un comportement asynchrone tout en utilisant la bibliothèque
`requests`.

```python
import asyncio
import requests

async def fetch(url):
   return await asyncio.to_thread(requests.get, url)

async def main():
   url = 'https://api.example.com/data'
   response = await fetch(url)
   print(response.status_code)
   print(response.text)

asyncio.run(main())
```

2. **Utiliser `aiohttp`** :
Pour une approche véritablement asynchrone,
vous pouvez utiliser la bibliothèque `aiohttp`, qui est conçue pour fonctionner
avec `asyncio`. Elle permet de faire des requêtes HTTP sans bloquer l'exécution
du programme.

```python
import aiohttp
import asyncio

async def fetch(url):
   async with aiohttp.ClientSession() as session:
       async with session.get(url) as response:
           return await response.text()

async def main():
   url = 'https://api.example.com/data'
   html = await fetch(url)
   print(html)

asyncio.run(main())
```

3. **Utiliser `requests-async`** :
Il existe une bibliothèque appelée `requests-async` qui apporte la syntaxe
`async/await` à l'API de `requests`. Cela vous permet d'utiliser
les fonctionnalités familières de `requests` tout en bénéficiant
d'une exécution asynchrone.

```python
import requests_async as requests

async def main():
   response = await requests.get('https://api.example.com/data')
   print(response.status_code)
   print(response.text)

asyncio.run(main())
```

Bien que la bibliothèque `requests` ne soit pas conçue pour être utilisée
de manière asynchrone, vous pouvez contourner cette limitation en utilisant
des méthodes comme `asyncio.to_thread()` ou en optant pour des bibliothèques
comme `aiohttp` ou `requests-async`. Ces approches permettent d'effectuer
des requêtes HTTP sans bloquer le fil d'exécution principal,
rendant votre code plus réactif et efficace, surtout lors de la gestion
de plusieurs requêtes simultanées.

### Références
- [1] https://www.geeksforgeeks.org/asynchronous-http-requests-with-python/
- [2] https://discuss.python.org/t/perform-asynchronous-http-requests/17718
- [3] https://www.calybre.global/post/asynchronous-api-calls-in-python-with-asyncio
- [4] https://superfastpython.com/python-async-requests/
- [5] https://www.twilio.com/en-us/blog/asynchronous-http-requests-in-python-with-aiohttp
- [6] https://pypi.org/project/requests-async/

