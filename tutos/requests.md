# Requests

## Introduction
Le module `requests` en Python est une bibliothèque populaire qui facilite
l'envoi de requêtes HTTP et la gestion des réponses. Voici un aperçu
de son utilisation, y compris des exemples pratiques.

### Installation
Pour utiliser le module `requests`, vous devez d'abord l'installer.
Vous pouvez le faire en utilisant pip :

```bash
pip install requests
```

### Importation du Module

Une fois installé, vous pouvez importer le module dans votre script Python :

```python
import requests
```

### Types de Requêtes HTTP

Le module `requests` prend en charge plusieurs types de requêtes HTTP,
notamment :

- **GET** : Récupérer des données à partir d'une URL.
- **POST** : Envoyer des données à un serveur.
- **PUT** : Mettre à jour des données sur un serveur.
- **DELETE** : Supprimer des ressources sur un serveur.
- **HEAD** : Obtenir les en-têtes d'une réponse sans le corps.

### Exemples d'Utilisation

#### 1. Faire une Requête GET

Pour récupérer des données d'une page web, vous pouvez utiliser
la méthode `get()` :

```python
response = requests.get('https://www.example.com')
print(response.text)  # Affiche le contenu de la page
```

#### 2. Faire une Requête POST

Pour envoyer des données (par exemple, un formulaire), vous pouvez utiliser
la méthode `post()` :

```python
data = {'username': 'user', 'password': 'pass'}
response = requests.post('https://www.example.com/login', data=data)
print(response.text)  # Affiche la réponse du serveur
```

#### 3. Gérer les Réponses

La réponse d'une requête est un objet qui contient plusieurs attributs
utiles :

- `response.text` : Le contenu de la réponse sous forme de chaîne (texte).
- `response.content` : Le contenu de la réponse sous forme de bytes.
- `response.json()` : Pour analyser les réponses JSON.
- `response.status_code` : Le code de statut HTTP (200, 404, etc.).
- `response.headers` : Les en-têtes de la réponse.

Exemple de Gestion de Réponse

```python
response = requests.get('https://api.github.com')

if response.status_code == 200:
    print("Succès :", response.json())  # Affiche les données JSON
else:
    print("Erreur :", response.status_code)
```

#### 4. Utilisation d'un Proxy

Si vous devez passer par un proxy pour faire vos requêtes,
vous pouvez le spécifier comme suit :

```python
proxy = {
    "http": "http://username:password@proxy:port",
    "https": "http://username:password@proxy:port"
}
response = requests.get('https://www.example.com', proxies=proxy)
print(response.text)
```

Le module `requests` simplifie considérablement le processus
d'envoi de requêtes HTTP en Python. Avec ses méthodes intuitives
et sa gestion facile des réponses, il est devenu un standard
pour interagir avec les API et les services web. Que ce soit
pour effectuer des requêtes simples ou gérer des scénarios plus complexes
comme l'authentification ou l'utilisation de proxies, `requests` offre
une interface conviviale et puissante.


### Références
- [1] https://www.w3schools.com/python/module_requests.asp
- [2] https://dridk.me/python-requests.html
- [3] https://realpython.com/python-requests/
- [4] https://www.geeksforgeeks.org/python-requests-tutorial/
- [5] https://www.tresfacile.net/le-module-python-requests/


## Utilisation des coockies
Pour utiliser des cookies avec le module `requests` en Python,
vous pouvez soit passer des cookies lors de l'envoi d'une requête,
soit utiliser un objet `Session` pour gérer les cookies automatiquement.
Voici comment procéder :

### 1. Passer des Cookies dans une Requête

Vous pouvez envoyer des cookies en tant que dictionnaire
via le paramètre `cookies` lors de l'envoi d'une requête. Voici un exemple :

```python
import requests

# URL de la requête
url = 'http://example.com'

# Dictionnaire de cookies à envoyer
cookies = {
    'session_id': '123456789',
    'user': 'JohnDoe'
}

# Faire une requête GET avec des cookies
response = requests.get(url, cookies=cookies)

# Afficher la réponse
print(response.text)
```

### 2. Utiliser un Objet Session

L'utilisation d'un objet `Session` est une méthode pratique
pour gérer les cookies sur plusieurs requêtes.
Cela vous permet de conserver les cookies entre les requêtes
sans avoir à les spécifier à chaque fois.

Exemple d'Utilisation d'un Objet Session

```python
import requests

# Créer une session
session = requests.Session()

# Faire une première requête pour obtenir des cookies
response = session.get('http://example.com')

# Les cookies reçus sont stockés dans l'objet session
print("Cookies après la première requête :", session.cookies)

# Faire une seconde requête avec les cookies automatiquement gérés
response = session.get('http://example.com/another-page')

# Afficher la réponse
print(response.text)

# Fermer la session lorsque vous avez terminé
session.close()
```

### 3. Récupérer les Cookies de la Réponse

Après avoir effectué une requête, vous pouvez accéder aux cookies renvoyés
par le serveur via l'attribut `cookies` de l'objet `Response`.

Exemple de Récupération des Cookies

```python
response = requests.get('http://example.com')

# Récupérer les cookies renvoyés par le serveur
received_cookies = response.cookies

# Afficher les cookies reçus
for cookie in received_cookies:
    print(f"{cookie.name}: {cookie.value}")
```

### 4. Sauvegarder et Charger des Cookies

Si vous souhaitez persister des cookies entre les exécutions de votre script,
vous pouvez les sauvegarder dans un fichier et les charger ultérieurement.
Voici un exemple simple :

Sauvegarder les Cookies

```python
import requests
import pickle

# Créer une session et faire une requête
session = requests.Session()
session.get('http://example.com')

# Sauvegarder les cookies dans un fichier
with open('cookies.pkl', 'wb') as f:
    pickle.dump(session.cookies, f)
```

Charger les Cookies

```python
import requests
import pickle

# Créer une nouvelle session
session = requests.Session()

# Charger les cookies depuis le fichier
with open('cookies.pkl', 'rb') as f:
    session.cookies.update(pickle.load(f))

# Faire une requête avec les cookies chargés
response = session.get('http://example.com/another-page')
print(response.text)
```


Le module `requests` facilite grandement la gestion des cookies en Python.
Que vous choisissiez de passer des cookies directement dans vos requêtes
ou d'utiliser un objet `Session` pour gérer automatiquement les cookies,
le module offre une interface simple et intuitive pour interagir
avec des sites web et des API qui utilisent des sessions basées
sur des cookies.

### Références
- [1] https://deptinfo-ensip.univ-poitiers.fr/ENS/doku/doku.php/stu:python_reseau:requests
- [2] https://stackoverflow.com/questions/31554771/how-can-i-use-cookies-in-python-requests
- [3] https://dridk.me/python-requests.html
- [4] https://brightdata.fr/blog/donnees-web/python-requests-guide
- [5] http://www.python-simple.com/python-modules-internet/requests.php
- [6] https://openclassrooms.com/forum/sujet/requests-et-cookies-mon-cookies-tiens-15-minutes
- [7] https://www.reddit.com/r/Python/comments/6n7l79/loading_cookies_in_requests_module/?tl=fr
- [8] https://docs.nest-js.fr/techniques/cookies


## Gestion des erreurs
Pour gérer les erreurs HTTP avec le module `requests` en Python,
vous pouvez utiliser plusieurs techniques pour détecter et traiter
les erreurs potentielles lors de l'envoi de requêtes HTTP.
Voici un guide détaillé sur la gestion des erreurs, accompagné
d'exemples pratiques.

### 1. Vérification du Code de Statut

Après avoir effectué une requête, vous pouvez vérifier le code de statut
de la réponse pour déterminer si la requête a réussi ou échoué.
Un code de statut 200 indique un succès, tandis que d'autres codes
(comme 404 ou 500) signalent des erreurs.

#### Exemple de Vérification du Code de Statut

```python
import requests

url = 'https://api.example.com/data'

response = requests.get(url)

if response.status_code == 200:
    print("Requête réussie !")
    print(response.json())  # Traiter la réponse JSON
elif response.status_code == 404:
    print("Erreur : Ressource non trouvée (404)")
else:
    print(f"Erreur : La requête a échoué avec le code {response.status_code}")
```

### 2. Gestion des Exceptions

Le module `requests` lève des exceptions pour certaines erreurs,
comme les problèmes de connexion ou les erreurs de temps d'attente.
Vous pouvez gérer ces exceptions en utilisant un bloc `try-except`.

#### Exemple de Gestion des Exceptions

```python
import requests

url = 'https://api.example.com/data'

try:
    response = requests.get(url)
    response.raise_for_status()
    #: Lève une exception pour les codes d'erreur HTTP

    print("Requête réussie !")
    print(response.json())
except requests.exceptions.HTTPError as http_err:
    print(f"Erreur HTTP : {http_err}")  # Gérer les erreurs HTTP spécifiques
except requests.exceptions.ConnectionError:
    print("Erreur : Problème de connexion.")
except requests.exceptions.Timeout:
    print("Erreur : La requête a expiré.")
except requests.exceptions.RequestException as err:
    print(f"Une erreur s'est produite : {err}")
    #: Gérer toutes les autres exceptions
```

### 3. Utilisation d'un Session pour Gérer les Cookies et les Erreurs

Lorsque vous utilisez un objet `Session`, il peut être utile pour maintenir
les cookies et gérer les erreurs sur plusieurs requêtes.

Exemple avec un Objet Session

```python
import requests

session = requests.Session()

try:
    response = session.get('https://api.example.com/data')
    response.raise_for_status()
    print("Requête réussie !")
    print(response.json())
except requests.exceptions.RequestException as e:
    print(f"Une erreur s'est produite : {e}")
finally:
    session.close()  # Fermer la session après utilisation
```


Le module `requests` facilite la gestion des erreurs HTTP grâce
à des vérifications simples du code de statut et à une gestion des exceptions
robuste. En utilisant ces techniques, vous pouvez rendre
votre code plus résilient et capable de gérer divers scénarios
d'erreur lors des interactions avec des API ou des ressources web.
Cela améliore l'expérience utilisateur et permet une meilleure gestion
des problèmes potentiels lors des requêtes HTTP.

### Références
- [1] https://brightdata.fr/blog/donnees-web/python-requests-guide
- [2] https://dridk.me/python-requests.html
- [3] https://www.tresfacile.net/le-module-python-requests/


## Envoyer des fichiers multimédias
Pour envoyer des fichiers multimédias avec le module `requests` en Python,
vous pouvez utiliser la méthode `post()` en spécifiant le fichier
à télécharger dans le paramètre `files`. Voici un guide détaillé
avec des exemples pratiques.

### Étapes pour Envoyer des Fichiers Multimédias

1. **Préparer le Fichier à Envoyer** : Ouvrir le fichier en mode binaire.
2. **Utiliser la Méthode `post()`** : Envoyer le fichier au serveur
avec la méthode `requests.post()`.
3. **Gérer la Réponse** : Vérifier si l'envoi a réussi et traiter la réponse.

### Exemple de Code

Voici un exemple complet qui montre comment envoyer une image à un serveur :

```python
import requests

# URL du serveur où vous souhaitez envoyer le fichier
url = 'https://example.com/upload'

# Chemin vers le fichier que vous souhaitez envoyer
file_path = 'path/to/your/image.jpg'

# Ouvrir le fichier en mode binaire
with open(file_path, 'rb') as file:
    # Préparer les fichiers à envoyer
    files = {'file': file}  # 'file' est le nom du champ attendu par le serveur

    # Envoyer la requête POST avec le fichier
    response = requests.post(url, files=files)

# Vérifier la réponse du serveur
if response.status_code == 200:
    print("Fichier envoyé avec succès !")
    print("Réponse du serveur :", response.text)
else:
    print(f"Erreur lors de l'envoi du fichier : {response.status_code}")
```

### Explication du Code

1. **Importation du Module** : Le module `requests` est importé pour gérer
les requêtes HTTP.

2. **Définition de l'URL** : L'URL où le fichier sera envoyé est définie.

3. **Ouverture du Fichier** : Le fichier est ouvert en mode binaire (`'rb'`),
ce qui est nécessaire pour les fichiers multimédias.

4. **Préparation des Fichiers** : Un dictionnaire `files` est créé,
où la clé correspond au nom du champ attendu par le serveur
(dans cet exemple, `'file'`).

5. **Envoi de la Requête POST** : La méthode `requests.post()`
est utilisée pour envoyer le fichier au serveur.

6. **Gestion de la Réponse** : Après l'envoi, le code vérifie
si la réponse a un code de statut 200 (succès) et affiche un message approprié.

### Envoi de Plusieurs Fichiers

Si vous souhaitez envoyer plusieurs fichiers à la fois,
vous pouvez les ajouter au dictionnaire `files` :

```python
# Chemins vers les fichiers que vous souhaitez envoyer
file_paths = ['path/to/your/image1.jpg', 'path/to/your/image2.png']

# Préparer les fichiers à envoyer
files = {}
for i, file_path in enumerate(file_paths):
    with open(file_path, 'rb') as file:
        files[f'file{i+1}'] = file
        #: Utiliser des noms de champs différents si nécessaire

# Envoyer la requête POST avec plusieurs fichiers
response = requests.post(url, files=files)

# Vérifier la réponse du serveur
if response.status_code == 200:
    print("Fichiers envoyés avec succès !")
else:
    print(f"Erreur lors de l'envoi des fichiers : {response.status_code}")
```

L'utilisation du module `requests` pour envoyer des fichiers multimédias
est simple et efficace. En utilisant la méthode `post()` avec le paramètre
`files`, vous pouvez facilement télécharger des fichiers vers un serveur
tout en gérant les réponses et les erreurs potentielles.
Assurez-vous que l'URL et les noms des champs correspondent aux attentes
du serveur pour garantir un envoi réussi.

### Références
- [1] https://developer.android.com/training/secure-file-sharing/request-file?hl=fr
- [2] https://developers.google.com/photos/library/legacy/guides/upload-media?hl=fr
- [3] https://brightdata.fr/blog/donnees-web/python-requests-guide
- [4] https://developer.mozilla.org/fr/docs/Learn/Forms/Sending_and_retrieving_form_data
- [5] https://massive.io/fr/comment-faire/comment-recevoir-des-fichiers-grands/
- [6] https://openclassrooms.com/forum/sujet/upload-de-fichiers-avec-le-paquetage-request


## Bonnes pratiques pour envoyer des fichiers multimédias
Pour envoyer des fichiers multimédias avec le module `requests`
en Python, il est important de suivre certaines bonnes pratiques
pour garantir que les fichiers sont correctement transmis
et que l'application fonctionne de manière fiable.
Voici quelques recommandations basées sur les meilleures pratiques :

### 1. Utiliser la Méthode POST

Lorsque vous envoyez des fichiers multimédias, utilisez la méthode HTTP `POST`,
qui est conçue pour envoyer des données au serveur. Assurez-vous que l'URL
à laquelle vous envoyez le fichier accepte les requêtes POST.

### 2. Spécifier le Type de Contenu

Lorsque vous envoyez des fichiers, il est généralement recommandé d'utiliser
le type de contenu `multipart/form-data`. Cela permet d'envoyer des fichiers
et des données de formulaire dans une seule requête.

### 3. Gérer les Fichiers Correctement

Ouvrez les fichiers en mode binaire (`'rb'`) pour éviter toute corruption
de données, surtout pour les fichiers multimédias comme les images
ou les vidéos.

#### Exemple de Code

Voici un exemple pratique qui montre comment envoyer un fichier image
à un serveur :

```python
import requests

# URL du serveur où vous souhaitez envoyer le fichier
url = 'https://example.com/upload'

# Chemin vers le fichier que vous souhaitez envoyer
file_path = 'path/to/your/image.jpg'

# Ouvrir le fichier en mode binaire
with open(file_path, 'rb') as file:
    # Préparer les fichiers à envoyer
    files = {'file': file}  # 'file' est le nom du champ attendu par le serveur

    # Envoyer la requête POST avec le fichier
    response = requests.post(url, files=files)

# Vérifier la réponse du serveur
if response.status_code == 200:
    print("Fichier envoyé avec succès !")
else:
    print(f"Erreur lors de l'envoi du fichier : {response.status_code}")
```

### 4. Gérer les Erreurs

Implémentez une gestion des erreurs pour traiter les échecs d'envoi
ou les réponses inattendues du serveur. Utilisez des blocs `try-except`
pour capturer les exceptions potentielles.

```python
try:
    response = requests.post(url, files=files)
    response.raise_for_status()  # Lève une exception pour les codes d'erreur HTTP
except requests.exceptions.HTTPError as http_err:
    print(f"Erreur HTTP : {http_err}")
except requests.exceptions.RequestException as req_err:
    print(f"Erreur lors de la requête : {req_err}")
```

### 5. Limiter la Taille des Fichiers

Soyez conscient des limites de taille imposées par le serveur pour éviter
d'échouer lors de l'envoi de fichiers trop volumineux.
Vous pouvez vérifier la documentation de l'API ou du serveur pour connaître
ces limites.

### 6. Utiliser des Sessions si Nécessaire

Si vous devez envoyer plusieurs fichiers ou faire plusieurs requêtes
au même serveur, envisagez d'utiliser un objet `Session` pour maintenir
les cookies et améliorer les performances.

```python
session = requests.Session()
response = session.post(url, files=files)
```

En suivant ces bonnes pratiques lors de l'envoi de fichiers multimédias
avec le module `requests`, vous pouvez garantir que vos requêtes
sont efficaces et fiables. Cela inclut l'utilisation correcte
des méthodes HTTP, la gestion appropriée des fichiers et des erreurs,
ainsi que la prise en compte des limitations du serveur.
Ces étapes contribueront à créer une expérience utilisateur fluide
et sans erreur lors du téléchargement de fichiers multimédias.

### Références
- [1] https://developer.mozilla.org/fr/docs/Learn/Forms/Sending_and_retrieving_form_data
- [2] https://brightdata.fr/blog/donnees-web/python-requests-guide
- [3] https://developer.android.com/training/data-storage/use-cases?hl=fr
- [4] https://developer.android.com/training/data-storage/shared/media?hl=fr
- [5] https://fr.linkedin.com/advice/1/how-can-you-use-email-attachments-links-make-your?lang=fr


## Les types MIME
Les types MIME (Multipurpose Internet Mail Extensions) sont utilisés
pour indiquer la nature et le format d'un document, et ils sont essentiels
pour le transfert de fichiers multimédias sur Internet. Voici les types MIME
les plus couramment utilisés pour les fichiers multimédias :

### Types MIME Courants pour les Fichiers Multimédias

1. **Audio**
   - `audio/mpeg` : Fichiers audio au format MP3.
   - `audio/wav` : Fichiers audio au format WAV.
   - `audio/ogg` : Fichiers audio au format OGG, souvent avec le codec Vorbis.
   - `audio/webm` : Fichiers audio au format WebM, utilisant généralement
les codecs Vorbis ou Opus.
   - `audio/x-ms-wma` : Windows Media Audio.

2. **Vidéo**
   - `video/mp4` : Fichiers vidéo au format MP4.
   - `video/webm` : Fichiers vidéo au format WebM, utilisant les codecs
VP8 ou VP9.
   - `video/ogg` : Fichiers vidéo au format OGG, souvent avec le codec Theora.
   - `video/x-ms-wmv` : Windows Media Video.
   - `video/x-flv` : Flash Video (FLV).

3. **Image**
   - `image/jpeg` : Images au format JPEG.
   - `image/png` : Images au format PNG.
   - `image/gif` : Images au format GIF.
   - `image/svg+xml` : Images au format SVG.

Ces types MIME sont largement utilisés dans les applications web
et les services API pour gérer le transfert de fichiers multimédias.
En spécifiant correctement le type MIME lors de l'envoi ou de la réception
de fichiers, vous garantissez que le contenu est traité correctement
par les navigateurs et autres applications. Pour plus d'informations
sur les types MIME, vous pouvez consulter des ressources
comme [MDN Web Docs](https://developer.mozilla.org/fr/docs/Web/HTTP/MIME_types)
ou d'autres documents techniques sur le sujet.

### Références
- [1] https://developer.mozilla.org/fr/docs/Web/HTTP/MIME_types
- [2] https://cedric.cnam.fr/sys/crolard/enseignement/NFA040/TP17.pdf
- [3] https://fr.wikipedia.org/wiki/Type_MIME
- [4] http://www.dg77.net/tekno/xhtml/mime.htm
- [5] https://www.alsacreations.com/article/lire/1918-Qu-est-ce-qu-un-type-MIME.html
- [6] https://code-garage.fr/blog/comprendre-les-types-mime
- [7] https://www.techno-science.net/glossaire-definition/Type-MIME.html

## Vérifier la sécurité des fichiers multimédias envoyés

Pour assurer la sécurité des fichiers multimédias envoyés
avec le module `requests`, il est essentiel de suivre certaines
bonnes pratiques et recommandations. Voici un résumé des meilleures
pratiques basées sur les résultats de recherche :

### 1. Authentification des Utilisateurs

Avant de permettre le téléchargement de fichiers,
assurez-vous que les utilisateurs sont authentifiés. Cela peut inclure
l'utilisation de mots de passe, d'authentification à deux facteurs (2FA)
ou d'autres méthodes pour garantir que seuls les utilisateurs autorisés
peuvent télécharger des fichiers.

### 2. Contrôles d'Accès

Mettez en place des contrôles d'accès stricts pour limiter qui peut télécharger
des fichiers et où ces fichiers peuvent être stockés. Cela aide à prévenir
les abus et les téléchargements non autorisés.

### 3. Restrictions sur les Types de Fichiers

Limitez les types de fichiers pouvant être téléchargés pour minimiser
le risque d'injection de logiciels malveillants. Par exemple,
vous pouvez interdire les fichiers exécutables (`.exe`, `.sh`, etc.)
et n'autoriser que les formats nécessaires, comme les images (`.jpg`, `.png`)
ou les documents (`.pdf`).

### 4. Validation et Assainissement des Noms de Fichiers

Assurez-vous que les noms de fichiers sont validés et assainis
avant d'être enregistrés sur le serveur. Cela inclut :
- Éviter l'utilisation de chemins relatifs ou absolus fournis
par l'utilisateur qui pourraient mener à des attaques par traversée de chemin.
- Renommer les fichiers pour éviter toute ambiguïté ou conflit.

### 5. Vérification des Fichiers Malveillants

Implémentez une vérification antivirus ou un scan des fichiers téléchargés
pour détecter tout contenu malveillant avant qu'il ne soit stocké
sur le serveur.

### 6. Stockage Sécurisé

Stockez les fichiers dans un emplacement sécurisé, idéalement en dehors
du répertoire racine du serveur web, pour éviter un accès direct via l'URL.

### 7. Journalisation et Audits

Mettez en place une journalisation des activités liées aux téléchargements
de fichiers et effectuez régulièrement des audits de sécurité pour détecter
toute activité suspecte ou non conforme.

### 8. Limitation de la Taille des Fichiers

Imposez des limites sur la taille des fichiers pouvant être téléchargés
pour éviter l'épuisement des ressources du serveur.

### 9. Utilisation d'un Environnement Isolé

Considérez l'utilisation d'un environnement isolé (sandbox)
pour traiter les fichiers téléchargés afin de réduire le risque d'exécution
de code malveillant.


En suivant ces bonnes pratiques, vous pouvez réduire considérablement
le risque associé à l'envoi et au stockage de fichiers multimédias
via le module `requests`. La sécurité doit être intégrée
dès la conception du système pour garantir une protection adéquate
contre les menaces potentielles.

### Références
- [1] https://meritis.fr/blog/5-failles-de-securite-expliquees-en-python/
- [2] https://massive.io/fr/securite-et-conformite/how-to-upload-files-securely/
- [3] https://developer.android.com/privacy-and-security/safetynet/attestation?hl=fr
- [4] https://web.dev/articles/fetch-metadata?hl=fr
- [5] https://brightdata.fr/blog/donnees-web/python-requests-guide
- [6] https://developer.mozilla.org/fr/docs/Learn/Forms/Sending_and_retrieving_form_data


## Limitations de la bibliothèque requests pour les fichiers de plus de 50 Mo
La bibliothèque `requests` en Python est largement utilisée
pour effectuer des requêtes HTTP, y compris l'envoi de fichiers.
Cependant, lors de l'envoi de fichiers volumineux (comme ceux dépassant 50 Mo),
plusieurs limitations et considérations doivent être prises en compte :

### Limitations de la Bibliothèque Requests pour les Fichiers Volumineux

1. **Limites du Serveur** :
  - Les serveurs web peuvent imposer des limites sur la taille des fichiers
pouvant être téléchargés. Par exemple, des serveurs comme **IIS**
ont une limite par défaut de 30 Mo pour la taille maximale d'une demande (`maxAllowedContentLength`). Si vous essayez d'envoyer un fichier
plus volumineux, le serveur peut renvoyer une erreur HTTP 413
(Request Entity Too Large) [1].

2. **Configuration de l'Application** :
   - Dans certaines applications web, comme celles basées sur **ASP.NET Core**,
la taille maximale des fichiers multipart peut être configurée
par défaut à 128 Mo. Cela signifie que si vous ne configurez pas explicitement
cette limite, vous pourriez rencontrer des problèmes
lors de l'envoi de fichiers volumineux [3].

3. **Gestion de la Mémoire** :
   - Lorsque vous envoyez de gros fichiers, la gestion
de la mémoire devient cruciale. La bibliothèque `requests` lit le fichier
en mémoire avant de l'envoyer, ce qui peut entraîner une utilisation excessive
de la mémoire si le fichier est très volumineux. Cela peut provoquer
des ralentissements ou même des échecs si la mémoire disponible
est insuffisante.

4. **Timeouts et Connexions Instables** :
   - Les fichiers volumineux peuvent prendre plus de temps à être téléchargés,
ce qui peut entraîner des timeouts si le serveur ou le client a des paramètres
de timeout trop courts. Assurez-vous d'augmenter les paramètres
de timeout si nécessaire.

5. **Erreurs Réseau** :
   - Les connexions réseau peuvent être instables lors du transfert
de fichiers volumineux, ce qui pourrait entraîner des interruptions.
Il est important d'implémenter une logique de reprise
ou de gestion des erreurs pour gérer ces cas.

### Recommandations pour Envoyer des Fichiers Volumineux

- **Utiliser un Envoi par Morceaux (Chunked Upload)** :
Si le serveur le permet, envisagez d'envoyer le fichier en morceaux plutôt
qu'en une seule fois pour éviter les problèmes liés à la taille.
  
- **Configurer les Paramètres du Serveur** : Si vous avez accès
à la configuration du serveur, ajustez les limites pour permettre des fichiers
plus volumineux.

- **Gérer les Exceptions et Timeouts** : Implémentez une gestion
robuste des erreurs et ajustez les timeouts pour tenir compte
du temps supplémentaire nécessaire pour envoyer des fichiers volumineux.

- **Vérifier l'Intégrité du Fichier** : Après l'envoi, vérifiez
que le fichier a été reçu correctement en utilisant des sommes de contrôle
ou d'autres méthodes d'intégrité.


Bien que `requests` soit une bibliothèque puissante
pour gérer les requêtes HTTP, il est essentiel d'être conscient
des limitations concernant l'envoi de fichiers volumineux. En suivant
ces recommandations et en tenant compte des limites du serveur
et des configurations applicatives, vous pouvez améliorer vos chances
de réussir à envoyer efficacement des fichiers dépassant 50 Mo.

### Références
- [1] https://learn.microsoft.com/fr-fr/iis/configuration/system.webserver/security/requestfiltering/requestlimits/
- [2] https://docs.github.com/fr/repositories/creating-and-managing-repositories/repository-limits
- [3] https://learn.microsoft.com/fr-fr/aspnet/core/mvc/models/file-uploads?view=aspnetcore-9.0
- [4] https://www.ibm.com/docs/fr/i/7.5?topic=capacities-file-system-limits



## Utilisation des certificats SSL
Pour utiliser des certificats SSL avec le module `requests` en Python,
vous devez savoir comment gérer les certificats lors de vos requêtes HTTP.
Voici un guide détaillé sur la manière de procéder.

### 1. Installation du Module Requests
Si vous ne l'avez pas encore fait, installez le module `requests` :

```bash
pip install requests
```

### 2. Utilisation des Certificats SSL

Le module `requests` permet d'utiliser des certificats SSL pour sécuriser
les connexions. Vous pouvez spécifier un certificat client
(pour l'authentification) et un fichier CA (Certificate Authority)
pour vérifier la validité du certificat du serveur.

#### a. Spécifier un Certificat Client

Si vous devez vous authentifier auprès d'un serveur via un certificat client,
vous pouvez le faire en utilisant les paramètres `cert` et `verify`.

```python
import requests

url = 'https://example.com/api'

# Chemin vers votre certificat client et votre clé privée
cert = ('/path/to/client.crt', '/path/to/client.key')

# Faire une requête avec le certificat client
response = requests.get(url, cert=cert)

print(response.text)
```

#### b. Vérification du Certificat du Serveur

Par défaut, `requests` vérifie le certificat SSL du serveur.
Si vous avez un fichier CA spécifique que vous souhaitez utiliser
pour vérifier le certificat du serveur, vous pouvez le spécifier
avec le paramètre `verify`.

```python
# Chemin vers le fichier CA
ca_cert = '/path/to/ca_bundle.crt'

# Faire une requête avec vérification du certificat
response = requests.get(url, verify=ca_cert)

print(response.text)
```

### 3. Désactiver la Vérification SSL (Non Recommandé)

Il est possible de désactiver la vérification SSL en passant `verify=False`,
mais cela n'est pas recommandé car cela expose votre application
à des attaques de type "man-in-the-middle".

```python
response = requests.get(url, verify=False)
print(response.text)
```

### 4. Gestion des Exceptions

Lorsque vous travaillez avec des connexions SSL, il est important de gérer
les exceptions qui peuvent survenir, par exemple lorsqu'un certificat
est invalide ou que la connexion échoue.

```python
try:
    response = requests.get(url, cert=cert, verify=ca_cert)
    response.raise_for_status()  # Vérifie si la requête a échoué
    print(response.text)
except requests.exceptions.SSLError as ssl_err:
    print(f"Erreur SSL : {ssl_err}")
except requests.exceptions.RequestException as req_err:
    print(f"Erreur de requête : {req_err}")
```

Le module `requests` facilite l'utilisation des certificats SSL pour sécuriser
les communications HTTP en Python. En spécifiant des certificats client
et en vérifiant les certificats du serveur, vous pouvez établir des connexions
sécurisées tout en gérant efficacement les erreurs potentielles
liées aux certificats. Assurez-vous toujours de vérifier les certificats
pour garantir la sécurité de vos communications réseau.


### Références
- [1] https://www.certeurope.fr/blog/guide-csr-certificat/
- [2] https://www.kaspersky.fr/resource-center/definitions/what-is-a-ssl-certificate
- [3] https://help.tableau.com/current/server/fr-fr/ssl_cert_create.htm
- [4] https://experienceleague.adobe.com/fr/docs/deliverability-learn/deliverability-best-practice-guide/additional-resources/campaign/ac-ssl-certificate-request
- [5] https://clients.genious.net/knowledgebase/85/Comment-configurer-et-installer-un-certificat-SSL-sur-cPanel-.html?language=portuguese-pt


## Gérer les fichiers de plus de 50 Mo
Pour gérer l'envoi de fichiers multimédias de plus de 50 Mo
avec le module `requests` en Python, il est important de prendre
en compte plusieurs aspects, notamment la gestion des limites de taille,
la fragmentation des fichiers si nécessaire, et l'utilisation
d'une connexion stable. Voici quelques conseils pratiques :

### 1. Vérifier les Limites du Serveur

Avant d'envoyer un fichier volumineux, assurez-vous que le serveur accepte
des fichiers de cette taille. De nombreux serveurs ont une limite de taille
pour les fichiers téléchargés (par exemple, 50 Mo). Si vous essayez d'envoyer
un fichier plus gros, vous pourriez recevoir une erreur
comme "413 Request Entity Too Large".

### 2. Utiliser des Requêtes Chunked

Si le serveur le permet, vous pouvez envoyer des fichiers en utilisant
une approche "chunked" (par morceaux). Cela consiste à envoyer le fichier
en plusieurs parties plutôt qu'en une seule requête.
Voici un exemple de code pour envoyer un fichier par morceaux :

```python
import requests

def upload_large_file(url, file_path):
    with open(file_path, 'rb') as f:
        # Lire et envoyer le fichier par morceaux
        while True:
            chunk = f.read(1024 * 1024)  # Lire 1 Mo à la fois
            if not chunk:
                break
            response = requests.post(url, data=chunk)
            if response.status_code != 200:
                print(f"Erreur lors de l'envoi : {response.status_code}")
                return
    print("Fichier envoyé avec succès !")

# Exemple d'utilisation
upload_large_file('https://example.com/upload', 'path/to/your/largefile.mp4')
```

### 3. Utiliser une Session

Utiliser une session `requests.Session()` peut améliorer
la gestion des connexions et la performance lors de l'envoi
de plusieurs fichiers ou de fichiers volumineux.

```python
session = requests.Session()
response = session.post(url, files={'file': open(file_path, 'rb')})
```

### 4. Gérer les Erreurs et les Exceptions

Implémentez une gestion des erreurs robuste pour traiter les problèmes
qui peuvent survenir lors de l'envoi de fichiers volumineux.

```python
try:
    response = requests.post(url, files={'file': open(file_path, 'rb')})
    response.raise_for_status()
    #: Lève une exception pour les codes d'erreur HTTP

except requests.exceptions.HTTPError as err:
    print(f"Erreur HTTP : {err}")
except requests.exceptions.ConnectionError:
    print("Erreur : Problème de connexion.")
except requests.exceptions.Timeout:
    print("Erreur : La requête a expiré.")
except Exception as e:
    print(f"Une erreur s'est produite : {e}")
```

### 5. Configurer le Serveur

Si vous avez accès au serveur, assurez-vous que sa configuration
permet le téléchargement de fichiers volumineux. Par exemple,
dans un serveur web comme Nginx ou Apache, vous devrez peut-être ajuster
les paramètres pour augmenter la limite de taille des fichiers.


En suivant ces étapes et conseils, vous pouvez gérer efficacement l'envoi
de fichiers multimédias de plus de 50 Mo avec le module `requests`.
Assurez-vous toujours que votre code gère les erreurs correctement
et que le serveur est configuré pour accepter des fichiers volumineux.

### Références
- [1] https://help.dropbox.com/fr-fr/share/manage-file-requests
- [2] https://massive.io/fr/comment-faire/comment-recevoir-des-fichiers-grands/
- [3] https://brightdata.fr/blog/donnees-web/python-requests-guide
- [4] https://www.docstring.fr/formations/scraping-avance-mise-en-production/recuperer-le-html-avec-requests-2060/
- [5] https://openclassrooms.com/forum/sujet/upload-de-fichiers-avec-le-paquetage-request
- [6] https://www.reddit.com/r/node/comments/1b0dtji/handle_large_file_with_limited_connection/?tl=fr
- [7] https://www.autodesk.com/fr/support/technical/article/caas/sfdcarticles/sfdcarticles/FRA/413-Request-Entity-Too-Large-when-trying-to-check-in-an-Inventor-file-bigger-than-50-MB-to-Vault.html


