# isiMonitor3D — Communiquer avec le système (guide AGV)

*Guide court et opérationnel pour l'intégrateur / automaticien AGV. Tout ce qu'il
faut pour se connecter et consommer le signal — la RFC complète reste la
référence détaillée.*

## 1. L'idée en 30 secondes

isiMonitor3D, ce sont les yeux de l'entrepôt : des caméras surveillent des
zones marquées au sol et publient en continu **ce qui s'y trouve** — palette
présente ou non, vide ou chargée, carton ou polybag, personne dans la zone.

Vous n'avez jamais à parler aux caméras ni aux algorithmes de vision. Tout ce
que le système sait est publié en **JSON** sur un **broker MQTT** central, et le
même contenu est disponible en **REST** via la passerelle **isicomms**. Votre
automate choisit la voie qui lui convient :

- **Voie A — MQTT** (recommandée) : vous vous abonnez une fois, les changements
  arrivent tout seuls (rafraîchissement ~1 s).
- **Voie B — REST** : vous interrogez une URL HTTP à votre rythme (1–2 Hz).

## 2. Le schéma

```
   caméras           PC entrepôt                 serveur central               vous (AGV/WMS)
 ┌─────────┐   ┌───────────────────┐         ┌─────────────────────────┐
 │  cam_a  ├──►│  Backbone         │  MQTT   │  Broker MQTT    :1883   │◄── A) abonnement MQTT
 │  cam_b  ├──►│  node "wh_pc_01"  ├───────► │           │             │
 └─────────┘   └───────────────────┘ publie  │           ▼             │
                (un par zone/PC ;            │  isicomms       :8080   │◄── B) GET REST
                 il ne fait que publier)     │  agrège tous les PCs    │
                                             │  en une API unique      │
                                             └─────────────────────────┘
```

Chaque PC d'entrepôt est un **nœud** (`node_id`). Ajouter un PC ne change rien
pour vous : il apparaît tout seul dans l'arbre des topics et dans l'API.

## 3. L'arbre des topics — chaque branche expliquée

Tous les topics suivent le même motif :

```
isiMonitor3D / v1 / <node_id> / <branche>
    │           │       │
    │           │       └─ quel PC parle (ex. wh_pc_01)
    │           └─ version du protocole
    └─ racine (configurable par déploiement)
```

L'arbre complet publié par chaque nœud :

```
isiMonitor3D/v1/<node_id>/
├── zone/<zone>                  ⭐ état courant d'une zone (retenu)  ← VOTRE signal
├── zone/<zone>/passings            événements entrée / sortie de zone
├── zone/<zone>/images/<id>         URL d'un instantané (jamais d'octets d'image)
├── track2d/<classe>                position au sol de chaque objet, en mètres, en continu
├── track3d/<classe>                position 3D (nœuds à deux caméras)
├── proximity                       alertes de proximité personne / engin (retenu)
├── config                          carte d'identité du nœud : zones, caméras, mode (retenu)
└── diagnostics/heartbeat           battement de cœur toutes les ~5 s : santé, fps, latence
```

| Branche | Ce qu'elle transporte | Pour un AGV |
|---|---|---|
| `zone/<zone>` | La liste complète des objets présents dans la zone, republiée à chaque changement. **Retenue** : l'état courant arrive dès la connexion. | **C'est votre signal principal.** |
| `zone/<zone>/passings` | Un événement par franchissement : `enter` ou `leave`. | Utile pour journaliser. |
| `zone/<zone>/images/<id>` | Une URL d'instantané — les images ne transitent jamais par MQTT. | À ignorer. |
| `track2d/<classe>` | Position + vitesse au sol (mètres entrepôt) de chaque objet suivi. | À ignorer, sauf navigation fine. |
| `track3d/<classe>` | Position 3D par stéréovision. | À ignorer. |
| `proximity` | Alertes de proximité personne / machine. | Optionnel (sécurité). |
| `config` | Zones définies, caméras, mode du nœud. Sert à la découverte automatique. | Consulter une fois si besoin. |
| `diagnostics/heartbeat` | Santé du nœud (fps, latence, caméras). | Surveiller que le nœud est vivant. |

**Attention :** le segment `<zone>` du topic est un identifiant interne stable
(ex. `zp_mr8z7cot`), pas le nom lisible. Filtrez sur le champ `zone` **du
payload** (ex. `"Sortie_1"`), jamais sur le topic.

## 4. Le message qui vous intéresse : `zone_state`

Un JSON par zone, **retenu** sur le broker : l'état courant arrive
immédiatement à la connexion, puis à chaque changement (~1 s tant que la zone
est occupée). Payload réel capturé sur le système en marche :

```json
{
  "schema_version": 6,
  "type": "zone_state",
  "ts": 1785156682.85,
  "zone": "Sortie_1",
  "zone_id": "zp_mr8z7cot",
  "objects": [
    { "track_id": 1220,
      "cls": "palette",
      "confidence": 0.87,
      "xy_m": [-1.00, -0.12],
      "occupancy_state": "full",
      "occupancy_content": "carton",
      "occupancy_confidence": 0.82 }
  ],
  "count": 1
}
```

| Champ | Signification |
|---|---|
| `zone` | Nom de la zone (ex. `Sortie_1`) — **filtrez sur ce champ**. |
| `objects[].cls` | Classe détectée : `palette`, `person`, `carton`, … |
| `objects[].confidence` | Confiance de détection, 0 à 1. |
| `objects[].xy_m` | Position dans l'entrepôt, en mètres (informatif ici). |
| `objects[].occupancy_state` | Palette uniquement : `"empty"` / `"full"`. |
| `objects[].occupancy_content` | Palette uniquement : `"carton"` / `"polybag"`. |
| `count` | Nombre d'objets dans la zone. |

Trois règles à retenir :

1. **Retenu** = à la connexion (ou reconnexion), vous recevez l'état courant en
   moins de 2 s, sans attendre un changement.
2. **Zone vide** = un message explicite avec `"objects": []`. *Vide* ne veut
   jamais dire *hors ligne* — le silence, si.
3. Pas de synchronisation d'horloge nécessaire : réagissez à l'arrivée du
   message ; `ts` est informatif.

## 5. Deux façons de se connecter

| | Voie A — MQTT (recommandée) | Voie B — REST |
|---|---|---|
| Adresse | `<IP_SERVEUR>:1883` (TCP, sans authentification en profil test LAN) | `http://<IP_SERVEUR>:8080` |
| Geste | S'abonner à `isiMonitor3D/v1/+/zone/+` en QoS 1 | `GET /v1/zones/Sortie_1` en boucle, 1–2 Hz |
| Modèle | Événementiel — les changements arrivent seuls | Interrogation — vous tirez l'état |
| Test rapide | `mosquitto_sub -h <IP> -t 'isiMonitor3D/v1/+/zone/+' -v` | `curl http://<IP>:8080/v1/zones` |

**Actuellement, `<IP_SERVEUR>` = `192.168.2.113`** — donc broker MQTT sur
`192.168.2.113:1883` et REST sur `http://192.168.2.113:8080`.

Les deux voies servent exactement les mêmes champs JSON. Pour explorer en
direct : console de test `http://<IP_SERVEUR>:8080/test` (une carte par
vérification, avec l'URL REST et le topic MQTT exacts à copier) et Swagger sur
`/docs`.

## 6. Programme d'exemple (testé sur le système réel)

C'est très probablement à ceci que ressemblera votre intégration : un client
MQTT qui surveille une zone et déclenche le pick. Python 3, une seule
dépendance (`pip install paho-mqtt`) :

```python
#!/usr/bin/env python3
"""Client minimal isiMonitor3D - déclencheur de pick AGV."""
import json
import paho.mqtt.client as mqtt

BROKER = "<IP_SERVEUR>"    # <-- IP du serveur isiMonitor3D
ZONE   = "Sortie_1"        # <-- zone à surveiller (champ "zone" du payload)
TOPIC  = "isiMonitor3D/v1/+/zone/+"

def on_connect(client, userdata, flags, reason, properties=None):
    print("connecte:", reason)
    client.subscribe(TOPIC, qos=1)

def on_message(client, userdata, msg):
    etat = json.loads(msg.payload)
    if etat.get("type") != "zone_state" or etat.get("zone") != ZONE:
        return
    palettes  = [o for o in etat["objects"] if o["cls"] == "palette"]
    personnes = [o for o in etat["objects"] if o["cls"] == "person"]
    if personnes:
        print(f"[{ZONE}] ATTENTE - personne dans la zone")
    elif palettes:
        p = palettes[0]
        print(f"[{ZONE}] PALETTE presente  conf={p['confidence']:.2f}  "
              f"etat={p['occupancy_state']}  contenu={p['occupancy_content']}  "
              f"pos=({p['xy_m'][0]:.2f}, {p['xy_m'][1]:.2f}) m")
        # >>> logique AGV ici : lancer le pick-and-place <<<
    else:
        print(f"[{ZONE}] vide - rien a prendre")

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER, 1883)
client.loop_forever()
```

Sortie sur le système réel :

```
connecte: Success
[Sortie_1] PALETTE presente  conf=0.87  etat=full  contenu=carton  pos=(-1.00, -0.12) m
```

Règles de décision suggérées (seuils à convenir le jour du test) : pick quand
une `palette` est présente ; `occupancy_state` / `occupancy_content` choisissent
la mission ; attente tant qu'une `person` est dans la zone ; zone confirmée
libre quand `objects` redevient vide après le pick.

L'équivalent REST, si votre environnement préfère HTTP :

```python
import requests, time
while True:
    z = requests.get("http://<IP_SERVEUR>:8080/v1/zones/Sortie_1").json()
    for entry in z["zones"]:
        print(entry["objects"])
    time.sleep(0.5)          # 2 Hz suffisent largement
```

## 7. Check-list réseau

- Client AGV et serveur isiMonitor3D sur le même LAN (ou routé, sans NAT/proxy).
- Joignable depuis l'AGV : `<IP_SERVEUR>` **TCP 1883** (MQTT) et **TCP 8080**
  (REST). Aucun port entrant nécessaire côté AGV.
- Profil test LAN : en clair, sans identifiants. (Un profil sécurisé
  TLS + jeton existe pour une exposition hors LAN.)
- AGV en Wi-Fi : aucun souci — keepalive MQTT + reconnexion automatique gèrent
  les coupures d'itinérance, et le message retenu restaure l'état courant
  instantanément à la reconnexion.
- IP fixe ou réservation DHCP pour le serveur ; `<IP_SERVEUR>` est communiquée
  avant le test.
