#!/usr/bin/env python3
"""Minimal isiMonitor3D client - AGV pick trigger (system test)."""
import json
import paho.mqtt.client as mqtt

BROKER = "<SERVER_IP>"     # <-- isiMonitor3D server IP
ZONE   = "Sortie_1"        # <-- zone to watch (payload field "zone")
TOPIC  = "isiMonitor3D/v1/+/zone/+"

def on_connect(client, userdata, flags, reason, properties=None):
    print("connected:", reason)
    client.subscribe(TOPIC, qos=1)

def on_message(client, userdata, msg):
    state = json.loads(msg.payload)
    if state.get("type") != "zone_state" or state.get("zone") != ZONE:
        return
    palettes = [o for o in state["objects"] if o["cls"] == "palette"]
    persons  = [o for o in state["objects"] if o["cls"] == "person"]
    if persons:
        print(f"[{ZONE}] HOLD - person in zone")
    elif palettes:
        p = palettes[0]
        print(f"[{ZONE}] PALETTE present  conf={p['confidence']:.2f}  "
              f"state={p['occupancy_state']}  content={p['occupancy_content']}  "
              f"pos=({p['xy_m'][0]:.2f}, {p['xy_m'][1]:.2f}) m")
        # >>> AGV logic here: launch pick-and-place <<<
    else:
        print(f"[{ZONE}] empty - nothing to pick")

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER, 1883)
client.loop_forever()
