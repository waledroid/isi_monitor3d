# ruff: noqa: RUF001 — en-dashes/middle-dots in UI copy are intentional
"""The /ui probe page as a module-level string — no templates, no static
files, no build step; the same dark glass design language as the isical /
isiGen studios (shared tokens: #07080c bg, glass panels, #4fc3f7 accent).

Three parts: the schema tree (left), the live cards (nodes / zones / tracks /
consumers — 2 s poll), and the AGV system-test cards merged in from the
former ``/test`` console (on-demand RUN checks; each shows the live answer
plus the exact REST endpoint / MQTT topic / mosquitto_sub line to copy)."""

UI_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>isicomms</title>
<style>
:root{
  --bg:#07080c; --text:#fff; --muted:#94a3b8; --muted2:#5c6470;
  --glass:rgba(18,22,33,.45); --border:rgba(255,255,255,.08);
  --accent:#4fc3f7; --ok:#2ed573; --bad:#ff4757; --warn:#f5ab35;
  --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  --shadow:0 8px 32px 0 rgba(0,0,0,.4), inset 0 1px 1px 0 rgba(255,255,255,.08);
}
*{box-sizing:border-box;margin:0}
html{scrollbar-color:rgba(255,255,255,.15) transparent}
body{background:var(--bg);color:var(--text);
  font:14px/1.45 "Segoe UI",system-ui,sans-serif;padding:24px 56px;min-height:100vh;
  max-width:1900px;margin:0 auto;
  background-image:radial-gradient(ellipse at 20% -10%,rgba(79,195,247,.07),transparent 55%),
                   radial-gradient(ellipse at 90% 110%,rgba(46,213,115,.05),transparent 50%);
  background-attachment:fixed}
header{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:24px}
h1{font-size:17px;font-weight:600;letter-spacing:.4px}
h1 b{color:var(--accent)}
.dot{width:11px;height:11px;border-radius:50%;background:var(--bad);
  display:inline-block;margin-left:4px;flex:0 0 auto}
.dot.ok{background:var(--ok);box-shadow:0 0 8px var(--ok)}
.dot.warn{background:var(--warn);box-shadow:0 0 8px var(--warn)}
.stats{color:var(--muted);font-size:12px}
.stats b{color:var(--text)}
#tok{margin-left:auto;background:var(--glass);border:1px solid var(--border);
  color:var(--text);border-radius:8px;padding:6px 10px;font-size:12px;width:210px}
#tok::placeholder{color:var(--muted)}
.layout{display:grid;grid-template-columns:300px 1fr;gap:26px;align-items:start}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:26px}
@media(max-width:1250px){.layout{grid-template-columns:1fr}.grid{grid-template-columns:1fr}}
/* schema tree (native <details> nesting = free expand/collapse) */
#tree{font:12.5px/1.7 ui-monospace,Consolas,monospace;max-height:78vh;overflow:auto}
#tree details{padding-left:14px}
#tree>details{padding-left:0}
#tree summary{cursor:pointer;color:var(--text);white-space:nowrap;list-style:none}
#tree summary::before{content:"▸";color:var(--muted);display:inline-block;
  width:13px;transition:transform .12s}
#tree details[open]>summary::before{transform:rotate(90deg)}
#tree .leaf{cursor:pointer;white-space:nowrap;padding-left:14px;color:var(--accent)}
#tree .leaf:hover,#tree summary:hover{background:rgba(255,255,255,.05);border-radius:6px}
#tree .cnt{color:var(--muted);font-size:11px;margin-left:6px}
#tree .age{color:var(--muted);font-size:11px;margin-left:4px}
#tree .zname{color:var(--muted);font-size:11px;font-style:italic;margin-left:4px}
#tree pre{white-space:pre-wrap;word-break:break-all;background:rgba(0,0,0,.35);
  border-radius:8px;padding:8px;margin:4px 0 6px;color:var(--text);font-size:11.5px}
.card{background:var(--glass);border:1px solid var(--border);border-radius:14px;
  box-shadow:var(--shadow);padding:16px 18px;backdrop-filter:blur(10px)}
.card h2{font-size:11px;text-transform:uppercase;letter-spacing:.12em;
  color:var(--accent);margin-bottom:8px;display:flex;align-items:center;gap:8px}
.card h2 .n{color:var(--muted);font-weight:400;letter-spacing:0;text-transform:none}
table{width:100%;border-collapse:collapse;font-size:12.5px}
th{color:var(--muted);text-align:left;font-weight:500;padding:3px 8px 5px 0;
  border-bottom:1px solid var(--border);white-space:nowrap}
td{padding:4px 8px 4px 0;border-bottom:1px solid rgba(255,255,255,.04);
  vertical-align:top;font-variant-numeric:tabular-nums}
.mut{color:var(--muted)} .ok{color:var(--ok)} .bad{color:var(--bad)} .warn{color:var(--warn)}
.tag{display:inline-block;border:1px solid var(--border);border-radius:6px;
  padding:0 6px;font-size:11px;color:var(--muted);margin-right:4px}
.empty{color:var(--muted);font-size:12px;padding:8px 0}
/* ---- AGV test section (merged /test console) ---- */
#testsec{grid-column:1/-1}
.testhead{display:flex;align-items:center;gap:12px;margin:2px 0 12px}
.testhead h2{font-size:11px;text-transform:uppercase;letter-spacing:.12em;
  color:var(--accent)}
.testhead .n{color:var(--muted);font-size:12px}
.runall{margin-left:auto;background:var(--accent);color:#04121b;border:0;
  border-radius:9px;padding:7px 15px;font-size:12.5px;font-weight:650;cursor:pointer;
  transition:transform .12s ease,filter .15s ease}
.runall:hover{filter:brightness(1.08)}
.runall:active{transform:scale(.96)}
.testgrid{display:grid;grid-template-columns:1fr 1fr;gap:26px}
@media(max-width:1250px){.testgrid{grid-template-columns:1fr}}
.tcard h2{margin-bottom:0}
.tcard h2 .sp{flex:1}
.run{background:transparent;border:1px solid var(--accent);color:var(--accent);
  border-radius:8px;padding:4px 13px;font-size:12px;font-weight:650;cursor:pointer;
  transition:background .15s ease,color .15s ease,transform .12s ease;white-space:nowrap}
.run:hover{background:var(--accent);color:#04121b}
.run:active{transform:scale(.94)}
.spin{width:13px;height:13px;border:2px solid var(--border);border-top-color:var(--accent);
  border-radius:50%;display:none;animation:rot .7s linear infinite}
@keyframes rot{to{transform:rotate(360deg)}}
.busy .spin{display:inline-block}
.busy .run{opacity:.4;pointer-events:none}
.result{display:none;margin-top:12px}
.result.show{display:block;animation:reveal .26s ease-out}
@keyframes reveal{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
@media(prefers-reduced-motion:reduce){.result.show{animation:none}.spin{animation-duration:1.4s}}
/* status chips — icon + label, never color alone */
.chip{display:inline-flex;align-items:center;gap:7px;border-radius:999px;
  padding:3px 12px 3px 9px;font-size:12px;font-weight:650;letter-spacing:.2px}
.chip .ic{font-size:11px}
.chip.good{background:rgba(46,213,115,.14);color:var(--ok)}
.chip.warn{background:rgba(245,171,53,.14);color:var(--warn)}
.chip.bad{background:rgba(255,71,87,.14);color:var(--bad)}
.chip.neutral{background:rgba(148,163,184,.14);color:var(--muted)}
.summary{margin:8px 0 4px;font-size:13px}
.zrow{display:flex;align-items:center;gap:10px;padding:7px 10px;border-radius:9px;
  background:rgba(255,255,255,.03);margin-top:8px;animation:reveal .26s ease-out both}
.zrow:nth-child(2){animation-delay:.04s}.zrow:nth-child(3){animation-delay:.08s}
.zrow:nth-child(4){animation-delay:.12s}
.zrow .zname{font-weight:650;font-size:13px;min-width:90px}
.zrow .zinfo{color:var(--muted);font-size:12px}
.endpoints{margin-top:12px;border-top:1px solid var(--border);padding-top:10px;
  display:grid;gap:6px}
.ep{display:flex;align-items:center;gap:8px;font-size:12px;min-width:0}
.ep .etag{color:var(--muted2);min-width:48px;font-weight:650;letter-spacing:.4px;
  font-size:10.5px}
.ep code{font-family:var(--mono);font-size:11.5px;color:var(--muted);
  background:rgba(0,0,0,.28);border-radius:6px;padding:3px 8px;overflow-x:auto;
  white-space:nowrap;flex:1;min-width:0}
.ep code a{color:var(--accent);text-decoration:none}
.ep code::-webkit-scrollbar{height:4px}
.ep code::-webkit-scrollbar-thumb{background:rgba(255,255,255,.14);border-radius:2px}
.ep code::-webkit-scrollbar-track{background:transparent}
.copy{background:none;border:0;color:var(--muted2);cursor:pointer;font-size:10.5px;
  font-weight:650;letter-spacing:.3px;padding:2px 5px;border-radius:5px;
  transition:color .15s ease,transform .1s ease}
.copy:hover{color:var(--accent)}
.copy:active{transform:scale(.85)}
.copy.okc{color:var(--ok)}
.tcard details{margin-top:10px}
.tcard details summary{color:var(--muted2);font-size:12px;cursor:pointer;user-select:none}
.tcard details pre{font-family:var(--mono);font-size:11px;line-height:1.45;
  color:var(--muted);background:rgba(0,0,0,.3);border-radius:9px;padding:10px 12px;
  overflow:auto;max-height:260px}
.testfoot{color:var(--muted2);font-size:12px;margin-top:14px}
.testfoot code{font-family:var(--mono)}
</style>
</head>
<body>
<header>
  <h1><b>isicomms</b></h1>
  <span class="stats">received <b id="s-rx">–</b> ·
    dropped <b id="s-drop">–</b> · <span id="s-note" class="mut"></span></span>
  <input id="tok" type="password" placeholder="API token (if required)"
         autocomplete="off">
  <span id="health" class="dot"
        title="green: backbone data flowing · amber: connected, nothing coming · red: gateway down"></span>
</header>

<div class="layout">
<div class="card" id="side"><h2>Schema tree <span class="n" id="n-tree"></span></h2>
  <div id="tree"><div class="empty">— no topics yet —</div></div></div>

<div class="grid">
  <div class="card"><h2>Nodes <span class="n" id="n-nodes"></span></h2>
    <div id="nodes"></div></div>
  <div class="card"><h2>Zones <span class="n" id="n-zones"></span></h2>
    <div id="zones"></div></div>
  <div class="card"><h2>Tracks <span class="n" id="n-tracks"></span></h2>
    <div id="tracks"></div></div>
  <div class="card"><h2>Consumers <span class="n" id="n-cons"></span></h2>
    <div id="consumers"></div></div>

  <div id="testsec">
    <div class="testhead">
      <h2>AGV system test</h2>
      <span class="n">each card answers one state the pick-and-place test needs —
        RUN shows the live answer + the exact REST / MQTT address to use</span>
      <button class="runall" id="runall">Run all checks</button>
    </div>
    <div class="testgrid" id="testgrid"></div>
    <div class="testfoot">MQTT broker: <code id="brokerline"></code> — plain TCP,
      no auth (test profile). Subscribe at QoS 1; <code>zone/*</code> and
      <code>config</code> are retained, the current state arrives immediately on
      connect. Full instructions: <em>AGV System Test — Minimal Integration
      Guide</em>.</div>
  </div>
</div>
</div>

<script>
"use strict";
const $=id=>document.getElementById(id);
const tok=$("tok");
tok.value=localStorage.getItem("isicomms_token")||"";
tok.addEventListener("change",()=>localStorage.setItem("isicomms_token",tok.value));
const host=location.hostname||"SERVER_IP";
$("brokerline").textContent=host+":1883";

async function j(path){ // never-raise fetch: parsed JSON, or {__err} on failure
  try{
    const h={"X-Client-Name":"ui-probe"};
    if(tok.value)h.Authorization="Bearer "+tok.value;
    const r=await fetch(path,{headers:h});
    if(r.status===401){$("s-note").textContent="401 — token required";return{__err:401};}
    if(!r.ok)return{__err:r.status};
    $("s-note").textContent="";
    return await r.json();
  }catch(e){return{__err:String(e)};}
}
const bad=d=>!d||d.__err!==undefined;
const esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const ago=ts=>{const d=Date.now()/1000-ts;return d<2?"now":d<60?d.toFixed(0)+"s":
  d<3600?(d/60).toFixed(0)+"m":(d/3600).toFixed(1)+"h";};

function table(rows,head){
  if(!rows.length)return '<div class="empty">— nothing yet —</div>';
  return "<table><tr>"+head.map(h=>"<th>"+h+"</th>").join("")+"</tr>"+
    rows.map(r=>"<tr>"+r.map(c=>"<td>"+c+"</td>").join("")+"</tr>").join("")+"</table>";
}

function renderNodes(d){
  if(bad(d))return;
  $("n-nodes").textContent=d.count;
  $("nodes").innerHTML=table((d.nodes||[]).map(n=>[
    "<b>"+esc(n.node_id)+"</b>"+(n.area?' <span class="tag">'+esc(n.area)+"</span>":""),
    n.status==="alive"?'<span class="ok">alive</span>':'<span class="warn">'+esc(n.status)+"</span>",
    esc(n.mode||"–"),
    (n.fps!=null?n.fps.toFixed(1):"–"),
    (n.latency_ms!=null?n.latency_ms.toFixed(0)+" ms":"–"),
    (n.cameras||[]).map(c=>'<span class="tag">'+esc(c)+"</span>").join(""),
    '<span class="mut">'+ago(n.last_seen)+"</span>",
  ]),["node","status","mode","fps","p95","cameras","seen"]);
}
function renderZones(d){
  if(bad(d))return;
  $("n-zones").textContent=d.count;
  $("zones").innerHTML=table((d.zones||[]).map(z=>[
    "<b>"+esc(z.name)+"</b>",
    '<span class="mut">'+esc(z.node_id)+"</span>",
    z.count==null?'<span class="mut">–</span>':String(z.count),
    (z.objects||[]).map(o=>'<span class="tag">'+esc(o.cls)+
      (o.occupancy_state?" · "+esc(o.occupancy_state):"")+"</span>").join("")||
      '<span class="mut">empty</span>',
    z.state_ts?'<span class="mut">'+ago(z.state_ts)+"</span>":'<span class="mut">–</span>',
  ]),["zone","node","count","objects","state"]);
}
function renderTracks(d){
  if(bad(d))return;
  $("n-tracks").textContent=d.count;
  $("tracks").innerHTML=table((d.tracks||[]).slice(-14).map(t=>[
    "#"+t.track_id,esc(t.cls),
    t.xyz_m?'<span class="tag" style="color:var(--accent);border-color:var(--accent)">3D</span>'+
      (t.single_view?'<span class="mut" title="single-view fallback — Z pinned to 0">sv</span>':"")
      :'<span class="mut">2D</span>',
    (t.xy_m?t.xy_m:t.xyz_m||[]).map(v=>v.toFixed(2)).join(", ")+" m",
    (t.confidence!=null?(t.confidence*100).toFixed(0)+"%":"–"),
    '<span class="mut">'+ago(t.ts)+"</span>",
  ]),["id","cls","dim","position","conf","seen"]);
}
function renderConsumers(d){
  if(bad(d))return;
  const cl=d.api_clients||[];
  const act=cl.filter(c=>c.active).length;
  $("n-cons").textContent=act+" active · MQTT "+
    (d.mqtt_connected!=null?d.mqtt_connected:"–");
  $("consumers").innerHTML=
    '<div class="mut" style="font-size:12px;margin-bottom:6px">MQTT clients connected: <b style="color:var(--text)">'+
    (d.mqtt_connected!=null?d.mqtt_connected:"–")+
    "</b> <span>(broker total — gateway + nodes + AGVs; identities not exposed)</span></div>"+
    table(cl.slice(0,10).map(c=>[
      c.name?"<b>"+esc(c.name)+'</b> <span class="tag">'+esc(c.ip)+"</span>"
            :"<b>"+esc(c.ip)+"</b>",
      c.active?'<span class="ok">active</span>':'<span class="mut">idle</span>',
      String(c.requests),
      '<span class="mut">'+ago(c.last_seen)+"</span>",
    ]),["client (X-Client-Name or IP)","state","reqs","seen"]);
}
function renderStats(d){
  if(bad(d))return;
  $("s-rx").textContent=d.stats.received;
  $("s-drop").textContent=d.stats.dropped_malformed+d.stats.dropped_version;
}

// ---- schema tree ------------------------------------------------------
// Built from the latest-per-topic map. Native <details> gives expand/
// collapse; open state and any expanded payload survive re-renders (the
// tree is rebuilt only when the TOPIC SET changes; counts/ages update in
// place every poll). Clicking a leaf toggles its latest payload inline.
let treeTopics="";          // signature of the current topic set
let openPaths=new Set(["isiMonitor3D"]);   // default: base expanded
let latestByTopic={};
let zoneNames={};           // zone_id → display name, from /zones (config adverts)

function buildTree(topics){
  const root={};
  Object.keys(topics).sort().forEach(t=>{
    let n=root;
    t.split("/").forEach(seg=>{n=n.children=n.children||{};n=n[seg]=n[seg]||{};});
    n.topic=t;
  });
  // Zone-id topic segments get their display name appended ("zp_x — “Sortie_1”"),
  // resolved from the /zones pairing (config adverts) — topics stay id-keyed.
  const label=seg=>esc(seg)+(zoneNames[seg]
    ?'<span class="zname">— “'+esc(zoneNames[seg])+'”</span>':"");
  const render=(nodes,path)=>Object.keys(nodes).map(seg=>{
    const node=nodes[seg],p=path?path+"/"+seg:seg;
    if(node.topic&&!node.children)
      return '<div class="leaf" data-topic="'+esc(node.topic)+'">'+label(seg)+
             '<span class="cnt" data-cnt="'+esc(node.topic)+'"></span>'+
             '<span class="age" data-age="'+esc(node.topic)+'"></span></div>';
    const inner=(node.children?render(node.children,p):"")+
      (node.topic?'<div class="leaf" data-topic="'+esc(node.topic)+'">(this level)'+
        '<span class="cnt" data-cnt="'+esc(node.topic)+'"></span></div>':"");
    return '<details data-path="'+esc(p)+'"'+(openPaths.has(p)?" open":"")+
           "><summary>"+label(seg)+"</summary>"+inner+"</details>";
  }).join("");
  $("tree").innerHTML=render(root.children||{},"")||'<div class="empty">— no topics yet —</div>';
  $("tree").querySelectorAll("details").forEach(d=>d.addEventListener("toggle",()=>{
    d.open?openPaths.add(d.dataset.path):openPaths.delete(d.dataset.path);}));
  $("tree").querySelectorAll(".leaf").forEach(el=>el.addEventListener("click",()=>{
    const pre=el.nextElementSibling&&el.nextElementSibling.tagName==="PRE"
      ?el.nextElementSibling:null;
    if(pre){pre.remove();return;}
    const m=latestByTopic[el.dataset.topic];if(!m)return;
    let body=m.payload;try{body=JSON.stringify(JSON.parse(m.payload),null,2);}catch(_e){}
    const p=document.createElement("pre");p.textContent=body;
    el.after(p);}));
}
function renderTree(topics){
  latestByTopic=topics;
  const names=Object.keys(topics).sort();
  $("n-tree").textContent=names.length+" topics";
  // Rebuild when topics OR zone display names change (renames re-annotate live).
  const sig=names.join("|")+"§"+
    Object.entries(zoneNames).sort().map(e=>e.join(":")).join(",");
  if(sig!==treeTopics){treeTopics=sig;buildTree(topics);}
  names.forEach(t=>{
    const c=document.querySelector('[data-cnt="'+CSS.escape(t)+'"]');
    if(c)c.textContent="×"+topics[t].count;
    const a=document.querySelector('[data-age="'+CSS.escape(t)+'"]');
    if(a)a.textContent=ago(topics[t].ts);
  });
}

// ---- AGV test cards (merged /test console) ----------------------------
const chip=(kind,ic,label)=>'<span class="chip '+kind+'"><span class="ic">'+ic+
  "</span>"+esc(label)+"</span>";
const CH={good:["good","●"],warn:["warn","▲"],bad:["bad","✕"],neutral:["neutral","○"]};
const st=(k,label)=>chip(CH[k][0],CH[k][1],label);

const CHECKS=[
 {id:"gw",title:"Gateway",q:"Is the REST gateway reachable?",
  rest:"/healthz",topic:null,
  interpret:d=>d&&d.ok?[st("good","REACHABLE"),"Gateway answering at "+location.host+"."]
                      :[st("bad","UNREACHABLE"),"No answer from the gateway."]},
 {id:"node",title:"Node",q:"Is the vision node alive?",
  rest:"/v1/nodes",topic:"isiMonitor3D/v1/+/diagnostics/heartbeat",
  interpret:d=>{
    const n=(d.nodes||[]);
    if(!n.length)return[st("bad","NO NODE"),"No Backbone node discovered yet."];
    return[n.every(x=>x.status==="alive")?st("good","ALIVE"):st("warn","STALE"),
      n.map(x=>"<b>"+esc(x.node_id)+"</b> — "+esc(x.status)+", "+
        (x.cameras||[]).length+" cam, "+(x.fps?x.fps.toFixed(1):"?")+" fps, "+
        (x.latency_ms?Math.round(x.latency_ms):"?")+" ms").join("<br>")];}},
 {id:"cfg",title:"Zones configured",q:"Which zones does the node advertise?",
  rest:"/v1/config",topic:"isiMonitor3D/v1/+/config",retained:true,
  interpret:d=>{
    const rows=[];
    for(const c of (d.nodes||[])){
      const zs=((c.config&&c.config.zones)||[]).map(z=>z.name);
      rows.push("<b>"+esc(c.node_id||"?")+"</b> — "+
        (zs.length?zs.map(esc).join(", "):"no zones"));
    }
    if(!rows.length)return[st("neutral","NONE"),"No retained config received."];
    return[st("good","ADVERTISED"),rows.join("<br>")];}},
 {id:"zones",title:"Zone contents",q:"Palette in the zone — and loaded with what?",
  rest:"/v1/zones",topic:"isiMonitor3D/v1/+/zone/+",retained:true,
  interpret:d=>{
    const zs=d.zones||[];
    if(!zs.length)return[st("neutral","NO ZONES"),"No zone state on the broker."];
    const rows=zs.map(z=>{
      const obj=z.objects||[];
      const pal=obj.filter(o=>o.cls==="palette");
      const per=obj.filter(o=>o.cls==="person");
      let c,info;
      if(per.length){c=st("warn","HOLD");info="person in zone";}
      else if(pal.length){
        const p=pal[0];
        c=st("good","PALETTE");
        info=esc(p.occupancy_state||"?")+
             (p.occupancy_content?" · "+esc(p.occupancy_content):"")+
             " · conf "+(p.confidence??0).toFixed(2);}
      else{c=st("neutral","EMPTY");info="nothing to pick";}
      return '<div class="zrow"><span class="zname">'+esc(z.name)+"</span>"+c+
             '<span class="zinfo">'+info+"</span></div>";});
    return[null,rows.join("")];}},
 {id:"tracks",title:"Live tracks",q:"What is being tracked right now?",
  rest:"/v1/tracks",topic:"isiMonitor3D/v1/+/track2d/+",
  interpret:d=>{
    const t=d.tracks||[];
    if(!t.length)return[st("neutral","QUIET"),"No active tracks."];
    const by={};t.forEach(x=>by[x.cls]=(by[x.cls]||0)+1);
    return[st("good",t.length+" TRACKED"),
      Object.entries(by).map(([k,v])=>v+"× "+esc(k)).join(" · ")];}},
 {id:"tracks3d",title:"3D localization",q:"Which objects have live XYZ positions?",
  rest:"/v1/tracks?type=track_3d",topic:"isiMonitor3D/v1/+/track3d/+",
  interpret:d=>{
    const t=d.tracks||[];
    if(!t.length)return[st("neutral","QUIET"),
      "No 3D tracks — needs both cameras seeing a subscribed class."];
    return[st("good",t.length+" LOCALIZED"),
      t.slice(-6).map(x=>"#"+x.track_id+" <b>"+esc(x.cls)+"</b> ["+
        (x.xyz_m||[]).map(v=>v.toFixed(2)).join(", ")+"] m"+
        (x.single_view?' <span class="mut">(single-view)</span>':"")).join("<br>")];}},
 {id:"pass",title:"Passings",q:"Recent zone entries / exits?",
  rest:"/v1/passings",topic:"isiMonitor3D/v1/+/zone/+/passings",
  interpret:d=>{
    const p=(d.passings||[]).slice(-5).reverse();
    if(!p.length)return[st("neutral","NONE"),"No boundary crossings recorded."];
    return[st("good",p.length+" RECENT"),
      p.map(x=>(x.direction==="enter"?"⟶":"⟵")+" <b>"+esc(x.cls)+"</b> "+
        (x.direction==="enter"?"entered":"left")+" <b>"+esc(x.zone)+"</b>").join("<br>")];}},
];

function epRow(tag,html,copyText){
  return '<div class="ep"><span class="etag">'+tag+"</span><code>"+html+"</code>"+
    '<button class="copy" data-copy="'+esc(copyText)+'" title="copy">copy</button></div>';
}
function cardHtml(c){
  const restUrl=location.origin+c.rest;
  let eps=epRow("REST",'GET <a href="'+c.rest+'" target="_blank">'+esc(restUrl)+"</a>",restUrl);
  if(c.topic){
    eps+=epRow("MQTT",esc(c.topic)+(c.retained?' <span style="color:var(--muted2)">(retained)</span>':""),c.topic);
    eps+=epRow("SHELL",esc("mosquitto_sub -h "+host+" -t '"+c.topic+"' -v"),
               "mosquitto_sub -h "+host+" -t '"+c.topic+"' -v");
  }
  return '<div class="card tcard" id="card-'+c.id+'">'+
    "<h2>"+esc(c.title)+' <span class="n">'+esc(c.q)+"</span>"+
    '<span class="sp"></span><span class="spin"></span>'+
    '<button class="run" data-id="'+c.id+'">RUN</button></h2>'+
    '<div class="result" id="res-'+c.id+'">'+
    '<div id="chip-'+c.id+'"></div>'+
    '<div class="summary" id="sum-'+c.id+'"></div>'+
    '<div class="endpoints">'+eps+"</div>"+
    '<details><summary>raw JSON</summary><pre id="raw-'+c.id+'">—</pre></details>'+
    "</div></div>";
}
$("testgrid").innerHTML=CHECKS.map(cardHtml).join("");

async function run(id){
  const c=CHECKS.find(x=>x.id===id),card=$("card-"+id);
  card.classList.add("busy");
  const d=await j(c.rest);
  card.classList.remove("busy");
  const res=$("res-"+id);res.classList.remove("show");void res.offsetWidth;
  let chipHtml,sum;
  if(bad(d)){chipHtml=st("bad","ERROR");sum="Request failed: "+esc(d?d.__err:"?");}
  else{[chipHtml,sum]=c.interpret(d||{});}
  $("chip-"+id).innerHTML=chipHtml||"";
  $("sum-"+id).innerHTML=sum;
  $("raw-"+id).textContent=(d&&!bad(d))?JSON.stringify(d,null,2).slice(0,20000):"—";
  res.classList.add("show");
}
document.addEventListener("click",e=>{
  const r=e.target.closest(".run");if(r)return run(r.dataset.id);
  const cp=e.target.closest(".copy");
  if(cp){navigator.clipboard.writeText(cp.dataset.copy).then(()=>{
    cp.classList.add("okc");cp.textContent="copied";
    setTimeout(()=>{cp.classList.remove("okc");cp.textContent="copy";},1200);});}
});
async function runAll(){for(const c of CHECKS)await run(c.id);}
$("runall").addEventListener("click",runAll);

// ---- live poll --------------------------------------------------------
async function tick(){
  const [h,nodes,zones,tracks,cons,tail]=await Promise.all([
    j("/healthz"),j("/nodes"),j("/zones"),j("/tracks"),
    j("/clients"),j("/recent?limit=80")]);
  const alive=!bad(nodes)&&(nodes.nodes||[]).some(n=>n.status==="alive");
  $("health").className="dot"+(!bad(h)&&h.ok?(alive?" ok":" warn"):"");
  const zn={};(bad(zones)?[]:(zones.zones||[])).forEach(z=>{if(z.zone_id)zn[z.zone_id]=z.name;});
  zoneNames=zn;
  renderNodes(nodes);renderZones(zones);renderTracks(tracks);
  renderConsumers(cons);renderStats(tail);
  if(!bad(tail)&&tail.topics)renderTree(tail.topics);
}
tick();
setInterval(tick,2000);
runAll();   // the test cards auto-run on load (/test?run=all lands here too)
</script>
</body>
</html>
"""
