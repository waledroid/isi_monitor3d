# ruff: noqa: RUF001 — en-dashes/middle-dots in UI copy are intentional
"""The /ui probe page as a module-level string — no templates, no static
files, no build step; the same dark glass design language as the isical /
isiGen studios (shared tokens: #07080c bg, glass panels, #4fc3f7 accent)."""

UI_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>isicomms</title>
<style>
:root{
  --bg:#07080c; --text:#fff; --muted:#94a3b8;
  --glass:rgba(18,22,33,.45); --border:rgba(255,255,255,.08);
  --accent:#4fc3f7; --ok:#2ed573; --bad:#ff4757; --warn:#f5ab35;
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
#tail{grid-column:1/-1}
#feed{max-height:340px;overflow-y:auto;font:12px/1.5 ui-monospace,Consolas,monospace}
.msg{padding:3px 6px;border-bottom:1px solid rgba(255,255,255,.04);cursor:pointer;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.msg:hover{background:rgba(255,255,255,.04)}
.msg .t{color:var(--muted)} .msg .topic{color:var(--accent)}
.msg pre{white-space:pre-wrap;word-break:break-all;color:var(--text);
  background:rgba(0,0,0,.35);border-radius:8px;padding:8px;margin-top:5px}
label.pause{font-size:12px;color:var(--muted);margin-left:auto;font-weight:400;
  text-transform:none;letter-spacing:0;cursor:pointer}
.empty{color:var(--muted);font-size:12px;padding:8px 0}
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
  <div class="card"><h2>Passings <span class="n" id="n-pass"></span></h2>
    <div id="passings"></div></div>
  <div class="card" id="tail"><h2>Raw MQTT tail
    <span class="n" id="n-tail"></span>
    <label class="pause"><input type="checkbox" id="pause"> pause</label></h2>
    <div id="feed"></div></div>
</div>
</div>

<script>
"use strict";
const $=id=>document.getElementById(id);
const tok=$("tok");
tok.value=localStorage.getItem("isicomms_token")||"";
tok.addEventListener("change",()=>localStorage.setItem("isicomms_token",tok.value));

async function j(path){ // never-raise fetch (comms_nodes.js pattern)
  try{
    const h=tok.value?{Authorization:"Bearer "+tok.value}:{};
    const r=await fetch(path,{headers:h});
    if(r.status===401){$("s-note").textContent="401 — token required";return null;}
    if(!r.ok)return null;
    $("s-note").textContent="";
    return await r.json();
  }catch(_e){return null;}
}
const esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const ago=ts=>{const d=Date.now()/1000-ts;return d<2?"now":d<60?d.toFixed(0)+"s":
  d<3600?(d/60).toFixed(0)+"m":(d/3600).toFixed(1)+"h";};
const hhmmss=ts=>new Date(ts*1000).toLocaleTimeString();

function table(rows,head){
  if(!rows.length)return '<div class="empty">— nothing yet —</div>';
  return "<table><tr>"+head.map(h=>"<th>"+h+"</th>").join("")+"</tr>"+
    rows.map(r=>"<tr>"+r.map(c=>"<td>"+c+"</td>").join("")+"</tr>").join("")+"</table>";
}

function renderNodes(d){
  if(!d)return;
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
  if(!d)return;
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
  if(!d)return;
  $("n-tracks").textContent=d.count;
  $("tracks").innerHTML=table((d.tracks||[]).slice(-14).map(t=>[
    "#"+t.track_id,esc(t.cls),
    (t.xy_m?t.xy_m:t.xyz_m||[]).map(v=>v.toFixed(2)).join(", ")+" m",
    (t.confidence!=null?(t.confidence*100).toFixed(0)+"%":"–"),
    '<span class="mut">'+ago(t.ts)+"</span>",
  ]),["id","cls","position","conf","seen"]);
}
function renderPassings(d){
  if(!d)return;
  $("n-pass").textContent=d.count;
  $("passings").innerHTML=table((d.passings||[]).slice(-12).reverse().map(p=>[
    '<span class="mut">'+hhmmss(p.ts)+"</span>",
    esc(p.zone),esc(p.cls)+" #"+p.track_id,
    p.direction==="enter"?'<span class="ok">enter</span>':'<span class="warn">leave</span>',
  ]),["time","zone","object","dir"]);
}
function renderTail(d){
  if(!d)return;
  $("s-rx").textContent=d.stats.received;
  $("s-drop").textContent=d.stats.dropped_malformed+d.stats.dropped_version;
  $("n-tail").textContent="last "+d.count;
  if($("pause").checked)return;
  const feed=$("feed");
  if(feed.querySelector("pre"))return; // a row is expanded — hold the feed
                                       // still until it is collapsed
  feed.innerHTML=(d.messages||[]).slice().reverse().map((m,i)=>
    '<div class="msg" data-i="'+i+'"><span class="t">'+hhmmss(m.ts)+
    '</span> <span class="topic">'+esc(m.topic)+"</span> "+
    esc(m.payload.slice(0,160))+"</div>").join("")||'<div class="empty">— no messages yet —</div>';
  const msgs=(d.messages||[]).slice().reverse();
  feed.querySelectorAll(".msg").forEach(el=>el.addEventListener("click",()=>{
    const pre=el.querySelector("pre");
    if(pre){pre.remove();return;}
    const m=msgs[+el.dataset.i];let body=m.payload;
    try{body=JSON.stringify(JSON.parse(m.payload),null,2);}catch(_e){}
    const p=document.createElement("pre");p.textContent=body;el.appendChild(p);
  }));
}

// ---- schema tree ------------------------------------------------------
// Built from the latest-per-topic map. Native <details> gives expand/
// collapse; open state and any expanded payload survive re-renders (the
// tree is rebuilt only when the TOPIC SET changes; counts/ages update in
// place every poll). Clicking a leaf toggles its latest payload inline.
let treeTopics="";          // signature of the current topic set
let openPaths=new Set(["isiMonitor3D"]);   // default: base expanded
let latestByTopic={};

function buildTree(topics){
  const root={};
  Object.keys(topics).sort().forEach(t=>{
    let n=root;
    t.split("/").forEach(seg=>{n=n.children=n.children||{};n=n[seg]=n[seg]||{};});
    n.topic=t;
  });
  const render=(nodes,path)=>Object.keys(nodes).map(seg=>{
    const node=nodes[seg],p=path?path+"/"+seg:seg;
    if(node.topic&&!node.children)
      return '<div class="leaf" data-topic="'+esc(node.topic)+'">'+esc(seg)+
             '<span class="cnt" data-cnt="'+esc(node.topic)+'"></span>'+
             '<span class="age" data-age="'+esc(node.topic)+'"></span></div>';
    const inner=(node.children?render(node.children,p):"")+
      (node.topic?'<div class="leaf" data-topic="'+esc(node.topic)+'">(this level)'+
        '<span class="cnt" data-cnt="'+esc(node.topic)+'"></span></div>':"");
    return '<details data-path="'+esc(p)+'"'+(openPaths.has(p)?" open":"")+
           "><summary>"+esc(seg)+"</summary>"+inner+"</details>";
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
  const sig=names.join("|");
  if(sig!==treeTopics){treeTopics=sig;buildTree(topics);}
  names.forEach(t=>{
    const c=document.querySelector('[data-cnt="'+CSS.escape(t)+'"]');
    if(c)c.textContent="×"+topics[t].count;
    const a=document.querySelector('[data-age="'+CSS.escape(t)+'"]');
    if(a)a.textContent=ago(topics[t].ts);
  });
}

async function tick(){
  const [h,nodes,zones,tracks,pass,tail]=await Promise.all([
    j("/healthz"),j("/nodes"),j("/zones"),j("/tracks"),
    j("/passings?limit=20"),j("/recent?limit=80")]);
  const alive=!!(nodes&&(nodes.nodes||[]).some(n=>n.status==="alive"));
  $("health").className="dot"+(h&&h.ok?(alive?" ok":" warn"):"");
  renderNodes(nodes);renderZones(zones);renderTracks(tracks);
  renderPassings(pass);renderTail(tail);
  if(tail&&tail.topics)renderTree(tail.topics);
}
tick();
setInterval(tick,2000);
</script>
</body>
</html>
"""
