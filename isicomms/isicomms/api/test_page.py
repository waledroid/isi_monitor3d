"""The AGV system-test console: ``GET /test``.

A single self-contained HTML page (inline CSS/JS, no static mount, no build
step — wheel-safe like ``/ui``) for the joint functional test with the AGV
team. One card per *state the test needs* (gateway up, node alive, zones
configured, zone contents, live tracks, passings). Clicking a card's RUN
button shows the state **and** the two ways to obtain it — the REST endpoint
URL and the MQTT topic — so the AGV integrators can lift the exact address
into their own client.

The page shell carries no data and is served token-free; its JavaScript sends
``Authorization: Bearer`` from the shared ``isicomms_token`` localStorage box
when the deployment requires one (same pattern as ``/ui``).
"""

from __future__ import annotations

TEST_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>isiMonitor3D — AGV System Test</title>
<style>
:root{
  --bg:#07080c; --panel:rgba(255,255,255,.045); --panel-edge:rgba(255,255,255,.09);
  --ink:#e8ecf1; --ink-2:#9aa4b2; --ink-3:#5c6470;
  --accent:#4fc3f7;
  --good:#37c978; --warn:#e8b93c; --bad:#ef6363; --neutral:#8b94a1;
  --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}
*{box-sizing:border-box}
html,body{margin:0;padding:0;background:var(--bg);color:var(--ink);
  font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1060px;margin:0 auto;padding:28px 20px 60px}

header{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;margin-bottom:6px}
h1{font-size:21px;font-weight:650;margin:0;letter-spacing:.2px}
h1 .dim{color:var(--ink-3);font-weight:500}
.sub{color:var(--ink-2);font-size:13.5px;margin:0 0 22px}
.toolbar{display:flex;gap:10px;align-items:center;margin-left:auto}
#tok{background:var(--panel);border:1px solid var(--panel-edge);color:var(--ink);
  border-radius:8px;padding:6px 10px;font-size:12.5px;width:170px}
#tok::placeholder{color:var(--ink-3)}

.runall{background:var(--accent);color:#04121b;border:0;border-radius:9px;
  padding:8px 16px;font-size:13.5px;font-weight:650;cursor:pointer;
  transition:transform .12s ease,filter .15s ease}
.runall:hover{filter:brightness(1.08)}
.runall:active{transform:scale(.96)}

.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(480px,1fr));gap:14px}
@media(max-width:560px){.grid{grid-template-columns:1fr}}

.card{background:var(--panel);border:1px solid var(--panel-edge);border-radius:14px;
  padding:16px 18px;transition:border-color .2s ease,transform .2s ease}
.card:hover{border-color:rgba(255,255,255,.16)}
.card h2{font-size:15px;font-weight:650;margin:0;display:flex;align-items:center;gap:10px}
.card h2 .q{color:var(--ink-2);font-weight:450;font-size:13px}
.card-head{display:flex;align-items:center;gap:12px}
.card-head .spacer{flex:1}

.run{background:transparent;border:1px solid var(--accent);color:var(--accent);
  border-radius:8px;padding:5px 14px;font-size:12.5px;font-weight:650;cursor:pointer;
  transition:background .15s ease,color .15s ease,transform .12s ease;white-space:nowrap}
.run:hover{background:var(--accent);color:#04121b}
.run:active{transform:scale(.94)}

/* spinner */
.spin{width:14px;height:14px;border:2px solid var(--panel-edge);border-top-color:var(--accent);
  border-radius:50%;display:none;animation:rot .7s linear infinite}
@keyframes rot{to{transform:rotate(360deg)}}
.busy .spin{display:inline-block}
.busy .run{opacity:.4;pointer-events:none}

/* result reveal */
.result{display:none;margin-top:14px}
.result.show{display:block;animation:reveal .26s ease-out}
@keyframes reveal{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
@media(prefers-reduced-motion:reduce){
  .result.show{animation:none}.spin{animation-duration:1.4s}}

/* status chips — icon + label, never color alone */
.chip{display:inline-flex;align-items:center;gap:7px;border-radius:999px;
  padding:3px 12px 3px 9px;font-size:12.5px;font-weight:650;letter-spacing:.2px}
.chip .ic{font-size:12px}
.chip.good{background:rgba(55,201,120,.14);color:var(--good)}
.chip.warn{background:rgba(232,185,60,.14);color:var(--warn)}
.chip.bad{background:rgba(239,99,99,.14);color:var(--bad)}
.chip.neutral{background:rgba(139,148,161,.14);color:var(--neutral)}

.summary{margin:10px 0 4px;font-size:14px;color:var(--ink)}
.summary .muted{color:var(--ink-2)}

/* per-zone rows */
.zrow{display:flex;align-items:center;gap:10px;padding:8px 10px;border-radius:9px;
  background:rgba(255,255,255,.03);margin-top:8px;animation:reveal .26s ease-out both}
.zrow:nth-child(2){animation-delay:.04s}.zrow:nth-child(3){animation-delay:.08s}
.zrow:nth-child(4){animation-delay:.12s}
.zrow .zname{font-weight:650;font-size:13.5px;min-width:90px}
.zrow .zinfo{color:var(--ink-2);font-size:12.5px}

/* endpoints block */
.endpoints{margin-top:12px;border-top:1px solid var(--panel-edge);padding-top:10px;
  display:grid;gap:6px}
.ep{display:flex;align-items:center;gap:8px;font-size:12px;min-width:0}
.ep .tag{color:var(--ink-3);min-width:52px;font-weight:650;letter-spacing:.4px;font-size:10.5px}
.ep code{font-family:var(--mono);font-size:11.5px;color:var(--ink-2);background:rgba(0,0,0,.28);
  border-radius:6px;padding:3px 8px;overflow-x:auto;white-space:nowrap;flex:1;min-width:0}
.ep code a{color:var(--accent);text-decoration:none}
.copy{background:none;border:0;color:var(--ink-3);cursor:pointer;font-size:10.5px;font-weight:650;letter-spacing:.3px;
  padding:2px 5px;border-radius:5px;transition:color .15s ease,transform .1s ease}
.copy:hover{color:var(--accent)}
.copy:active{transform:scale(.85)}
.ep code::-webkit-scrollbar{height:4px}
.ep code::-webkit-scrollbar-thumb{background:rgba(255,255,255,.14);border-radius:2px}
.ep code::-webkit-scrollbar-track{background:transparent}
.copy.ok{color:var(--good)}

details{margin-top:10px}
details summary{color:var(--ink-3);font-size:12px;cursor:pointer;user-select:none}
details pre{font-family:var(--mono);font-size:11px;line-height:1.45;color:var(--ink-2);
  background:rgba(0,0,0,.3);border-radius:9px;padding:10px 12px;overflow:auto;max-height:260px}

footer{margin-top:26px;color:var(--ink-3);font-size:12px}
footer code{font-family:var(--mono)}

/* live pulse on the header dot */
.dot{width:9px;height:9px;border-radius:50%;background:var(--neutral);display:inline-block}
.dot.on{background:var(--good);animation:pulse 2.2s ease-out infinite}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(55,201,120,.45)}
  70%{box-shadow:0 0 0 8px rgba(55,201,120,0)}100%{box-shadow:0 0 0 0 rgba(55,201,120,0)}}
</style>
</head>
<body>
<div class="wrap">
<header>
  <span class="dot" id="livedot"></span>
  <h1>isiMonitor3D <span class="dim">/ AGV system test</span></h1>
  <div class="toolbar">
    <input id="tok" type="password" placeholder="API token (if required)">
    <button class="runall" id="runall">Run all checks</button>
  </div>
</header>
<p class="sub">Each card answers one state the pick-and-place test needs. RUN shows the live
answer and the exact REST endpoint / MQTT topic your client should use.</p>

<div class="grid" id="grid"></div>

<footer>MQTT broker: <code id="brokerline"></code> — plain TCP, no auth (test profile).
Subscribe topics at QoS&nbsp;1; <code>zone/*</code> and <code>config</code> are retained, the
current state arrives immediately on connect. Full instructions: <em>AGV System Test — Minimal
Integration Guide</em>.</footer>
</div>

<script>
"use strict";
const $=id=>document.getElementById(id);
const host=location.hostname||"SERVER_IP";
const tok=$("tok");
tok.value=localStorage.getItem("isicomms_token")||"";
tok.addEventListener("change",()=>localStorage.setItem("isicomms_token",tok.value));
$("brokerline").textContent=host+":1883";

async function j(path){
  try{
    const h=tok.value?{Authorization:"Bearer "+tok.value}:{};
    const r=await fetch(path,{headers:h});
    if(!r.ok)return{__err:r.status};
    return await r.json();
  }catch(e){return{__err:String(e)}}
}
const esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const chip=(kind,ic,label)=>`<span class="chip ${kind}"><span class="ic">${ic}</span>${esc(label)}</span>`;
const CH={good:["good","●"],warn:["warn","▲"],bad:["bad","✕"],neutral:["neutral","○"]};
const st=(k,label)=>chip(CH[k][0],CH[k][1],label);

/* ---- the checks ------------------------------------------------------- */
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
      n.map(x=>`<b>${esc(x.node_id)}</b> — ${esc(x.status)}, `+
        `${(x.cameras||[]).length} cam, ${x.fps?x.fps.toFixed(1):"?"} fps, `+
        `${x.latency_ms?Math.round(x.latency_ms):"?"} ms`).join("<br>")];}},
 {id:"cfg",title:"Zones configured",q:"Which zones does the node advertise?",
  rest:"/v1/config",topic:"isiMonitor3D/v1/+/config",retained:true,
  interpret:d=>{
    const rows=[];
    for(const c of (d.nodes||[])){
      const zs=((c.config&&c.config.zones)||[]).map(z=>z.name);
      rows.push(`<b>${esc(c.node_id||"?")}</b> — ${zs.length? zs.map(esc).join(", "):"no zones"}`);
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
        info=`${esc(p.occupancy_state||"?")}`+
             (p.occupancy_content?` · ${esc(p.occupancy_content)}`:"")+
             ` · conf ${(p.confidence??0).toFixed(2)}`;}
      else{c=st("neutral","EMPTY");info="nothing to pick";}
      return `<div class="zrow"><span class="zname">${esc(z.name)}</span>${c}`+
             `<span class="zinfo">${info}</span></div>`;});
    return[null,rows.join("")];}},
 {id:"tracks",title:"Live tracks",q:"What is being tracked right now?",
  rest:"/v1/tracks",topic:"isiMonitor3D/v1/+/track2d/+",
  interpret:d=>{
    const t=d.tracks||[];
    if(!t.length)return[st("neutral","QUIET"),"No active tracks."];
    const by={};t.forEach(x=>by[x.cls]=(by[x.cls]||0)+1);
    return[st("good",t.length+" TRACKED"),
      Object.entries(by).map(([k,v])=>`${v}× ${esc(k)}`).join(" · ")];}},
 {id:"pass",title:"Passings",q:"Recent zone entries / exits?",
  rest:"/v1/passings",topic:"isiMonitor3D/v1/+/zone/+/passings",
  interpret:d=>{
    const p=(d.passings||[]).slice(-5).reverse();
    if(!p.length)return[st("neutral","NONE"),"No boundary crossings recorded."];
    return[st("good",p.length+" RECENT"),
      p.map(x=>`${x.direction==="enter"?"⟶":"⟵"} <b>${esc(x.cls)}</b> `+
        `${x.direction==="enter"?"entered":"left"} <b>${esc(x.zone)}</b>`).join("<br>")];}},
];

/* ---- render ----------------------------------------------------------- */
function epRow(tag,html,copyText){
  return `<div class="ep"><span class="tag">${tag}</span><code>${html}</code>
    <button class="copy" data-copy="${esc(copyText)}" title="copy">copy</button></div>`;
}
function cardHtml(c){
  const restUrl=location.origin+c.rest;
  let eps=epRow("REST",`GET <a href="${c.rest}" target="_blank">${esc(restUrl)}</a>`,restUrl);
  if(c.topic){
    eps+=epRow("MQTT",esc(c.topic)+(c.retained?' <span style="color:var(--ink-3)">(retained)</span>':""),c.topic);
    eps+=epRow("SHELL",esc(`mosquitto_sub -h ${host} -t '${c.topic}' -v`),
               `mosquitto_sub -h ${host} -t '${c.topic}' -v`);
  }
  return `<div class="card" id="card-${c.id}">
    <div class="card-head">
      <h2>${esc(c.title)} <span class="q">${esc(c.q)}</span></h2>
      <span class="spacer"></span><span class="spin"></span>
      <button class="run" data-id="${c.id}">RUN</button>
    </div>
    <div class="result" id="res-${c.id}">
      <div id="chip-${c.id}"></div>
      <div class="summary" id="sum-${c.id}"></div>
      <div class="endpoints">${eps}</div>
      <details><summary>raw JSON</summary><pre id="raw-${c.id}">—</pre></details>
    </div></div>`;
}
$("grid").innerHTML=CHECKS.map(cardHtml).join("");

async function run(id){
  const c=CHECKS.find(x=>x.id===id),card=$("card-"+id);
  card.classList.add("busy");
  const d=await j(c.rest);
  card.classList.remove("busy");
  const res=$("res-"+id);res.classList.remove("show");void res.offsetWidth;
  let chipHtml,sum;
  if(d&&d.__err!==undefined){chipHtml=st("bad","ERROR");sum="Request failed: "+esc(d.__err);}
  else{[chipHtml,sum]=c.interpret(d||{});}
  $("chip-"+id).innerHTML=chipHtml||"";
  $("sum-"+id).innerHTML=sum;
  $("raw-"+id).textContent=d?JSON.stringify(d,null,2).slice(0,20000):"—";
  res.classList.add("show");
  if(id==="gw")$("livedot").classList.toggle("on",!!(d&&d.ok));
}
document.addEventListener("click",e=>{
  const r=e.target.closest(".run");if(r)return run(r.dataset.id);
  const cp=e.target.closest(".copy");
  if(cp){navigator.clipboard.writeText(cp.dataset.copy).then(()=>{
    cp.classList.add("ok");cp.textContent="copied";
    setTimeout(()=>{cp.classList.remove("ok");cp.textContent="copy";},1200);});}
});
async function runAll(){for(const c of CHECKS)await run(c.id);}
$("runall").addEventListener("click",runAll);
if(new URLSearchParams(location.search).get("run")==="all")runAll();
else run("gw");
</script>
</body>
</html>
"""
