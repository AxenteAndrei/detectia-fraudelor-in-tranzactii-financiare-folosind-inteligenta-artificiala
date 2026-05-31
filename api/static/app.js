// ============================================================
// Frontend SPA pentru demo-ul de detectie a fraudelor.
// Comunica cu /predict, /generate, /health.
// ============================================================

// ---------------------------------------------------------------- configuratie
const TOP = ["V14","V10","V12","V17","V16"];
const TOP_LABEL = {
  V14:"V14 * (imp=0.2065)", V10:"V10 * (imp=0.1292)",
  V12:"V12 * (imp=0.1148)", V17:"V17 * (imp=0.1106)", V16:"V16 * (imp=0.0785)"
};
const HELP = {
  Amount: "Suma tranzactiei in EUR. Fraudele au tipic sume mici (mediana 9.25 EUR) dar uneori si mari.",
  Time:   "Secunde de la prima tranzactie din dataset (48 ore total). Fraudele sunt distribuite uniform, legitimele au pattern ciclic zi/noapte.",
  V14:    "Componenta PCA cel mai importanta (imp=0.2065). Valori sub -4 sunt puternic asociate cu frauda.",
  V10:    "A doua componenta ca importanta (imp=0.1292). Valori negative indica activitate neobisnuita.",
  V12:    "Componenta PCA cu importanta 0.1148. Asociata cu frecventa tranzactiilor recente.",
  V17:    "Componenta PCA cu importanta 0.1106. Captureaza anomalii in istoricul tranzactional.",
  V16:    "Componenta PCA cu importanta 0.0785. Indicator de consistenta comportamentala.",
  V:      "Componenta PCA secundara. Valori tipice in intervalul [-3, 3]. Importanta mai mica in detectia fraudei."
};

// definitia tuturor sliderilor
const FEATURES = [];
FEATURES.push({name:"Amount", min:0, max:5000, step:1, dec:2, val:100, unit:" EUR", label:"Amount (EUR)", grp:"real"});
FEATURES.push({name:"Time", min:0, max:172792, step:1, dec:0, val:50000, unit:" s", label:"Time (secunde)", grp:"real"});
TOP.forEach(n => FEATURES.push({name:n, min:-5, max:5, step:0.01, dec:2, val:0, unit:"", label:TOP_LABEL[n], grp:"top", top:true}));
for(let i=1;i<=14;i++){ const n="V"+i; if(!TOP.includes(n)) FEATURES.push({name:n,min:-5,max:5,step:0.01,dec:2,val:0,unit:"",label:n,grp:"rest1"}); }
for(let i=15;i<=28;i++){ const n="V"+i; if(!TOP.includes(n)) FEATURES.push({name:n,min:-5,max:5,step:0.01,dec:2,val:0,unit:"",label:n,grp:"rest2"}); }

const CFG = {}; FEATURES.forEach(f => CFG[f.name]=f);

// ---------------------------------------------------------------- construire UI
function fmt(name, v){ const f=CFG[name]; return Number(v).toFixed(f.dec)+f.unit; }

function escTip(s){ return String(s).replace(/"/g,'&quot;'); }

function buildSliders(){
  const groups={real:"grp-real", top:"grp-top", rest1:"grp-rest1", rest2:"grp-rest2"};
  FEATURES.forEach(f=>{
    const help = HELP[f.name] || HELP.V;
    const div=document.createElement("div");
    div.className="slider"+(f.top?" top":"");
    div.innerHTML=`
      <div class="lab">
        <span class="name${f.top?' top':''}">${f.label}<span class="info-icon" data-tip="${escTip(help)}">?</span></span>
        <span class="val" id="val-${f.name}">${fmt(f.name,f.val)}</span>
      </div>
      <input type="range" id="sl-${f.name}" min="${f.min}" max="${f.max}" step="${f.step}" value="${f.val}">`;
    document.getElementById(groups[f.grp]).appendChild(div);
    const sl=div.querySelector("input");
    sl.addEventListener("input", ()=>{ document.getElementById("val-"+f.name).textContent=fmt(f.name,sl.value); });
  });
}

function setVal(name,v){
  const f=CFG[name]; if(!f) return;
  v=Math.min(f.max, Math.max(f.min, Number(v)));
  document.getElementById("sl-"+name).value=v;
  document.getElementById("val-"+name).textContent=fmt(name, v);
}
function getVal(name){ return parseFloat(document.getElementById("sl-"+name).value); }

function toggle(id){
  document.getElementById("grp-"+id).classList.toggle("open");
  document.getElementById("t-"+id).classList.toggle("open");
}

function buildGuide(){
  const dl=document.getElementById("guide");
  const entries=[
    ["Amount", HELP.Amount],["Time", HELP.Time],
    ["V14 *", HELP.V14],["V10 *", HELP.V10],["V12 *", HELP.V12],["V17 *", HELP.V17],["V16 *", HELP.V16],
    ["V1 - V28 (general)", HELP.V]
  ];
  dl.innerHTML=entries.map(([t,d])=>`<dt>${t}</dt><dd>${d}</dd>`).join("");
}

// ---------------------------------------------------------------- tooltip global
function setupTooltip(){
  const tip = document.getElementById('tip');
  function show(el){
    tip.textContent = el.dataset.tip;
    tip.hidden = false;
    const r = el.getBoundingClientRect();
    const tw = tip.offsetWidth, th = tip.offsetHeight;
    const pad = 8;
    let left = r.left + r.width/2 - tw/2;
    let top  = r.top - th - 10;
    if (top < pad) top = r.bottom + 10;                                 // sub icon, daca nu incape sus
    left = Math.max(pad, Math.min(left, window.innerWidth - tw - pad)); // clamp orizontal
    tip.style.left = left + 'px';
    tip.style.top  = top  + 'px';
  }
  function hide(){ tip.hidden = true; }
  document.body.addEventListener('mouseover', e => {
    const el = e.target.closest('.info-icon');
    if (el) show(el);
  });
  document.body.addEventListener('mouseout', e => {
    if (e.target.closest('.info-icon')) hide();
  });
  // ascunde la scroll (sidebar are scroll propriu) ca sa nu ramana orfan
  document.querySelectorAll('.sidebar').forEach(el => el.addEventListener('scroll', hide, {passive:true}));
  window.addEventListener('scroll', hide, {passive:true});
}

// ---------------------------------------------------------------- API calls
async function health(){
  const dot=document.getElementById("hdot"), txt=document.getElementById("htext");
  try{
    const r=await fetch("/health"); const j=await r.json();
    dot.className="dot ok"; txt.textContent=`${j.model} online (AUC-PR ${j.auc_pr})`;
  }catch(e){ dot.className="dot err"; txt.textContent="backend indisponibil"; }
}

async function generate(type){
  try{
    const r=await fetch("/generate/"+type);
    if(!r.ok) throw new Error(await r.text());
    const j=await r.json();
    Object.keys(j).forEach(k=> setVal(k, j[k]));
  }catch(e){ alert("Eroare la generare: "+e.message); }
}

async function verify(){
  const payload={}; FEATURES.forEach(f=> payload[f.name]=getVal(f.name));
  try{
    const r=await fetch("/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
    if(!r.ok) throw new Error(await r.text());
    const j=await r.json();
    renderResult(j, payload);
    pushHistory(j, payload);
  }catch(e){ alert("Eroare la predictie: "+e.message); }
}

// ---------------------------------------------------------------- randare rezultat
function renderResult(j, payload){
  const fraud = j.label==="FRAUDA";
  const p = j.probability*100;                       // probabilitate frauda
  const conf = fraud ? p : (100-p);
  const badge = fraud
    ? `<span class="badge fraud">FRAUDA &middot; ${p.toFixed(2)}%</span>`
    : `<span class="badge legit">LEGITIMA &middot; ${conf.toFixed(2)}% siguranta</span>`;
  const barClass = fraud ? "fraud" : "legit";
  const rows = j.top5_features.map(f=>`
    <tr><td><b>${f.feature}</b></td><td class="num">${f.value}</td>
        <td class="num">${f.importance}</td><td class="num">${f.contribution}</td></tr>`).join("");
  document.getElementById("result").innerHTML=`
    <div class="res-head">${badge}</div>
    <div class="bar-wrap">
      <div class="bar-lab"><span>Probabilitate de frauda</span><span>${p.toFixed(2)}%</span></div>
      <div class="bar"><span class="${barClass}" style="width:${p.toFixed(1)}%;"></span></div>
    </div>
    <div class="msg">${j.message}</div>
    <div class="res-sub-title">Top 5 caracteristici care au influentat decizia</div>
    <table>
      <thead><tr><th>Feature</th><th class="num">Valoare</th><th class="num">Importanta RF</th><th class="num">Contributie</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

// ---------------------------------------------------------------- istoric (in memorie)
let HISTORY=[];
function pushHistory(j, payload){
  HISTORY.unshift({amount:payload.Amount, time:payload.Time, p:(j.probability*100), label:j.label});
  HISTORY=HISTORY.slice(0,5);
  const rows=HISTORY.map(h=>`
    <tr><td class="num">${Number(h.amount).toFixed(2)}</td>
        <td class="num">${Number(h.time).toFixed(0)}</td>
        <td class="num">${h.p.toFixed(2)}%</td>
        <td><span class="pill ${h.label==='FRAUDA'?'fraud':'legit'}">${h.label}</span></td></tr>`).join("");
  document.getElementById("history").innerHTML=`
    <table>
      <thead><tr><th class="num">Amount</th><th class="num">Time</th><th class="num">P(frauda)</th><th>Decizie</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

// ---------------------------------------------------------------- event wiring
function wireControls(){
  document.getElementById('btn-gen-legit').addEventListener('click', () => generate('legitimate'));
  document.getElementById('btn-gen-fraud').addEventListener('click', () => generate('fraud'));
  document.getElementById('btn-verify').addEventListener('click', verify);
  document.querySelectorAll('.grp-toggle[data-target]').forEach(el =>
    el.addEventListener('click', () => toggle(el.dataset.target))
  );
}

// ---------------------------------------------------------------- init
window.addEventListener('DOMContentLoaded', () => {
  buildSliders();
  buildGuide();
  setupTooltip();
  wireControls();
  health();
});
