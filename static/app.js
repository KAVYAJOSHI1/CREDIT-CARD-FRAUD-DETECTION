const API='/api';
const view=document.getElementById("view");
const titles={overview:"Risk Operations Dashboard",transactions:"Transaction Intelligence",analyzer:"AI Fraud Analyzer",models:"Model Performance",alerts:"Fraud Alerts Center",reports:"Fraud & Risk Reports",settings:"Platform Settings"};
const FEATURES=["Time",...Array.from({length:28},(_,i)=>"V"+(i+1)),"Amount"];
const esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const pct=(x,d=1)=>(x*100).toFixed(d)+"%";
const money=x=>"$"+Number(x).toLocaleString(undefined,{maximumFractionDigits:0});
const get=async u=>(await fetch(API+u)).json();
function route(){return location.hash.slice(1)||"overview"}
function shell(){document.querySelectorAll(".nav a").forEach(a=>a.classList.toggle("active",a.dataset.route===route()));document.getElementById("pageTitle").textContent=titles[route()]||titles.overview}
const best=ms=>ms.find(m=>m.name==="Stacked Ensemble");
const ago=t=>{const m=Math.round((Date.now()/1000-t)/60);return m<60?m+" min ago":Math.round(m/60)+" h ago"};

async function overview(){
  const [s,m]=await Promise.all([get("/stats"),get("/models")]);const e=best(m.metrics);
  const acc=s.labelled?pct(s.correct/s.labelled):"–";
  return '<section class="grid4"><div class="card"><div class="label">Transactions scored</div><div class="metric">'+s.total+'</div><div class="delta">Logged in SQLite</div></div><div class="card"><div class="label">Blocked</div><div class="metric">'+s.blocked+'</div><div class="delta">'+money(s.amount_blocked)+' held</div></div><div class="card"><div class="label">Test recall / precision</div><div class="metric">'+pct(e.recall,0)+' / '+pct(e.precision,0)+'</div><div class="delta">Held-out test, 99 frauds</div></div><div class="card"><div class="label">Live accuracy on logged txns</div><div class="metric">'+acc+'</div><div class="delta">'+s.correct+' of '+s.labelled+' labelled</div></div></section><section class="hero section"><div class="eyebrow">Deep ensemble engine</div><h2>Five neural networks and an autoencoder, stacked.</h2><p>A residual MLP (3 seeds), an FT-Transformer (2 seeds) and a denoising autoencoder vote through a stacked meta-learner, with MC-dropout uncertainty and Integrated Gradients explanations on every decision.</p><div class="tabs"><a class="btn primary" href="#analyzer">Analyze transaction</a><a class="btn" href="#models">View model metrics</a></div></section><section class="bottom"><div class="card"><div class="sectionhead"><h2>Architecture</h2></div><div class="list"><div class="rowbox"><div><strong>Residual MLP ×3</strong><span>4 residual blocks · BatchNorm · GELU · Dropout · focal loss</span></div><span class="pill low">ACTIVE</span></div><div class="rowbox"><div><strong>FT-Transformer ×2</strong><span>per-feature tokens · [CLS] · 3 self-attention blocks</span></div><span class="pill low">ACTIVE</span></div><div class="rowbox"><div><strong>Denoising Autoencoder</strong><span>trained on legitimate txns only · reconstruction error</span></div><span class="pill low">ACTIVE</span></div></div></div><div class="card"><div class="sectionhead"><h2>System status</h2><span class="pill low">HEALTHY</span></div><div class="list"><div class="rowbox"><div><strong>Block threshold</strong><span>stacked probability ≥ '+m.threshold.toFixed(4)+'</span></div></div><div class="rowbox"><div><strong>Database</strong><span>SQLite · '+s.total+' rows</span></div><b>✓</b></div></div></div></section>'
}

async function transactionsView(){
  const data=await get("/transactions?limit=50");
  return '<section class="card"><div class="sectionhead"><h2>Scored transactions (latest 50)</h2><button class="btn" onclick="render()">Refresh</button></div><div class="table"><div class="thead" style="grid-template-columns:.5fr 1fr 1fr 1fr 1fr 1fr;"><div>ID</div><div>Amount</div><div>Probability</div><div>Uncertainty</div><div>Risk</div><div>Status / truth</div></div>'+data.map(t=>'<div class="tr" style="grid-template-columns:.5fr 1fr 1fr 1fr 1fr 1fr;"><div><div class="merchant">#'+t.id+'</div><div class="sub">'+esc(t.source)+' · '+ago(t.created_at)+'</div></div><div>'+money(t.amount)+'</div><div>'+pct(t.probability,2)+'</div><div>±'+t.uncertainty.toFixed(3)+'</div><div><span class="pill '+t.level+'">'+esc(t.level)+'</span></div><div class="'+t.status+'">'+t.status.toUpperCase()+(t.true_label==null?'':' <span class="sub">(actual: '+(t.true_label?'fraud':'legit')+')</span>')+'</div></div>').join("")+'</div></section>'
}

function analyzer(){
  const inputs=FEATURES.map(f=>'<div class="field"><label>'+f+'</label><input id="f_'+f+'" type="number" step="any" value="0"></div>').join("");
  return '<section class="bottom"><div class="card"><div class="sectionhead"><h2>Transaction Analyzer</h2><span class="pill low">30 FEATURES</span></div><p class="label" style="line-height:1.6">Features V1–V28 are anonymised PCA components (Kaggle credit-card dataset), so load a real held-out transaction, then edit values to see how the model reacts.</p><div class="tabs" style="margin:10px 0"><button class="btn primary" onclick="loadSample(\'fraud\')">Load real FRAUD</button><button class="btn" onclick="loadSample(\'legit\')">Load real LEGIT</button></div><div id="truth"></div><div class="form" style="grid-template-columns:repeat(4,1fr)">'+inputs+'</div><button class="btn primary" style="margin-top:12px" onclick="analyze()">Run deep ensemble</button><div id="result"></div></div><div class="card"><h2>What happens on submit</h2><div class="list"><div class="rowbox"><div><strong>01 · Preprocess</strong><span>log(Amount), time-of-day, robust scaling.</span></div></div><div class="rowbox"><div><strong>02 · Five networks + autoencoder</strong><span>ResMLP ×3, FT-Transformer ×2, reconstruction error.</span></div></div><div class="rowbox"><div><strong>03 · Stacker</strong><span>Logistic meta-learner → fraud probability.</span></div></div><div class="rowbox"><div><strong>04 · Uncertainty + explanation</strong><span>30 MC-dropout passes and Integrated Gradients.</span></div></div></div></div></section>'
}
let trueLabel=null;
async function loadSample(kind){
  const x=await get("/sample?kind="+kind);trueLabel=x.true_label;
  FEATURES.forEach(f=>document.getElementById("f_"+f).value=x[f]);
  document.getElementById("truth").innerHTML='<div class="notice">Loaded a real held-out transaction (ground truth hidden until you run the model).</div>';
  document.getElementById("result").innerHTML="";
}
async function analyze(){
  const p={};FEATURES.forEach(f=>p[f]=document.getElementById("f_"+f).value);if(trueLabel!=null)p.true_label=trueLabel;
  const box=document.getElementById("result");box.innerHTML='<div class="notice">Running 5 networks, autoencoder, MC-dropout and Integrated Gradients…</div>';
  try{
    const r=await fetch(API+"/analyze",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(p)});
    const x=await r.json();if(!r.ok)throw new Error(x.error);
    const mx=Math.max(...x.top_features.map(t=>Math.abs(t.attribution)))||1;
    box.innerHTML='<div class="result"><div class="riskrow"><div><div class="label">Fraud probability</div><div class="score">'+pct(x.probability,2)+'</div></div><div><span class="pill '+x.level+'">'+esc(x.level)+'</span> <b class="'+x.status+'">'+x.status.toUpperCase()+'</b></div></div>'+(trueLabel!=null?'<div class="notice">Ground truth: <b>'+(trueLabel?'FRAUD':'LEGIT')+'</b> — model was '+((x.status==="blocked")===(trueLabel===1)?'correct':'wrong')+'</div>':'')+'<p class="label" style="margin-top:12px">Per-model scores</p><div>'+[["ResMLP ×3",x.resmlp],["FT-Transformer ×2",x.ft_transformer]].map(a=>'<span class="reason">'+a[0]+': '+pct(a[1],2)+'</span>').join("")+'<span class="reason">Autoencoder error: '+x.ae_error.toFixed(3)+'</span><span class="reason">Uncertainty ±'+x.uncertainty.toFixed(3)+' ('+esc(x.confidence)+' confidence)</span></div><p class="label" style="margin-top:12px">Top drivers (Integrated Gradients)</p>'+x.top_features.map(t=>'<div style="margin:6px 0"><div class="label">'+t.feature+' = '+t.raw_value.toFixed(2)+' → '+(t.attribution>0?'raises':'lowers')+' risk</div><div class="bar"><i style="width:'+(Math.abs(t.attribution)/mx*100)+'%;background:'+(t.attribution>0?'#fb7185':'#34d399')+'"></i></div></div>').join("")+'</div>';
  }catch(e){box.innerHTML='<div class="notice" style="color:#fb7185">'+esc(e.message||"Error contacting server")+'</div>'}
}

async function models(){
  const m=await get("/models"),u=m.uncertainty;
  const rows=m.metrics.map(r=>'<div class="tr" style="grid-template-columns:1.6fr repeat(5,1fr)"><div class="merchant">'+esc(r.name)+'</div><div>'+r.pr_auc.toFixed(3)+'</div><div>'+r.roc_auc.toFixed(3)+'</div><div>'+pct(r.precision,1)+'</div><div>'+pct(r.recall,1)+'</div><div>'+r.f1.toFixed(3)+'</div></div>').join("");
  return '<section class="card"><div class="sectionhead"><h2>Held-out test set (99 frauds / 56,962 transactions)</h2></div><div class="table"><div class="thead" style="grid-template-columns:1.6fr repeat(5,1fr)"><div>Model</div><div>PR-AUC</div><div>ROC-AUC</div><div>Precision</div><div>Recall</div><div>F1</div></div>'+rows+'</div><div class="notice">Thresholds tuned on validation, numbers reported on test. With only 99 frauds, PR-AUC differences under ~0.02 are within noise; the deep models match, rather than beat, the XGBoost baseline.</div></section><section class="bottom section"><div class="card"><h2>MC-dropout uncertainty</h2><div class="list"><div class="rowbox"><div><strong>Error rate, confident predictions</strong></div><b>'+pct(u.error_rate_low_uncertainty,3)+'</b></div><div class="rowbox"><div><strong>Error rate, top-10% uncertain</strong></div><b>'+pct(u.error_rate_top10pct_uncertainty,2)+'</b></div><div class="rowbox"><div><strong>Mean σ fraud vs legit</strong></div><b>'+u.mean_std_fraud.toFixed(3)+' vs '+u.mean_std_legit.toFixed(3)+'</b></div></div></div><div class="card"><h2>Feature attribution</h2><img src="/reports/feature_attribution.png" style="width:100%;border-radius:10px;background:#fff"></div></section><section class="card section"><img src="/reports/model_comparison.png" style="width:100%;border-radius:10px;background:#fff"></section>'
}

async function alerts(){
  const d=await get("/alerts");
  return '<section class="hero"><div class="eyebrow">Priority queue</div><h2>'+d.length+' blocked transactions</h2><p>Highest-recency blocks from the scored log. Uncertainty shows how sure the ensemble is.</p></section><section class="list">'+d.map(t=>'<div class="rowbox"><div><strong>'+esc(t.level)+' · #'+t.id+' · '+money(t.amount)+'</strong><span>'+(t.top_features.length?'Drivers: '+t.top_features.slice(0,3).map(f=>f.feature).join(", "):"")+' · ±'+t.uncertainty.toFixed(3)+(t.true_label==null?'':' · actual: '+(t.true_label?'fraud':'legit'))+'</span></div><span class="pill '+t.level+'">'+pct(t.probability,2)+'</span></div>').join("")+'</section>'
}

async function reports(){
  const s=await get("/stats");
  return '<section class="grid4"><div class="card"><div class="label">Scored</div><div class="metric">'+s.total+'</div></div><div class="card"><div class="label">Blocked</div><div class="metric">'+s.blocked+'</div></div><div class="card"><div class="label">Value held</div><div class="metric">'+money(s.amount_blocked)+'</div><div class="delta">Sum of blocked amounts in log</div></div><div class="card"><div class="label">Correct (labelled)</div><div class="metric">'+s.correct+'/'+s.labelled+'</div></div></section><section class="card section"><div class="notice">The log is a replay of real held-out transactions with fraud over-sampled for demonstration, plus anything run through the Analyzer — so these are not production fraud rates.</div></section>'
}

async function settings(){
  const m=await get("/models");
  return '<section class="bottom"><div class="card"><h2>Detection policy</h2><div class="list"><div class="rowbox"><div><strong>Block threshold (stacked probability)</strong><span>Chosen on out-of-fold validation predictions to maximise F1</span></div><b>'+m.threshold.toFixed(4)+'</b></div><div class="rowbox"><div><strong>Risk levels</strong><span>critical ≥ threshold · high ≥ 0.90 · medium ≥ 0.50</span></div></div></div></div><div class="card"><h2>Platform</h2><div class="list"><div class="rowbox"><div><strong>AI backend</strong><span>Flask + Keras 3 / TensorFlow</span></div></div><div class="rowbox"><div><strong>Database</strong><span>SQLite (db/fraudshield.sqlite)</span></div></div></div></div></section>'
}

async function render(){const r=route();shell();view.innerHTML='<div class="notice">Loading '+esc(titles[r]||"")+'…</div>';
  const pages={overview,transactions:transactionsView,analyzer,models,alerts,reports,settings};
  if(!pages[r]){location.hash="overview";return}
  try{view.innerHTML=await pages[r]()}catch(e){view.innerHTML='<div class="notice" style="color:#fb7185">Failed to load: '+esc(e.message)+'</div>'}}
window.addEventListener("hashchange",render);render();
