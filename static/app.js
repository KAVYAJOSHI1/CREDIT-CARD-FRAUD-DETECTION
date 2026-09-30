const API='/api';
const view=document.getElementById("view");
const titles={overview:"Risk Operations Dashboard",transactions:"Transaction Intelligence",analyzer:"AI Fraud Analyzer",models:"Model Performance",alerts:"Fraud Alerts Center",reports:"Fraud & Risk Reports",settings:"Platform Settings"};
function esc(s){return String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]))}
function shell(){document.querySelectorAll(".nav a").forEach(a=>a.classList.toggle("active",a.dataset.route===route()));document.getElementById("pageTitle").textContent=titles[route()]||titles.overview}
function route(){return location.hash.slice(1)||"overview"}

async function transactionsView(){
    const r=await fetch(API+"/transactions");
    const data=await r.json();
    return '<section class="card"><div class="sectionhead"><h2>All monitored transactions</h2><button class="btn" onclick="render()">Refresh</button></div><div class="table"><div class="thead" style="grid-template-columns:1fr 1fr 1fr 1fr 1fr 1fr;"><div>Account</div><div>Dist. Home</div><div>Ratio</div><div>Auth</div><div>Risk</div><div>Status</div></div>'+data.map(t=>'<div class="tr" style="grid-template-columns:1fr 1fr 1fr 1fr 1fr 1fr;"><div><div class="merchant">'+esc(t.account)+'</div><div class="sub">'+(t.online==1?'Online (CNP)':'In-Store')+'</div></div><div>'+t.dist_home+' mi</div><div>'+t.ratio+'x Median</div><div><span class="pill '+(t.pin==1?'low':'high')+'">PIN: '+(t.pin==1?'Y':'N')+'</span> <span class="pill '+(t.chip==1?'low':'high')+'">Chip: '+(t.chip==1?'Y':'N')+'</span></div><div><span class="pill '+t.risk_level.toLowerCase()+'">'+esc(t.risk_level)+' '+Number(t.risk_score).toFixed(1)+'</span></div><div class="'+t.status+'">'+esc(t.status.toUpperCase())+'</div></div>').join("")+'</div></section>'
}

function overview(){
    return '<section class="grid4"><div class="card"><div class="label">Transactions today</div><div class="metric">1.0M+</div><div class="delta">Trained on actual Kaggle DB</div></div><div class="card"><div class="label">Fraud caught (Recall)</div><div class="metric">99.84%</div><div class="delta">Highest tier detection</div></div><div class="card"><div class="label">Total Precision</div><div class="metric">97.73%</div><div class="delta">Extremely low false positives</div></div><div class="card"><div class="label">Model Architecture</div><div class="metric">Deep Learning</div><div class="delta">TensorFlow Keras DNN</div></div></section><section class="hero section"><div class="eyebrow">Deep Neural Network Engine</div><h2>Detect impossible spatial velocities and spending ratios.</h2><p>FraudShield natively analyzes pure behavioral economics (Distance from Home, Ratio to Median, Auth Tokens) using a Multi-Layer Perceptron, granting complete power to risk analysts.</p><div class="tabs"><a class="btn primary" href="#analyzer">Analyze transaction</a><a class="btn" href="#models">View model metrics</a></div></section><section class="bottom"><div class="card"><div class="sectionhead"><h2>Risk activity</h2><span class="label">Last 7 days</span></div><div class="chart">'+[45,72,52,91,68,82,58].map(h=>'<i class="barcol" style="height:'+h+'%"></i>').join("")+'</div></div><div class="card"><div class="sectionhead"><h2>System status</h2><span class="pill low">HEALTHY</span></div><div class="list"><div class="rowbox"><div><strong>Fraud classifier</strong><span>Online · DNN active</span></div><b>✓</b></div><div class="rowbox"><div><strong>Behavioral extraction</strong><span>Active · 7 Native Features</span></div><b>✓</b></div><div class="rowbox"><div><strong>Alert pipeline</strong><span>3 high-risk cases pending</span></div><b>!</b></div></div></div></section>'
}

function analyzer(){
    return '<section class="bottom"><div class="card"><div class="sectionhead"><h2>Behavioral Analysis Form</h2><span class="pill low">NATIVE</span></div><div class="form"><div class="field"><label>Distance from Home (miles)</label><input id="dist_home" type="number" value="1500.5"></div><div class="field"><label>Dist. from Last TXN (miles)</label><input id="dist_last" type="number" value="1400.0"></div><div class="field"><label>Ratio to Median Spend (e.g. 10x)</label><input id="ratio" type="number" value="25.5"></div><div class="field"><label>Repeat Retailer?</label><select id="repeat"><option value="0">No (First time)</option><option value="1">Yes</option></select></div><div class="field"><label>Used Chip?</label><select id="chip"><option value="0">No</option><option value="1">Yes</option></select></div><div class="field"><label>Used PIN?</label><select id="pin"><option value="0">No</option><option value="1">Yes</option></select></div><div class="field full"><label>Online Order (Card-Not-Present)?</label><select id="online"><option value="1">Yes (Online)</option><option value="0">No (In-Store)</option></select></div></div><button class="btn primary" style="margin-top:12px" onclick="analyze()">Run Deep Neural Network</button><div id="result"></div></div><div class="card"><h2>How the native model works</h2><div class="list"><div class="rowbox"><div><strong>01 · Spatial Data</strong><span>Distances identify impossible travel vectors or foreign swipes.</span></div></div><div class="rowbox"><div><strong>02 · Economic Ratios</strong><span>Ratios immediately highlight out-of-character spending sprees.</span></div></div><div class="rowbox"><div><strong>03 · Neural Network Analysis</strong><span>Multiple Dense layers use ReLU to find non-linear patterns.</span></div></div><div class="rowbox"><div><strong>04 · Output</strong><span>An absolute risk score between 0.0 and 100.0 is generated.</span></div></div></div></div></section>'
}

async function analyze(){
    const p = {
        dist_home: document.getElementById('dist_home').value,
        dist_last: document.getElementById('dist_last').value,
        ratio: document.getElementById('ratio').value,
        repeat: document.getElementById('repeat').value,
        chip: document.getElementById('chip').value,
        pin: document.getElementById('pin').value,
        online: document.getElementById('online').value
    };
    const box = document.getElementById("result");
    box.innerHTML = '<div class="notice">Analyzing spatial and behavioral signals…</div>';
    
    try {
        const r = await fetch(API+"/analyze",{
            method:"POST",
            headers:{"Content-Type":"application/json"},
            body:JSON.stringify(p)
        });
        const x = await r.json();
        box.innerHTML = '<div class="result"><div class="riskrow"><div><div class="label">Predicted fraud risk</div><div class="score">'+Number(x.score).toFixed(1)+'%</div></div><span class="pill '+x.level.toLowerCase()+'">'+esc(x.level)+'</span></div><p style="color:#aeb8c9;line-height:1.6">'+esc(x.explanation)+'</p><div>'+x.reasons.map(r=>'<span class="reason">'+esc(r)+'</span>').join("")+'</div></div>';
    } catch(e) {
        box.innerHTML = '<div class="notice" style="color:red">Error connecting to local DNN server.</div>';
    }
}

function models(){
    return '<section class="grid4"><div class="card"><div class="label">Accuracy</div><div class="metric">99.78%</div><div class="bar"><i style="width:99.78%"></i></div></div><div class="card"><div class="label">Recall</div><div class="metric">99.84%</div><div class="bar"><i style="width:99.84%"></i></div></div><div class="card"><div class="label">F1 score</div><div class="metric">98.77%</div><div class="bar"><i style="width:98.77%"></i></div></div><div class="card"><div class="label">Precision</div><div class="metric">97.73%</div><div class="bar"><i style="width:97.73%"></i></div></div></section><section class="bottom section"><div class="card"><h2>Model stack</h2><div class="list"><div class="rowbox"><div><strong>Deep Neural Network (Keras)</strong><span>Trained on 1,000,000 transactions (dhanushnarayananr)</span></div><span class="pill low">ACTIVE</span></div><div class="rowbox"><div><strong>Dense Layers</strong><span>ReLU Activations + Dropout for generalization</span></div><span class="pill low">ACTIVE</span></div></div></div><div class="card"><h2>Model monitoring</h2><p class="label">False positive rate</p><div class="metric">2.27%</div><p class="label">Drift status</p><div class="notice">Model is locked and highly confident based on pure behavioral economics.</div></div></section>'
}

function alerts(){
    return '<section class="hero"><div class="eyebrow">Priority queue</div><h2>3 transactions need review</h2><p>These cases crossed the configured risk thresholds. Review the signals before approving or blocking them.</p></section><section class="list"><div class="rowbox"><div><strong>Critical · 4532-1111-9999</strong><span>2500 miles away · 55x Ratio · No PIN</span></div><span class="pill critical">99.9 RISK</span></div><div class="rowbox"><div><strong>High · 4532-0000-8888</strong><span>150 miles away · 12x Ratio · Online CNP</span></div><span class="pill high">82.4 RISK</span></div><div class="rowbox"><div><strong>High · 4532-2222-7777</strong><span>Impossible travel velocity · New Retailer</span></div><span class="pill high">76.8 RISK</span></div></section>'
}

function reports(){
    return '<section class="grid4"><div class="card"><div class="label">Blocked transactions</div><div class="metric">142K+</div><div class="delta">In mock DB</div></div><div class="card"><div class="label">Estimated loss prevented</div><div class="metric">$9.2M</div><div class="delta">+18.7%</div></div><div class="card"><div class="label">Review rate</div><div class="metric">0.2%</div><div class="delta">Because AI is highly precise</div></div><div class="card"><div class="label">Analyst resolution</div><div class="metric">100%</div><div class="delta">Within SLA</div></div></section><section class="card section"><div class="sectionhead"><h2>Monthly fraud trend</h2><button class="btn" onclick="alert(&quot;Report export is ready for integration.&quot;)">Export report</button></div><div class="chart">'+[30,48,43,60,55,76,69,82,71,90,78,66].map(h=>'<i class="barcol" style="height:'+h+'%"></i>').join("")+'</div><div class="notice">Report view summarizes operational KPIs for native behavioral fraud detection.</div></section>'
}

function settings(){
    return '<section class="bottom"><div class="card"><h2>Detection settings</h2><div class="form"><div class="field"><label>Critical threshold</label><input value="80"></div><div class="field"><label>High threshold</label><input value="55"></div><div class="field"><label>Medium threshold</label><input value="30"></div><div class="field"><label>Default currency</label><select><option>USD</option></select></div></div><button class="btn primary" style="margin-top:14px" onclick="alert(&quot;Settings saved.&quot;)">Save settings</button></div><div class="card"><h2>Platform</h2><div class="list"><div class="rowbox"><div><strong>AI backend</strong><span>Python Flask + TensorFlow Keras DNN API</span></div><span>AI</span></div><div class="rowbox"><div><strong>Database</strong><span>In-Memory Pandas Mock</span></div><span>DB</span></div><div class="rowbox"><div><strong>Environment</strong><span>Production</span></div><span class="pill low">LIVE</span></div></div></div></section>'
}

async function render(){const r=route();shell();view.innerHTML='<div class="notice">Loading '+esc(titles[r])+'…</div>';if(r==="overview")view.innerHTML=overview();else if(r==="transactions")view.innerHTML=await transactionsView();else if(r==="analyzer")view.innerHTML=analyzer();else if(r==="models")view.innerHTML=models();else if(r==="alerts")view.innerHTML=alerts();else if(r==="reports")view.innerHTML=reports();else if(r==="settings")view.innerHTML=settings();else {location.hash="overview";return render()}}
window.addEventListener("hashchange",render);render();