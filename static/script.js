document.addEventListener('DOMContentLoaded', () => {
    
    // --- Navigation ---
    const navItems = document.querySelectorAll('.nav-item');
    const views = document.querySelectorAll('.view-section');
    
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            navItems.forEach(n => n.classList.remove('active'));
            views.forEach(v => v.classList.remove('active'));
            
            item.classList.add('active');
            document.getElementById(item.dataset.target).classList.add('active');
            
            if(item.dataset.target === 'view-overview') {
                loadOverviewData();
            }
        });
    });

    // --- Overview Data Fetching ---
    async function loadOverviewData() {
        try {
            const res = await fetch('/api/stats');
            const data = await res.json();
            
            // Update KPIs
            document.getElementById('kpi-tx').innerText = data.kpis.total_transactions.toLocaleString();
            document.getElementById('kpi-fraud').innerText = data.kpis.threats_blocked.toLocaleString();
            document.getElementById('kpi-users').innerText = data.kpis.active_users.toLocaleString();
            document.getElementById('kpi-vol').innerText = "$" + (data.kpis.total_volume).toLocaleString('en-US', {minimumFractionDigits: 0, maximumFractionDigits: 0});
            
            // Update Table
            const tbody = document.querySelector('#recent-table tbody');
            tbody.innerHTML = '';
            
            data.recent_transactions.forEach(t => {
                const tr = document.createElement('tr');
                const isFraud = t.fraud === 1 || t.fraud === 1.0;
                
                tr.innerHTML = `
                    <td style="font-family: monospace; color: var(--text-muted)">${t.Transaction_ID.substring(0,10)}</td>
                    <td>${t.Account_Number}</td>
                    <td>${parseFloat(t.distance_from_home).toFixed(1)}mi</td>
                    <td>${parseFloat(t.ratio_to_median_purchase_price).toFixed(2)}x</td>
                    <td>${t.online_order == 1 ? 'Yes' : 'No'}</td>
                    <td><span class="badge ${isFraud ? '' : 'safe'}">${isFraud ? 'FRAUD' : 'SAFE'}</span></td>
                `;
                
                // Add click listener to open modal
                tr.addEventListener('click', () => openUserModal(t, isFraud));
                
                tbody.appendChild(tr);
            });
            
        } catch(e) {
            console.error("Failed to load overview data:", e);
        }
    }
    
    // Load data initially
    loadOverviewData();

    // --- Modal Logic ---
    const modal = document.getElementById('user-modal');
    const btnCloseModal = document.getElementById('btn-close-modal');
    
    btnCloseModal.addEventListener('click', () => modal.classList.add('hidden'));
    
    const mockNames = ["James Wilson", "Emma Thompson", "Robert Chen", "Sophia Garcia", "Michael Chang", "Olivia Smith", "David Miller", "Isabella Davis"];
    
    function openUserModal(tx, isFraud) {
        modal.classList.remove('hidden');
        
        const nameIdx = parseInt(tx.Account_Number.split('-')[1]) % mockNames.length;
        document.getElementById('modal-name').innerText = mockNames[nameIdx];
        document.getElementById('modal-account').innerText = tx.Account_Number;
        
        document.getElementById('modal-dh').innerText = parseFloat(tx.distance_from_home).toFixed(2) + " miles";
        document.getElementById('modal-dl').innerText = parseFloat(tx.distance_from_last_transaction).toFixed(2) + " miles";
        document.getElementById('modal-ratio').innerText = parseFloat(tx.ratio_to_median_purchase_price).toFixed(2) + "x Median";
        document.getElementById('modal-online').innerText = tx.online_order == 1 ? "Yes (CNP)" : "No (In-Store)";
        document.getElementById('modal-pin').innerText = tx.used_pin_number == 1 ? "Yes" : "No";
        
        const alertBox = document.getElementById('modal-alert');
        const analysis = document.getElementById('modal-analysis');
        
        const prob = tx.Probability !== undefined ? tx.Probability : (isFraud ? 0.99 : 0.01);
        
        if(isFraud) {
            alertBox.className = 'alert-box danger';
            analysis.innerText = `HIGH RISK (${(prob*100).toFixed(1)}%). Transaction blocked. Pattern matches known fraud vectors (e.g., Extreme Ratio or Distance without PIN).`;
        } else {
            alertBox.className = 'alert-box safe';
            analysis.innerText = `CLEARED (${(prob*100).toFixed(1)}% Risk). Normal behavioral pattern.`;
        }
    }

    // --- Load Suspicious Profile ---
    const btnPrefill = document.getElementById('btn-prefill');
    if(btnPrefill) {
        btnPrefill.addEventListener('click', () => {
            document.getElementById('inp-account').value = "4892-1234-9988-0000";
            document.getElementById('inp-dist-home').value = "250.5"; // Very far
            document.getElementById('inp-dist-last').value = "100.0"; 
            document.getElementById('inp-ratio').value = "25.0"; // Spending 25x normal
            document.getElementById('inp-repeat').value = "0"; // Not repeat
            document.getElementById('inp-chip').value = "0"; // No chip
            document.getElementById('inp-pin').value = "0"; // No pin
            document.getElementById('inp-online').value = "1"; // Online
        });
    }

    // --- Single Prediction ---
    const manualForm = document.getElementById('manual-form');
    if(manualForm) {
        const resultPanel = document.getElementById('single-result-panel');
        const analysisContent = document.querySelector('.analysis-content');
        
        manualForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            // Build payload exactly as model expects
            const features = {
                distance_from_home: parseFloat(document.getElementById('inp-dist-home').value),
                distance_from_last_transaction: parseFloat(document.getElementById('inp-dist-last').value),
                ratio_to_median_purchase_price: parseFloat(document.getElementById('inp-ratio').value),
                repeat_retailer: parseFloat(document.getElementById('inp-repeat').value),
                used_chip: parseFloat(document.getElementById('inp-chip').value),
                used_pin_number: parseFloat(document.getElementById('inp-pin').value),
                online_order: parseFloat(document.getElementById('inp-online').value)
            };
            
            // UI Loading
            resultPanel.className = 'result-panel';
            document.querySelector('.empty-state').classList.add('hidden');
            analysisContent.classList.remove('hidden');
            
            const ring = document.getElementById('status-ring');
            const text = document.getElementById('status-text');
            const probVal = document.getElementById('single-prob');
            
            ring.className = 'ring';
            text.innerText = 'ANALYZING...';
            text.style.color = 'var(--text-primary)';
            probVal.innerText = '--';
            
            try {
                const res = await fetch('/api/predict_single', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ features })
                });
                const data = await res.json();
                
                if(data.success) {
                    probVal.innerText = (data.probability * 100).toFixed(2) + '%';
                    if(data.is_fraud) {
                        resultPanel.classList.add('fraud');
                        text.innerText = 'FRAUD DETECTED';
                    } else {
                        resultPanel.classList.add('safe');
                        text.innerText = 'TRANSACTION SAFE';
                    }
                    
                    loadOverviewData();
                } else {
                    alert("Prediction Error: " + data.error);
                }
            } catch(e) {
                alert("Network Error");
            }
        });
    }

    // --- Bulk Processing ---
    const fileInput = document.getElementById('csv-upload');
    const uploadZone = document.getElementById('upload-zone');
    const bulkLoading = document.getElementById('bulk-loading');
    const bulkResults = document.getElementById('bulk-results');
    const tbodyBulk = document.querySelector('#threat-table tbody');
    
    if(uploadZone) {
        uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadZone.classList.add('dragover');
        });
        uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
        uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadZone.classList.remove('dragover');
            if(e.dataTransfer.files.length) {
                fileInput.files = e.dataTransfer.files;
                processBulkFile(fileInput.files[0]);
            }
        });
        fileInput.addEventListener('change', () => {
            if(fileInput.files.length) processBulkFile(fileInput.files[0]);
        });
    }
    
    async function processBulkFile(file) {
        if(!file.name.endsWith('.csv')) return alert("Please upload a CSV file.");
        
        uploadZone.classList.add('hidden');
        bulkLoading.classList.remove('hidden');
        bulkResults.classList.add('hidden');
        
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const res = await fetch('/api/predict_bulk', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            
            if(data.success) {
                document.getElementById('bulk-total').innerText = data.total_processed.toLocaleString();
                document.getElementById('bulk-flagged').innerText = data.flagged_count.toLocaleString();
                
                tbodyBulk.innerHTML = '';
                data.flagged_transactions.forEach(t => {
                    const tr = document.createElement('tr');
                    tr.innerHTML = `
                        <td>#${t.row_id}</td>
                        <td style="font-family: monospace;">${t.account}</td>
                        <td>${t.dist_home.toFixed(1)} mi</td>
                        <td style="color: var(--danger)">${(t.probability*100).toFixed(2)}%</td>
                        <td><span class="badge">FRAUD</span></td>
                    `;
                    tbodyBulk.appendChild(tr);
                });
                
                bulkResults.classList.remove('hidden');
            } else {
                alert("Bulk Processing Error: " + data.error);
                uploadZone.classList.remove('hidden');
            }
        } catch(e) {
            alert("Upload failed.");
            uploadZone.classList.remove('hidden');
        } finally {
            bulkLoading.classList.add('hidden');
            fileInput.value = '';
        }
    }
});
