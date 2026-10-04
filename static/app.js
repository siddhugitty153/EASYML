/**
 * EasyML — Frontend Logic
 * ================================
 * Handles: step navigation, file upload, multi-CSV merge,
 * pipeline config & execution, results rendering, charts,
 * AI insights, chat panel, and auto-explore mode.
 */

// ── Theme Toggle ──
function toggleTheme() {
    const isLight = document.documentElement.getAttribute('data-theme') === 'light';
    const newTheme = isLight ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    document.getElementById('themeIcon').textContent = newTheme === 'light' ? '🌙' : '☀️';

    // For Chart.js text colors, we'd ideally trigger a re-render
}

if (localStorage.getItem('theme') === 'light') {
    document.documentElement.setAttribute('data-theme', 'light');
    document.addEventListener('DOMContentLoaded', () => {
        const icon = document.getElementById('themeIcon');
        if (icon) icon.textContent = '🌙';
    });
}

// ── State ──
let currentStep = 0;
let dataPreview = null;
let pipelineReport = null;
let compChart = null;
let timeChart = null;
let featureImpChart = null;
let cachedFeatureImportance = null;
let shapGlobalChart = null;
let pdpChart = null;
let whatIfShapChart = null;
let _agentState = null;       // last polled agent state
let _agentPollTimer = null;   // setInterval handle

// ═══════════════════════════════════════════════════════
// Step Navigation
// ═══════════════════════════════════════════════════════

function goToStep(n) {
    document.querySelectorAll('.step-panel').forEach((p, i) => {
        p.classList.toggle('active', i === n);
    });
    document.querySelectorAll('.step-dot').forEach((d, i) => {
        d.classList.toggle('active', i <= n);
        d.classList.toggle('completed', i < n);
    });
    currentStep = n;
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ═══════════════════════════════════════════════════════
// File Upload
// ═══════════════════════════════════════════════════════

const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');

uploadArea.addEventListener('click', () => fileInput.click());
uploadArea.addEventListener('dragover', e => {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
});
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('drag-over'));
uploadArea.addEventListener('drop', e => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        uploadFiles(e.dataTransfer.files);
    }
});
fileInput.addEventListener('change', () => {
    if (fileInput.files.length) uploadFiles(fileInput.files);
});

async function uploadFiles(files) {
    showLoading('Uploading files...');
    const form = new FormData();
    for (const f of files) form.append('file', f);

    try {
        const res = await fetch('/api/upload', { method: 'POST', body: form });
        const data = await res.json();
        hideLoading();

        if (data.error) {
            showToast(data.error, 'error');
            return;
        }

        if (data.multi) {
            // Show merge panel
            showMergePanel(data.files);
        } else {
            dataPreview = data;
            renderDataPreview(data);
            showToast(`Loaded ${data.rows.toLocaleString()} rows`, 'success');
        }
    } catch (err) {
        hideLoading();
        showToast('Upload failed: ' + err.message, 'error');
    }
}

// ── Sample Datasets ──

async function loadSample(name) {
    showLoading(`Loading ${name} dataset...`);
    try {
        const res = await fetch('/api/load-sample', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ dataset: name })
        });
        const data = await res.json();
        hideLoading();

        if (data.error) { showToast(data.error, 'error'); return; }
        dataPreview = data;
        renderDataPreview(data);
        showToast(`Loaded ${name} (${data.rows} rows)`, 'success');
    } catch (err) {
        hideLoading();
        showToast('Failed: ' + err.message, 'error');
    }
}

// ── Cloud Import ──

async function importFromUrl() {
    const url = document.getElementById('urlInput').value.trim();
    if (!url) {
        showToast('Please enter a valid URL', 'error');
        return;
    }

    showLoading('Importing from URL...');
    try {
        const res = await fetch('/api/data/import_url', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url })
        });
        const data = await res.json();
        hideLoading();

        if (data.error) { showToast(data.error, 'error'); return; }

        dataPreview = data;
        renderDataPreview(data);
        showToast(`Imported ${data.rows.toLocaleString()} rows`, 'success');
        document.getElementById('urlInput').value = ''; // clear
    } catch (err) {
        hideLoading();
        showToast('Import failed: ' + err.message, 'error');
    }
}

// ═══════════════════════════════════════════════════════
// Multi-File Data Configuration
// ═══════════════════════════════════════════════════════

let uploadedFilesMeta = []; // Store file metadata for configuration

function showMergePanel(files) {
    uploadedFilesMeta = files;
    const panel = document.getElementById('mergePanel');
    const list = document.getElementById('fileRolesList');

    list.innerHTML = files.map((f, i) => {
        // Auto-detect likely role from filename
        const nameLower = f.name.toLowerCase();
        let defaultRole = 'train';
        if (nameLower.includes('test')) defaultRole = 'test';
        else if (nameLower.includes('rul') || nameLower.includes('meta') || nameLower.includes('label')) defaultRole = 'supplement';

        return `
        <div class="file-role-card">
            <div class="file-role-info">
                <span class="file-role-icon">📄</span>
                <div>
                    <strong>${f.name}</strong>
                    <span class="file-role-stats">${f.rows.toLocaleString()} rows · ${f.n_columns} columns</span>
                </div>
            </div>
            <select class="select-input file-role-select" id="fileRole_${i}" onchange="onRoleChange()">
                <option value="train" ${defaultRole === 'train' ? 'selected' : ''}>🏋️ Training Data</option>
                <option value="test" ${defaultRole === 'test' ? 'selected' : ''}>🧪 Test Data</option>
                <option value="supplement" ${defaultRole === 'supplement' ? 'selected' : ''}>🔗 Supplementary (merge)</option>
                <option value="ignore">🚫 Ignore</option>
            </select>
        </div>`;
    }).join('');

    panel.classList.remove('hidden');
    showToast(`${files.length} files uploaded — assign roles to continue`, 'info');
    onRoleChange();
}

function onRoleChange() {
    const hasSupp = uploadedFilesMeta.some((_, i) =>
        document.getElementById(`fileRole_${i}`).value === 'supplement'
    );
    document.getElementById('supplementMergeRow').classList.toggle('hidden', !hasSupp);

    const trainCols = new Set();
    uploadedFilesMeta.forEach((f, i) => {
        const role = document.getElementById(`fileRole_${i}`).value;
        if (role === 'train' || role === 'test') {
            f.columns.forEach(c => trainCols.add(c));
        }
    });

    const section = document.getElementById('columnExcludeSection');
    const checklist = document.getElementById('columnChecklist');
    if (trainCols.size > 0) {
        checklist.innerHTML = [...trainCols].map(col => `
            <label class="checkbox-label column-check-item">
                <input type="checkbox" checked value="${col}">
                ${col}
            </label>
        `).join('');
        section.classList.remove('hidden');
    } else {
        section.classList.add('hidden');
    }
}

async function applyDataConfig() {
    const files = uploadedFilesMeta.map((f, i) => ({
        name: f.name,
        role: document.getElementById(`fileRole_${i}`).value
    }));

    const excludeColumns = [];
    document.querySelectorAll('#columnChecklist input[type=checkbox]').forEach(cb => {
        if (!cb.checked) excludeColumns.push(cb.value);
    });

    const mergeKey = document.getElementById('supplementMergeKey').value || null;
    const mergeConfig = mergeKey ? { on: mergeKey, how: 'inner' } : {};

    showLoading('Configuring data...');
    try {
        const res = await fetch('/api/data/configure', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ files, merge_config: mergeConfig, exclude_columns: excludeColumns })
        });
        const data = await res.json();
        hideLoading();

        if (data.error) { showToast(data.error, 'error'); return; }

        dataPreview = data;
        renderDataPreview(data);
        document.getElementById('mergePanel').classList.add('hidden');

        let msg = `Configured! ${data.rows} rows, ${data.n_columns} columns`;
        if (data.has_test_data) msg += ` (+ separate test set: ${data.test_rows} rows)`;
        showToast(msg, 'success');
    } catch (err) {
        hideLoading();
        showToast('Configuration failed: ' + err.message, 'error');
    }
}

function toggleMergeKey() {}
function mergeFiles() { applyDataConfig(); }


// ═══════════════════════════════════════════════════════
// Data Preview Rendering
// ═══════════════════════════════════════════════════════

function renderDataPreview(data) {
    const preview = document.getElementById('dataPreview');
    preview.classList.remove('hidden');

    // Stats
    const stats = document.getElementById('dataStats');
    stats.innerHTML = `
        <span class="stat">📊 <strong>${data.rows.toLocaleString()}</strong> rows</span>
        <span class="stat">📋 <strong>${data.n_columns}</strong> columns</span>
        <span class="stat">${data.total_missing ? '⚠️' : '✅'} <strong>${data.total_missing}</strong> missing</span>
        <span class="stat">${data.duplicates ? '⚠️' : '✅'} <strong>${data.duplicates}</strong> duplicates</span>
    `;

    // Column cards
    const cards = document.getElementById('columnCards');
    cards.innerHTML = data.columns.slice(0, 12).map(col => `
        <div class="col-card">
            <div class="col-name">${col.name}</div>
            <div class="col-type">${col.dtype}</div>
            <div class="col-info">${col.unique} unique${col.missing > 0 ? ` • ${col.missing} missing` : ''}</div>
        </div>
    `).join('') + (data.columns.length > 12 ? `<div class="col-card col-more">+${data.columns.length - 12} more</div>` : '');

    // Preview table
    const head = document.getElementById('previewHead');
    const body = document.getElementById('previewBody');
    if (data.head && data.head.length > 0) {
        const cols = Object.keys(data.head[0]);
        head.innerHTML = `<tr>${cols.map(c => `<th>${c}</th>`).join('')}</tr>`;
        body.innerHTML = data.head.slice(0, 20).map(row =>
            `<tr>${cols.map(c => `<td>${row[c] != null ? row[c] : '<span class="null-val">null</span>'}</td>`).join('')}</tr>`
        ).join('');
    }

    // Populate target selector
    const targetSelect = document.getElementById('targetCol');
    targetSelect.innerHTML = data.columns.map(c =>
        `<option value="${c.name}">${c.name} (${c.dtype})</option>`
    ).join('');
}

// ═══════════════════════════════════════════════════════
// Pipeline Execution
// ═══════════════════════════════════════════════════════

async function planPipelineWithAI() {
    const targetCol = document.getElementById('targetCol').value;
    if (!targetCol) {
        showToast('Please select a target column first', 'error');
        return;
    }

    const taskRadios = document.querySelectorAll('[name="task"]');
    let task = null;
    taskRadios.forEach(r => { if (r.checked && r.value !== 'auto') task = r.value; });
    
    const btn = document.getElementById('btnAiPlan');
    const originalText = btn.innerHTML;
    btn.innerHTML = '<div class="spinner" style="width:16px;height:16px;margin:auto;"></div>';
    btn.disabled = true;
    
    try {
        const res = await fetch('/api/pipeline/ai-plan', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ target_col: targetCol, task: task || 'classification' })
        });
        
        const data = await res.json();
        
        if (data.error) {
            showToast('AI Planning failed: ' + data.error, 'error');
        } else if (data.config) {
            const cfg = data.config;
            
            // Set Task
            if (cfg.task) {
                document.querySelectorAll('[name="task"]').forEach(r => {
                    if (r.value === cfg.task) r.checked = true;
                });
            }
            
            // Set Data Cleaning
            if (cfg.cleaning) {
                if (cfg.cleaning.missing_values) document.getElementById('cfgImputation').value = cfg.cleaning.missing_values;
                if (cfg.cleaning.outliers) document.getElementById('cfgOutlier').value = cfg.cleaning.outliers;
            }
            
            // Set Transformation
            if (cfg.transformation) {
                if (cfg.transformation.encoding) document.getElementById('cfgEncoding').value = cfg.transformation.encoding;
                if (cfg.transformation.scaling) document.getElementById('cfgScaling').value = cfg.transformation.scaling;
            }
            
            // Set Feature Engineering
            if (cfg.feature_engineering) {
                if (cfg.feature_engineering.polynomial !== undefined) document.getElementById('cfgPolynomial').checked = cfg.feature_engineering.polynomial;
                if (cfg.feature_engineering.interactions !== undefined) document.getElementById('cfgInteractions').checked = cfg.feature_engineering.interactions;
                if (cfg.feature_engineering.selection) document.getElementById('cfgSelection').value = cfg.feature_engineering.selection === 'none' ? '' : cfg.feature_engineering.selection;
            }
            
            // Set Imbalance
            if (cfg.imbalance && cfg.imbalance.method) {
                document.getElementById('cfgImbalance').value = cfg.imbalance.method;
            }
            
            // Set Models
            if (cfg.models && Array.isArray(cfg.models)) {
                document.querySelectorAll('#modelGrid input[type="checkbox"]').forEach(cb => {
                    cb.checked = cfg.models.includes(cb.value);
                });
            }
            
            // Set Tuning
            if (cfg.tuning && cfg.tuning.enable) {
                document.getElementById('cfgTuning').value = 'bayesian';
            } else {
                document.getElementById('cfgTuning').value = '';
            }
            
            // Auto-Explore logic: Disable auto-explore if using AI plan
            document.getElementById('cfgAutoExplore').checked = false;
            
            // Show Reasoning
            if (data.reasoning) {
                const reasoningPanel = document.getElementById('aiReasoningPanel');
                const reasoningContent = document.getElementById('aiReasoningContent');
                reasoningContent.innerHTML = marked.parse(data.reasoning);
                reasoningPanel.classList.remove('hidden');
            }
            
            showToast('AI pipeline configuration applied successfully!', 'success');
        }
    } catch (err) {
        showToast('AI Planning failed: ' + err.message, 'error');
    } finally {
        btn.innerHTML = originalText;
        btn.disabled = false;
    }
}


async function runPipeline() {
    const targetCol = document.getElementById('targetCol').value;
    if (!targetCol) { showToast('Select a target column', 'error'); return; }

    goToStep(2);

    const taskRadios = document.querySelectorAll('[name="task"]');
    let task = null;
    taskRadios.forEach(r => { if (r.checked && r.value !== 'auto') task = r.value; });

    // Build config
    const config = {
        target_col: targetCol,
        task,
        auto_explore: document.getElementById('cfgAutoExplore').checked,
        cleaning: {
            imputation: document.getElementById('cfgImputation').value,
            outlier_method: document.getElementById('cfgOutlier').value,
            outlier_action: 'clip',
            remove_duplicates: document.getElementById('cfgDedup').checked,
        },
        transformation: {
            encoding: document.getElementById('cfgEncoding').value,
            scaling: document.getElementById('cfgScaling').value,
        },
        feature_engineering: {
            time_features: document.getElementById('cfgTimeFeatures').checked,
            polynomial: document.getElementById('cfgPolynomial').checked,
            interactions: document.getElementById('cfgInteractions').checked,
            selection_method: document.getElementById('cfgSelection').value || null,
        },
        splitting: {
            method: document.getElementById('cfgSplit').value,
            test_size: parseInt(document.getElementById('cfgTestSize').value) / 100,
        },
    };

    // Imbalance
    const imbalance = document.getElementById('cfgImbalance').value;
    if (imbalance !== 'none') config.imbalance = { method: imbalance };

    // Models
    const selectedModels = [];
    document.querySelectorAll('#modelGrid input[type="checkbox"]:checked').forEach(cb => {
        selectedModels.push(cb.value);
    });
    if (selectedModels.length > 0) config.models = selectedModels;

    // Tuning
    const tuning = document.getElementById('cfgTuning').value;
    if (tuning) config.tuning = { method: tuning, n_trials: 30 };

    // Animate progress
    animateProgress();

    try {
        const res = await fetch('/api/pipeline/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        const report = await res.json();

        // Better error detection: check for both top-level and nested pipeline error
        const pipelineErr = report.error || report.pipeline?.error;

        if (pipelineErr) {
            stopProgress();
            if (report.ai_explanation) {
                showErrorModal(report.ai_explanation);
            } else {
                showToast('Pipeline error: ' + pipelineErr, 'error');
            }
            console.error('Pipeline failed:', pipelineErr, report.traceback || report.pipeline?.traceback);
            return;
        }

        pipelineReport = report;
        finishProgress();
        setTimeout(() => {
            renderResults(report);
            goToStep(3);
        }, 600);
    } catch (err) {
        stopProgress();
        showToast('Pipeline failed: ' + err.message, 'error');
    }
}

// ── Progress Animation ──

let progressInterval = null;
let progressValue = 0;

function animateProgress() {
    progressValue = 0;
    const ring = document.getElementById('progressRing');
    const text = document.getElementById('progressText');
    const status = document.getElementById('progressStatus');
    const circumference = 2 * Math.PI * 85;
    ring.style.strokeDasharray = circumference;

    const steps = [
        'Cleaning data...', 'Transforming features...',
        'Engineering features...', 'Splitting data...',
        'Training models...', 'Evaluating performance...',
        'Generating insights...'
    ];

    let stepIdx = 0;
    progressInterval = setInterval(() => {
        if (progressValue < 90) {
            progressValue += Math.random() * 3 + 1;
            progressValue = Math.min(progressValue, 90);
        }
        const offset = circumference - (progressValue / 100) * circumference;
        ring.style.strokeDashoffset = offset;
        text.textContent = Math.round(progressValue) + '%';

        if (progressValue > (stepIdx + 1) * (90 / steps.length) && stepIdx < steps.length - 1) {
            stepIdx++;
        }
        status.textContent = steps[stepIdx];
    }, 300);
}

function finishProgress() {
    if (progressInterval) clearInterval(progressInterval);
    progressValue = 100;
    const ring = document.getElementById('progressRing');
    const text = document.getElementById('progressText');
    const status = document.getElementById('progressStatus');
    const circumference = 2 * Math.PI * 85;
    ring.style.strokeDashoffset = 0;
    text.textContent = '100%';
    status.textContent = 'Complete! ✅';
}

function stopProgress() {
    if (progressInterval) clearInterval(progressInterval);
}

// ═══════════════════════════════════════════════════════
// Results Rendering
// ═══════════════════════════════════════════════════════

function renderResults(report) {
    renderExploreResults(report);
    renderBestModel(report);
    renderLeaderboard(report);
    renderCharts(report);
    renderFeatureInsights(report);
    renderShapGlobal(report);
    renderPDPSelector(report);
    renderPipelineReasoning(report);
    renderWhatIfForm();
    renderInsights(report);
    renderPipelineSummary(report);
    loadRunHistory();
}

// ── Auto-Explore Results ──

function renderExploreResults(report) {
    const container = document.getElementById('exploreResults');
    const grid = document.getElementById('exploreGrid');
    const explore = report.pipeline?.auto_explore;

    if (!explore) {
        container.classList.add('hidden');
        return;
    }

    container.classList.remove('hidden');
    const { details, winner } = explore;

    grid.innerHTML = Object.entries(details).map(([name, d]) => `
        <div class="explore-preset ${name === winner ? 'winner' : ''}">
            <div class="preset-name">${d.label || name} ${name === winner ? '🏆' : ''}</div>
            <div class="preset-score">${d.cv_score != null ? d.cv_score.toFixed(4) : 'ERR'}</div>
            <div class="preset-detail">
                ${d.cv_std != null ? `±${d.cv_std.toFixed(4)}` : d.error || ''}
                ${d.n_features ? ` • ${d.n_features} features` : ''}
            </div>
        </div>
    `).join('');
}

// ── Best Model Hero Card ──

function renderBestModel(report) {
    const card = document.getElementById('bestModelCard');
    const best = report.best_model;
    const comparison = report.model_comparison || {};
    const metrics = comparison[best];

    if (!best || !metrics || metrics.error) {
        card.innerHTML = `
            <div class="empty-state">
                <span class="empty-icon">⚠️</span>
                <p>No models trained successfully.</p>
                ${report.pipeline?.error ? `<p class="error-detail">${report.pipeline.error}</p>` : ''}
            </div>
        `;
        return;
    }

    const display = metrics.display_name || best;
    const task = report.pipeline?.task;
    const primaryMetric = task === 'classification' ? 'f1' : 'r2';
    const primaryValue = metrics[primaryMetric] ?? metrics.accuracy ?? '?';

    // Metric cards
    const metricKeys = task === 'classification'
        ? ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        : ['r2', 'mse', 'rmse', 'mae'];

    const metricCards = metricKeys
        .filter(k => metrics[k] != null)
        .map(k => `
            <div class="metric-item">
                <div class="metric-label">${k.toUpperCase()}</div>
                <div class="metric-value">${typeof metrics[k] === 'number' ? metrics[k].toFixed(4) : metrics[k]}</div>
            </div>
        `).join('');

    card.innerHTML = `
        <div class="best-model-header">
            <span class="trophy">🏆</span>
            <div>
                <h3 class="best-model-name">${display}</h3>
                <p class="best-model-score">${primaryMetric.toUpperCase()}: ${typeof primaryValue === 'number' ? primaryValue.toFixed(4) : primaryValue}</p>
            </div>
        </div>
        <div class="best-model-metrics">${metricCards}</div>
    `;
}

// ── Leaderboard ──

function renderLeaderboard(report) {
    const head = document.getElementById('leaderboardHead');
    const body = document.getElementById('leaderboardBody');
    const comparison = report.model_comparison || {};
    const task = report.pipeline?.task;

    const cols = task === 'classification'
        ? ['Rank', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'Time']
        : ['Rank', 'Model', 'R²', 'RMSE', 'MAE', 'Time'];

    head.innerHTML = `<tr>${cols.map(c => `<th>${c}</th>`).join('')}</tr>`;

    const sorted = Object.entries(comparison)
        .filter(([, m]) => !m.error)
        .sort((a, b) => (a[1].rank || 99) - (b[1].rank || 99));

    body.innerHTML = sorted.map(([name, m]) => {
        const cells = task === 'classification' ? [
            m.rank || '-',
            m.display_name || name,
            fmt(m.accuracy), fmt(m.precision), fmt(m.recall), fmt(m.f1),
            m.train_time != null ? m.train_time.toFixed(1) + 's' : '-'
        ] : [
            m.rank || '-',
            m.display_name || name,
            fmt(m.r2), fmt(m.rmse), fmt(m.mae),
            m.train_time != null ? m.train_time.toFixed(1) + 's' : '-'
        ];
        const isFirst = m.rank === 1;
        return `<tr class="${isFirst ? 'best-row' : ''}">${cells.map(c => `<td>${c}</td>`).join('')}</tr>`;
    }).join('');
}

function fmt(val) {
    if (val == null) return '-';
    return typeof val === 'number' ? val.toFixed(4) : val;
}

// ── Charts ──

function renderCharts(report) {
    const comparison = report.model_comparison || {};
    const task = report.pipeline?.task;
    const primaryMetric = task === 'classification' ? 'f1' : 'r2';

    const sorted = Object.entries(comparison)
        .filter(([, m]) => !m.error && m[primaryMetric] != null)
        .sort((a, b) => (b[1][primaryMetric] || 0) - (a[1][primaryMetric] || 0));

    const labels = sorted.map(([n, m]) => m.display_name || n);
    const scores = sorted.map(([, m]) => m[primaryMetric]);
    const times = sorted.map(([, m]) => m.train_time || 0);

    // Comparison chart
    if (compChart) compChart.destroy();
    const compCtx = document.getElementById('comparisonChart').getContext('2d');
    compChart = new Chart(compCtx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: primaryMetric.toUpperCase(),
                data: scores,
                backgroundColor: scores.map((_, i) =>
                    i === 0 ? 'rgba(16, 185, 129, 0.7)' : 'rgba(99, 102, 241, 0.5)'),
                borderColor: scores.map((_, i) =>
                    i === 0 ? '#10b981' : '#6366f1'),
                borderWidth: 1,
                borderRadius: 6,
            }]
        },
        options: {
            responsive: true,
            plugins: { legend: { display: false } },
            scales: {
                y: { beginAtZero: true, max: 1, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
                x: { grid: { display: false }, ticks: { color: '#94a3b8', maxRotation: 45 } }
            }
        }
    });

    // Time chart
    if (timeChart) timeChart.destroy();
    const timeCtx = document.getElementById('timeChart').getContext('2d');
    timeChart = new Chart(timeCtx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Training Time (s)',
                data: times,
                backgroundColor: 'rgba(245, 158, 11, 0.5)',
                borderColor: '#f59e0b',
                borderWidth: 1,
                borderRadius: 6,
            }]
        },
        options: {
            responsive: true,
            plugins: { legend: { display: false } },
            scales: {
                y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
                x: { grid: { display: false }, ticks: { color: '#94a3b8', maxRotation: 45 } }
            }
        }
    });
}

// ── Feature Insights ──

function renderFeatureInsights(report) {
    const section = document.getElementById('featureInsightsSection');
    const pipeline = report.pipeline || {};
    const featEng = report.feature_engineering || {};
    const allImportances = pipeline.feature_importance || {};
    const featuresUsed = pipeline.features_used || [];

    // ── 1. Feature Importance Chart ──
    cachedFeatureImportance = allImportances;
    const modelSelect = document.getElementById('featureImpModelSelect');
    const modelNames = Object.keys(allImportances);

    if (modelNames.length === 0) {
        // No importance data — try to get from feature selection report
        const selected = featEng.selected || {};
        let fallbackScores = null;
        let fallbackMethod = '';
        for (const [method, info] of Object.entries(selected)) {
            const scores = info.scores || info.importances;
            if (scores && Object.keys(scores).length > 0) {
                fallbackScores = scores;
                fallbackMethod = method;
                break;
            }
        }
        if (fallbackScores) {
            cachedFeatureImportance = { [fallbackMethod + ' (selection)']: fallbackScores };
            modelSelect.innerHTML = `<option>${fallbackMethod} (selection)</option>`;
            drawFeatureChart(fallbackScores);
        } else {
            // Completely hide chart if no data
            const chartCard = document.querySelector('.feature-chart-card');
            if (chartCard) chartCard.style.display = 'none';
        }
    } else {
        // Prefer best model first
        const best = report.best_model;
        const orderedNames = best && allImportances[best]
            ? [best, ...modelNames.filter(n => n !== best)]
            : modelNames;

        modelSelect.innerHTML = orderedNames.map(n => {
            const display = n.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
            return `<option value="${n}">${display}${n === best ? ' 🏆' : ''}</option>`;
        }).join('');

        // Draw chart for the first (best) model
        drawFeatureChart(allImportances[orderedNames[0]]);
    }

    // ── 2. Features Used Tags ──
    const tagsContainer = document.getElementById('featureTags');
    const countContainer = document.getElementById('featureCount');

    if (featuresUsed.length > 0) {
        tagsContainer.innerHTML = featuresUsed.map(f =>
            `<span class="feature-tag">${f}</span>`
        ).join('');
        countContainer.textContent = `${featuresUsed.length} feature${featuresUsed.length !== 1 ? 's' : ''} used in training`;
    } else {
        tagsContainer.innerHTML = '<span class="text-muted">Feature list not available</span>';
        countContainer.textContent = '';
    }

    // ── 3. Feature Engineering Summary ──
    const summaryContainer = document.getElementById('featureEngineeringSummary');
    const items = [];

    // Created features
    const created = featEng.created || {};
    if (created.polynomial && created.polynomial.count > 0) {
        items.push(`✨ Created <strong>${created.polynomial.count}</strong> polynomial features (degree ${created.polynomial.degree || 2})`);
    }
    if (created.interactions && created.interactions.count > 0) {
        items.push(`✨ Created <strong>${created.interactions.count}</strong> interaction features`);
    }
    if (created.time && created.time.features) {
        items.push(`⏰ Extracted <strong>${created.time.features.length}</strong> time features from <code>${created.time.source_column}</code>`);
    }
    if (created.aggregations && created.aggregations.count > 0) {
        items.push(`📊 Created <strong>${created.aggregations.count}</strong> rolling aggregation features`);
    }

    // Feature selection
    const selected = featEng.selected || {};
    for (const [method, info] of Object.entries(selected)) {
        const n = info.count || (info.selected || []).length;
        items.push(`🎯 Feature selection (<strong>${method}</strong>) kept <strong>${n}</strong> features`);
    }

    // Actions summary
    const actions = featEng.actions || [];
    for (const action of actions) {
        if (!items.some(i => i.includes(action.split(' ').slice(-2).join(' ')))) {
            items.push(`▸ ${action}`);
        }
    }

    if (items.length === 0) {
        items.push('No feature engineering or selection was applied — all original numeric features were used as-is.');
    }

    summaryContainer.innerHTML = items.map(i => `<div class="feature-summary-item">${i}</div>`).join('');

    section.classList.remove('hidden');
}

function drawFeatureChart(importances) {
    if (!importances || Object.keys(importances).length === 0) return;

    // Take top 15 features
    const entries = Object.entries(importances)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 15);

    const labels = entries.map(([name]) => name.length > 25 ? name.slice(0, 22) + '...' : name);
    const values = entries.map(([, val]) => val);
    const maxVal = Math.max(...values);

    // Color gradient — top feature is bright, rest fade
    const colors = values.map((v, i) => {
        const ratio = v / (maxVal || 1);
        if (i === 0) return 'rgba(16, 185, 129, 0.85)';
        return `rgba(99, 102, 241, ${0.3 + ratio * 0.5})`;
    });
    const borderColors = values.map((v, i) => {
        if (i === 0) return '#10b981';
        return '#6366f1';
    });

    if (featureImpChart) featureImpChart.destroy();

    const ctx = document.getElementById('featureImportanceChart').getContext('2d');
    featureImpChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Importance',
                data: values,
                backgroundColor: colors,
                borderColor: borderColors,
                borderWidth: 1,
                borderRadius: 4,
                borderSkipped: false,
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        title: (items) => entries[items[0].dataIndex][0],
                        label: (item) => `Importance: ${item.raw.toFixed(6)}`
                    }
                }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#94a3b8', font: { size: 11 } }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#cbd5e1', font: { size: 11 } }
                }
            }
        }
    });
}

function updateFeatureChart() {
    const select = document.getElementById('featureImpModelSelect');
    const modelName = select.value;
    if (cachedFeatureImportance && cachedFeatureImportance[modelName]) {
        drawFeatureChart(cachedFeatureImportance[modelName]);
    }
}

// ── AI Insights ──

function renderInsights(report) {
    const grid = document.getElementById('insightsGrid');
    const exp = report.explanations || {};

    if (!Object.keys(exp).length) {
        grid.innerHTML = '<p style="color:var(--text-muted)">No insights available.</p>';
        return;
    }

    const sections = [
        { key: 'data', icon: '📊', title: 'Data Quality' },
        { key: 'preprocessing', icon: '🔄', title: 'Preprocessing' },
        { key: 'best_model', icon: '🏆', title: 'Best Model' },
        { key: 'overfitting_check', icon: '🛡️', title: 'Overfitting Check' },
        { key: 'feature_importance', icon: '📈', title: 'Feature Importance' },
        { key: 'suggestions', icon: '💡', title: 'Suggestions' },
    ];

    grid.innerHTML = sections
        .filter(s => exp[s.key] && exp[s.key].length)
        .map(s => `
            <div class="insight-card">
                <h4>${s.icon} ${s.title}</h4>
                <ul>
                    ${exp[s.key].map(item => `<li>${simpleMarkdown(item)}</li>`).join('')}
                </ul>
            </div>
        `).join('');
}

function simpleMarkdown(text) {
    // Convert **bold** and `code` to HTML
    return text
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.+?)\*/g, '<em>$1</em>')
        .replace(/`(.+?)`/g, '<code>$1</code>');
}

// ═══════════════════════════════════════════════════════
// SHAP Global Importance Chart
// ═══════════════════════════════════════════════════════

function renderShapGlobal(report) {
    const panel = document.getElementById('shapGlobalPanel');
    const shapData = report.pipeline?.shap_global;

    if (!shapData || shapData.error) {
        panel.classList.add('hidden');
        return;
    }

    panel.classList.remove('hidden');
    const methodBadge = document.getElementById('shapMethodBadge');
    methodBadge.innerHTML = `Computed via <strong>${shapData.method || 'SHAP'}</strong>`;

    const entries = Object.entries(shapData.mean_abs_shap || {})
        .sort((a, b) => b[1] - a[1])
        .slice(0, 15);

    const labels = entries.map(([name]) => name.length > 25 ? name.slice(0, 22) + '...' : name);
    const values = entries.map(([, val]) => val);

    if (shapGlobalChart) shapGlobalChart.destroy();

    const ctx = document.getElementById('shapGlobalChart').getContext('2d');
    shapGlobalChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Mean |SHAP|',
                data: values,
                backgroundColor: values.map((_, i) => {
                    const hue = 250 - (i / values.length) * 120;
                    return `hsla(${hue}, 80%, 65%, 0.7)`;
                }),
                borderColor: values.map((_, i) => {
                    const hue = 250 - (i / values.length) * 120;
                    return `hsla(${hue}, 80%, 55%, 1)`;
                }),
                borderWidth: 1,
                borderRadius: 4,
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: {
                    beginAtZero: true,
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#94a3b8', font: { size: 11 } },
                    title: { display: true, text: 'Mean |SHAP Value|', color: '#94a3b8' }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#cbd5e1', font: { size: 11 } }
                }
            }
        }
    });
}

// ═══════════════════════════════════════════════════════
// Partial Dependence Plots
// ═══════════════════════════════════════════════════════

function renderPDPSelector(report) {
    const select = document.getElementById('pdpFeatureSelect');
    const features = report.pipeline?.features_used || [];

    if (features.length === 0) {
        document.getElementById('pdpPanel').classList.add('hidden');
        return;
    }

    select.innerHTML = features.slice(0, 20).map(f =>
        `<option value="${f}">${f}</option>`
    ).join('');
}

async function loadPDP() {
    const select = document.getElementById('pdpFeatureSelect');
    const feature = select.value;
    if (!feature) return;

    try {
        const res = await fetch('/api/explain/pdp', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ feature })
        });
        const data = await res.json();
        if (data.error) { showToast('PDP error: ' + data.error, 'error'); return; }
        drawPDPChart(data);
    } catch (err) {
        showToast('PDP failed: ' + err.message, 'error');
    }
}

function drawPDPChart(data) {
    if (pdpChart) pdpChart.destroy();

    const ctx = document.getElementById('pdpChart').getContext('2d');
    pdpChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.values.map(v => v.toFixed(2)),
            datasets: [{
                label: `Effect of ${data.feature}`,
                data: data.predictions,
                borderColor: '#8b5cf6',
                backgroundColor: 'rgba(139, 92, 246, 0.1)',
                fill: true,
                tension: 0.3,
                pointRadius: 2,
                borderWidth: 2,
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { display: false },
                title: {
                    display: true,
                    text: `Partial Dependence: ${data.feature}`,
                    color: '#e2e8f0',
                    font: { size: 13, weight: '600' }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#94a3b8', maxTicksLimit: 10 },
                    title: { display: true, text: data.feature, color: '#94a3b8' }
                },
                y: {
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#94a3b8' },
                    title: { display: true, text: 'Avg. Prediction', color: '#94a3b8' }
                }
            }
        }
    });
}

// ═══════════════════════════════════════════════════════
// Pipeline Design Reasoning
// ═══════════════════════════════════════════════════════

function renderPipelineReasoning(report) {
    const grid = document.getElementById('reasoningGrid');
    const exp = report.explanations || {};
    const reasoning = exp.pipeline_reasoning || [];

    if (!reasoning.length) {
        document.getElementById('pipelineReasoningSection').classList.add('hidden');
        return;
    }

    grid.innerHTML = reasoning.map(item => {
        // Extract the bold title from the reasoning item
        const match = item.match(/^\*\*(.+?):\*\*\s*(.*)$/s);
        const title = match ? match[1] : '';
        const body = match ? match[2] : item;
        const icon = getReasoningIcon(title);

        return `
            <div class="reasoning-card">
                <div class="reasoning-header">
                    <span class="reasoning-icon">${icon}</span>
                    <strong>${title || 'Insight'}</strong>
                </div>
                <p>${simpleMarkdown(body)}</p>
            </div>
        `;
    }).join('');
}

function getReasoningIcon(title) {
    const icons = {
        'Task Detection': '🎯',
        'Imputation Choice': '🩹',
        'Outlier Handling': '📊',
        'Scaling Decision': '⚖️',
        'Feature Selection': '✂️',
        'Model Selection Reasoning': '🏆',
        'Auto-Explore Result': '🔬',
    };
    return icons[title] || '💡';
}

// ═══════════════════════════════════════════════════════
// What-If Predictor
// ═══════════════════════════════════════════════════════

async function renderWhatIfForm() {
    const form = document.getElementById('whatIfForm');
    try {
        const res = await fetch('/api/predict/whatif-info');
        const data = await res.json();
        if (data.error) {
            form.innerHTML = '<p class="text-muted">What-If Predictor not available yet.</p>';
            return;
        }

        const features = data.features || [];
        let html = '<div class="whatif-fields">';
        for (const f of features.slice(0, 20)) {
            html += `
                <div class="whatif-field">
                    <label title="Range: ${f.min} — ${f.max}">${f.name}</label>
                    <input type="number" 
                        id="whatif_${f.name}" 
                        value="${f.median}" 
                        step="any" 
                        min="${f.min}" 
                        max="${f.max}"
                        class="whatif-input">
                    <span class="whatif-range">${f.min.toFixed(1)} — ${f.max.toFixed(1)}</span>
                </div>
            `;
        }
        html += '</div>';
        html += `<button class="btn-primary whatif-predict-btn" onclick="runWhatIfPrediction()">🎯 Predict</button>`;
        if (features.length > 20) {
            html += `<p class="text-muted" style="margin-top:0.5rem">Showing top 20 of ${features.length} features</p>`;
        }
        form.innerHTML = html;
    } catch (err) {
        form.innerHTML = '<p class="text-muted">Could not load feature info.</p>';
    }
}

async function runWhatIfPrediction() {
    const features = {};
    const inputs = document.querySelectorAll('.whatif-input');
    inputs.forEach(input => {
        const name = input.id.replace('whatif_', '');
        features[name] = parseFloat(input.value) || 0;
    });

    try {
        const res = await fetch('/api/predict/whatif', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ features })
        });
        const data = await res.json();
        if (data.error) { showToast('Prediction error: ' + data.error, 'error'); return; }

        renderWhatIfResult(data);
    } catch (err) {
        showToast('Prediction failed: ' + err.message, 'error');
    }
}

function renderWhatIfResult(data) {
    const card = document.getElementById('whatIfPredCard');
    const probas = data.probabilities;

    let probaHTML = '';
    if (probas && probas.length > 0) {
        probaHTML = `
            <div class="whatif-probabilities">
                <h5>Class Probabilities</h5>
                <div class="proba-bars">
                    ${probas.map((p, i) => `
                        <div class="proba-bar">
                            <span class="proba-label">Class ${i}</span>
                            <div class="proba-track">
                                <div class="proba-fill" style="width:${(p * 100).toFixed(0)}%"></div>
                            </div>
                            <span class="proba-value">${(p * 100).toFixed(1)}%</span>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }

    card.innerHTML = `
        <div class="whatif-prediction">
            <span class="whatif-pred-label">Prediction</span>
            <span class="whatif-pred-value">${data.prediction}</span>
        </div>
        ${probaHTML}
    `;

    // Draw SHAP waterfall
    const shapData = data.shap_explanation;
    if (shapData && shapData.contributions) {
        drawWhatIfShapChart(shapData);
        document.getElementById('whatIfShapContainer').classList.remove('hidden');
    }
}

function drawWhatIfShapChart(shapData) {
    const contributions = shapData.contributions.slice(0, 12);
    const labels = contributions.map(c => c.feature.length > 20 ? c.feature.slice(0, 17) + '...' : c.feature);
    const values = contributions.map(c => c.contribution);

    if (whatIfShapChart) whatIfShapChart.destroy();

    const ctx = document.getElementById('whatIfShapChart').getContext('2d');
    whatIfShapChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'SHAP Contribution',
                data: values,
                backgroundColor: values.map(v =>
                    v > 0 ? 'rgba(239, 68, 68, 0.6)' : 'rgba(59, 130, 246, 0.6)'
                ),
                borderColor: values.map(v =>
                    v > 0 ? '#ef4444' : '#3b82f6'
                ),
                borderWidth: 1,
                borderRadius: 4,
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (item) => `${item.raw > 0 ? '+' : ''}${item.raw.toFixed(4)} (${item.raw > 0 ? 'increases' : 'decreases'} prediction)`
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#94a3b8' },
                    title: { display: true, text: 'SHAP Value (impact on prediction)', color: '#94a3b8' }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#cbd5e1', font: { size: 11 } }
                }
            }
        }
    });
}

// ═══════════════════════════════════════════════════════
// Export Dropdown
// ═══════════════════════════════════════════════════════

function toggleExportMenu() {
    const menu = document.getElementById('exportMenu');
    menu.classList.toggle('hidden');
}

// Close export menu on outside click
document.addEventListener('click', (e) => {
    const dropdown = document.querySelector('.export-dropdown');
    const menu = document.getElementById('exportMenu');
    if (dropdown && menu && !dropdown.contains(e.target)) {
        menu.classList.add('hidden');
    }
});

// ── Pipeline Summary ──

function renderPipelineSummary(report) {
    const grid = document.getElementById('pipelineSummary');
    const p = report.pipeline || {};

    const items = [
        { label: 'Task', value: p.task || '?' },
        { label: 'Target', value: p.target_col || '?' },
        { label: 'Total Time', value: p.total_time ? p.total_time.toFixed(1) + 's' : '?' },
        { label: 'Steps', value: (p.steps_completed || []).join(' → ') },
        { label: 'Status', value: p.status === 'completed' ? '✅ completed' : p.status },
    ];

    // Checkpoints
    const checks = p.checkpoints || [];
    if (checks.length) {
        items.push({
            label: 'Checkpoints',
            value: checks.map(c => `${c.step}${c.rows ? ` (${c.rows}×${c.cols})` : ''}`).join(', ')
        });
    }

    grid.innerHTML = items.map(i => `
        <div class="summary-item">
            <span class="summary-label">${i.label}</span>
            <span class="summary-value">${i.value}</span>
        </div>
    `).join('');
}

// ═══════════════════════════════════════════════════════
// Chat Panel
// ═══════════════════════════════════════════════════════

function toggleChat() {
    const panel = document.getElementById('chatPanel');
    const toggle = document.getElementById('chatToggle');
    const icon = document.getElementById('chatToggleIcon');
    const isOpen = !panel.classList.contains('hidden');

    panel.classList.toggle('hidden');
    toggle.classList.toggle('active');
    icon.textContent = isOpen ? '💬' : '✕';

    if (!isOpen) {
        setTimeout(() => document.getElementById('chatInput').focus(), 100);
    }
}

function sendQuickChat(msg) {
    document.getElementById('chatInput').value = msg;
    sendChatMessage();
}

async function sendChatMessage() {
    const input = document.getElementById('chatInput');
    const msg = input.value.trim();
    if (!msg) return;

    addChatMessage(msg, 'user');
    input.value = '';

    try {
        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: msg })
        });
        const data = await res.json();

        addChatMessage(data.reply, 'bot');

        // Handle actions
        if (data.action) handleChatAction(data.action, data.params || {});
    } catch (err) {
        addChatMessage('Sorry, something went wrong. ❌', 'bot');
    }
}

function addChatMessage(text, sender) {
    const container = document.getElementById('chatMessages');
    const div = document.createElement('div');
    div.className = `chat-msg ${sender}`;
    div.innerHTML = `<div class="chat-bubble">${simpleMarkdown(text)}</div>`;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}

function handleChatAction(action, params) {
    switch (action) {
        case 'load_sample':
            loadSample(params.dataset);
            break;
        case 'set_target':
            const sel = document.getElementById('targetCol');
            if (sel) {
                sel.value = params.column;
                goToStep(1);
            }
            break;
        case 'run_pipeline':
            goToStep(1);
            setTimeout(() => runPipeline(), 300);
            break;
        case 'run_auto_explore':
            document.getElementById('cfgAutoExplore').checked = true;
            goToStep(1);
            setTimeout(() => runPipeline(), 300);
            break;
        case 'show_results':
            if (pipelineReport) goToStep(3);
            break;
        case 'show_explain':
            if (pipelineReport) {
                goToStep(3);
                setTimeout(() => {
                    document.getElementById('insightsSection').scrollIntoView({ behavior: 'smooth' });
                }, 400);
            }
            break;
    }
}

// ═══════════════════════════════════════════════════════
// Data Profiler (EDA)
// ═══════════════════════════════════════════════════════

async function profileData() {
    const modal = document.getElementById('profilerModal');
    const body = document.getElementById('profilerBody');
    modal.classList.remove('hidden');
    body.innerHTML = '<div class="profiler-loading"><div class="spinner"></div><p>Analyzing your data...</p></div>';

    try {
        const res = await fetch('/api/profile');
        const data = await res.json();
        if (data.error) { showToast('Profile error: ' + data.error, 'error'); modal.classList.add('hidden'); return; }
        renderProfileReport(data);
    } catch (err) {
        showToast('Profiling failed: ' + err.message, 'error');
        modal.classList.add('hidden');
    }
}

function closeProfiler() {
    document.getElementById('profilerModal').classList.add('hidden');
}

function renderProfileReport(data) {
    const body = document.getElementById('profilerBody');
    const ov = data.overview || {};
    const missing = data.missing || {};
    const cols = data.columns || [];
    const corr = data.correlations || {};
    const alerts = data.alerts || [];

    let html = '';

    // Overview cards
    html += '<div class="profiler-overview">';
    html += `<div class="prof-card"><span class="prof-num">${ov.rows?.toLocaleString() || 0}</span><span class="prof-label">Rows</span></div>`;
    html += `<div class="prof-card"><span class="prof-num">${ov.columns || 0}</span><span class="prof-label">Columns</span></div>`;
    html += `<div class="prof-card"><span class="prof-num">${ov.numeric_cols || 0}</span><span class="prof-label">Numeric</span></div>`;
    html += `<div class="prof-card"><span class="prof-num">${ov.categorical_cols || 0}</span><span class="prof-label">Categorical</span></div>`;
    html += `<div class="prof-card"><span class="prof-num">${ov.duplicates || 0}</span><span class="prof-label">Duplicates</span></div>`;
    html += `<div class="prof-card"><span class="prof-num">${ov.memory_mb || 0} MB</span><span class="prof-label">Memory</span></div>`;
    html += '</div>';

    // Alerts
    if (alerts.length > 0) {
        html += '<div class="profiler-alerts">';
        html += '<h3>⚠️ Data Quality Alerts</h3>';
        html += alerts.map(a => {
            const cls = a.level === 'danger' ? 'alert-danger' : a.level === 'warning' ? 'alert-warning' : a.level === 'success' ? 'alert-success' : 'alert-info';
            return `<div class="prof-alert ${cls}"><span class="prof-alert-badge">${a.category}</span> ${a.message}</div>`;
        }).join('');
        html += '</div>';
    }

    // Missing values
    if (missing.total > 0) {
        html += '<div class="profiler-section">';
        html += `<h3>🕳️ Missing Values <span class="badge-count">${missing.total} total</span></h3>`;
        html += '<div class="missing-bars">';
        html += (missing.details || []).map(d => `
            <div class="missing-row">
                <span class="missing-col">${d.column}</span>
                <div class="missing-track"><div class="missing-fill" style="width:${Math.min(d.pct, 100)}%"></div></div>
                <span class="missing-pct">${d.pct}% (${d.count})</span>
            </div>
        `).join('');
        html += '</div></div>';
    }

    // Column stats table
    html += '<div class="profiler-section">';
    html += '<h3>📋 Column Statistics</h3>';
    html += '<div class="table-wrapper"><table class="prof-table"><thead><tr>';
    html += '<th>Column</th><th>Type</th><th>Missing</th><th>Unique</th><th>Min</th><th>Max</th><th>Mean</th><th>Std</th>';
    html += '</tr></thead><tbody>';
    for (const c of cols) {
        html += '<tr>';
        html += `<td><strong>${c.name}</strong></td>`;
        html += `<td><span class="type-badge type-${c.type}">${c.type}</span></td>`;
        html += `<td>${c.missing_pct > 0 ? `<span class="text-warn">${c.missing_pct}%</span>` : '✅ 0%'}</td>`;
        html += `<td>${c.unique}${c.unique_pct ? ` (${c.unique_pct}%)` : ''}</td>`;
        html += `<td>${c.min !== undefined ? c.min : '—'}</td>`;
        html += `<td>${c.max !== undefined ? c.max : '—'}</td>`;
        html += `<td>${c.mean !== undefined ? c.mean : (c.mode || '—')}</td>`;
        html += `<td>${c.std !== undefined ? c.std : '—'}</td>`;
        html += '</tr>';
    }
    html += '</tbody></table></div></div>';

    // Top correlations
    const topPairs = corr.top_pairs || [];
    if (topPairs.length > 0) {
        html += '<div class="profiler-section">';
        html += '<h3>🔗 Top Correlations</h3>';
        html += '<div class="corr-list">';
        html += topPairs.slice(0, 10).map(p => {
            const strength = Math.abs(p.correlation);
            const color = strength > 0.7 ? 'corr-strong' : strength > 0.4 ? 'corr-moderate' : 'corr-weak';
            return `<div class="corr-pair ${color}">
                <span class="corr-cols">${p.col1} ↔ ${p.col2}</span>
                <span class="corr-val">${p.correlation}</span>
            </div>`;
        }).join('');
        html += '</div></div>';
    }

    // Heatmap (canvas-based)
    if (corr.matrix && corr.matrix.length > 1 && corr.matrix.length <= 20) {
        html += '<div class="profiler-section">';
        html += '<h3>🌡️ Correlation Heatmap</h3>';
        html += '<div class="heatmap-container"><canvas id="corrHeatmap"></canvas></div>';
        html += '</div>';
    }

    body.innerHTML = html;

    // Draw heatmap if present
    if (corr.matrix && corr.matrix.length > 1 && corr.matrix.length <= 20) {
        drawCorrelationHeatmap(corr.columns, corr.matrix);
    }
}

function drawCorrelationHeatmap(columns, matrix) {
    const canvas = document.getElementById('corrHeatmap');
    if (!canvas) return;
    const n = columns.length;
    const cellSize = Math.min(50, Math.max(25, 500 / n));
    const labelWidth = 100;
    const canvasW = labelWidth + n * cellSize;
    const canvasH = labelWidth + n * cellSize;
    canvas.width = canvasW;
    canvas.height = canvasH;
    const ctx = canvas.getContext('2d');

    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, canvasW, canvasH);

    // Draw cells
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            const val = matrix[i][j];
            const x = labelWidth + j * cellSize;
            const y = labelWidth + i * cellSize;

            // Color: blue (-1) -> white (0) -> red (+1)
            const r = val > 0 ? Math.round(220 * val + 30) : 30;
            const g = 30;
            const b = val < 0 ? Math.round(220 * Math.abs(val) + 30) : 30;
            const alpha = 0.3 + Math.abs(val) * 0.7;

            ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`;
            ctx.fillRect(x, y, cellSize - 1, cellSize - 1);

            // Value text
            if (cellSize >= 30) {
                ctx.fillStyle = '#e2e8f0';
                ctx.font = `${Math.max(9, cellSize / 4)}px Inter, sans-serif`;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(val.toFixed(1), x + cellSize / 2, y + cellSize / 2);
            }
        }
    }

    // Draw labels
    ctx.fillStyle = '#94a3b8';
    ctx.font = `${Math.max(9, cellSize / 3.5)}px Inter, sans-serif`;
    for (let i = 0; i < n; i++) {
        // Y-axis labels
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        const lbl = columns[i].length > 12 ? columns[i].slice(0, 10) + '..' : columns[i];
        ctx.fillText(lbl, labelWidth - 5, labelWidth + i * cellSize + cellSize / 2);
        // X-axis labels
        ctx.save();
        ctx.translate(labelWidth + i * cellSize + cellSize / 2, labelWidth - 5);
        ctx.rotate(-Math.PI / 4);
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        ctx.fillText(lbl, 0, 0);
        ctx.restore();
    }
}


// ═══════════════════════════════════════════════════════
// Experiment History
// ═══════════════════════════════════════════════════════

async function loadRunHistory() {
    try {
        const res = await fetch('/api/runs/history');
        const data = await res.json();
        if (data.runs) renderRunHistory(data.runs);
    } catch (err) {
        console.warn('Could not load run history:', err);
    }
}

function renderRunHistory(runs) {
    const head = document.getElementById('runHistoryHead');
    const body = document.getElementById('runHistoryBody');

    head.innerHTML = '<tr><th>#</th><th>Timestamp</th><th>Task</th><th>Target</th><th>Best Model</th><th>Score</th><th></th></tr>';

    if (!runs || runs.length === 0) {
        body.innerHTML = '<tr><td colspan="7" class="empty-state">No runs recorded yet. Train a model to get started!</td></tr>';
        return;
    }

    body.innerHTML = runs.map((run, i) => {
        const ts = run.timestamp ? new Date(run.timestamp).toLocaleString() : '?';
        const metrics = run.metrics || {};
        const score = metrics.f1 !== undefined ? `F1: ${Number(metrics.f1).toFixed(3)}` :
            metrics.r2 !== undefined ? `R²: ${Number(metrics.r2).toFixed(3)}` :
                metrics.accuracy !== undefined ? `Acc: ${Number(metrics.accuracy).toFixed(3)}` : '—';
        const taskBadge = run.task_type === 'classification' ? '<span class="badge-cls">CLS</span>' : '<span class="badge-reg">REG</span>';
        const exploreBadge = run.auto_explore ? ' <span class="badge-explore">Auto</span>' : '';

        return `<tr>
            <td class="run-id">${run.id}</td>
            <td class="run-ts">${ts}</td>
            <td>${taskBadge}${exploreBadge}</td>
            <td><code>${run.target_col || '?'}</code></td>
            <td><strong>${(run.best_model || '?').replace(/_/g, ' ')}</strong></td>
            <td class="run-score">${score}</td>
            <td><button class="btn-icon btn-delete" onclick="deleteRunEntry('${run.id}')" title="Delete">🗑</button></td>
        </tr>`;
    }).join('');
}

async function deleteRunEntry(id) {
    try {
        await fetch(`/api/runs/${id}`, { method: 'DELETE' });
        loadRunHistory();
        showToast('Run deleted', 'info');
    } catch (err) {
        showToast('Delete failed', 'error');
    }
}

async function clearRunHistory() {
    if (!confirm('Delete all experiment history?')) return;
    try {
        await fetch('/api/runs/clear', { method: 'DELETE' });
        loadRunHistory();
        showToast('History cleared', 'info');
    } catch (err) {
        showToast('Clear failed', 'error');
    }
}


// ═══════════════════════════════════════════════════════
// UI Utilities
// ═══════════════════════════════════════════════════════

function showLoading(text = 'Loading...') {
    document.getElementById('loadingOverlay').classList.remove('hidden');
    document.getElementById('loadingText').textContent = text;
}

function hideLoading() {
    document.getElementById('loadingOverlay').classList.add('hidden');
}

function showToast(msg, type = 'info') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = msg;
    container.appendChild(toast);
    setTimeout(() => toast.remove(), 4000);
}

// ── Test size slider label ──
const testSlider = document.getElementById('cfgTestSize');
if (testSlider) {
    testSlider.addEventListener('input', () => {
        document.getElementById('testSizeLabel').textContent = testSlider.value + '%';
    });
}

// ── Init ──
document.addEventListener('DOMContentLoaded', () => {
    goToStep(0);
});

// ═══════════════════════════════════════════════════════
// Error Modal Support (Phase 5)
// ═══════════════════════════════════════════════════════

function showErrorModal(markdownText) {
    const modal = document.getElementById('errorModal');
    const content = document.getElementById('errorExplanationContent');
    content.innerHTML = window.marked ? window.marked.parse(markdownText) : markdownText;
    modal.style.display = 'flex';
}

function closeErrorModal() {
    const modal = document.getElementById('errorModal');
    modal.style.display = 'none';
}

// ═══════════════════════════════════════════════════════
// Autonomous Agent Controller
// ═══════════════════════════════════════════════════════

async function launchAgent() {
    const targetCol = document.getElementById('targetCol').value;
    if (!targetCol) {
        showToast('Please select a target column first.', 'error');
        return;
    }

    const metric = document.getElementById('agentMetric').value;
    const target = parseFloat(document.getElementById('agentTarget').value);
    const maxIter = parseInt(document.getElementById('agentMaxIter').value);

    const btn = document.getElementById('btnAgentLaunch');
    btn.disabled = true;
    btn.innerHTML = '<span class="btn-agent-icon">...</span> Launching...';

    try {
        const resp = await fetch('/api/agent/run', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                target_col: targetCol,
                task: document.querySelector('input[name="task"]:checked')?.value || 'auto',
                goal: {metric, target},
                max_iterations: maxIter,
            }),
        });
        const data = await resp.json();
        if (!resp.ok) {
            showToast(data.error || 'Failed to launch agent', 'error');
            btn.disabled = false;
            btn.innerHTML = '<span class="btn-agent-icon">⚡</span> Launch Autonomous Agent';
            return;
        }

        // Open monitor modal
        document.getElementById('agentModal').classList.remove('hidden');
        document.getElementById('agentIterMax').textContent = maxIter;
        document.getElementById('agentLogBox').innerHTML = '';
        document.getElementById('agentHistoryBody').innerHTML =
            '<tr><td colspan="5" class="empty-state">Waiting for first iteration...</td></tr>';
        document.getElementById('btnViewAgentResults').classList.add('hidden');
        document.getElementById('btnStopAgent').disabled = false;
        document.getElementById('agentModalIcon').textContent = '⚡';
        document.getElementById('agentModalTitle').textContent = 'Autonomous Agent Running';

        // Start polling
        if (_agentPollTimer) clearInterval(_agentPollTimer);
        _agentPollTimer = setInterval(_agentPollLoop, 2000);
        showToast('Agent launched! Autonomous loop started.', 'success');
    } catch (err) {
        showToast('Network error: ' + err.message, 'error');
        btn.disabled = false;
        btn.innerHTML = '<span class="btn-agent-icon">⚡</span> Launch Autonomous Agent';
    }
}

async function _agentPollLoop() {
    try {
        const resp = await fetch('/api/agent/status');
        const state = await resp.json();
        _agentState = state;
        _renderAgentState(state);

        const done = ['success', 'finished', 'failed', 'stopped'].includes(state.status);
        if (done) {
            clearInterval(_agentPollTimer);
            _agentPollTimer = null;

            // Re-enable launch button
            const btn = document.getElementById('btnAgentLaunch');
            if (btn) {
                btn.disabled = false;
                btn.innerHTML = '<span class="btn-agent-icon">⚡</span> Launch Autonomous Agent';
            }

            // Show view results button
            document.getElementById('btnViewAgentResults').classList.remove('hidden');
            document.getElementById('btnStopAgent').disabled = true;

            const icon = document.getElementById('agentModalIcon');
            const title = document.getElementById('agentModalTitle');
            if (state.status === 'success') {
                icon.textContent = '✅';
                title.textContent = 'Goal Achieved!';
            } else if (state.status === 'failed') {
                icon.textContent = '❌';
                title.textContent = 'Agent Failed';
            } else {
                icon.textContent = '⏹️';
                title.textContent = 'Agent Finished';
            }
        }
    } catch (err) {
        console.warn('Agent poll error:', err);
    }
}

function _renderAgentState(state) {
    // Phase pill
    const phasePill = document.getElementById('agentPhasePill');
    if (phasePill) phasePill.textContent = state.current_phase || 'idle';

    // Iteration counter
    const iterEl = document.getElementById('agentIterCurrent');
    if (iterEl) iterEl.textContent = state.iteration || 0;

    // Score pill
    const scorePill = document.getElementById('agentScorePill');
    if (scorePill) {
        scorePill.textContent = state.best_score != null
            ? `Best ${(state.goal_metric || '').toUpperCase()}: ${state.best_score}`
            : 'Best: —';
    }

    // Progress fill
    const fill = document.getElementById('agentProgressFill');
    if (fill && state.max_iterations) {
        const pct = Math.round((state.iteration / state.max_iterations) * 100);
        fill.style.width = pct + '%';
    }

    // Phase timeline highlight
    const phases = ['perceive', 'plan', 'act', 'observe', 'reflect'];
    const currentIdx = phases.indexOf(state.current_phase);
    document.querySelectorAll('.agent-timeline-step').forEach((el, i) => {
        el.classList.toggle('active', el.dataset.phase === state.current_phase);
        el.classList.toggle('done', i < currentIdx);
    });

    // Iteration history table
    const tbody = document.getElementById('agentHistoryBody');
    if (tbody && state.history && state.history.length > 0) {
        tbody.innerHTML = '';
        state.history.forEach(h => {
            const models = ((h.config && h.config.models) || []).join(', ');
            const score = h.score != null ? parseFloat(h.score).toFixed(4) : '—';
            const isBest = h.is_best ? '★' : '';
            const ovStr = h.overfitting ? '⚠️' : '—';
            const statusBadge = h.status === 'ok'
                ? `<span style="color:var(--success)">✓</span>`
                : `<span style="color:var(--error)">${h.status}</span>`;
            const tr = document.createElement('tr');
            if (h.is_best) tr.style.background = 'rgba(99,102,241,0.08)';
            tr.innerHTML = `
                <td>#${h.iteration} ${isBest}</td>
                <td style="font-size:0.8rem">${models}</td>
                <td><strong>${score}</strong></td>
                <td>${ovStr}</td>
                <td>${statusBadge}</td>
            `;
            tbody.appendChild(tr);
        });
    }

    // Live log — append only new entries
    const logBox = document.getElementById('agentLogBox');
    if (logBox && state.log) {
        const currentCount = logBox.children.length;
        const newEntries = state.log.slice(currentCount);
        newEntries.forEach(entry => {
            const line = document.createElement('div');
            line.className = 'agent-log-line';
            if (entry.includes('GOAL ACHIEVED') || entry.includes('New best')) {
                line.style.color = '#4ade80';
            } else if (entry.includes('ERROR') || entry.includes('crashed') || entry.includes('FATAL')) {
                line.style.color = '#f87171';
            } else if (entry.includes('WARNING') || entry.includes('Overfitting')) {
                line.style.color = '#fbbf24';
            } else if (entry.includes('PLAN') || entry.includes('REFLECT') || entry.includes('Gemini')) {
                line.style.color = '#a78bfa';
            }
            line.textContent = entry;
            logBox.appendChild(line);
        });
        logBox.scrollTop = logBox.scrollHeight;
    }
}

function closeAgentModal() {
    document.getElementById('agentModal').classList.add('hidden');
    // Polling continues in background if agent is still running
}

async function stopAgent() {
    try {
        await fetch('/api/agent/stop', {method: 'POST'});
        showToast('Stop requested. Current iteration will complete first.', 'info');
    } catch (err) {
        showToast('Could not reach server: ' + err, 'error');
    }
}

function viewAgentResults() {
    document.getElementById('agentModal').classList.add('hidden');
    if (_agentState && _agentState.best_run_report &&
        Object.keys(_agentState.best_run_report).length > 0) {
        pipelineReport = _agentState.best_run_report;
        renderResults(pipelineReport);
        _renderAgentSummary(_agentState);
        loadRunHistory();
    }
    goToStep(3);
}

function _renderAgentSummary(state) {
    const banner = document.getElementById('agentSummaryBanner');
    if (!banner) return;
    banner.classList.remove('hidden');

    // Status badge
    const statusEl = document.getElementById('agentSummaryStatus');
    const statusMap = {
        success:  {label: '✅ Goal Achieved'},
        finished: {label: '⏹️ Finished'},
        failed:   {label: '❌ Failed'},
        stopped:  {label: '⏹️ Stopped'},
    };
    const sm = statusMap[state.status] || {label: state.status};
    statusEl.textContent = sm.label;

    // Stats row
    const statsEl = document.getElementById('agentSummaryStats');
    const metricLabel = (state.goal_metric || '').toUpperCase();
    const bestModels = (state.best_config && state.best_config.models) || [];
    statsEl.innerHTML = `
        <div class="agent-stat">
            <div class="agent-stat-value">${state.iteration}</div>
            <div class="agent-stat-label">Iterations</div>
        </div>
        <div class="agent-stat">
            <div class="agent-stat-value">${state.best_score != null ? state.best_score : '—'}</div>
            <div class="agent-stat-label">Best ${metricLabel}</div>
        </div>
        <div class="agent-stat">
            <div class="agent-stat-value">${state.goal_target}</div>
            <div class="agent-stat-label">Target</div>
        </div>
        <div class="agent-stat">
            <div class="agent-stat-value">${bestModels.length}</div>
            <div class="agent-stat-label">Models in best run</div>
        </div>
    `;

    // Narrative
    const narrativeEl = document.getElementById('agentSummaryNarrative');
    if (state.final_summary) {
        narrativeEl.innerHTML = window.marked
            ? window.marked.parse(state.final_summary)
            : state.final_summary;
    }

    // Insert banner at top of step3
    const step3 = document.getElementById('step3');
    if (step3 && step3.firstChild !== banner) {
        step3.insertBefore(banner, step3.firstChild);
    }
}
