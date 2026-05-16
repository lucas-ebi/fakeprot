const WARN_BROWSER_CELLS = 2500000;
const HARD_BROWSER_CELLS = 5000000;

const ICON_EYE = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 14" width="16" height="12" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M1 7C4.5 1 15.5 1 19 7C15.5 13 4.5 13 1 7Z"/><circle cx="10" cy="7" r="2.5"/></svg>`;
const ICON_DL  = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><line x1="8" y1="1" x2="8" y2="10"/><polyline points="4,7 8,11 12,7"/><line x1="2" y1="14" x2="14" y2="14"/></svg>`;

function fileMeta(name) {
  if (name.includes('all_sequences'))          return {type:'MSA',  view:'msa',  desc:'All sequences including ancestral nodes'};
  if (name.includes('current_sequences'))      return {type:'MSA',  view:'msa',  desc:'Extant (leaf) sequences only'};
  if (name.includes('gene_tree'))              return {type:'TREE', view:'tree', desc:'Gene phylogeny with branch lengths'};
  if (name.includes('species_cladogram'))      return {type:'TREE', view:'tree', desc:'Species tree (topology only)'};
  if (name.includes('OG_'))                    return {type:'MSA',  view:'msa',  desc:'Ortholog group alignment'};
  if (name.includes('ortholog_groups'))        return {type:'CSV',  view:'csv',  desc:'Sequence‑to‑ortholog group mapping'};
  if (name.includes('physicochemical_groups')) return {type:'JSON', view:'json', desc:'Per‑column amino acid frequencies and physicochemical class per ortholog group'};
  if (name.includes('run_info'))               return {type:'JSON', view:'json', desc:'Run parameters and timestamp'};
  return {type:'—', view:null, desc:''};
}

function setStatus(msg, state = 'loading') {
  document.getElementById('status-text').textContent = msg;
  document.getElementById('status-dot').className = 'dot ' + state;
}

function addWarning(msg, container = document.getElementById('warnings')) {
  const div = document.createElement('div');
  div.className = 'warning';
  div.textContent = msg;
  container.appendChild(div);
  return div;
}

function addPreviewWarning(container, msg) {
  const div = addWarning(msg, container);
  div.classList.add('preview-warning');
  return div;
}

function switchTab(btn) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
}

// ── Worker setup ───────────────────────────────────────────────────────────
let worker = null;
try {
  worker = new Worker('worker.js');
} catch (_) {
  setStatus(
    'Cannot start worker — serve docs/ with a local HTTP server (e.g. python -m http.server).',
    'error'
  );
}
const _pending = new Map();
let _msgId = 0;

function workerRequest(msg) {
  if (!worker) return Promise.reject(new Error('Worker unavailable'));
  return new Promise((resolve, reject) => {
    const id = _msgId++;
    _pending.set(id, {resolve, reject});
    worker.postMessage({...msg, id});
  });
}

if (worker) {
  worker.onmessage = ({data}) => {
    const {type, id, ...rest} = data;
    if (type === 'status')    { setStatus(rest.msg); return; }
    if (type === 'ready')     {
      setStatus('Ready', 'ready');
      document.getElementById('run-btn').disabled = false;
      document.getElementById('btn-label').textContent = 'Run simulation';
      document.getElementById('sim-form').addEventListener('submit', async e => {
        e.preventDefault();
        await runSimulation();
      });
      return;
    }
    if (type === 'initError') { setStatus('Initialisation failed: ' + rest.message, 'error'); return; }
    const p = _pending.get(id);
    if (!p) return;
    _pending.delete(id);
    if (type === 'error') p.reject(new Error(rest.message));
    else p.resolve(rest);
  };

  worker.onerror = err => {
    setStatus('Worker error: ' + err.message, 'error');
    console.error(err);
    for (const {reject} of _pending.values()) reject(new Error(err.message || 'Worker error'));
    _pending.clear();
    worker = null;
  };
}

// ── File access ────────────────────────────────────────────────────────────
const textCache = new Map();
let activePreviewBtn = null;

async function getFileText(filename) {
  if (textCache.has(filename)) return textCache.get(filename);
  const {text} = await workerRequest({type: 'getText', filename});
  textCache.set(filename, text);
  return text;
}

async function downloadFile(filename) {
  const {buffer} = await workerRequest({type: 'getBytes', filename});
  const url = URL.createObjectURL(new Blob([buffer], {type: 'text/plain'}));
  const a = document.createElement('a');
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

function ogLabelFromFilename(filename) {
  const match = filename.match(/_OG_([^./]+)\./);
  return match ? `OG ${match[1]}` : filename;
}

// ── Run simulation ─────────────────────────────────────────────────────────
async function runSimulation() {
  const btn = document.getElementById('run-btn');
  btn.disabled = true;
  btn.classList.add('running');
  document.getElementById('btn-label').textContent = 'Simulating…';
  setStatus('Running simulation…');
  textCache.clear();
  closePreview();

  ['panel-current-msa','panel-current-logo','panel-gene-tree','panel-species-tree'].forEach(id => {
    const el = document.getElementById(id);
    el.innerHTML = '';
    el.classList.remove('visible');
  });
  document.getElementById('empty-sequences').style.display  = '';
  document.getElementById('empty-logo').style.display       = '';
  document.getElementById('empty-gene-tree').style.display  = '';
  document.getElementById('empty-species-tree').style.display = '';
  document.getElementById('msg-sequences').textContent    = 'Execute a simulation to view the alignment';
  document.getElementById('msg-logo').textContent         = 'Execute a simulation to view the sequence logo';
  document.getElementById('msg-gene-tree').textContent    = 'Execute a simulation to view the gene tree';
  document.getElementById('msg-species-tree').textContent = 'Execute a simulation to view the species tree';
  document.getElementById('files-content').style.display = 'none';
  document.getElementById('empty-files').style.display   = '';
  document.getElementById('warnings').innerHTML = '';

  const size        = parseInt(document.getElementById('size').value);
  const length      = parseInt(document.getElementById('length').value);
  const n_orthologs = parseInt(document.getElementById('n_orthologs').value);
  const gamma_shape = parseFloat(document.getElementById('gamma_shape').value);
  const p_del_raw   = document.getElementById('p_del').value.trim();
  const p_ins_raw   = document.getElementById('p_ins').value.trim();
  const seed_raw    = document.getElementById('seed').value.trim();
  const msa_format  = document.getElementById('msa_format').value;
  const tree_format = document.getElementById('tree_format').value;
  const estimatedCells = size * length;

  if (estimatedCells > HARD_BROWSER_CELLS) {
    addWarning(`Requested simulation (${estimatedCells.toLocaleString()} sequence cells) exceeds browser memory limits. Reduce sequence count or length, or use the command-line version for larger jobs.`);
    setStatus('Run is too large for the browser', 'error');
    btn.disabled = false;
    btn.classList.remove('running');
    document.getElementById('btn-label').textContent = 'Run simulation';
    return;
  }
  if (estimatedCells > WARN_BROWSER_CELLS) {
    addWarning(`Large simulation (${estimatedCells.toLocaleString()} sequence cells). This may take considerable time and use substantial browser memory.`);
  }

  try {
    const config = {
      size, length, n_orthologs, gamma_shape, msa_format, tree_format,
      p_del: p_del_raw ? parseFloat(p_del_raw) : 'None',
      p_ins: p_ins_raw ? parseFloat(p_ins_raw) : 'None',
      seed:  seed_raw  ? parseInt(seed_raw)    : 'None',
    };
    const {filenames: files, warnings} = await workerRequest({type: 'run', config});

    if (warnings) addWarning(warnings);

    const order = ['all_sequences','current_sequences','OG_','gene_tree',
                   'species_cladogram','ortholog_groups','physicochemical_groups','run_info'];
    files.sort((a, b) => {
      const ai = order.findIndex(k => a.includes(k));
      const bi = order.findIndex(k => b.includes(k));
      return (ai === -1 ? 99 : ai) - (bi === -1 ? 99 : bi);
    });

    const AUTO_KEYS = ['current_sequences','gene_tree','species_cladogram'];
    const tbody = document.getElementById('output-tbody');
    tbody.innerHTML = '';

    for (const filename of files) {
      const meta = fileMeta(filename);
      const isAuto = AUTO_KEYS.some(k => filename.includes(k));
      const canView = !isAuto && (
        meta.view === 'msa' ||
        (meta.view === 'tree' && filename.endsWith('.newick')) ||
        meta.view === 'csv' || meta.view === 'json'
      );
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td class="file-name">${filename}</td>
        <td class="file-type">${meta.type}</td>
        <td class="file-desc">${meta.desc}</td>
        <td style="text-align:center">${canView
          ? `<button type="button" class="icon-btn view-btn" onclick='showPreview(${JSON.stringify(filename)},this)' title="Preview ${filename}" aria-label="Preview ${filename}">${ICON_EYE}</button>`
          : `<span style="color:var(--border)">—</span>`}</td>
        <td style="text-align:center"><button type="button" class="icon-btn" onclick='downloadFile(${JSON.stringify(filename)})' title="Download ${filename}" aria-label="Download ${filename}">${ICON_DL}</button></td>`;
      tbody.appendChild(tr);
    }
    document.getElementById('empty-files').style.display = 'none';
    document.getElementById('files-content').style.display = 'block';

    const msaFile = files.find(f => f.includes('current_sequences'));
    if (msaFile) {
      const text = await getFileText(msaFile);
      const el = document.getElementById('panel-current-msa');
      const logoEl = document.getElementById('panel-current-logo');
      document.getElementById('empty-sequences').style.display = 'none';
      document.getElementById('empty-logo').style.display = 'none';
      el.classList.add('visible');
      logoEl.classList.add('visible');
      renderMSA(text, el);
      renderLogoSection('Current sequences', text, logoEl);

      const ogLogoFiles = files.filter(f => f.includes('OG_') && f.endsWith('.fasta'));
      for (const ogFile of ogLogoFiles) {
        renderLogoSection(ogLabelFromFilename(ogFile), await getFileText(ogFile), logoEl);
      }
      setupLogoScrollSync(logoEl);
    } else {
      document.getElementById('msg-sequences').textContent = 'No alignment file was generated.';
      document.getElementById('msg-logo').textContent = 'No alignment file was generated.';
    }

    const geneTreeFile = files.find(f => f.includes('gene_tree') && f.endsWith('.newick'));
    if (geneTreeFile) {
      const el = document.getElementById('panel-gene-tree');
      document.getElementById('empty-gene-tree').style.display = 'none';
      el.classList.add('visible');
      renderTree(await getFileText(geneTreeFile), el);
    } else {
      document.getElementById('msg-gene-tree').textContent = 'Tree preview requires Newick format.';
    }

    const specTreeFile = files.find(f => f.includes('species_cladogram') && f.endsWith('.newick'));
    if (specTreeFile) {
      const el = document.getElementById('panel-species-tree');
      document.getElementById('empty-species-tree').style.display = 'none';
      el.classList.add('visible');
      renderTree(await getFileText(specTreeFile), el);
    } else {
      document.getElementById('msg-species-tree').textContent = 'Species tree preview requires Newick format.';
    }

    switchTab(document.querySelector('.tab-btn[data-tab="sequences"]'));
    setStatus(`${files.length} file${files.length !== 1 ? 's' : ''} generated`, 'ready');

  } catch (err) {
    setStatus('Error: ' + err.message, 'error');
    console.error(err);
  } finally {
    btn.disabled = false;
    btn.classList.remove('running');
    document.getElementById('btn-label').textContent = 'Run simulation';
  }
}

// ── File preview (Files tab) ───────────────────────────────────────────────
async function showPreview(filename, btn) {
  const preview = document.getElementById('file-preview');
  if (activePreviewBtn === btn && preview.style.display !== 'none') {
    closePreview(); return;
  }
  if (activePreviewBtn) activePreviewBtn.classList.remove('active');
  activePreviewBtn = btn;
  btn.classList.add('active');
  document.getElementById('preview-title').textContent = filename;
  const content = document.getElementById('preview-content');
  content.innerHTML = '';
  const text = await getFileText(filename);
  const meta = fileMeta(filename);
  if      (meta.view === 'msa')  renderMSAPreview(text, content);
  else if (meta.view === 'tree') renderTree(text, content);
  else if (meta.view === 'csv')  renderCSV(text, content);
  else if (meta.view === 'json') renderJSON(text, content);
  preview.style.display = 'block';
  preview.scrollIntoView({behavior:'smooth', block:'nearest'});
}

function setupLogoScrollSync(container) {
  const wraps = Array.from(container.querySelectorAll('.logo-wrap'));
  if (wraps.length < 2) return;
  let syncing = false;
  wraps.forEach(wrap => {
    wrap.addEventListener('scroll', () => {
      if (syncing) return;
      syncing = true;
      wraps.forEach(w => { if (w !== wrap) w.scrollLeft = wrap.scrollLeft; });
      syncing = false;
    });
  });
}

function closePreview() {
  document.getElementById('file-preview').style.display = 'none';
  document.getElementById('preview-content').innerHTML = '';
  if (activePreviewBtn) { activePreviewBtn.classList.remove('active'); activePreviewBtn = null; }
}
