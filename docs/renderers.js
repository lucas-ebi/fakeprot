// ClustalX colour scheme
const AA_COLORS = {
  'A':'#80a0f0','V':'#80a0f0','I':'#80a0f0','L':'#80a0f0',
  'M':'#80a0f0','F':'#80a0f0','W':'#80a0f0',
  'K':'#f01505','R':'#f01505',
  'D':'#c048c0','E':'#c048c0',
  'S':'#15c015','T':'#15c015','N':'#15c015','Q':'#15c015',
  'H':'#15a4a4','Y':'#15a4a4',
  'C':'#f08080','G':'#f09048','P':'#c0c000','-':'#d8d8d8',
};
const MSA_LEGEND = [
  {label:'Hydrophobic (A,V,I,L,M,F,W)',color:'#80a0f0'},
  {label:'Positive (K,R)',              color:'#f01505'},
  {label:'Negative (D,E)',              color:'#c048c0'},
  {label:'Polar (S,T,N,Q)',             color:'#15c015'},
  {label:'Aromatic (H,Y)',              color:'#15a4a4'},
  {label:'Cysteine',                    color:'#f08080'},
  {label:'Glycine',                     color:'#f09048'},
  {label:'Proline',                     color:'#c0c000'},
  {label:'Gap',                         color:'#d8d8d8'},
];
const MAX_MSA_PREVIEW_SEQS = 500;
const MAX_MSA_PREVIEW_COLS = 500;
const MAX_TREE_PREVIEW_LEAVES = 1000;
const LOGO_AAS = 'ARNDCQEGHILKMFPSTWYV'.split('');
const LOGO_MAX_BITS = Math.log2(LOGO_AAS.length);

// ── Parsers ────────────────────────────────────────────────────────────────
function parseFASTA(text) {
  const seqs = [];
  let cur = null;
  for (const line of text.split('\n')) {
    const t = line.trim();
    if (!t) continue;
    if (t.startsWith('>')) { cur = {id: t.slice(1).trim().split(/\s+/)[0], seq: ''}; seqs.push(cur); }
    else if (cur) cur.seq += t;
  }
  return seqs;
}

function parseCSV(text) {
  const lines = text.trim().split('\n').filter(l => l.trim());
  if (!lines.length) return {headers:[], rows:[]};
  const parse = l => {
    const fields = [];
    let f = '', inQ = false;
    for (const c of l) {
      if (c === '"') inQ = !inQ;
      else if (c === ',' && !inQ) { fields.push(f); f = ''; }
      else f += c;
    }
    fields.push(f);
    return fields.map(v => v.trim());
  };
  return {headers: parse(lines[0]), rows: lines.slice(1).map(parse)};
}

function parseNewick(s) {
  s = s.trim().replace(/;+$/, '').replace(/\s+/g, '');
  let i = 0;
  function node() {
    const n = {name:'', length:0, children:[]};
    if (s[i] === '(') {
      i++;
      n.children.push(node());
      while (s[i] === ',') { i++; n.children.push(node()); }
      i++;
    }
    const ls = i;
    while (i < s.length && ':,)'.indexOf(s[i]) === -1) i++;
    n.name = s.slice(ls, i);
    if (s[i] === ':') {
      i++;
      const ds = i;
      while (i < s.length && ',)'.indexOf(s[i]) === -1) i++;
      n.length = parseFloat(s.slice(ds, i)) || 0;
    }
    return n;
  }
  return node();
}

function layoutTree(root) {
  let idx = 0;
  function setY(n) {
    if (!n.children.length) { n._y = idx++; return; }
    n.children.forEach(setY);
    n._y = (n.children[0]._y + n.children[n.children.length - 1]._y) / 2;
  }
  function setX(n, x) { n._x = x + n.length; n.children.forEach(c => setX(c, n._x)); }
  setY(root); root.length = 0; setX(root, 0);
  let maxX = 0;
  function findMax(n) { if (n._x > maxX) maxX = n._x; n.children.forEach(findMax); }
  findMax(root);
  if (maxX < 1e-10) {
    function setDepth(n, d) { n._x = d; n.children.forEach(c => setDepth(c, d + 1)); }
    setDepth(root, 0); maxX = 0; findMax(root);
  }
  return {nLeaves: idx, maxX: maxX || 1};
}

function countNewickLeaves(text) {
  const s = text.trim().replace(/\s+/g, '');
  let leaves = 0;
  for (let i = 0; i < s.length; i++) {
    const prev = s[i - 1];
    const cur = s[i];
    if ((prev === '(' || prev === ',') && cur !== '(' && cur !== ')' && cur !== ',' && cur !== ':' && cur !== ';') {
      leaves++;
    }
  }
  return leaves;
}

// ── Renderers ──────────────────────────────────────────────────────────────
function prepareMSAPreview(content, container, showLimitWarning = true) {
  const allSeqs = parseFASTA(content);
  if (!allSeqs.length) {
    const p = document.createElement('p');
    p.style.cssText = 'color:var(--muted);font-size:.875rem;padding:1rem';
    p.textContent = 'No sequences found. Visualisation requires FASTA format.';
    container.appendChild(p);
    return null;
  }

  let totalCols = 0;
  for (const seq of allSeqs) {
    if (seq.seq.length > totalCols) totalCols = seq.seq.length;
  }

  const seqs = allSeqs.slice(0, MAX_MSA_PREVIEW_SEQS).map(seq => ({
    id: seq.id,
    seq: seq.seq.slice(0, MAX_MSA_PREVIEW_COLS),
  }));
  const isLimited = allSeqs.length > seqs.length || totalCols > MAX_MSA_PREVIEW_COLS;
  if (isLimited && showLimitWarning) {
    addPreviewWarning(
      container,
      `Preview limited to first ${seqs.length.toLocaleString()} of ${allSeqs.length.toLocaleString()} sequences and first ${Math.min(totalCols, MAX_MSA_PREVIEW_COLS).toLocaleString()} of ${totalCols.toLocaleString()} columns. Download the full files for complete data.`
    );
  }

  let nCol = 0;
  for (const seq of seqs) {
    if (seq.seq.length > nCol) nCol = seq.seq.length;
  }
  return {seqs, nCol};
}

function renderMSA(content, container, showLimitWarning = true) {
  const preview = prepareMSAPreview(content, container, showLimitWarning);
  if (!preview) return;
  const {seqs, nCol} = preview;

  const legend = document.createElement('div');
  legend.className = 'msa-legend';
  MSA_LEGEND.forEach(({label, color}) => {
    const sp = document.createElement('span');
    sp.className = 'leg';
    sp.innerHTML = `<span class="leg-sw" style="background:${color}"></span>${label}`;
    legend.appendChild(sp);
  });
  container.appendChild(legend);

  const nSeq = seqs.length;
  const CW = 14, CH = 18, LW = 150;
  const canvas = document.createElement('canvas');
  canvas.width  = LW + nCol * CW;
  canvas.height = nSeq * CH;
  canvas.style.display = 'block';
  const ctx = canvas.getContext('2d');
  seqs.forEach((s, row) => {
    const y = row * CH;
    ctx.fillStyle = '#1a1c1a'; ctx.textAlign = 'right';
    ctx.textBaseline = 'middle'; ctx.font = '10px monospace';
    ctx.fillText(s.id.slice(0, 22), LW - 4, y + CH / 2);
    for (let col = 0; col < s.seq.length; col++) {
      const aa = s.seq[col].toUpperCase();
      const x  = LW + col * CW;
      ctx.fillStyle = AA_COLORS[aa] || '#cccccc';
      ctx.fillRect(x, y, CW - 1, CH - 1);
      ctx.fillStyle = '#fff'; ctx.textAlign = 'center';
      ctx.textBaseline = 'middle'; ctx.font = 'bold 10px monospace';
      ctx.fillText(aa, x + CW / 2, y + CH / 2);
    }
  });
  const wrap = document.createElement('div');
  wrap.className = 'msa-wrap';
  wrap.appendChild(canvas);
  container.appendChild(wrap);
}

function renderLogo(content, container, showLimitWarning = true) {
  const preview = prepareMSAPreview(content, container, showLimitWarning);
  if (!preview) return;
  const {seqs, nCol} = preview;

  const CW = 20, AXIS_W = 42, TOP = 12, LOGO_H = 150, GAP_H = 18, BOTTOM = 28;
  const W = AXIS_W + nCol * CW + 12;
  const H = TOP + LOGO_H + GAP_H + BOTTOM;
  const canvas = document.createElement('canvas');
  canvas.width = W;
  canvas.height = H;
  canvas.style.display = 'block';
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, W, H);

  const x0 = AXIS_W;
  const yBase = TOP + LOGO_H;
  ctx.strokeStyle = '#d0d0ce';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x0 - 6, TOP);
  ctx.lineTo(x0 - 6, yBase);
  ctx.lineTo(W - 8, yBase);
  ctx.stroke();

  ctx.fillStyle = '#595c56';
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  [0, 2, 4].forEach(bits => {
    const y = yBase - (bits / LOGO_MAX_BITS) * LOGO_H;
    ctx.strokeStyle = '#ededed';
    ctx.beginPath();
    ctx.moveTo(x0 - 4, y);
    ctx.lineTo(W - 8, y);
    ctx.stroke();
    ctx.fillText(bits.toString(), x0 - 10, y);
  });
  ctx.save();
  ctx.translate(10, TOP + LOGO_H / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = 'center';
  ctx.fillText('bits', 0, 0);
  ctx.restore();

  for (let col = 0; col < nCol; col++) {
    const counts = Object.fromEntries(LOGO_AAS.map(aa => [aa, 0]));
    let nonGap = 0;
    let gap = 0;
    for (const seq of seqs) {
      const aa = (seq.seq[col] || '-').toUpperCase();
      if (aa === '-') {
        gap++;
      } else if (counts[aa] !== undefined) {
        counts[aa]++;
        nonGap++;
      }
    }

    const x = x0 + col * CW;
    if (nonGap > 0) {
      let entropy = 0;
      const items = [];
      for (const aa of LOGO_AAS) {
        const p = counts[aa] / nonGap;
        if (p <= 0) continue;
        entropy -= p * Math.log2(p);
        items.push({aa, p});
      }
      const info = Math.max(0, LOGO_MAX_BITS - entropy);
      let y = yBase;
      items
        .map(item => ({...item, h: item.p * info / LOGO_MAX_BITS * LOGO_H}))
        .filter(item => item.h >= 1)
        .sort((a, b) => a.h - b.h)
        .forEach(item => {
          y -= item.h;
          ctx.save();
          ctx.translate(x + CW / 2, y + item.h);
          ctx.scale(1, item.h / 16);
          ctx.fillStyle = AA_COLORS[item.aa] || '#999';
          ctx.font = 'bold 18px Arial, Helvetica, sans-serif';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'bottom';
          ctx.fillText(item.aa, 0, 0);
          ctx.restore();
        });
    }

    const total = nonGap + gap;
    const gapFrac = total ? gap / total : 0;
    if (gapFrac > 0) {
      ctx.fillStyle = '#d8d8d8';
      ctx.fillRect(x + 2, yBase + 4 + (1 - gapFrac) * GAP_H, CW - 4, gapFrac * GAP_H);
    }

    if ((col + 1) % 10 === 0 || col === 0) {
      ctx.fillStyle = '#595c56';
      ctx.font = '9px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText(String(col + 1), x + CW / 2, yBase + GAP_H + 8);
    }
  }

  const wrap = document.createElement('div');
  wrap.className = 'logo-wrap';
  wrap.appendChild(canvas);
  container.appendChild(wrap);
}

function renderLogoSection(title, content, container, showLimitWarning = true) {
  const section = document.createElement('section');
  section.className = 'logo-section';
  const heading = document.createElement('div');
  heading.className = 'logo-title';
  heading.textContent = title;
  section.appendChild(heading);
  renderLogo(content, section, showLimitWarning);
  container.appendChild(section);
}

function renderMSAPreview(content, container) {
  const logoPanel = document.createElement('div');
  logoPanel.className = 'preview-logo-panel';
  renderLogo(content, logoPanel);
  container.appendChild(logoPanel);
  renderMSA(content, container, false);
}

function renderTree(content, container) {
  const estimatedLeaves = countNewickLeaves(content);
  if (estimatedLeaves > MAX_TREE_PREVIEW_LEAVES) {
    addPreviewWarning(
      container,
      `Tree preview skipped because the tree contains ~${estimatedLeaves.toLocaleString()} leaves (exceeds ${MAX_TREE_PREVIEW_LEAVES.toLocaleString()} limit). Download the Newick file for full visualisation.`
    );
    return;
  }

  let root;
  try { root = parseNewick(content); }
  catch (e) { container.textContent = 'Parse error: ' + e.message; return; }
  const {nLeaves, maxX} = layoutTree(root);
  if (!nLeaves) { container.textContent = 'Empty tree.'; return; }
  if (nLeaves > MAX_TREE_PREVIEW_LEAVES) {
    addPreviewWarning(
      container,
      `Tree preview skipped because tree has ${nLeaves.toLocaleString()} leaves (limit ${MAX_TREE_PREVIEW_LEAVES.toLocaleString()}). Download Newick file for full tree.`
    );
    return;
  }

  const ROW = Math.max(14, Math.min(20, Math.floor(360 / nLeaves)));
  const TW = 360, LW = 130;
  const PAD = {t:8, r:10, b:32, l:16};
  const W = PAD.l + TW + 6 + LW + PAD.r;
  const H = PAD.t + nLeaves * ROW + PAD.b;
  const NS = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('width', W); svg.setAttribute('height', H);
  const g = document.createElementNS(NS, 'g');
  g.setAttribute('transform', `translate(${PAD.l},${PAD.t})`);
  svg.appendChild(g);
  const tx = x => x / maxX * TW;
  const ty = y => y * ROW + ROW / 2;
  // Floating branch-length tooltip element (reused across all branches)
  const tip = document.createElementNS(NS, 'g');
  tip.setAttribute('pointer-events', 'none');
  tip.style.display = 'none';
  const tipRect = document.createElementNS(NS, 'rect');
  tipRect.setAttribute('rx', '2'); tipRect.setAttribute('fill', '#1a1c1a');
  tipRect.setAttribute('opacity', '0.78');
  const tipText = document.createElementNS(NS, 'text');
  tipText.setAttribute('font-size', '10'); tipText.setAttribute('font-family', 'monospace');
  tipText.setAttribute('fill', '#fff'); tipText.setAttribute('dominant-baseline', 'middle');
  tip.appendChild(tipRect); tip.appendChild(tipText);
  g.appendChild(tip);

  function seg(x1, y1, x2, y2, label) {
    if (label) {
      const hit = document.createElementNS(NS, 'line');
      hit.setAttribute('x1', x1); hit.setAttribute('y1', y1);
      hit.setAttribute('x2', x2); hit.setAttribute('y2', y2);
      hit.setAttribute('stroke', 'transparent'); hit.setAttribute('stroke-width', '10');
      hit.style.cursor = 'default';
      hit.addEventListener('mouseenter', () => {
        tipText.textContent = label;
        const tw = label.length * 6.2 + 6;
        tipRect.setAttribute('x', (x1 + x2) / 2 - tw / 2 - 2);
        tipRect.setAttribute('y', y1 - 10);
        tipRect.setAttribute('width', tw + 4); tipRect.setAttribute('height', 14);
        tipText.setAttribute('x', (x1 + x2) / 2 - tw / 2 + 1);
        tipText.setAttribute('y', y1 - 3);
        tip.style.display = '';
      });
      hit.addEventListener('mouseleave', () => { tip.style.display = 'none'; });
      g.appendChild(hit);
    }
    const el = document.createElementNS(NS, 'line');
    el.setAttribute('x1', x1); el.setAttribute('y1', y1);
    el.setAttribute('x2', x2); el.setAttribute('y2', y2);
    el.setAttribute('stroke', '#595c56'); el.setAttribute('stroke-width', '1');
    if (label) el.setAttribute('pointer-events', 'none');
    g.appendChild(el);
  }
  function draw(n) {
    const nx = tx(n._x), ny = ty(n._y);
    if (n.children.length) {
      seg(nx, ty(n.children[0]._y), nx, ty(n.children[n.children.length-1]._y));
      n.children.forEach(c => {
        seg(nx, ty(c._y), tx(c._x), ty(c._y), c.length > 0 ? c.length.toFixed(5) : null);
        draw(c);
      });
    } else {
      const dot = document.createElementNS(NS, 'circle');
      dot.setAttribute('cx', nx); dot.setAttribute('cy', ny);
      dot.setAttribute('r', '2'); dot.setAttribute('fill', '#3b6fb6');
      g.appendChild(dot);
      const txt = document.createElementNS(NS, 'text');
      txt.setAttribute('x', nx + 5); txt.setAttribute('y', ny);
      txt.setAttribute('font-size', ROW <= 14 ? '10' : '11');
      txt.setAttribute('font-family', 'monospace');
      txt.setAttribute('dominant-baseline', 'middle');
      txt.setAttribute('fill', '#1a1c1a');
      txt.textContent = n.name;
      g.appendChild(txt);
    }
  }
  draw(root);
  const barVal = Math.pow(10, Math.floor(Math.log10(maxX)));
  const barPx  = barVal / maxX * TW;
  const barY   = nLeaves * ROW + 14;
  seg(0, barY, barPx, barY);
  seg(0, barY - 3, 0, barY + 3);
  seg(barPx, barY - 3, barPx, barY + 3);
  const st = document.createElementNS(NS, 'text');
  st.setAttribute('x', barPx / 2); st.setAttribute('y', barY + 10);
  st.setAttribute('font-size', '9'); st.setAttribute('font-family', 'sans-serif');
  st.setAttribute('text-anchor', 'middle'); st.setAttribute('fill', '#595c56');
  st.textContent = barVal < 0.01 ? barVal.toExponential(0) : barVal.toString();
  g.appendChild(st);
  const wrap = document.createElement('div');
  wrap.className = 'tree-wrap';
  wrap.appendChild(svg);
  container.appendChild(wrap);
}

function renderCSV(content, container) {
  const {headers, rows} = parseCSV(content);
  const wrap = document.createElement('div');
  wrap.className = 'csv-wrap';
  const table = document.createElement('table');
  table.className = 'csv-table';
  const thead = document.createElement('thead');
  const hRow  = document.createElement('tr');
  headers.forEach(h => { const th = document.createElement('th'); th.textContent = h; hRow.appendChild(th); });
  thead.appendChild(hRow); table.appendChild(thead);
  const tbody = document.createElement('tbody');
  rows.forEach(row => {
    const tr = document.createElement('tr');
    row.forEach(cell => { const td = document.createElement('td'); td.textContent = cell; tr.appendChild(td); });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody); wrap.appendChild(table); container.appendChild(wrap);
}

function renderJSON(content, container) {
  const pre = document.createElement('pre');
  pre.className = 'json-view';
  try { pre.textContent = JSON.stringify(JSON.parse(content), null, 2); }
  catch { pre.textContent = content; }
  container.appendChild(pre);
}
