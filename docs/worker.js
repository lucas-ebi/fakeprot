// Web Worker: owns Pyodide, runs simulations, serves output files.
// Communicates with the main thread via postMessage.
//
// Spontaneous messages (no id):
//   {type:'status', msg}     — loading progress
//   {type:'ready'}           — Pyodide + FakeProt initialised
//   {type:'initError', message}
//
// Request/response (matching id):
//   run      → done     {filenames, warnings}
//   getText  → text     {filename, text}
//   getBytes → bytes    {filename, buffer}   (buffer is Transferable)
//   any      → error    {message}

importScripts('https://cdn.jsdelivr.net/pyodide/v0.27.0/full/pyodide.js');

const BASE = 'https://raw.githubusercontent.com/lucas-ebi/fakeprot/main/fakeprot/';
const FAKEPROT_FILES = [
  '__init__.py', 'config.py', 'substitution.py', 'simulation.py',
  'models/__init__.py', 'models/sequence.py', 'models/species.py', 'models/msa_store.py',
  'evolution/__init__.py', 'evolution/mutation.py', 'evolution/tree.py',
  'io/__init__.py', 'io/output.py',
];

let pyodide;

async function init() {
  postMessage({type: 'status', msg: 'Initialising Pyodide…'});
  pyodide = await loadPyodide();

  postMessage({type: 'status', msg: 'Installing packages…'});
  await pyodide.loadPackage(['numpy', 'scipy', 'pandas', 'micropip']);
  await pyodide.runPythonAsync(
    'import micropip\nawait micropip.install(["networkx","biopython"])'
  );

  postMessage({type: 'status', msg: 'Loading FakeProt…'});
  for (const dir of ['fakeprot','fakeprot/models','fakeprot/evolution','fakeprot/io','output']) {
    try { pyodide.FS.mkdir('/' + dir); } catch (_) {}
  }
  await Promise.all(FAKEPROT_FILES.map(async file => {
    const res = await fetch(BASE + file);
    if (!res.ok) throw new Error(`Failed to fetch fakeprot/${file} (${res.status})`);
    pyodide.FS.writeFile('/fakeprot/' + file, await res.text());
  }));
  pyodide.runPython('import sys\nif "/" not in sys.path: sys.path.insert(0, "/")');

  postMessage({type: 'ready'});
}

self.onmessage = async ({data}) => {
  const {id, type, ...args} = data;
  try {
    if (type === 'run') {
      const cfg = args.config;

      try {
        pyodide.FS.readdir('/output')
          .filter(f => f !== '.' && f !== '..')
          .forEach(f => pyodide.FS.unlink('/output/' + f));
      } catch (_) {}

      await pyodide.runPythonAsync(`
import sys, io as _io
_cap = _io.StringIO()
_real_stdout = sys.stdout
sys.stdout = _cap
try:
    from fakeprot.config import SimulationConfig
    from fakeprot.simulation import run
    _cfg = SimulationConfig(
        size=${cfg.size},
        length=${cfg.length},
        p_gap=${cfg.p_gap},
        n_orthologs=${cfg.n_orthologs},
        gamma_shape=${cfg.gamma_shape},
        gamma_scale=${cfg.gamma_scale},
        seed=${cfg.seed},
        out="/output/result",
        msa_format="${cfg.msa_format}",
        tree_format="${cfg.tree_format}",
    )
    run(_cfg)
finally:
    sys.stdout = _real_stdout
_warnings = _cap.getvalue().strip()
`);
      const warnings  = pyodide.globals.get('_warnings') || '';
      const filenames = pyodide.FS.readdir('/output').filter(f => f !== '.' && f !== '..');
      postMessage({type: 'done', id, filenames, warnings});

    } else if (type === 'getText') {
      const raw  = pyodide.FS.readFile('/output/' + args.filename);
      const text = new TextDecoder().decode(raw);
      postMessage({type: 'text', id, filename: args.filename, text});

    } else if (type === 'getBytes') {
      const raw = pyodide.FS.readFile('/output/' + args.filename);
      const buf = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
      postMessage({type: 'bytes', id, filename: args.filename, buffer: buf}, [buf]);
    }

  } catch (err) {
    if (type === 'run') {
      try { await pyodide.runPythonAsync('import sys\nsys.stdout = sys.__stdout__'); } catch (_) {}
    }
    postMessage({type: 'error', id, message: err.message});
  }
};

init().catch(err => postMessage({type: 'initError', message: err.message}));
