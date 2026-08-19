/* Front end for the compute API.
 *
 * Holds no parameter knowledge of its own: every form is built from
 * GET /computations/<name>, which returns each field's expected type, default
 * and hint. Registering a new computation server-side gives it a working form
 * here with no change to this file.
 */
'use strict';

const POLL_MS = 4000;
const $ = (id) => document.getElementById(id);

let SCHEMA = null;          // schema of the selected computation
let SELECTED = null;        // job id shown in the detail panel

// ---------------------------------------------------------------- utilities
async function api(path, opts) {
  const r = await fetch(path, opts);
  let body = null;
  try { body = await r.json(); } catch (_) { /* non-JSON error page */ }
  if (!r.ok) {
    const err = new Error((body && body.error) || `${r.status} ${r.statusText}`);
    err.status = r.status;
    err.fields = body && body.fields;
    throw err;
  }
  return body;
}

function banner(msg, kind) {
  const el = $('banner');
  el.textContent = msg;
  el.className = 'banner ' + (kind || '');
}

function fmtBytes(n) {
  if (n < 1024) return n + ' B';
  if (n < 1024 * 1024) return (n / 1024).toFixed(1) + ' KB';
  return (n / 1048576).toFixed(1) + ' MB';
}

function fmtDuration(hours) {
  if (hours == null) return '—';
  if (hours < 1 / 60) return '<1 min';
  if (hours < 1) return (hours * 60).toFixed(1) + ' min';
  return hours.toFixed(2) + ' hr';
}

function wallHours(job) {
  if (!job.started_at || !job.finished_at) return null;
  const a = Date.parse(job.started_at), b = Date.parse(job.finished_at);
  return (isNaN(a) || isNaN(b)) ? null : (b - a) / 3600000;
}

/* Cost model: the scheme's stability limit gives K ~ G^4 timesteps over G^2
 * patches, so total work ~ G^6. Anchored on a measured 96x96 run. The 1.10
 * factor is the observed overshoot across the completed grid-study jobs. */
const COST_ANCHOR_GRID = 96, COST_ANCHOR_HOURS = 3.28, COST_FUDGE = 1.10;
function estimateHours(rg, ry, T) {
  if (!rg || !ry) return null;
  const g = Math.sqrt(rg * ry);                       // non-square: use the mean
  return COST_ANCHOR_HOURS * COST_FUDGE * Math.pow(g / COST_ANCHOR_GRID, 6)
         * (T && T > 0 ? T : 1);
}

// ------------------------------------------------------------- computations
/* Grouped the way COMPUTATION_MENU.md groups them: 18 similar names in one flat
 * list is hard to scan. Anything unmatched falls into "Other" rather than being
 * dropped, so a newly registered computation still appears. */
const GROUPS = [
  ['Scalar results', (n) => /^Compute MFPT|^Time Until/.test(n)],
  ['Characteristic time', (n) => /^Characteristic Time/.test(n)],
  ['Mass analysis', (n) => /^Mass Analysis|^Jrr/.test(n)],
  ['MFPT collections', (n) => /^MFPT point/.test(n)],
  ['Density profiles', (n) => /^Phi/.test(n)],
  ['Heatmaps', (n) => /^Static Heatplot/.test(n)],
  ['Combined & matrix', (n) => /^Full Analysis|^Angular Trajectory/.test(n)],
  ['Parameter sweeps', (n) => /^BC Parameter/.test(n)],
];

function fillComputations(names) {
  const sel = $('comp');
  sel.innerHTML = '<option value="">— choose a computation —</option>';
  const used = new Set();
  for (const [label, test] of GROUPS) {
    const hits = names.filter((n) => test(n));
    if (!hits.length) continue;
    const g = document.createElement('optgroup');
    g.label = label;
    for (const n of hits) {
      used.add(n);
      g.appendChild(new Option(n, n));
    }
    sel.appendChild(g);
  }
  const rest = names.filter((n) => !used.has(n));
  if (rest.length) {
    const g = document.createElement('optgroup');
    g.label = 'Other';
    rest.forEach((n) => g.appendChild(new Option(n, n)));
    sel.appendChild(g);
  }
}

// --------------------------------------------------------------- form build
function inputFor(field, value) {
  const wrap = document.createElement('div');
  wrap.className = 'field';
  wrap.dataset.param = field.name;
  wrap.dataset.type = field.type;

  const lab = document.createElement('label');
  lab.textContent = field.name;
  lab.htmlFor = 'f_' + field.name;
  if (field.required) lab.className = 'req';
  wrap.appendChild(lab);

  let el;
  if (field.type === 'bool') {
    el = document.createElement('input');
    el.type = 'checkbox';
    el.checked = value === true;
  } else if (field.type === 'int' || field.type === 'float') {
    el = document.createElement('input');
    el.type = 'number';
    el.step = field.type === 'int' ? '1' : 'any';
    if (value !== undefined && value !== null) el.value = value;
  } else {
    el = document.createElement('input');
    el.type = 'text';
    if (Array.isArray(value)) el.value = value.join(', ');
    else if (value !== undefined && value !== null) el.value = value;
    if (field.type === 'list') el.placeholder = 'comma separated, e.g. 0, 4, 8';
  }
  el.id = 'f_' + field.name;
  wrap.appendChild(el);

  if (field.hint) {
    const h = document.createElement('div');
    h.className = 'hint';
    // hints are written "name: description (type)"; the label already shows name
    h.textContent = field.hint.replace(new RegExp('^' + field.name + ':\\s*'), '');
    wrap.appendChild(h);
  }
  const e = document.createElement('div');
  e.className = 'err';
  wrap.appendChild(e);

  el.addEventListener('input', onFormChange);
  el.addEventListener('change', onFormChange);
  return wrap;
}

function buildForm(schema) {
  SCHEMA = schema;
  const form = $('form');
  form.innerHTML = '';
  $('compApproach').textContent = schema.approach
    ? 'approach: ' + schema.approach.join(', ') : '';

  for (const f of schema.required) {
    form.appendChild(inputFor({ ...f, required: true }, undefined));
  }

  // N_LIST is the most error-prone field to enter by hand, so offer a helper
  // that asks the server for the positions rather than computing them here.
  if (schema.required.some((f) => f.name === 'N_LIST')) {
    form.appendChild(tubeHelper());
  }

  // show_plt opens a plot window on the server; never expose it, always false.
  const optional = schema.optional.filter((f) => f.name !== 'show_plt');
  if (optional.length) {
    const d = document.createElement('details');
    const s = document.createElement('summary');
    s.textContent = `Advanced (${optional.length} optional, prefilled with defaults)`;
    d.appendChild(s);
    for (const f of optional) d.appendChild(inputFor(f, f.default));
    form.appendChild(d);
  }
  onFormChange();
}

function tubeHelper() {
  const wrap = document.createElement('div');
  wrap.className = 'field';
  wrap.innerHTML =
    '<label>N_LIST helper</label>' +
    '<div class="row">' +
      '<input id="tubes" type="number" step="1" min="2" placeholder="tube count">' +
      '<button class="mini" id="fillTubes" type="button">compute positions</button>' +
    '</div>' +
    '<div class="hint">Asks the server for evenly spaced positions, so they ' +
    'match what the solver uses exactly.</div>';
  setTimeout(() => {
    $('fillTubes').addEventListener('click', async () => {
      const rays = Number($('f_ry_param') && $('f_ry_param').value);
      const tubes = Number($('tubes').value);
      if (!rays || !tubes) {
        banner('Set ry_param and a tube count first.', 'bad');
        return;
      }
      try {
        const r = await api(`/helpers/n_list?rays=${rays}&tubes=${tubes}`);
        $('f_N_LIST').value = r.N_LIST.join(', ');
        banner(r.distinct ? `Filled ${tubes} positions for ${rays} rays.`
                          : 'Positions are not distinct — reduce the tube count.',
               r.distinct ? 'ok' : 'bad');
        onFormChange();
      } catch (e) { banner(e.message, 'bad'); }
    });
  }, 0);
  return wrap;
}

// -------------------------------------------------------- collect & coerce
/* Convert the form's strings into the JSON types the API validates against,
 * using the type the schema reported for each field. Returns
 * {params, errors} rather than throwing so every bad field can be shown at
 * once instead of one at a time. */
function collect() {
  const params = {}, errors = {};
  for (const wrap of $('form').querySelectorAll('.field[data-param]')) {
    const name = wrap.dataset.param, type = wrap.dataset.type;
    const el = $('f_' + name);
    if (!el) continue;
    if (type === 'bool') { params[name] = el.checked; continue; }

    const raw = el.value.trim();
    if (raw === '') continue;                 // omitted: server default applies

    if (type === 'int' || type === 'float') {
      const n = Number(raw);
      if (!isFinite(n)) { errors[name] = 'not a number'; continue; }
      if (type === 'int' && !Number.isInteger(n)) {
        errors[name] = 'must be a whole number'; continue;
      }
      params[name] = n;
    } else if (type === 'list') {
      const parts = raw.replace(/[[\]]/g, '').split(',')
                       .map((s) => s.trim()).filter((s) => s !== '');
      if (!parts.length) { errors[name] = 'empty list'; continue; }
      const nums = parts.map(Number);
      if (nums.some((n) => !isFinite(n))) {
        errors[name] = 'list must contain only numbers'; continue;
      }
      params[name] = nums;
    } else {
      params[name] = raw;
    }
  }
  if (SCHEMA && SCHEMA.optional.some((f) => f.name === 'show_plt')) {
    params.show_plt = false;
  }
  return { params, errors };
}

function showFieldErrors(errors) {
  for (const wrap of $('form').querySelectorAll('.field[data-param]')) {
    const name = wrap.dataset.param;
    const msg = errors[name];
    wrap.classList.toggle('bad', Boolean(msg));
    const e = wrap.querySelector('.err');
    if (e) e.textContent = msg || '';
  }
  const other = Object.entries(errors)
    .filter(([k]) => !$('form').querySelector(`.field[data-param="${k}"]`));
  if (other.length) {
    banner(other.map(([k, v]) => `${k}: ${v}`).join('\n'), 'bad');
  }
}

function onFormChange() {
  if (!SCHEMA) return;
  const { params, errors } = collect();
  const missing = SCHEMA.required.filter(
    (f) => params[f.name] === undefined).map((f) => f.name);
  $('submit').disabled = missing.length > 0 || Object.keys(errors).length > 0;

  const est = $('est');
  const hrs = estimateHours(params.rg_param, params.ry_param, params.T_param);
  if (hrs == null) { est.hidden = true; return; }
  est.hidden = false;
  est.className = 'est' + (hrs >= 1 ? ' heavy' : '');
  est.textContent = `Estimated wall time: ${fmtDuration(hrs)}`
    + (hrs >= 1 ? '  — long run, and there is no checkpointing: a reboot loses it.' : '');
}

// ------------------------------------------------------------------ submit
async function submitJob() {
  const { params, errors } = collect();
  if (Object.keys(errors).length) { showFieldErrors(errors); return; }

  const hrs = estimateHours(params.rg_param, params.ry_param, params.T_param);
  if (hrs != null && hrs >= 1 &&
      !confirm(`This is estimated at ${fmtDuration(hrs)} and cannot be `
               + `resumed if interrupted.\n\nQueue it?`)) return;

  $('submit').disabled = true;
  try {
    const r = await api('/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        computation: $('comp').value,
        params,
        submitted_by: $('who').value.trim() || null,
      }),
    });
    showFieldErrors({});
    banner(`Queued ${r.job_id.slice(0, 8)}.`, 'ok');
    refresh();
  } catch (e) {
    // the API returns a per-field map, so highlight the offending inputs
    if (e.fields) { showFieldErrors(e.fields); banner('Rejected — see fields.', 'bad'); }
    else banner(e.message, 'bad');
  } finally {
    onFormChange();
  }
}

// ------------------------------------------------------------------- queue
function renderJobs(jobs) {
  const tb = $('jobs');
  if (!jobs.length) {
    tb.innerHTML = '<tr><td colspan="6" class="empty">No jobs yet.</td></tr>';
    return;
  }
  tb.innerHTML = '';
  for (const j of jobs) {
    const tr = document.createElement('tr');
    if (j.id === SELECTED) tr.className = 'sel';
    const p = j.params || {};
    const grid = (p.rg_param && p.ry_param) ? `${p.rg_param}x${p.ry_param}`
               : (p.grid_list ? 'sweep' : '—');
    tr.innerHTML =
      `<td class="mono">${j.id.slice(0, 8)}</td>` +
      `<td>${j.computation}</td>` +
      `<td class="mono">${grid}</td>` +
      `<td class="st st-${j.status}">${j.status}</td>` +
      `<td class="mono">${fmtDuration(wallHours(j))}</td>` +
      `<td></td>`;
    if (j.status === 'queued') {
      const b = document.createElement('button');
      b.className = 'mini';
      b.textContent = 'cancel';
      b.addEventListener('click', async (ev) => {
        ev.stopPropagation();
        try { await api('/jobs/' + j.id, { method: 'DELETE' }); refresh(); }
        catch (e) { banner(e.message, 'bad'); }
      });
      tr.lastElementChild.appendChild(b);
    }
    tr.addEventListener('click', () => { SELECTED = j.id; refresh(); });
    tb.appendChild(tr);
  }
}

// ------------------------------------------------------------------ detail
async function renderDetail(id) {
  const box = $('detail');
  let job;
  try { job = await api('/jobs/' + id); }
  catch (e) { box.innerHTML = `<div class="empty">${e.message}</div>`; return; }

  const parts = [];
  parts.push('<div class="kv">');
  parts.push(`<div><span>job</span><span class="mono">${job.id}</span></div>`);
  parts.push(`<div><span>status</span><span class="st st-${job.status}">${job.status}</span></div>`);
  parts.push(`<div><span>wall time</span><span>${fmtDuration(wallHours(job))}</span></div>`);
  if (job.submitted_by) parts.push(`<div><span>submitted by</span><span>${job.submitted_by}</span></div>`);
  parts.push('</div>');

  if (job.error) {
    parts.push(`<div class="banner bad" style="display:block;margin-top:10px">${job.error}</div>`);
  }
  if (job.result != null) {
    parts.push('<h2 style="margin-top:14px">Result</h2>');
    parts.push(`<pre class="mono" style="font-size:12px;white-space:pre-wrap">${
      JSON.stringify(job.result, null, 2)}</pre>`);
  }

  parts.push('<h2 style="margin-top:14px">Parameters</h2><div class="kv">');
  for (const [k, v] of Object.entries(job.params || {})) {
    parts.push(`<div><span>${k}</span><span class="mono">${
      Array.isArray(v) ? '[' + v.join(', ') + ']' : String(v)}</span></div>`);
  }
  parts.push('</div>');
  parts.push('<button class="mini" id="clone" style="margin-top:10px">load these parameters into the form</button>');

  box.innerHTML = parts.join('');

  $('clone').addEventListener('click', async () => {
    $('comp').value = job.computation;
    await onComputationChange();
    for (const [k, v] of Object.entries(job.params || {})) {
      const el = $('f_' + k);
      if (!el) continue;
      if (el.type === 'checkbox') el.checked = Boolean(v);
      else el.value = Array.isArray(v) ? v.join(', ') : v;
    }
    onFormChange();
    banner('Parameters loaded. Adjust and queue.', 'ok');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });

  if (job.status !== 'succeeded') return;

  let listing;
  try { listing = await api(`/jobs/${id}/outputs`); } catch (_) { return; }
  const files = listing.files || [];
  const sec = document.createElement('div');
  sec.innerHTML = '<h2 style="margin-top:14px">Outputs</h2>';
  if (!files.length) {
    sec.innerHTML += '<div class="empty">No files.</div>';
  } else {
    const ul = document.createElement('ul');
    ul.className = 'files';
    for (const f of files) {
      const href = `/jobs/${id}/outputs/${f.path.split('/').map(encodeURIComponent).join('/')}`;
      const li = document.createElement('li');
      li.innerHTML = `<a href="${href}" target="_blank">${f.path}</a>` +
                     `<span class="sz">${fmtBytes(f.bytes)}</span>`;
      ul.appendChild(li);
    }
    sec.appendChild(ul);
    // images are served inline (image/* mimetype), so a plain <img> previews them
    for (const f of files.filter((x) => /\.png$/i.test(x.path))) {
      const d = document.createElement('div');
      d.className = 'preview';
      const href = `/jobs/${id}/outputs/${f.path.split('/').map(encodeURIComponent).join('/')}`;
      d.innerHTML = `<img src="${href}" alt="${f.path}">`;
      sec.appendChild(d);
    }
  }
  box.appendChild(sec);
}

// ------------------------------------------------------------------ refresh
async function refresh() {
  try {
    const h = await api('/health');
    $('healthDot').style.background = 'var(--good)';
    $('healthText').textContent = 'server ok';
    const q = h.queue || {};
    $('queueChips').innerHTML = Object.keys(q).length
      ? Object.entries(q).sort()
          .map(([k, v]) => `<span class="chip">${k}: ${v}</span>`).join('')
      : '<span class="chip">queue empty</span>';
  } catch (_) {
    $('healthDot').style.background = 'var(--bad)';
    $('healthText').textContent = 'server unreachable';
  }
  try {
    const r = await api('/jobs?limit=40');
    renderJobs(r.jobs || []);
  } catch (_) { /* header already reports the outage */ }
  if (SELECTED) renderDetail(SELECTED);
}

async function onComputationChange() {
  const name = $('comp').value;
  showFieldErrors({});
  banner('', '');
  if (!name) {
    $('form').innerHTML = '';
    $('est').hidden = true;
    $('submit').disabled = true;
    SCHEMA = null;
    return;
  }
  try {
    buildForm(await api('/computations/' + encodeURIComponent(name)));
  } catch (e) { banner(e.message, 'bad'); }
}

// --------------------------------------------------------------------- init
(async function init() {
  $('authNote').textContent =
    'No authentication: anyone who can reach this page can queue and cancel '
    + 'work. "submitted by" is unverified, for your own bookkeeping.';

  $('who').value = localStorage.getItem('itcm_who') || '';
  $('who').addEventListener('change',
    () => localStorage.setItem('itcm_who', $('who').value.trim()));

  $('comp').addEventListener('change', onComputationChange);
  $('submit').addEventListener('click', submitJob);

  try {
    const r = await api('/computations');
    fillComputations(r.computations || []);
  } catch (e) {
    banner('Could not load computations: ' + e.message, 'bad');
  }
  refresh();
  setInterval(refresh, POLL_MS);
})();
