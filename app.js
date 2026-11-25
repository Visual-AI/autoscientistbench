'use strict';

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const esc = (value) =>
  String(value ?? '').replace(/[&<>"']/g, (char) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  }[char]));

const fmt = (value) =>
  value === null || value === undefined || !Number.isFinite(Number(value))
    ? '-'
    : new Intl.NumberFormat('en', { maximumSignificantDigits: 5 }).format(value);

const percent = (value) => (Number.isFinite(value) ? Number(value).toFixed(1) : '-');
const percentage = (value) => (Number.isFinite(value) ? `${percent(value)}%` : '-');

const DOMAIN_PALETTE = {
  computer_science: '#0052ff',
  bioinformatics: '#0d9488',
  medicine: '#e11d48',
  mathematics: '#4f46e5',
  physics: '#d97706',
  economics: '#0284c7',
  chemistry: '#ea580c',
  statistics: '#7c3aed',
  environmental_science: '#15803d',
};

const MODEL_GRADIENTS = {
  gpt61sol: ['#059669', '#34d399'],
  kimi31: ['#1d4ed8', '#60a5fa'],
  deepseek41: ['#0284c7', '#38bdf8'],
  gpt6luna: ['#0d9488', '#5eead4'],
  qwen38flash: ['#4f46e5', '#818cf8'],
  qwen3827b: ['#6366f1', '#a5b4fc'],
  qwen36935: ['#7c3aed', '#c4b5fd'],
  qwen359b: ['#64748b', '#cbd5e1'],
};

const MODEL_EFFORT_FALLBACK = {
  gpt61sol: 'xhigh',
  gpt6luna: 'xhigh',
  deepseek41: 'max',
  qwen38flash: 'xhigh',
  qwen3827b: 'xhigh',
  qwen36935: 'xhigh',
  qwen359b: 'xhigh',
  kimi31: 'high',
};

const PROVIDER_SWATCHES = {
  OpenAI: '#059669',
  Moonshot: '#1d4ed8',
  DeepSeek: '#0284c7',
  Qwen: '#4f46e5',
};

const state = {
  data: null,
  visualManifest: null,
  visuals: null,
  taskMeta: {},
  atlasMode: 'art',
  atlasModel: 'all',
  paperDomain: 'all',
  paperIndex: 0,
  paperAutoPlay: !window.matchMedia('(prefers-reduced-motion: reduce)').matches,
  paperHoverPaused: false,
  paperInView: false,
  domain: 'all',
  query: '',
  layout: window.matchMedia('(max-width: 720px)').matches ? 'rows' : 'columns',
  sort: 'score',
  order: -1,
  taskDomain: 'all',
  taskQuery: '',
  taskPage: 1,
  taskView: 'grid',
  taskSort: 'original',
  taskShowAll: false,
  dialog: null,
};

const PAGE_SIZE = 12;
const PAPER_AUTOPLAY_MS = 3600;
let toastTimer;
let paperTimer;
let paperFilmstripKey = '';
let refreshing = false;

function domainColor(domainId, fallback = '#0052ff') {
  return DOMAIN_PALETTE[domainId] || fallback;
}

function modelGradient(model) {
  return MODEL_GRADIENTS[model.id] || [model.color || '#0052ff', '#60a5fa'];
}

function modelEffort(model) {
  return model.effort || MODEL_EFFORT_FALLBACK[model.id] || '';
}

function taskLabel(task) {
  const entry = state.taskMeta[task.id];
  if (entry) {
    return {
      num: entry.num,
      shortName: entry.shortName,
      displayCode: `${entry.num} · ${entry.shortName}`,
    };
  }
  const idx = state.data ? state.data.tasks.findIndex((t) => t.id === task.id) + 1 : 0;
  const num = String(Math.max(1, idx)).padStart(3, '0');
  const fallbackName = task.title.split(':')[0].trim().slice(0, 20);
  return { num, shortName: fallbackName, displayCode: `${num} · ${fallbackName}` };
}

function referenceImportDate() {
  return new Date(`${state.data.references.importedOn}T00:00:00Z`).toLocaleDateString('en-GB', {
    day: 'numeric',
    month: 'short',
    year: 'numeric',
    timeZone: 'UTC',
  });
}

function logo(model, className = '') {
  return `<img class="${className}" src="assets/${esc(model.logo)}.svg" alt="${esc(model.provider)}" width="20" height="20">`;
}

// Strictly expose ONLY the 8 published leaderboard models across the entire website
function publishedModels() {
  return state.data.models
    .filter((model) => model.publication === 'results')
    .sort((a, b) => (b.score ?? -Infinity) - (a.score ?? -Infinity) || a.name.localeCompare(b.name));
}

function selectedTasks(domain = state.domain) {
  return state.data.tasks.filter((task) => domain === 'all' || task.domain === domain);
}

// Uses the selected collection's own task count as denominator (104 overall, N for a discipline)
function summarize(model, tasks, denominator = tasks.length || state.data.metric.denominator) {
  const cells = tasks.map((task) => task.cells[model.id]);
  const passed = cells.filter((cell) => cell.passed).length;
  const values = cells.filter((cell) => Number.isFinite(cell.relativeScore));
  return {
    ...model,
    passed,
    total: tasks.length,
    score: values.length ? values.reduce((sum, cell) => sum + cell.relativeScore / denominator, 0) : null,
    scoreCoverage: values.length,
    scorePartial: values.length !== cells.length,
    scoreDenominator: denominator,
  };
}

function rankedModels() {
  const tasks = selectedTasks();
  const denom = tasks.length || state.data.metric.denominator;
  const models = publishedModels()
    .map((model) => summarize(model, tasks, denom))
    .sort((a, b) => (b.score ?? -Infinity) - (a.score ?? -Infinity) || a.name.localeCompare(b.name));

  let previousScore;
  let rank = 0;
  models.forEach((model, index) => {
    if (model.score === null) {
      model.rank = null;
      return;
    }
    if (model.score !== previousScore) rank = index + 1;
    model.rank = rank;
    previousScore = model.score;
  });

  const q = state.query.trim().toLowerCase();
  return models.filter(
    (model) =>
      !q ||
      `${model.name} ${modelEffort(model)} ${model.provider} ${model.harness}`
        .toLowerCase()
        .includes(q)
  );
}

function taskSummary(task) {
  const pub = publishedModels();
  let passCount = 0;
  let bestScore = -Infinity;
  let bestModel = null;
  let scoreSum = 0;
  let scoreCount = 0;

  const spectrum = pub.map((model) => {
    const cell = task.cells[model.id];
    if (cell.passed) passCount += 1;
    if (Number.isFinite(cell.relativeScore)) {
      scoreSum += cell.relativeScore;
      scoreCount += 1;
      if (cell.relativeScore > bestScore) {
        bestScore = cell.relativeScore;
        bestModel = model;
      }
    }
    return { model, passed: cell.passed, score: cell.relativeScore };
  });

  return {
    passCount,
    totalPublished: pub.length,
    meanScore: scoreCount ? scoreSum / scoreCount : null,
    bestScore: Number.isFinite(bestScore) ? bestScore : null,
    bestModel,
    spectrum,
  };
}

function scoreScale(values) {
  const finite = values.filter(Number.isFinite);
  const low = Math.min(0, ...finite);
  const high = Math.max(100, ...finite);
  const target = (high - low) / 4;
  const power = 10 ** Math.floor(Math.log10(target || 1));
  const step = ([1, 2, 2.5, 5, 10].find((n) => n * power >= target) || 10) * power;
  const min = Math.floor(low / step) * step;
  const max = Math.ceil(high / step) * step;
  const ticks = [];
  for (let value = max; value >= min - step / 10; value -= step) {
    ticks.push(Number(value.toPrecision(10)));
  }
  return { min, max, ticks, range: max - min, zero: ((0 - min) / (max - min)) * 100 };
}

function scoreGeometry(value, scale) {
  return {
    start: ((Math.min(value, 0) - scale.min) / scale.range) * 100,
    size: (Math.abs(value) / scale.range) * 100,
  };
}

function heatColors(ratio) {
  const t = Math.max(0, Math.min(1, ratio));
  if (t <= 0.001) return { bg: '#edf1f7', ink: '#64748b' };
  if (t < 0.25) return { bg: '#dbeafe', ink: '#1e3a8a' };
  if (t < 0.5) return { bg: '#93c5fd', ink: '#0f172a' };
  if (t < 0.75) return { bg: '#2563eb', ink: '#ffffff' };
  if (t < 0.99) return { bg: '#0d9488', ink: '#ffffff' };
  return { bg: '#047857', ink: '#ffffff' };
}

function toast(message) {
  clearTimeout(toastTimer);
  const el = $('#toast');
  el.textContent = message;
  el.hidden = false;
  toastTimer = setTimeout(() => {
    el.hidden = true;
  }, 3200);
}

/* ============================================================
   Floating Tooltip Helpers
   ============================================================ */
function showTooltip(html, event) {
  const tip = $('#hover-tooltip');
  tip.innerHTML = html;
  tip.hidden = false;
  positionTooltip(event);
}

function positionTooltip(event) {
  const tip = $('#hover-tooltip');
  if (tip.hidden) return;
  const pad = 16;
  const rect = tip.getBoundingClientRect();
  let x = event.clientX + 16;
  let y = event.clientY + 16;
  if (x + rect.width + pad > window.innerWidth) {
    x = Math.max(pad, event.clientX - rect.width - 16);
  }
  if (y + rect.height + pad > window.innerHeight) {
    y = Math.max(pad, event.clientY - rect.height - 16);
  }
  tip.style.left = `${x}px`;
  tip.style.top = `${y}px`;
}

function hideTooltip() {
  $('#hover-tooltip').hidden = true;
}

/* ============================================================
   104-Task Interactive Visual Overview
   ============================================================ */
function renderAtlasCluster(domainId, cols, fillSpan = 0) {
  const domain = state.data.domains.find((d) => d.id === domainId);
  if (!domain) return '';
  const color = domainColor(domain.id, domain.color);
  const tasks = state.data.tasks.filter((task) => task.domain === domain.id);
  const pub = publishedModels();
  const focusModel = state.atlasModel === 'all' ? null : pub.find((m) => m.id === state.atlasModel);

  const tilesHtml = tasks
    .map((task) => {
      const info = taskSummary(task);
      const lbl = taskLabel(task);
      let extraClass = state.atlasMode === 'art' ? 'art-mode' : state.atlasMode === 'papers' ? 'paper-mode' : 'heat-mode';
      let styleVars = `--d-color:${color};`;
      let inner = '';

      if (state.atlasMode === 'heatmap') {
        if (focusModel) {
          const cell = task.cells[focusModel.id];
          const val = Number.isFinite(cell.relativeScore) ? cell.relativeScore : 0;
          const colors = heatColors(val / 100);
          styleVars += `--heat-bg:${colors.bg};--heat-ink:${colors.ink};`;
          inner = `<span class="heat-val">${Number.isFinite(cell.relativeScore) ? Math.round(cell.relativeScore) : '-'}</span>`;
        } else {
          // Mean improvement score across all 8 leaderboard models
          const val = Number.isFinite(info.meanScore) ? info.meanScore : 0;
          const colors = heatColors(val / 80);
          styleVars += `--heat-bg:${colors.bg};--heat-ink:${colors.ink};`;
          inner = `<span class="heat-val">${Number.isFinite(info.meanScore) ? `${Math.round(info.meanScore)}` : '-'}</span>`;
        }
      } else {
        const src =
          state.atlasMode === 'art'
            ? `assets/task-art/${esc(task.id)}.svg`
            : `assets/papers/${esc(task.id)}.webp`;
        if (focusModel) {
          const cell = task.cells[focusModel.id];
          extraClass += cell.passed ? ' overlay-pass' : ' overlay-dim';
        }
        inner = `<img src="${src}" alt="${esc(lbl.shortName)}: ${esc(task.title)}"${state.atlasMode === 'papers' ? ' loading="lazy"' : ''}>`;
      }

      return `<button type="button" class="atlas-tile ${extraClass}" data-task="${esc(task.id)}" data-atlas-tile="${esc(task.id)}" style="${styleVars}" aria-label="${esc(lbl.displayCode)} (${esc(domain.name)})">${inner}</button>`;
    })
    .join('');

  let fillerHtml = '';
  if (fillSpan > 0) {
    const refModel = focusModel || pub[0];
    const domainStat = refModel ? summarize(refModel, tasks, tasks.length) : null;
    const fillerLabel = focusModel ? 'Model mean' : fillSpan >= 2 ? `${esc(refModel.short.replace('\n', ' '))} mean` : 'Top mean';
    fillerHtml = `<div class="cluster-metric-cell" style="--span:${fillSpan};--d-color:${color}">
      <strong>${domainStat ? percentage(domainStat.score) : `${tasks.length}`}</strong>
      <span>${fillerLabel}</span>
    </div>`;
  }

  return `<section class="discipline-cluster" style="--d-color:${color}" aria-label="${esc(domain.name)} (${tasks.length} tasks)">
    <button type="button" class="cluster-head" data-explore-domain="${esc(domain.id)}" title="Explore ${esc(domain.name)} in task collection">
      <span class="cluster-title"><i class="cluster-dot"></i><span>${esc(domain.name)}</span></span>
      <span class="cluster-count">${tasks.length}</span>
    </button>
    <div class="cluster-grid" style="--cols:${cols}">${tilesHtml}${fillerHtml}</div>
  </section>`;
}

function renderAtlas() {
  const pub = publishedModels();
  const focusModel = state.atlasModel === 'all' ? null : pub.find((m) => m.id === state.atlasModel);

  if (focusModel) {
    const sum = summarize(focusModel, state.data.tasks, state.data.tasks.length);
    $('#atlas-headline').textContent = `${focusModel.name} (${modelEffort(focusModel)}): ${percentage(sum.score)} improvement score (${sum.passed} / ${state.data.totalTasks} passed)`;
    $('#atlas-caption').textContent =
      state.atlasMode === 'heatmap'
        ? 'Each tile displays this model\'s improvement score (%) relative to baseline (0) and target ceiling (100).'
        : 'Passed tasks are highlighted in emerald; non-passing tasks are dimmed. Click any tile for details.';
  } else {
    $('#atlas-headline').textContent = `${state.data.totalTasks} research tasks across ${state.data.domains.length} disciplines`;
    $('#atlas-caption').textContent =
      state.atlasMode === 'heatmap'
        ? `Each tile shows the mean improvement score (%) across all ${pub.length} evaluated models.`
        : 'Hover any tile to inspect metrics and improvement scores, or click to open its full task record.';
  }

  const mosaicHtml = `
    <div class="atlas-tier-top">
      ${renderAtlasCluster('computer_science', 7, 0)}
      <div class="atlas-stack">
        ${renderAtlasCluster('bioinformatics', 4, 0)}
        ${renderAtlasCluster('statistics', 4, 0)}
      </div>
      <div class="atlas-stack">
        ${renderAtlasCluster('mathematics', 5, 0)}
        ${renderAtlasCluster('medicine', 5, 2)}
      </div>
    </div>
    <div class="atlas-tier-bottom">
      ${renderAtlasCluster('physics', 5, 1)}
      ${renderAtlasCluster('economics', 4, 0)}
      ${renderAtlasCluster('chemistry', 3, 1)}
      ${renderAtlasCluster('environmental_science', 2, 0)}
    </div>
  `;
  $('#atlas-mosaic').innerHTML = mosaicHtml;

  const legend = $('#atlas-legend-items');
  if (state.atlasMode === 'heatmap') {
    const stops = [
      { label: '0%', color: '#edf1f7' },
      { label: '1 to 24%', color: '#dbeafe' },
      { label: '25 to 49%', color: '#93c5fd' },
      { label: '50 to 74%', color: '#2563eb' },
      { label: '75 to 99%', color: '#0d9488' },
      { label: '100%+', color: '#047857' },
    ];
    legend.innerHTML = stops
      .map((item) => `<span class="legend-item"><i class="legend-swatch" style="--swatch:${item.color}"></i>${item.label}</span>`)
      .join('');
  } else if (focusModel) {
    legend.innerHTML = `
      <span class="legend-item"><i class="legend-swatch" style="--swatch:#059669"></i>Passed by ${esc(focusModel.name)}</span>
      <span class="legend-item"><i class="legend-swatch" style="--swatch:#cbd5e1"></i>Below threshold or failed</span>
    `;
  } else {
    const leader = pub[0];
    legend.innerHTML = `
      <span class="legend-item"><i class="legend-swatch" style="--swatch:#0052ff"></i>${state.data.totalTasks} executable tasks</span>
      <span class="legend-item"><i class="legend-swatch" style="--swatch:#059669"></i>Top improvement score: ${esc(leader.name)} (${percentage(leader.score)})</span>
      <span class="legend-item"><i class="legend-swatch" style="--swatch:#0a0e17"></i>${pub.length} evaluated models</span>
    `;
  }

  $$('[data-atlas-mode]').forEach((btn) => {
    const active = btn.dataset.atlasMode === state.atlasMode;
    btn.classList.toggle('selected', active);
    btn.setAttribute('aria-pressed', active);
  });
}

/* ============================================================
   Leaderboard Charts (Columns, Rows, Frontier Scatter, Matrix)
   ============================================================ */
function renderProviderLegend(models) {
  const seen = new Set();
  const providers = [];
  models.forEach((model) => {
    if (!seen.has(model.provider)) {
      seen.add(model.provider);
      providers.push({ name: model.provider, color: PROVIDER_SWATCHES[model.provider] || model.color });
    }
  });
  $('#provider-legend').innerHTML = providers
    .map((p) => `<span><i style="--color:${p.color}"></i>${esc(p.name)}</span>`)
    .join('');
}

function renderFrontierPlot(models) {
  const totalTasks = selectedTasks().length || 104;
  const maxScore = Math.max(55, Math.ceil((Math.max(...models.map((m) => m.score || 0)) + 8) / 10) * 10);
  const W = 1120;
  const H = 370;
  const pad = { top: 28, right: 44, bottom: 52, left: 64 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  const xPos = (passed) => pad.left + (passed / totalTasks) * plotW;
  const yPos = (score) => pad.top + plotH - (Math.max(0, score) / maxScore) * plotH;

  const sorted = [...models].sort((a, b) => a.passed - b.passed || a.score - b.score);
  const pareto = [];
  sorted.forEach((model) => {
    const dominated = models.some(
      (other) =>
        other.id !== model.id &&
        other.passed >= model.passed &&
        other.score >= model.score &&
        (other.passed > model.passed || other.score > model.score)
    );
    if (!dominated) pareto.push(model);
  });
  pareto.sort((a, b) => a.passed - b.passed);

  const yTicks = [];
  const yStep = maxScore <= 30 ? 5 : 10;
  for (let v = 0; v <= maxScore; v += yStep) yTicks.push(v);

  const xTickCount = Math.min(5, totalTasks);
  const xTicks = Array.from({ length: xTickCount + 1 }, (_, i) => Math.round((totalTasks * i) / xTickCount));

  const labelOffsets = {
    gpt61sol: { dx: -16, dy: 4, anchor: 'end' },
    kimi31: { dx: -16, dy: -6, anchor: 'end' },
    deepseek41: { dx: 16, dy: 4, anchor: 'start' },
    gpt6luna: { dx: 16, dy: 14, anchor: 'start' },
    qwen38flash: { dx: -16, dy: 5, anchor: 'end' },
    qwen3827b: { dx: -16, dy: -8, anchor: 'end' },
    qwen36935: { dx: 16, dy: 4, anchor: 'start' },
    qwen359b: { dx: 16, dy: -12, anchor: 'start' },
  };

  const gridY = yTicks
    .map(
      (v) => `<g>
        <line x1="${pad.left}" x2="${W - pad.right}" y1="${yPos(v)}" y2="${yPos(v)}" stroke="${v === 0 ? '#cbd3e1' : '#e5eaf2'}" stroke-dasharray="${v === 0 ? '' : '3 4'}"/>
        <text x="${pad.left - 12}" y="${yPos(v) + 4}" text-anchor="end" fill="#525e70" font-size="13">${v}%</text>
      </g>`
    )
    .join('');

  const gridX = xTicks
    .map(
      (v) => `<g>
        <line x1="${xPos(v)}" x2="${xPos(v)}" y1="${pad.top}" y2="${pad.top + plotH}" stroke="#eef2f8"/>
        <text x="${xPos(v)}" y="${pad.top + plotH + 22}" text-anchor="middle" fill="#525e70" font-size="13">${v}</text>
      </g>`
    )
    .join('');

  const paretoPath =
    pareto.length > 1
      ? `<path d="M ${pareto.map((m) => `${xPos(m.passed).toFixed(1)} ${yPos(m.score).toFixed(1)}`).join(' L ')}" fill="none" stroke="#0052ff" stroke-width="2" stroke-dasharray="5 5" opacity="0.5"/>`
      : '';

  const points = models
    .map((model) => {
      const cx = xPos(model.passed);
      const cy = yPos(model.score);
      const [gTop] = modelGradient(model);
      const offset = labelOffsets[model.id] || { dx: 16, dy: 4, anchor: 'start' };
      return `<g class="frontier-point" data-model="${esc(model.id)}" data-frontier-model="${esc(model.id)}" tabindex="0" role="button" aria-label="${esc(model.name)} (${esc(modelEffort(model))}): ${percentage(model.score)} score, ${model.passed} passed">
        <circle class="ring" cx="${cx}" cy="${cy}" r="10" fill="${gTop}" fill-opacity="0.16" stroke="${gTop}" stroke-width="2"/>
        <circle cx="${cx}" cy="${cy}" r="4.5" fill="${gTop}"/>
        <text x="${cx + offset.dx}" y="${cy + offset.dy}" text-anchor="${offset.anchor}" fill="#0a0e17" font-size="13.5" font-weight="600">${esc(model.name)}</text>
        <text x="${cx + offset.dx}" y="${cy + offset.dy + 15}" text-anchor="${offset.anchor}" fill="#525e70" font-size="12">${percentage(model.score)} · ${model.passed}/${totalTasks} · ${esc(modelEffort(model))}</text>
      </g>`;
    })
    .join('');

  return `<div class="frontier-wrap">
    <svg class="frontier-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Scatter plot comparing tasks passed against improvement score">
      ${gridX}
      ${gridY}
      ${paretoPath}
      ${points}
      <text x="${pad.left + plotW / 2}" y="${H - 8}" text-anchor="middle" fill="#525e70" font-size="13.5" font-weight="500">Tasks passed (out of ${totalTasks})</text>
      <text x="18" y="${pad.top + plotH / 2}" text-anchor="middle" fill="#525e70" font-size="13.5" font-weight="500" transform="rotate(-90 18 ${pad.top + plotH / 2})">Improvement score (%)</text>
    </svg>
  </div>`;
}

function renderDisciplineMatrix(models) {
  const domains = state.data.domains;
  const shortDomainName = {
    computer_science: 'CS',
    bioinformatics: 'Bioinfo.',
    mathematics: 'Math',
    environmental_science: 'Env. Sci.',
  };
  const headerCols = domains
    .map(
      (domain) => `<th>
        <button type="button" class="matrix-col-btn" data-matrix-domain="${esc(domain.id)}" title="${esc(domain.name)}: click to filter leaderboard">
          <span>${esc(shortDomainName[domain.id] || domain.name)}</span>
          <small>${domain.count}</small>
        </button>
      </th>`
    )
    .join('');

  const bodyRows = models
    .map((model) => {
      const cells = domains
        .map((domain) => {
          const domainTasks = state.data.tasks.filter((t) => t.domain === domain.id);
          const mean = summarize(model, domainTasks, domainTasks.length);
          const ratio = Number.isFinite(mean.score) ? mean.score / 85 : 0;
          const colors = heatColors(ratio);
          return `<td class="matrix-cell"
            data-matrix-cell="${esc(model.id)}:${esc(domain.id)}"
            data-model="${esc(model.id)}"
            style="--cell-bg:${colors.bg};--cell-ink:${colors.ink}"
            title="${esc(model.name)} (${esc(modelEffort(model))}) in ${esc(domain.name)}: ${percentage(mean.score)} mean (${mean.passed}/${domainTasks.length} passed)">
            ${percent(mean.score)}
          </td>`;
        })
        .join('');

      const overall = summarize(model, state.data.tasks, state.data.tasks.length);
      const overallColors = heatColors((overall.score || 0) / 85);
      return `<tr>
        <td class="matrix-model-cell">
          <button type="button" class="matrix-model-btn" data-model="${esc(model.id)}">
            ${logo(model)}
            <span>${esc(model.name)} <small style="font-weight:450;color:#6b7688">${esc(modelEffort(model))}</small></span>
          </button>
        </td>
        ${cells}
        <td class="matrix-cell matrix-overall" data-model="${esc(model.id)}" style="--cell-bg:${overallColors.bg};--cell-ink:${overallColors.ink}">
          ${percent(overall.score)}
        </td>
      </tr>`;
    })
    .join('');

  return `<div class="matrix-wrap">
    <p class="matrix-caption">Within-discipline mean improvement score (%). Select any discipline header to filter the leaderboard, or select a model row for details.</p>
    <table class="matrix-table">
      <thead>
        <tr>
          <th>Model</th>
          ${headerCols}
          <th>Overall<br><small style="font-weight:450;color:#525e70">104</small></th>
        </tr>
      </thead>
      <tbody>${bodyRows}</tbody>
    </table>
  </div>`;
}

function renderChart(models) {
  const chart = $('#performance-chart');
  const domain = state.data.domains.find((d) => d.id === state.domain);
  const total = selectedTasks().length;
  renderProviderLegend(models);

  $('#stage-scope').textContent = domain
    ? `${models.length} models · ${domain.name} (${total} tasks)`
    : `${models.length} models · All ${total} tasks`;

  if (state.layout === 'matrix') {
    $('#chart-subtitle').textContent =
      'Within-discipline mean improvement score (%) across all nine sciences. Overall averages across all 104 tasks.';
  } else if (state.layout === 'frontier') {
    $('#chart-subtitle').textContent = domain
      ? `${domain.name} (${total} tasks): tasks passed versus within-discipline improvement score.`
      : 'Tasks passed versus improvement score across all 104 tasks. Dashed curve traces the Pareto frontier.';
  } else {
    $('#chart-subtitle').textContent = domain
      ? `${domain.name}: mean improvement score across ${total} tasks (baseline = 0%, target ceiling = 100%).`
      : 'Improvement score relative to baseline (0%) and target ceiling (100%), averaged across all 104 tasks.';
  }

  const valid = models.filter((m) => Number.isFinite(m.score));
  if (!valid.length) {
    chart.innerHTML = '<div class="empty-panel">No models match the current filter.</div>';
    return;
  }

  if (state.layout === 'matrix') {
    chart.innerHTML = renderDisciplineMatrix(valid);
    return;
  }

  if (state.layout === 'frontier') {
    chart.innerHTML = renderFrontierPlot(valid);
    return;
  }

  const scale = scoreScale(valid.map((m) => m.score));

  if (state.layout === 'columns') {
    const grid = scale.ticks
      .map(
        (value) =>
          `<div class="grid-line ${value === 0 ? 'zero-line' : ''}" style="bottom:${((value - scale.min) / scale.range) * 100}%"><span>${fmt(value)}</span></div>`
      )
      .join('');
    const refLine = `<div class="reference-line" style="bottom:${((100 - scale.min) / scale.range) * 100}%"><span>Target ceiling 100%</span></div>`;
    const bars = valid
      .map((model) => {
        const geom = scoreGeometry(model.score, scale);
        const [gradTop, gradBottom] = modelGradient(model);
        const effort = modelEffort(model);
        return `<button type="button" class="bar-column" data-model="${esc(model.id)}" aria-label="${esc(model.name)} (${esc(effort)}): ${percentage(model.score)} improvement score" style="--grad-top:${gradTop};--grad-bottom:${gradBottom};--height:${geom.size}%;--bottom:${geom.start}%">
          <span class="bar-space">
            <span class="bar-fill">
              <span class="bar-value">${percent(model.score)}${model.scorePartial ? '*' : ''}</span>
            </span>
          </span>
          <span class="bar-label">
            ${logo(model)}
            <span class="bar-name">${esc(model.short)}</span>
            <span class="bar-effort">${esc(effort)}</span>
          </span>
        </button>`;
      })
      .join('');
    chart.innerHTML = `<div class="column-chart"><div class="chart-grid" aria-hidden="true">${grid}${refLine}</div><div class="chart-bars">${bars}</div></div>`;
  } else {
    const refPct = ((100 - scale.min) / scale.range) * 100;
    chart.innerHTML = `<div class="row-chart">${valid
      .map((model) => {
        const geom = scoreGeometry(model.score, scale);
        const [gradTop, gradBottom] = modelGradient(model);
        const effort = modelEffort(model);
        return `<button type="button" class="row-bar" data-model="${esc(model.id)}" style="--grad-top:${gradTop};--grad-bottom:${gradBottom};--start:${geom.start}%;--width:${geom.size}%;--ref:${refPct}%">
          <span class="row-bar-label">${logo(model)}<span>${esc(model.name)} <small style="font-weight:450;color:#6b7688">${esc(effort)}</small></span></span>
          <span class="row-track"><i></i><span class="row-ref-mark"></span></span>
          <span class="row-bar-stats"><strong>${percentage(model.score)}${model.scorePartial ? '*' : ''}</strong><span>${model.passed}/${total}</span></span>
        </button>`;
      })
      .join('')}</div>`;
  }
}

function renderLeaderboard() {
  const models = rankedModels();
  const sorted = [...models].sort((a, b) => {
    if (state.sort === 'name') return state.order * a.name.localeCompare(b.name);
    if (state.sort === 'passed') return state.order * (a.passed - b.passed) || (b.score ?? 0) - (a.score ?? 0);
    if (a.score === null) return b.score === null ? 0 : 1;
    if (b.score === null) return -1;
    return state.order * (a.score - b.score);
  });

  const total = selectedTasks().length;
  const maxBar = Math.max(50, ...models.map((m) => m.score || 0));

  $('#ranking-count').textContent = `${models.length} model${models.length === 1 ? '' : 's'}`;
  $('#score-note').textContent = `Improvement score = sum of task scores / ${total} (${state.domain === 'all' ? 'all active tasks' : 'tasks in selected discipline'}).${state.data.metric.clipNegative ? ' Negative task scores are clipped to 0.' : ''}`;
  $('#no-models').hidden = Boolean(models.length);

  $('#rankings-body').innerHTML = sorted
    .map((model) => {
      const [color] = modelGradient(model);
      const effort = modelEffort(model);
      const widthPct = Math.max(0, Math.min(100, ((model.score || 0) / maxBar) * 100));
      return `<tr class="${model.rank === 1 ? 'first-place' : ''}">
        <td class="rank-cell">${model.rank ?? '-'}</td>
        <td>
          <div class="model-cell">
            ${logo(model)}
            <div class="model-meta">
              <div class="model-name-line">
                <button type="button" class="model-name" data-model="${esc(model.id)}">${esc(model.name)}</button>
                <span class="model-effort">${esc(effort)}</span>
              </div>
              <span class="model-sub">${esc(model.provider)}</span>
            </div>
          </div>
        </td>
        <td class="harness-cell">${esc(model.harness)}</td>
        <td class="numeric passed-cell">${model.passed} / ${total}</td>
        <td class="numeric score-cell">
          <div class="score-inline">
            <span class="score-micro-track" aria-hidden="true"><span style="width:${widthPct}%;--color:${color}"></span></span>
            <span class="score-figure">${percent(model.score)}${model.score === null ? '' : '%'}${model.scorePartial ? '*' : ''}</span>
          </div>
        </td>
        <td class="col-action"><button type="button" class="row-open" data-model="${esc(model.id)}" aria-label="Inspect ${esc(model.name)}">↗</button></td>
      </tr>`;
    })
    .join('');

  $$('.sort-trigger').forEach((btn) => {
    const active = btn.dataset.sort === state.sort;
    btn.querySelector('span').textContent = active ? (state.order === 1 ? '↑' : '↓') : '↕';
    btn.closest('th').setAttribute('aria-sort', active ? (state.order === 1 ? 'ascending' : 'descending') : 'none');
  });

  renderChart(models);
}

/* ============================================================
   Interactive 3D Archival Paper Observatory (Gliding Auto-Play)
   ============================================================ */
function filteredPapers() {
  return state.data.tasks.filter((t) => state.paperDomain === 'all' || t.domain === state.paperDomain);
}

function updatePlayButton() {
  const btn = $('#paper-play');
  if (!btn) return;
  const playing = state.paperAutoPlay;
  btn.setAttribute('aria-pressed', String(playing));
  btn.setAttribute('aria-label', playing ? 'Pause auto-play' : 'Resume auto-play');
  btn.title = playing ? 'Pause auto-play' : 'Resume auto-play';
  btn.innerHTML = playing
    ? '<svg viewBox="0 0 20 20" width="14" height="14" fill="currentColor" aria-hidden="true"><rect x="4.5" y="3.5" width="3.5" height="13" rx="1"/><rect x="12" y="3.5" width="3.5" height="13" rx="1"/></svg>'
    : '<svg viewBox="0 0 20 20" width="14" height="14" fill="currentColor" aria-hidden="true"><path d="M5.5 3.8a1 1 0 0 1 1.5-.86l9.6 6.2a1 1 0 0 1 0 1.72l-9.6 6.2a1 1 0 0 1-1.5-.86V3.8Z"/></svg>';
}

function syncPaperAutoplay(resetProgress = true) {
  clearTimeout(paperTimer);
  const fill = $('#deck-progress-fill');
  const dialogOpen = $('#detail-dialog')?.open;
  const active =
    Boolean(state.data) &&
    state.paperAutoPlay &&
    !state.paperHoverPaused &&
    state.paperInView &&
    !dialogOpen &&
    !document.hidden &&
    filteredPapers().length > 1;

  if (fill) {
    if (!state.paperAutoPlay || !state.paperInView || filteredPapers().length <= 1) {
      fill.className = '';
      fill.style.transform = 'scaleX(0)';
    } else if (state.paperHoverPaused || dialogOpen) {
      fill.classList.add('paused');
    } else if (resetProgress) {
      fill.className = '';
      void fill.offsetWidth;
      fill.className = 'running';
    } else {
      fill.classList.remove('paused');
      if (!fill.classList.contains('running')) fill.classList.add('running');
    }
  }

  if (active) {
    paperTimer = setTimeout(() => {
      if (!state.data) return;
      state.paperIndex += 1;
      renderPaperObservatory(1);
    }, PAPER_AUTOPLAY_MS);
  }
}

function applyDeckTransform(card, off, stepX) {
  const abs = Math.abs(off);
  const tx = off * stepX;
  const tz = -abs * 110;
  const ry = -off * 18;
  const sc = abs === 0 ? 1 : abs === 1 ? 0.84 : abs === 2 ? 0.68 : 0.56;
  const op = abs === 0 ? 1 : abs === 1 ? 0.82 : abs === 2 ? 0.45 : 0;
  const zi = Math.max(1, 10 - abs);
  card.style.setProperty('--tx', `${tx}px`);
  card.style.setProperty('--tz', `${tz}px`);
  card.style.setProperty('--ry', `${ry}deg`);
  card.style.setProperty('--sc', String(sc));
  card.style.setProperty('--op', String(op));
  card.style.setProperty('--zi', String(zi));
  card.classList.toggle('is-center', abs === 0);
}

function renderPaperObservatory(direction = 0) {
  const papers = filteredPapers();
  if (!papers.length) return;
  state.paperIndex = ((state.paperIndex % papers.length) + papers.length) % papers.length;

  const tabs = [{ id: 'all', name: 'All', count: state.data.totalTasks, color: '#0052ff' }, ...state.data.domains];
  $('#paper-domain-tabs').innerHTML = tabs
    .map((d) => {
      const active = d.id === state.paperDomain;
      const color = d.id === 'all' ? '#0052ff' : domainColor(d.id, d.color);
      const tabLabel = d.id === 'environmental_science' ? 'Env. Science' : d.name;
      return `<button type="button" class="paper-tab ${active ? 'selected' : ''}" role="tab" aria-selected="${active}" data-paper-domain="${esc(d.id)}" style="--d-color:${color}" title="${esc(d.name)}">
        <i></i><span>${esc(tabLabel)}</span><span class="tab-count">${d.count}</span>
      </button>`;
    })
    .join('');

  const activeTask = papers[state.paperIndex];
  const activeDomain = state.data.domains.find((d) => d.id === activeTask.domain);
  const activeColor = domainColor(activeDomain.id, activeDomain.color);
  const activeLabel = taskLabel(activeTask);
  const activeMeta = taskSummary(activeTask);

  $('#paper-counter').innerHTML = `${String(state.paperIndex + 1).padStart(3, '0')} <small>/ ${String(papers.length).padStart(3, '0')}</small>`;
  $('#paper-observatory').style.setProperty('--active-color', activeColor);

  // Persistent keyed 3D coverflow card updates so cards smoothly glide across positions
  const deck = $('#paper-deck');
  const offsets = papers.length >= 5 ? [-2, -1, 0, 1, 2] : papers.length >= 3 ? [-1, 0, 1] : [0];
  const isCompact = window.matchMedia('(max-width: 720px)').matches;
  const stepX = isCompact ? 102 : 162;

  const targetById = new Map();
  offsets.forEach((off) => {
    const idx = (state.paperIndex + off + papers.length) % papers.length;
    const task = papers[idx];
    targetById.set(task.id, { off, idx, task });
  });

  const existing = new Map();
  deck.querySelectorAll('.deck-card[data-paper-id]').forEach((el) => {
    if (!el.classList.contains('is-exiting')) {
      existing.set(el.dataset.paperId, el);
    }
  });

  // Smoothly glide exiting cards off the far wing before removal
  existing.forEach((el, pid) => {
    if (!targetById.has(pid)) {
      const prevOff = Number(el.dataset.offset || 0);
      const exitOff = direction !== 0 ? (direction > 0 ? -3 : 3) : prevOff < 0 ? -3 : 3;
      el.classList.add('is-exiting');
      applyDeckTransform(el, exitOff, stepX);
      setTimeout(() => el.remove(), 640);
    }
  });

  // Update retained cards in place, and smoothly animate newly entering cards from the wing
  targetById.forEach(({ off, idx, task }, pid) => {
    const lbl = taskLabel(task);
    let card = existing.get(pid);
    const isNew = !card;
    if (isNew) {
      card = document.createElement('button');
      card.type = 'button';
      card.className = 'deck-card';
      card.dataset.paperId = pid;
      card.innerHTML = `<img src="assets/papers/${esc(task.id)}.webp" alt="First page of ${esc(task.title)}">`;
      const enterOff = direction > 0 ? 3 : direction < 0 ? -3 : off;
      applyDeckTransform(card, enterOff, stepX);
      deck.appendChild(card);
      if (enterOff !== off) {
        void card.offsetWidth;
      }
    }

    card.dataset.offset = String(off);
    if (off === 0) {
      card.dataset.task = task.id;
      delete card.dataset.paperJump;
      card.setAttribute('aria-label', `Inspect ${lbl.displayCode}`);
    } else {
      card.dataset.paperJump = String(idx);
      delete card.dataset.task;
      card.setAttribute('aria-label', `Focus ${lbl.displayCode}`);
    }
    applyDeckTransform(card, off, stepX);
  });

  // Synchronized Live Dossier with smooth entrance animation
  const dossier = $('#paper-dossier');
  dossier.innerHTML = `
    <div class="dossier-animate">
      <div class="dossier-meta">
        <span class="dossier-code"><i></i>${esc(activeLabel.displayCode)}</span>
        <span class="dossier-domain">${esc(activeDomain.name)}</span>
      </div>
      <h3 class="dossier-title">${esc(activeTask.title)}</h3>
      <p class="dossier-objective">${esc(activeTask.objective || 'Inspect baseline code, research contract, and model evaluations.')}</p>
    </div>

    <div class="dossier-preview-row dossier-animate">
      <button type="button" class="dossier-diagram" data-task="${esc(activeTask.id)}" aria-label="Open full diagram for ${esc(activeLabel.shortName)}">
        <img src="assets/task-art/${esc(activeTask.id)}.svg" alt="Scientific diagram for ${esc(activeLabel.shortName)}">
      </button>
      <div class="dossier-stats">
        <div><span>Metric</span><strong>${activeTask.higherIsBetter ? '↑' : '↓'} ${esc(activeTask.metric)}</strong></div>
        <div><span>Models passed</span><strong>${activeMeta.passCount} / ${activeMeta.totalPublished}</strong></div>
        <div><span>Baseline / Ceiling</span><strong>${fmt(activeTask.baseline)} → ${fmt(activeTask.humanReference)}</strong></div>
        <div><span>Top score</span><strong>${percentage(activeMeta.bestScore)}</strong></div>
      </div>
    </div>

    <div class="dossier-actions">
      <button type="button" class="studio-cta" data-task="${esc(activeTask.id)}">Inspect task results ↗</button>
      <a class="studio-link" href="assets/papers/${esc(activeTask.id)}.pdf" target="_blank" rel="noopener">Open first-page PDF ↗</a>
    </div>
  `;

  // Reuse micro-filmstrip DOM unless filter collection changed (prevents image flicker during auto-play!)
  const strip = $('#paper-filmstrip');
  const stripKey = `${state.paperDomain}:${papers.length}`;
  if (paperFilmstripKey !== stripKey) {
    paperFilmstripKey = stripKey;
    strip.innerHTML = papers
      .map((task, idx) => {
        const dom = state.data.domains.find((d) => d.id === task.domain);
        const lbl = taskLabel(task);
        return `<button type="button" class="filmstrip-item" data-paper-jump="${idx}" style="--d-color:${domainColor(dom.id, dom.color)}" title="${esc(lbl.displayCode)}: ${esc(task.title)}" aria-label="${esc(lbl.displayCode)}">
          <img src="assets/papers/${esc(task.id)}.webp" alt="" loading="lazy">
        </button>`;
      })
      .join('');
  }

  strip.querySelectorAll('.filmstrip-item').forEach((el, idx) => {
    el.classList.toggle('active', idx === state.paperIndex);
  });

  const activeThumb = strip.querySelector('.filmstrip-item.active');
  if (activeThumb) {
    const offsetLeft = activeThumb.offsetLeft - strip.clientWidth / 2 + activeThumb.clientWidth / 2;
    strip.scrollTo({ left: Math.max(0, offsetLeft), behavior: 'smooth' });
  }

  updatePlayButton();
  syncPaperAutoplay(true);
}

/* ============================================================
   Task Collection Explorer (001-104 + Paper Short Name)
   ============================================================ */
function renderTaskFilters() {
  const items = [{ id: 'all', name: 'All disciplines', count: state.data.totalTasks }, ...state.data.domains];
  $('#domain-filters').innerHTML = items
    .map((domain) => {
      const selected = domain.id === state.taskDomain;
      const color = domain.id === 'all' ? '#0052ff' : domainColor(domain.id, domain.color);
      return `<button type="button" class="rail-option ${selected ? 'selected' : ''}" data-task-domain="${esc(domain.id)}" aria-pressed="${selected}" style="--d-color:${color}">
        <span class="rail-check" aria-hidden="true"></span>
        ${domain.id === 'all' ? '' : '<i class="rail-dot" aria-hidden="true"></i>'}
        <span class="rail-name">${esc(domain.name)}</span>
        <span class="rail-count">${domain.count}</span>
      </button>`;
    })
    .join('');
}

function renderTasks() {
  const q = state.taskQuery.trim().toLowerCase();
  const tasks = state.data.tasks
    .map((task) => ({ task, meta: taskSummary(task), lbl: taskLabel(task) }))
    .filter(({ task, lbl }) => {
      if (state.taskDomain !== 'all' && state.taskDomain !== task.domain) return false;
      if (!q) return true;
      return `${lbl.num} ${lbl.shortName} ${task.title} ${task.metric} ${task.objective}`.toLowerCase().includes(q);
    });

  if (state.taskSort === 'hardest') {
    tasks.sort((a, b) => a.meta.passCount - b.meta.passCount || (a.meta.bestScore ?? 0) - (b.meta.bestScore ?? 0));
  } else if (state.taskSort === 'easiest') {
    tasks.sort((a, b) => b.meta.passCount - a.meta.passCount || (b.meta.bestScore ?? 0) - (a.meta.bestScore ?? 0));
  } else if (state.taskSort === 'topscore') {
    tasks.sort((a, b) => (b.meta.bestScore ?? -1) - (a.meta.bestScore ?? -1));
  } else if (state.taskSort === 'shortname') {
    tasks.sort((a, b) => a.lbl.shortName.localeCompare(b.lbl.shortName));
  } else if (state.taskSort === 'title') {
    tasks.sort((a, b) => a.task.title.localeCompare(b.task.title));
  } else if (state.taskSort === 'domain') {
    tasks.sort((a, b) => a.task.domain.localeCompare(b.task.domain) || a.lbl.num.localeCompare(b.lbl.num));
  }

  const pages = Math.max(1, Math.ceil(tasks.length / PAGE_SIZE));
  state.taskPage = Math.min(state.taskPage, pages);
  const visible = state.taskShowAll
    ? tasks
    : tasks.slice((state.taskPage - 1) * PAGE_SIZE, state.taskPage * PAGE_SIZE);

  $('#task-count').textContent = `${tasks.length} of ${state.data.totalTasks} tasks`;
  $('#task-list-columns').hidden = state.taskView !== 'list';

  const grid = $('#task-grid');
  grid.className = `task-collection ${state.taskView === 'list' ? 'list-mode' : 'grid-mode'}`;

  if (state.taskView === 'list') {
    grid.innerHTML = visible
      .map(({ task, meta, lbl }) => {
        const domain = state.data.domains.find((d) => d.id === task.domain);
        const color = domainColor(domain.id, domain.color);
        const ticks = meta.spectrum
          .map(
            (item) =>
              `<span class="spectrum-tick ${item.passed ? 'passed' : 'failed'}" title="${esc(item.model.name)} (${esc(modelEffort(item.model))}): ${item.passed ? 'Passed' : 'Did not pass'} (${percentage(item.score)})"></span>`
          )
          .join('');
        return `<button type="button" class="task-row" data-task="${esc(task.id)}" style="--d-color:${color}">
          <span class="task-thumb">
            <img src="assets/task-art/${esc(task.id)}.svg" alt="Scientific diagram for ${esc(lbl.shortName)}" loading="lazy">
          </span>
          <span class="task-main">
            <span class="task-discipline-line"><i></i>${esc(domain.name)} <span class="task-code">${esc(lbl.displayCode)}</span></span>
            <h3>${esc(task.title)}</h3>
            <span class="task-obj">${esc(task.objective || 'Inspect baseline, research contract, and model evaluations.')}</span>
          </span>
          <span class="task-spectrum">
            <span class="spectrum-bars" aria-hidden="true">${ticks}</span>
            <span class="spectrum-caption">${meta.passCount}/${meta.totalPublished} passed · Top ${percentage(meta.bestScore)}</span>
          </span>
          <span class="task-metric-col">
            <span class="task-metric-name">${task.higherIsBetter ? '↑' : '↓'} ${esc(task.metric)}</span>
            <span class="task-metric-dir">${task.higherIsBetter ? 'Higher is better' : 'Lower is better'}</span>
          </span>
        </button>`;
      })
      .join('');
  } else {
    grid.innerHTML = visible
      .map(({ task, meta, lbl }) => {
        const domain = state.data.domains.find((d) => d.id === task.domain);
        const color = domainColor(domain.id, domain.color);
        return `<button type="button" class="task-gallery-item" data-task="${esc(task.id)}" style="--d-color:${color}">
          <span class="task-thumb">
            <img src="assets/task-art/${esc(task.id)}.svg" alt="Scientific diagram for ${esc(lbl.shortName)}" loading="lazy">
          </span>
          <span class="task-gallery-body">
            <span class="task-discipline-line"><i></i>${esc(domain.name)} <span class="task-code">${esc(lbl.displayCode)}</span></span>
            <h3>${esc(task.title)}</h3>
            <span class="task-obj">${esc(task.objective || 'Inspect baseline, research contract, and model evaluations.')}</span>
            <span class="task-gallery-footer">
              <span>${meta.passCount}/${meta.totalPublished} passed</span>
              <span>${task.higherIsBetter ? '↑' : '↓'} ${esc(task.metric)}</span>
            </span>
          </span>
        </button>`;
      })
      .join('');
  }

  $$('[data-task-view]').forEach((btn) => {
    const active = btn.dataset.taskView === state.taskView;
    btn.classList.toggle('selected', active);
    btn.setAttribute('aria-pressed', active);
  });

  $('#no-tasks').hidden = Boolean(tasks.length);
  $('#pagination-bar').hidden = tasks.length <= PAGE_SIZE;
  $('#page-counter').textContent = state.taskShowAll ? `Showing all ${tasks.length}` : `${state.taskPage} / ${pages}`;
  $('#tasks-prev').disabled = state.taskShowAll || state.taskPage <= 1;
  $('#tasks-next').disabled = state.taskShowAll || state.taskPage >= pages;
  $('#tasks-show-all').textContent = state.taskShowAll ? 'Paginate (12 per page)' : `Show all ${tasks.length}`;
}

/* ============================================================
   Inspector Modal (Model & Task — strictly 8 leaderboard models)
   ============================================================ */
function showModel(id, open = true) {
  const model = publishedModels().find((m) => m.id === id);
  if (!model) return;
  state.dialog = { type: 'model', id };
  $('#dialog-context').textContent = 'Model evaluation record';

  const effort = modelEffort(model);
  const summaries = state.data.domains.map((domain) => {
    const domainTasks = selectedTasks(domain.id);
    return {
      ...domain,
      color: domainColor(domain.id, domain.color),
      mean: summarize(model, domainTasks, domainTasks.length),
    };
  });

  const topTasks = [...state.data.tasks]
    .map((task) => ({ task, cell: task.cells[model.id], lbl: taskLabel(task) }))
    .filter((item) => Number.isFinite(item.cell.relativeScore) && item.cell.relativeScore > 0)
    .sort((a, b) => b.cell.relativeScore - a.cell.relativeScore)
    .slice(0, 5);

  $('#dialog-content').innerHTML = `
    <div class="dialog-title-row">
      ${logo(model)}
      <h2 id="dialog-title">${esc(model.name)}</h2>
    </div>
    <div class="dialog-sub">
      <span>${esc(model.provider)} · ${esc(model.harness)} · Reasoning effort: ${esc(effort)}</span>
    </div>

    <div class="dialog-metrics">
      <div>
        <strong>${percentage(model.score)}</strong>
        <span>Improvement score</span>
      </div>
      <div>
        <strong>${model.passed} / ${model.total}</strong>
        <span>Tasks passed</span>
      </div>
      <div>
        <strong>${esc(effort)}</strong>
        <span>Reasoning effort</span>
      </div>
    </div>

    <h3 class="dialog-heading">Performance by discipline (within-discipline mean)</h3>
    <div class="breakdown-list">
      ${summaries
        .map((domain) => {
          const pct = Math.min(100, Math.max(0, domain.mean.score || 0));
          return `<div class="breakdown-row">
            <span>${esc(domain.name)} (${domain.count})</span>
            <span class="breakdown-track" style="--d-color:${domain.color}"><span style="width:${pct}%"></span></span>
            <span class="breakdown-val">${percentage(domain.mean.score)}</span>
          </div>`;
        })
        .join('')}
    </div>

    ${
      topTasks.length
        ? `<h3 class="dialog-heading">Highest-scoring tasks</h3>
           <table class="detail-table">
             <thead><tr><th>Task</th><th class="numeric">Improvement</th><th class="numeric">Verdict</th></tr></thead>
             <tbody>
               ${topTasks
                 .map(
                   ({ task, cell, lbl }) => `<tr>
                     <td><button type="button" class="model-name" data-task="${esc(task.id)}">${esc(lbl.displayCode)}: ${esc(task.title)}</button></td>
                     <td class="numeric">${percentage(cell.relativeScore)}</td>
                     <td class="numeric ${cell.passed ? 'result-pass' : 'result-fail'}">${cell.passed ? 'Passed' : 'Below threshold'}</td>
                   </tr>`
                 )
                 .join('')}
             </tbody>
           </table>`
        : ''
    }

    <p class="dialog-note">
      Overall improvement score averages task scores over all ${model.total} tasks. ${state.data.metric.clipNegative ? 'Negative task scores are clipped to 0 before aggregation. ' : ''}Target ceiling anchors synchronized ${esc(referenceImportDate())}.
    </p>
  `;

  if (open && !$('#detail-dialog').open) {
    $('#detail-dialog').showModal();
    syncPaperAutoplay(false);
  }
}

function showTask(id, open = true) {
  const task = state.data.tasks.find((t) => t.id === id);
  if (!task) return;
  state.dialog = { type: 'task', id };
  const domain = state.data.domains.find((d) => d.id === task.domain);
  const lbl = taskLabel(task);
  $('#dialog-context').textContent = `${domain.name} · ${lbl.displayCode}`;

  const statusText = (cell) =>
    cell.relativeScoreStatus === 'user_zero'
      ? 'Score 0'
      : cell.passed
      ? 'Passed'
      : cell.category === 'scored'
      ? 'Below threshold'
      : cell.category === 'agent'
      ? 'Agent failure'
      : 'Not passed';

  // Strictly show ONLY the 8 published leaderboard models
  const orderedModels = publishedModels();

  $('#dialog-content').innerHTML = `
    <h2 id="dialog-title">${esc(task.title)}</h2>
    <div class="dialog-sub">
      <span>${task.higherIsBetter ? '↑ Higher is better' : '↓ Lower is better'}</span>
      <span>Metric: ${esc(task.metric)}</span>
      <span>${esc(task.seeds)} evaluation seed${task.seeds === 1 ? '' : 's'}</span>
    </div>
    ${task.objective ? `<p style="margin-top:14px;color:var(--ink-secondary)">${esc(task.objective)}</p>` : ''}

    <div class="dialog-visual-grid">
      <div class="dialog-art">
        <img src="assets/task-art/${esc(task.id)}.svg" alt="Scientific diagram for ${esc(lbl.shortName)}">
      </div>
      <a class="dialog-paper-card" href="assets/papers/${esc(task.id)}.pdf" target="_blank" rel="noopener">
        <img src="assets/papers/${esc(task.id)}.webp" alt="First page preview of ${esc(task.title)}">
        <span>Open paper PDF ↗</span>
      </a>
    </div>

    <div class="dialog-metrics">
      <div>
        <strong>${fmt(task.baseline)}</strong>
        <span>Reproduced baseline (0%)</span>
      </div>
      <div>
        <strong>${fmt(task.humanReference)}</strong>
        <span>Target ceiling (100%)</span>
      </div>
      <div>
        <strong>${fmt(task.humanReference === null ? null : task.humanReference - task.baseline)}</strong>
        <span>Target span</span>
      </div>
    </div>

    <h3 class="dialog-heading">Model results on this task</h3>
    <table class="detail-table">
      <thead>
        <tr>
          <th>Model</th>
          <th class="numeric">Metric (xi)</th>
          <th class="numeric">Improvement</th>
          <th class="numeric">Outcome</th>
        </tr>
      </thead>
      <tbody>
        ${orderedModels
          .map((model) => {
            const cell = task.cells[model.id];
            const cls = cell.passed ? 'result-pass' : 'result-fail';
            return `<tr>
              <td><button type="button" class="model-name" data-model="${esc(model.id)}" style="font-size:inherit">${esc(model.name)}</button> <span class="model-effort">${esc(modelEffort(model))}</span></td>
              <td class="numeric">${fmt(cell.metric)}</td>
              <td class="numeric">${percentage(cell.relativeScore)}</td>
              <td class="numeric ${cls}">${statusText(cell)}</td>
            </tr>`;
          })
          .join('')}
      </tbody>
    </table>

    <p class="dialog-note">
      Formula: (xi - baseline) / (target_ceiling - baseline) × 100%. Target ceilings synchronized ${esc(referenceImportDate())}. ${state.data.metric.clipNegative ? 'Negative task scores are clipped to 0.' : 'Negative task scores are retained.'} Scores above 100% are uncapped.
    </p>
  `;

  if (open && !$('#detail-dialog').open) {
    $('#detail-dialog').showModal();
    syncPaperAutoplay(false);
  }
}

/* ============================================================
   Render All & Live Data Refresh
   ============================================================ */
function renderAll() {
  const data = state.data;
  const results = publishedModels();

  const scorePolicyEl = $('#score-policy');
  if (scorePolicyEl) {
    scorePolicyEl.textContent = data.metric.clipNegative
      ? 'Negative task scores are clipped to 0% before aggregation, while solutions that surpass the target ceiling remain uncapped above 100%.'
      : 'Negative task scores and scores above 100% are retained directly as computed.';
  }

  const dateText = new Date(data.updatedAt).toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    timeZone: 'UTC',
  });
  $$('.snapshot-date').forEach((el) => {
    el.textContent = `Updated ${dateText}`;
    el.title = `Source export: ${new Date(data.updatedAt).toISOString()}`;
  });

  $('#atlas-model').innerHTML = `<option value="all">All models (improvement score)</option>${results
    .map((m) => `<option value="${esc(m.id)}">${esc(m.name)} · ${esc(modelEffort(m))} (${percentage(m.score)})</option>`)
    .join('')}`;
  $('#atlas-model').value = state.atlasModel;

  $('#leader-domain').innerHTML = `<option value="all">All ${data.totalTasks} tasks</option>${data.domains
    .map((d) => `<option value="${esc(d.id)}">${esc(d.name)} (${d.count})</option>`)
    .join('')}`;
  $('#leader-domain').value = state.domain;

  renderAtlas();
  renderLeaderboard();
  renderPaperObservatory();
  renderTaskFilters();
  renderTasks();

  if ($('#detail-dialog').open && state.dialog) {
    if (state.dialog.type === 'model') showModel(state.dialog.id, false);
    else showTask(state.dialog.id, false);
  }
}

async function refresh(manual = false) {
  if (refreshing) return;
  refreshing = true;
  $$('.refresh-button').forEach((btn) => (btn.disabled = true));
  try {
    const [response, visualResponse, metaResponse] = await Promise.all([
      fetch(`data.json?t=${Date.now()}`, { cache: 'no-store' }),
      fetch(`assets/task-art/manifest.json?t=${Date.now()}`, { cache: 'no-store' }),
      fetch(`assets/task-meta.json?t=${Date.now()}`, { cache: 'no-store' }),
    ]);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    if (!visualResponse.ok) throw new Error(`Manifest HTTP ${visualResponse.status}`);
    const data = await response.json();
    const visualManifest = await visualResponse.json();
    const metaJson = metaResponse.ok ? await metaResponse.json() : { tasks: {} };

    if (!Array.isArray(data.tasks) || data.tasks.length !== data.totalTasks || !Array.isArray(data.models)) {
      throw new Error('Invalid benchmark snapshot');
    }
    const metaChanged = JSON.stringify(state.taskMeta) !== JSON.stringify(metaJson.tasks || {});
    const changed =
      state.data?.exportedAt !== data.exportedAt ||
      state.data?.sourceHash !== data.sourceHash ||
      JSON.stringify(state.data?.models) !== JSON.stringify(data.models) ||
      JSON.stringify(state.visualManifest) !== JSON.stringify(visualManifest) ||
      metaChanged;

    state.data = data;
    state.visualManifest = visualManifest;
    state.visuals = Object.fromEntries(visualManifest.visuals.map((v) => [v.taskId, v]));
    state.taskMeta = metaJson.tasks || {};
    $('#load-error').hidden = true;
    if (changed) renderAll();
    if (manual) toast(changed ? 'Benchmark results refreshed.' : 'Already viewing the latest snapshot.');
  } catch (error) {
    $('#load-error').textContent = state.data
      ? 'Could not refresh benchmark results. Showing the previous snapshot.'
      : 'Benchmark data could not be loaded. Ensure the webpage server is running and reload.';
    $('#load-error').hidden = false;
    if (manual) toast('Refresh failed. Please retry.');
    console.error('Benchmark data load failed:', error);
  } finally {
    refreshing = false;
    $$('.refresh-button').forEach((btn) => (btn.disabled = false));
  }
}

function downloadFile(content, type, filename) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/* ============================================================
   Event Wiring & Scroll Spy
   ============================================================ */
document.addEventListener('click', (event) => {
  const frontierTarget = event.target.closest('[data-frontier-model]');
  if (frontierTarget && state.data) {
    showModel(frontierTarget.dataset.frontierModel);
    return;
  }

  const matrixCell = event.target.closest('[data-matrix-cell]');
  if (matrixCell && state.data) {
    showModel(matrixCell.dataset.model);
    return;
  }

  const button = event.target.closest('button');
  if (!button) return;

  if (button.classList.contains('refresh-button')) {
    refresh(true);
    return;
  }
  if (!state.data) return;

  if (button.dataset.atlasMode) {
    state.atlasMode = button.dataset.atlasMode;
    renderAtlas();
    return;
  }

  if (button.dataset.paperDomain) {
    state.paperDomain = button.dataset.paperDomain;
    state.paperIndex = 0;
    $('#paper-deck').innerHTML = '';
    renderPaperObservatory(0);
    return;
  }

  if (button.dataset.paperJump !== undefined) {
    const targetIdx = Number(button.dataset.paperJump);
    const papers = filteredPapers();
    const len = papers.length;
    const forward = (targetIdx - state.paperIndex + len) % len;
    const backward = (state.paperIndex - targetIdx + len) % len;
    const dir = forward === 0 ? 0 : forward <= backward ? 1 : -1;
    state.paperIndex = targetIdx;
    renderPaperObservatory(dir);
    return;
  }

  if (button.dataset.layout) {
    state.layout = button.dataset.layout;
    $$('[data-layout]').forEach((b) => {
      const active = b === button;
      b.classList.toggle('selected', active);
      b.setAttribute('aria-pressed', active);
    });
    renderLeaderboard();
    return;
  }

  if (button.dataset.matrixDomain) {
    state.domain = button.dataset.matrixDomain;
    $('#leader-domain').value = state.domain;
    state.layout = 'columns';
    $$('[data-layout]').forEach((b) => {
      const active = b.dataset.layout === 'columns';
      b.classList.toggle('selected', active);
      b.setAttribute('aria-pressed', active);
    });
    renderLeaderboard();
    toast(`Filtered leaderboard to ${state.data.domains.find((d) => d.id === state.domain)?.name}.`);
    return;
  }

  if (button.dataset.sort) {
    if (state.sort === button.dataset.sort) state.order *= -1;
    else {
      state.sort = button.dataset.sort;
      state.order = state.sort === 'name' ? 1 : -1;
    }
    renderLeaderboard();
    return;
  }

  if (button.dataset.task) {
    hideTooltip();
    showTask(button.dataset.task);
    return;
  }

  if (button.dataset.model) {
    hideTooltip();
    showModel(button.dataset.model);
    return;
  }

  if (button.dataset.taskView) {
    state.taskView = button.dataset.taskView;
    state.taskPage = 1;
    renderTasks();
    return;
  }

  if (button.dataset.taskDomain) {
    state.taskDomain = button.dataset.taskDomain;
    state.taskPage = 1;
    renderTaskFilters();
    renderTasks();
    return;
  }

  if (button.dataset.exploreDomain) {
    state.taskDomain = button.dataset.exploreDomain;
    state.taskPage = 1;
    state.taskQuery = '';
    $('#task-search').value = '';
    renderTaskFilters();
    renderTasks();
    $('#tasks').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
});

$('#paper-play').addEventListener('click', () => {
  state.paperAutoPlay = !state.paperAutoPlay;
  updatePlayButton();
  syncPaperAutoplay(true);
});

$('#paper-prev').addEventListener('click', () => {
  if (!state.data) return;
  state.paperIndex -= 1;
  renderPaperObservatory(-1);
});

$('#paper-next').addEventListener('click', () => {
  if (!state.data) return;
  state.paperIndex += 1;
  renderPaperObservatory(1);
});

$('#paper-shuffle').addEventListener('click', () => {
  if (!state.data) return;
  const list = filteredPapers();
  if (list.length <= 1) return;
  let next = state.paperIndex;
  while (next === state.paperIndex) next = Math.floor(Math.random() * list.length);
  state.paperIndex = next;
  renderPaperObservatory(1);
});

$('#paper-deck').addEventListener('keydown', (event) => {
  if (!state.data) return;
  if (event.key === 'ArrowLeft') {
    event.preventDefault();
    state.paperIndex -= 1;
    renderPaperObservatory(-1);
  } else if (event.key === 'ArrowRight') {
    event.preventDefault();
    state.paperIndex += 1;
    renderPaperObservatory(1);
  }
});

// Pause rolling playback while user hovers/interacts with the observatory or opens a dialog
const paperObservatoryEl = $('#paper-observatory');
paperObservatoryEl.addEventListener('pointerenter', () => {
  state.paperHoverPaused = true;
  syncPaperAutoplay(false);
});
paperObservatoryEl.addEventListener('pointerleave', () => {
  state.paperHoverPaused = false;
  syncPaperAutoplay(true);
});

const paperVisibilityObserver = new IntersectionObserver(
  (entries) => {
    state.paperInView = entries.some((entry) => entry.isIntersecting);
    syncPaperAutoplay(true);
  },
  { threshold: 0.15 }
);
paperVisibilityObserver.observe(document.getElementById('papers'));

document.addEventListener('visibilitychange', () => {
  syncPaperAutoplay(true);
});

// Hover telemetry tooltip for Overview tiles and Frontier scatter points
document.addEventListener('pointerover', (event) => {
  if (!state.data) return;
  const tile = event.target.closest('[data-atlas-tile]');
  if (tile) {
    const task = state.data.tasks.find((t) => t.id === tile.dataset.atlasTile);
    if (!task) return;
    const domain = state.data.domains.find((d) => d.id === task.domain);
    const lbl = taskLabel(task);
    const info = taskSummary(task);
    const focusModel = state.atlasModel === 'all' ? null : publishedModels().find((m) => m.id === state.atlasModel);
    const focusCell = focusModel ? task.cells[focusModel.id] : null;
    const html = `
      <div class="tooltip-domain" style="--d-color:${domainColor(domain.id, domain.color)}"><i></i>${esc(domain.name)} · ${esc(lbl.displayCode)}</div>
      <div class="tooltip-title">${esc(task.title)}</div>
      <div class="tooltip-grid">
        <div><span>Metric</span><strong>${task.higherIsBetter ? '↑' : '↓'} ${esc(task.metric)}</strong></div>
        <div><span>Mean improvement</span><strong>${percentage(info.meanScore)}</strong></div>
        ${
          focusModel
            ? `<div><span>${esc(focusModel.name)}</span><strong>${percentage(focusCell.relativeScore)}</strong></div>
               <div><span>Verdict</span><strong>${focusCell.passed ? 'Passed' : 'Did not pass'}</strong></div>`
            : `<div><span>Top model</span><strong>${info.bestModel ? esc(info.bestModel.name) : '-'}</strong></div>
               <div><span>Best improvement</span><strong>${percentage(info.bestScore)}</strong></div>`
        }
      </div>
    `;
    showTooltip(html, event);
    return;
  }

  const point = event.target.closest('[data-frontier-model]');
  if (point) {
    const model = rankedModels().find((m) => m.id === point.dataset.frontierModel);
    if (!model) return;
    const [color] = modelGradient(model);
    const html = `
      <div class="tooltip-domain" style="--d-color:${color}"><i></i>${esc(model.provider)} · ${esc(model.harness)} · ${esc(modelEffort(model))}</div>
      <div class="tooltip-title">${esc(model.name)}</div>
      <div class="tooltip-grid">
        <div><span>Improvement score</span><strong>${percentage(model.score)}</strong></div>
        <div><span>Tasks passed</span><strong>${model.passed} / ${model.total}</strong></div>
      </div>
    `;
    showTooltip(html, event);
  }
});

document.addEventListener('pointermove', (event) => {
  if (!$('#hover-tooltip').hidden) positionTooltip(event);
});

document.addEventListener('pointerout', (event) => {
  if (event.target.closest('[data-atlas-tile]') || event.target.closest('[data-frontier-model]')) {
    hideTooltip();
  }
});

$('#atlas-model').addEventListener('change', (event) => {
  state.atlasModel = event.target.value;
  if (state.data) renderAtlas();
});

$('#leader-domain').addEventListener('change', (event) => {
  state.domain = event.target.value;
  if (state.data) renderLeaderboard();
});

$('#model-search').addEventListener('input', (event) => {
  state.query = event.target.value;
  if (state.data) renderLeaderboard();
});

$('#task-search').addEventListener('input', (event) => {
  state.taskQuery = event.target.value;
  state.taskPage = 1;
  if (state.data) renderTasks();
});

$('#task-sort').addEventListener('change', (event) => {
  state.taskSort = event.target.value;
  state.taskPage = 1;
  if (state.data) renderTasks();
});

$('#tasks-prev').addEventListener('click', () => {
  if (state.taskPage > 1) {
    state.taskPage -= 1;
    renderTasks();
    $('#tasks').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
});

$('#tasks-next').addEventListener('click', () => {
  state.taskPage += 1;
  renderTasks();
  $('#tasks').scrollIntoView({ behavior: 'smooth', block: 'start' });
});

$('#tasks-show-all').addEventListener('click', () => {
  state.taskShowAll = !state.taskShowAll;
  state.taskPage = 1;
  renderTasks();
});

$('#close-dialog').addEventListener('click', () => $('#detail-dialog').close());

$('#detail-dialog').addEventListener('click', (event) => {
  if (event.target === $('#detail-dialog')) {
    const rect = event.target.getBoundingClientRect();
    if (
      event.clientX < rect.left ||
      event.clientX > rect.right ||
      event.clientY < rect.top ||
      event.clientY > rect.bottom
    ) {
      event.target.close();
    }
  }
});

$('#detail-dialog').addEventListener('close', () => {
  state.dialog = null;
  syncPaperAutoplay(true);
});

$('#download-results').addEventListener('click', () => {
  if (!state.data) return;
  const models = rankedModels();
  const rows = [
    [
      'Rank',
      'Model',
      'Reasoning effort',
      'Harness',
      'Collection',
      'Passed',
      'Tasks in collection',
      'Improvement score (%)',
      'Source updated (UTC)',
    ],
    ...models.map((m) => [
      m.rank,
      m.name,
      modelEffort(m),
      m.harness,
      state.domain,
      m.passed,
      m.total,
      m.score ?? '',
      state.data.updatedAt,
    ]),
  ];
  const quote = (v) => `"${String(v).replace(/"/g, '""')}"`;
  downloadFile(
    `\ufeff${rows.map((r) => r.map(quote).join(',')).join('\r\n')}`,
    'text/csv;charset=utf-8',
    'autoscientistbench-leaderboard.csv'
  );
  toast('Leaderboard CSV downloaded.');
});

function syncHeaderSurface() {
  $('#site-header').classList.toggle('scrolled', window.scrollY > 12);
}
window.addEventListener('scroll', syncHeaderSurface, { passive: true });
syncHeaderSurface();

const sections = ['overview', 'leaderboard', 'papers', 'tasks', 'protocol'];
const observer = new IntersectionObserver(
  (entries) => {
    const visible = entries
      .filter((entry) => entry.isIntersecting)
      .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
    if (!visible) return;
    const id = visible.target.id;
    $$('[data-nav]').forEach((link) => {
      const active = link.dataset.nav === id;
      link.classList.toggle('active', active);
      if (active) link.setAttribute('aria-current', 'page');
      else link.removeAttribute('aria-current');
    });
  },
  { rootMargin: '-18% 0px -58% 0px', threshold: [0.05, 0.25, 0.5] }
);
sections.forEach((id) => {
  const el = document.getElementById(id);
  if (el) observer.observe(el);
});

$$('[data-layout]').forEach((btn) => {
  const active = btn.dataset.layout === state.layout;
  btn.classList.toggle('selected', active);
  btn.setAttribute('aria-pressed', active);
});

function handleInitialLocation() {
  const raw = location.hash.slice(1) || location.pathname.split('/').filter(Boolean).pop() || '';
  if (!raw) return;
  const target = document.getElementById(raw);
  if (target) {
    requestAnimationFrame(() => target.scrollIntoView({ behavior: 'instant', block: 'start' }));
  }
}

refresh().then(handleInitialLocation);
setInterval(() => {
  if (!document.hidden) refresh();
}, 60000);
