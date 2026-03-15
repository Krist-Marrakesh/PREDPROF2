const PLOTLY_LAYOUT = {
  paper_bgcolor: 'transparent',
  plot_bgcolor: 'transparent',
  font: { color: '#e2e8f0', family: 'Inter, system-ui, sans-serif', size: 12 },
  margin: { t: 20, r: 20, b: 50, l: 60 },
  xaxis: { gridcolor: '#1e2d45', linecolor: '#1e2d45', zerolinecolor: '#1e2d45' },
  yaxis: { gridcolor: '#1e2d45', linecolor: '#1e2d45', zerolinecolor: '#1e2d45' },
  legend: { bgcolor: 'transparent', bordercolor: '#1e2d45' },
  colorway: ['#00b4d8', '#f77f00', '#7c3aed', '#10b981', '#f59e0b'],
};

const CONFIG = { responsive: true, displayModeBar: true, scrollZoom: true };

async function loadAnalytics() {
  const res = await fetch('/api/analytics/data');
  const data = await res.json();

  if (!data.ready) {
    document.getElementById('charts-placeholder').style.display = 'block';
    return;
  }

  document.getElementById('charts-placeholder').style.display = 'none';
  document.getElementById('charts-content').style.display = 'block';

  if (data.best_val_acc !== undefined) {
    const el = document.getElementById('best-acc');
    if (el) el.textContent = (data.best_val_acc * 100).toFixed(1) + '%';
  }

  renderAccuracyChart(data);
  renderLossChart(data);
  renderClassDistChart(data);
  renderTop5Chart(data);
  if (data.predictions && data.predictions.length > 0) {
    renderPredictionsChart(data.predictions);
    document.getElementById('pred-section').style.display = 'block';
  }
}

function renderAccuracyChart(data) {
  const traces = [
    { x: data.epochs, y: data.train_acc, name: 'Train', mode: 'lines', line: { color: '#00b4d8', width: 2 } },
    { x: data.epochs, y: data.val_acc,   name: 'Val',   mode: 'lines', line: { color: '#f77f00', width: 2 } },
    {
      x: [data.epochs[0], data.epochs[data.epochs.length-1]],
      y: [0.8, 0.8], name: '80% target', mode: 'lines',
      line: { color: '#10b981', width: 1, dash: 'dash' },
    },
  ];
  Plotly.newPlot('chart-accuracy', traces, {
    ...PLOTLY_LAYOUT,
    xaxis: { ...PLOTLY_LAYOUT.xaxis, title: 'Эпоха' },
    yaxis: { ...PLOTLY_LAYOUT.yaxis, title: 'Accuracy', range: [0, 1] },
  }, CONFIG);
}

function renderLossChart(data) {
  const traces = [
    { x: data.epochs, y: data.train_loss, name: 'Train', mode: 'lines', line: { color: '#00b4d8', width: 2 } },
    { x: data.epochs, y: data.val_loss,   name: 'Val',   mode: 'lines', line: { color: '#f77f00', width: 2 } },
  ];
  Plotly.newPlot('chart-loss', traces, {
    ...PLOTLY_LAYOUT,
    xaxis: { ...PLOTLY_LAYOUT.xaxis, title: 'Эпоха' },
    yaxis: { ...PLOTLY_LAYOUT.yaxis, title: 'Loss' },
  }, CONFIG);
}

function renderClassDistChart(data) {
  if (!data.train_class_counts || Object.keys(data.train_class_counts).length === 0) return;
  const sorted = Object.entries(data.train_class_counts).sort((a, b) => b[1] - a[1]);
  const trace = {
    x: sorted.map(x => x[1]),
    y: sorted.map(x => x[0]),
    type: 'bar', orientation: 'h',
    marker: { color: '#00b4d8', opacity: 0.85 },
  };
  Plotly.newPlot('chart-classes', [trace], {
    ...PLOTLY_LAYOUT,
    margin: { ...PLOTLY_LAYOUT.margin, l: 120 },
    xaxis: { ...PLOTLY_LAYOUT.xaxis, title: 'Количество записей' },
    yaxis: { ...PLOTLY_LAYOUT.yaxis, automargin: true },
  }, CONFIG);
}

function renderTop5Chart(data) {
  if (!data.top5_valid || data.top5_valid.classes.length === 0) return;
  const trace = {
    x: data.top5_valid.counts,
    y: data.top5_valid.classes,
    type: 'bar', orientation: 'h',
    marker: { color: '#7c3aed', opacity: 0.85 },
  };
  Plotly.newPlot('chart-top5', [trace], {
    ...PLOTLY_LAYOUT,
    margin: { ...PLOTLY_LAYOUT.margin, l: 130 },
    xaxis: { ...PLOTLY_LAYOUT.xaxis, title: 'Записей' },
    yaxis: { ...PLOTLY_LAYOUT.yaxis, automargin: true },
  }, CONFIG);
}

function renderPredictionsChart(predictions) {
  const trace = {
    x: predictions.map(p => p.index),
    y: predictions.map(p => p.confidence),
    text: predictions.map(p => p.predicted_class),
    type: 'bar',
    marker: {
      color: predictions.map(p => p.confidence),
      colorscale: [[0, '#1e2d45'], [0.5, '#00b4d8'], [1, '#7c3aed']],
      showscale: false,
    },
    hovertemplate: 'Запись #%{x}<br>Класс: %{text}<br>Confidence: %{y:.3f}<extra></extra>',
  };
  Plotly.newPlot('chart-predictions', [trace], {
    ...PLOTLY_LAYOUT,
    xaxis: { ...PLOTLY_LAYOUT.xaxis, title: 'Индекс записи' },
    yaxis: { ...PLOTLY_LAYOUT.yaxis, title: 'Confidence', range: [0, 1] },
  }, CONFIG);
}

document.addEventListener('DOMContentLoaded', loadAnalytics);
