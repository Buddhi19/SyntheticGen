document.addEventListener("DOMContentLoaded", () => {
  const benchmarkResults = {
    indomain: [
      { label: "U-Net", original: { miou: 39.77, mf1: 58.22, oa: 55.35 }, synthetic: { miou: 51.36, mf1: 70.24, oa: 66.85 } },
      { label: "PSPNet", original: { miou: 43.01, mf1: 64.20, oa: 59.14 }, synthetic: { miou: 47.45, mf1: 67.11, oa: 63.26 } },
      { label: "FactSeg", original: { miou: 46.01, mf1: 64.83, oa: 62.16 }, synthetic: { miou: 49.95, mf1: 65.57, oa: 68.11 } },
      { label: "HRNet", original: { miou: 47.43, mf1: 65.55, oa: 63.68 }, synthetic: { miou: 53.01, mf1: 71.87, oa: 68.20 } },
      { label: "AerialFormer", original: { miou: 53.06, mf1: 68.40, oa: 65.57 }, synthetic: { miou: 54.26, mf1: 69.61, oa: 68.35 } },
    ],
    domain: [
      { label: "FactSeg U-R", original: { miou: 30.36, mf1: 44.55, oa: 61.44 }, synthetic: { miou: 35.52, mf1: 50.87, oa: 63.48 } },
      { label: "HRNet U-R", original: { miou: 28.82, mf1: 42.84, oa: 57.29 }, synthetic: { miou: 34.77, mf1: 49.04, oa: 64.20 } },
      { label: "FactSeg R-U", original: { miou: 39.98, mf1: 56.07, oa: 58.38 }, synthetic: { miou: 50.45, mf1: 66.71, oa: 65.16 } },
      { label: "HRNet R-U", original: { miou: 43.95, mf1: 60.30, oa: 60.51 }, synthetic: { miou: 53.79, mf1: 69.52, oa: 68.53 } },
    ],
  };

  const metricLabels = {
    miou: "mIoU (%)",
    mf1: "mF1 (%)",
    oa: "OA (%)",
  };

  function niceCeil(value) {
    const step = value <= 40 ? 5 : 10;
    return Math.ceil((value + step * 0.35) / step) * step;
  }

  function renderMetricPlot(targetId, groupKey, metricKey) {
    const target = document.getElementById(targetId);
    if (!target) {
      return;
    }

    const rows = benchmarkResults[groupKey];
    const width = 620;
    const height = 360;
    const margin = { top: 24, right: 18, bottom: 104, left: 50 };
    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;
    const maxValue = Math.max(
      ...rows.flatMap((row) => [row.original[metricKey], row.synthetic[metricKey]]),
    );
    const yMax = niceCeil(maxValue);
    const tickStep = yMax <= 40 ? 5 : 10;
    const ticks = [];
    for (let tick = 0; tick <= yMax; tick += tickStep) {
      ticks.push(tick);
    }

    const yFor = (value) => margin.top + chartHeight - (value / yMax) * chartHeight;
    const groupWidth = chartWidth / rows.length;
    const barWidth = Math.min(34, groupWidth * 0.24);

    const bars = rows.map((row, index) => {
      const center = margin.left + groupWidth * index + groupWidth / 2;
      const originalX = center - barWidth - 3;
      const syntheticX = center + 3;
      const originalY = yFor(row.original[metricKey]);
      const syntheticY = yFor(row.synthetic[metricKey]);
      const originalHeight = margin.top + chartHeight - originalY;
      const syntheticHeight = margin.top + chartHeight - syntheticY;
      const labelY = margin.top + chartHeight + 18;
      const delta = row.synthetic[metricKey] - row.original[metricKey];

      return `
        <g>
          <title>${row.label}: ${row.original[metricKey].toFixed(2)} to ${row.synthetic[metricKey].toFixed(2)} ${metricLabels[metricKey]} (${delta >= 0 ? "+" : ""}${delta.toFixed(2)})</title>
          <rect x="${originalX.toFixed(2)}" y="${originalY.toFixed(2)}" width="${barWidth}" height="${originalHeight.toFixed(2)}" fill="#c9c9c9"></rect>
          <rect x="${syntheticX.toFixed(2)}" y="${syntheticY.toFixed(2)}" width="${barWidth}" height="${syntheticHeight.toFixed(2)}" fill="#3273dc"></rect>
          <text x="${(originalX + barWidth / 2).toFixed(2)}" y="${(originalY - 6).toFixed(2)}" text-anchor="middle" class="dot-value">${row.original[metricKey].toFixed(1)}</text>
          <text x="${(syntheticX + barWidth / 2).toFixed(2)}" y="${(syntheticY - 6).toFixed(2)}" text-anchor="middle" class="dot-value">${row.synthetic[metricKey].toFixed(1)}</text>
          <text x="${center.toFixed(2)}" y="${labelY}" text-anchor="end" transform="rotate(-35 ${center.toFixed(2)} ${labelY})" class="x-label">${row.label}</text>
        </g>
      `;
    }).join("");

    const grid = ticks.map((tick) => {
      const y = yFor(tick);
      return `
        <g>
          <line x1="${margin.left}" y1="${y.toFixed(2)}" x2="${margin.left + chartWidth}" y2="${y.toFixed(2)}" class="grid-line"></line>
          <text x="${margin.left - 10}" y="${(y + 4).toFixed(2)}" text-anchor="end" class="tick-label">${tick}</text>
        </g>
      `;
    }).join("");

    target.innerHTML = `
      <svg class="metric-svg" viewBox="0 0 ${width} ${height}" aria-label="${metricLabels[metricKey]} comparison">
        ${grid}
        <line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${margin.top + chartHeight}" class="axis-line"></line>
        <line x1="${margin.left}" y1="${margin.top + chartHeight}" x2="${margin.left + chartWidth}" y2="${margin.top + chartHeight}" class="axis-line"></line>
        <text x="${margin.left - 38}" y="${margin.top + chartHeight / 2}" text-anchor="middle" transform="rotate(-90 ${margin.left - 38} ${margin.top + chartHeight / 2})" class="axis-title">${metricLabels[metricKey]}</text>
        <g transform="translate(${margin.left + chartWidth - 166}, ${margin.top + 8})">
          <rect x="0" y="0" width="12" height="12" fill="#c9c9c9"></rect>
          <text x="18" y="10" class="tick-label">Original</text>
          <rect x="88" y="0" width="12" height="12" fill="#3273dc"></rect>
          <text x="106" y="10" class="tick-label">Orig.+Syn.</text>
        </g>
        ${bars}
      </svg>
    `;
  }

  function updatePlots() {
    const select = document.getElementById("metric-select");
    const metricKey = select ? select.value : "miou";
    renderMetricPlot("plot-indomain", "indomain", metricKey);
    renderMetricPlot("plot-domain", "domain", metricKey);
  }

  const metricSelect = document.getElementById("metric-select");
  if (metricSelect) {
    metricSelect.addEventListener("change", updatePlots);
    updatePlots();
  }

  const burgers = Array.from(document.querySelectorAll(".navbar-burger"));

  burgers.forEach((burger) => {
    burger.addEventListener("click", () => {
      const targetId = burger.dataset.target;
      const target = targetId ? document.getElementById(targetId) : null;

      burger.classList.toggle("is-active");
      if (target) {
        target.classList.toggle("is-active");
      }
    });
  });

  document.querySelectorAll(".navbar-menu a[href^='#']").forEach((link) => {
    link.addEventListener("click", () => {
      document.querySelectorAll(".navbar-burger, .navbar-menu").forEach((node) => {
        node.classList.remove("is-active");
      });
    });
  });
});
