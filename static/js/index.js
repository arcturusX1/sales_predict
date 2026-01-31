// Extracted from templates/index.html
// API Base URL
const API_URL = "http://localhost:8000/api";

// Chart instances storage
let monthlySalesChart = null;
let dailyTrendChart = null;

// Initialize
document.addEventListener("DOMContentLoaded", () => {
  loadStatus();
  loadDataSummary();
  loadModels();
  loadOutlets();
  setDefaultDates();
});

function setDefaultDates() {
  const today = new Date();
  const start = new Date(today.getTime() - 30 * 24 * 60 * 60 * 1000);

  document.getElementById("startDate").valueAsDate = new Date();
  document.getElementById("endDate").valueAsDate = new Date(
    today.getTime() + 30 * 24 * 60 * 60 * 1000,
  );
  document.getElementById("dataStartDate").valueAsDate = start;
  document.getElementById("dataEndDate").valueAsDate = today;
}

async function loadStatus() {
  try {
    const response = await fetch(API_URL + "/status");
    const data = await response.json();

    let html = "";
    if (data.models && data.models.length > 0) {
      html += `<div class="metric-box">
                        <div class="metric-label">Models Trained</div>
                        <div class="metric-value">${data.models.length}</div>
                    </div>`;
    }

    if (data.data_range) {
      const start = new Date(data.data_range.start).toLocaleDateString();
      const end = new Date(data.data_range.end).toLocaleDateString();
      html += `<div class="status-item">
                        <div class="status-label">Data Range</div>
                        <div class="status-value">${start} to ${end}</div>
                    </div>`;
      html += `<div class="status-item">
                        <div class="status-label">Total Records</div>
                        <div class="status-value">${data.data_range.total_records.toLocaleString()}</div>
                    </div>`;
    }

    document.getElementById("statusContent").innerHTML =
      html || "<p>No status available</p>";
  } catch (error) {
    console.error("Error loading status:", error);
    document.getElementById("statusContent").innerHTML =
      '<div class="alert alert-error">Error loading status</div>';
  }
}

async function loadDataSummary() {
  try {
    const response = await fetch(API_URL + "/data/summary");
    const data = await response.json();

    let html = `
                    <div class="metric-box">
                        <div class="metric-label">Outlets</div>
                        <div class="metric-value">${data.outlets}</div>
                    </div>
                    <div class="status-item">
                        <div class="status-label">Avg Daily Sales</div>
                        <div class="status-value">$${Math.round(data.daily_stats.average_sales).toLocaleString()}</div>
                    </div>
                    <div class="status-item">
                        <div class="status-label">Max Daily Sales</div>
                        <div class="status-value">$${Math.round(data.daily_stats.max_sales).toLocaleString()}</div>
                    </div>
                `;

    document.getElementById("dataContent").innerHTML = html;
  } catch (error) {
    console.error("Error loading data summary:", error);
    document.getElementById("dataContent").innerHTML =
      '<div class="alert alert-error">Error loading data</div>';
  }
}

async function loadModels() {
  try {
    const response = await fetch(API_URL + "/models");
    const data = await response.json();

    let html =
      "<table><thead><tr><th>Model</th><th>Type</th><th>MAE</th><th>RMSE</th><th>R²</th><th>Action</th></tr></thead><tbody>";

    const models = data.models || {};
    const modelSelect = document.getElementById("modelSelect");
    modelSelect.innerHTML = "";

    for (const [name, info] of Object.entries(models)) {
      const metrics = info.metrics || {};
      html += `<tr>
                        <td>${name}</td>
                        <td>${info.model_type || "Unknown"}</td>
                        <td>${metrics.mae ? "$" + Math.round(metrics.mae).toLocaleString() : "N/A"}</td>
                        <td>${metrics.rmse ? "$" + Math.round(metrics.rmse).toLocaleString() : "N/A"}</td>
                        <td>${metrics.r2 ? metrics.r2.toFixed(4) : "N/A"}</td>
                        <td><button onclick="predictWithModel('${name}')" style="padding: 6px 12px; font-size: 12px;">Use</button></td>
                    </tr>`;

      const option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      modelSelect.appendChild(option);
    }

    html += "</tbody></table>";
    if (Object.keys(models).length === 0) {
      html = "<p>No models trained yet. Train one to get started!</p>";
    }

    document.getElementById("modelsContent").innerHTML = html;
  } catch (error) {
    console.error("Error loading models:", error);
    document.getElementById("modelsContent").innerHTML =
      '<div class="alert alert-error">Error loading models</div>';
  }
}

async function loadOutlets() {
  try {
    const response = await fetch(API_URL + "/data/summary");
    const data = await response.json();

    const outletSelect = document.getElementById("outletSelect");
    const dataOutlet = document.getElementById("dataOutlet");

    data.outlets_list.forEach((outlet) => {
      const option1 = document.createElement("option");
      option1.value = outlet;
      option1.textContent = outlet;
      outletSelect.appendChild(option1);

      const option2 = document.createElement("option");
      option2.value = outlet;
      option2.textContent = outlet;
      dataOutlet.appendChild(option2);
    });
  } catch (error) {
    console.error("Error loading outlets:", error);
  }
}

async function trainModel() {
  const btn = document.getElementById("trainBtn");
  const alertDiv = document.getElementById("trainingAlert");

  btn.disabled = true;
  btn.innerHTML = '<span class="loading"></span> Training...';
  alertDiv.innerHTML = "";

  try {
    const modelType = document.getElementById("modelType").value;
    const trainingPeriod =
      document.getElementById("trainingPeriod").value;

    const response = await fetch(API_URL + "/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model_type: modelType,
        training_period: trainingPeriod || null,
      }),
    });

    const data = await response.json();

    if (response.ok) {
      alertDiv.innerHTML = `<div class="alert alert-success">
                        <strong>✓ Training Complete!</strong> Model: ${data.model_name}
                        <br>MAE: $${Math.round(data.metrics.mae).toLocaleString()}, 
                        RMSE: $${Math.round(data.metrics.rmse).toLocaleString()}, 
                        R²: ${data.metrics.r2.toFixed(4)}
                    </div>`;
      loadModels();
    } else {
      alertDiv.innerHTML = `<div class="alert alert-error"><strong>Error:</strong> ${data.detail}</div>`;
    }
  } catch (error) {
    alertDiv.innerHTML = `<div class="alert alert-error"><strong>Error:</strong> ${error.message}</div>`;
  } finally {
    btn.disabled = false;
    btn.innerHTML = "Start Training";
  }
}

async function makePredictions() {
  const btn = document.getElementById("predictBtn");
  const alertDiv = document.getElementById("predictionsAlert");
  const resultsDiv = document.getElementById("predictionResults");

  btn.disabled = true;
  btn.innerHTML = '<span class="loading"></span> Generating...';
  alertDiv.innerHTML = "";
  resultsDiv.innerHTML =
    '<div class="spinner"><div class="loading"></div></div>';

  try {
    const modelName = document.getElementById("modelSelect").value;
    const outlet = document.getElementById("outletSelect").value;
    const startDate = document.getElementById("startDate").value;
    const endDate = document.getElementById("endDate").value;
    const showActuals = document.getElementById("showActuals")?.checked;

    if (!modelName) {
      alertDiv.innerHTML =
        '<div class="alert alert-error">Please select a model</div>';
      return;
    }

    const params = new URLSearchParams({
      model_name: modelName,
      start_date: startDate,
      end_date: endDate,
    });
    if (outlet) params.append("outlet", outlet);

    const response = await fetch(API_URL + `/predict?${params}`, {
      method: "POST",
    });
    const data = await response.json();

    let actualRecords = null;
    if (response.ok) {
      alertDiv.innerHTML = `<div class="alert alert-success">
                        ✓ Generated ${data.predictions_count} predictions
                    </div>`;

      if (showActuals) {
        try {
          const dataParams = new URLSearchParams({ start_date: startDate, end_date: endDate });
          if (outlet) dataParams.append('outlet', outlet);
          const r = await fetch(API_URL + `/data/daily?${dataParams}`);
          const actualData = await r.json();
          actualRecords = actualData.records || null;
        } catch (e) {
          console.warn('Failed to load actuals:', e);
        }
      }

      renderPredictionCharts(data, actualRecords);
    } else {
      alertDiv.innerHTML = `<div class="alert alert-error"><strong>Error:</strong> ${data.detail}</div>`;
    }
  } catch (error) {
    alertDiv.innerHTML = `<div class="alert alert-error"><strong>Error:</strong> ${error.message}</div>`;
  } finally {
    btn.disabled = false;
    btn.innerHTML = "Generate Predictions";
  }
}

function renderPredictionCharts(data, actualRecords = null) {
  const resultsDiv = document.getElementById("predictionResults");
  let html = `<div class="charts-grid">
                <div class="card">
                    <h3>Monthly Forecast Summary</h3>
                    <div class="chart-container">
                        <canvas id="monthlySalesChart"></canvas>
                    </div>
                </div>
                <div class="card">
                    <h3>Daily Predictions Trend</h3>
                    <div class="chart-container">
                        <canvas id="dailyTrendChart"></canvas>
                    </div>
                </div>
            </div>`;
  html += '<h3 style="margin-top: 20px;">Monthly Summary Table</h3>';
  html +=
    "<table><thead><tr><th>Outlet</th><th>Month</th><th>Predicted Sales</th></tr></thead><tbody>";

  data.monthly_summary.forEach((row) => {
    html += `<tr><td>${row.outlet}</td><td>${row.month}</td><td>$${Math.round(row.prediction).toLocaleString()}</td></tr>`;
  });
  html += "</tbody></table>";
  resultsDiv.innerHTML = html;

  setTimeout(() => {
    renderMonthlySalesChart(data.monthly_summary, actualRecords);
    renderDailyTrendChart(data.daily_predictions, actualRecords);
  }, 100);
}

function renderMonthlySalesChart(monthlySummary, actualRecords = null) {
  const ctx = document.getElementById("monthlySalesChart");
  if (!ctx) return;

  // Determine months (ordered)
  const months = Array.from(new Set(monthlySummary.map((r) => r.month))).sort();

  // Prepare predicted monthly data per outlet aligned with months
  const outletData = {};
  monthlySummary.forEach((row) => {
    if (!outletData[row.outlet]) outletData[row.outlet] = months.map(() => 0);
    const idx = months.indexOf(row.month);
    outletData[row.outlet][idx] += parseFloat(row.prediction);
  });

  const outlets = Object.keys(outletData);
  const predictedDatasets = outlets.map((outlet, idx) => ({
    label: outlet.substring(0, 20),
    data: outletData[outlet],
    backgroundColor: `hsla(${(idx * 360) / Math.max(1, outlets.length)}, 70%, 60%, 0.7)`,
    type: 'bar'
  }));

  // Prepare actual monthly data if available
  const actualDatasets = [];
  if (actualRecords && actualRecords.length > 0) {
    const actualMap = {}; // outlet -> month -> sum
    actualRecords.forEach((r) => {
      const datePart = (r.date || '').split('T')[0];
      const month = datePart ? datePart.slice(0,7) : null;
      if (!month) return;
      if (!actualMap[r.outlet]) actualMap[r.outlet] = {};
      actualMap[r.outlet][month] = (actualMap[r.outlet][month] || 0) + parseFloat(r.sales || 0);
    });

    Object.entries(actualMap).forEach(([outlet, mObj], idx) => {
      const dataArr = months.map((m) => mObj[m] ? mObj[m] : 0);
      actualDatasets.push({
        label: `${outlet.substring(0,20)} (actual)` ,
        data: dataArr,
        borderColor: '#e44',
        backgroundColor: 'rgba(228,68,68,0.05)',
        fill: false,
        tension: 0.3,
        type: 'line',
        pointRadius: 3,
        borderDash: [4,3]
      });
    });
  }

  const datasets = [...predictedDatasets, ...actualDatasets];

  if (monthlySalesChart) monthlySalesChart.destroy();

  monthlySalesChart = new Chart(ctx, {
    data: {
      labels: months,
      datasets: datasets,
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "bottom" },
      },
      scales: {
        y: {
          beginAtZero: true,
          title: { display: true, text: "Sales ($)" },
        },
      },
    },
  });
}

function renderDailyTrendChart(dailyPredictions, actualRecords = null) {
  const ctx = document.getElementById("dailyTrendChart");
  if (!ctx) return;

  const predDateMap = {};
  dailyPredictions.forEach((pred) => {
    const date = pred.date.split("T")[0];
    if (!predDateMap[date]) predDateMap[date] = 0;
    predDateMap[date] += parseFloat(pred.prediction);
  });

  const actualDateMap = {};
  if (actualRecords && actualRecords.length > 0) {
    actualRecords.forEach((r) => {
      const date = (r.date || '').split('T')[0];
      if (!date) return;
      actualDateMap[date] = (actualDateMap[date] || 0) + parseFloat(r.sales || 0);
    });
  }

  const allDates = Array.from(new Set([...Object.keys(predDateMap), ...Object.keys(actualDateMap)])).sort();
  const predSales = allDates.map((d) => predDateMap[d] || 0);
  const actualSales = allDates.map((d) => actualDateMap[d] || 0);

  const datasets = [
    {
      label: "Total Daily Predictions",
      data: predSales,
      borderColor: "#667eea",
      backgroundColor: "rgba(102, 126, 234, 0.1)",
      fill: true,
      tension: 0.4,
    },
  ];

  if (actualRecords && actualRecords.length > 0) {
    datasets.push({
      label: 'Actual Daily Sales',
      data: actualSales,
      borderColor: '#e44',
      backgroundColor: 'rgba(228,68,68,0.08)',
      fill: false,
      tension: 0.3,
      borderDash: [4,3]
    });
  }

  if (dailyTrendChart) dailyTrendChart.destroy();

  dailyTrendChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: allDates,
      datasets: datasets,
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: true },
      },
      scales: {
        y: { beginAtZero: true },
      },
    },
  });
}

async function loadData() {
  const resultsDiv = document.getElementById("dataResults");
  resultsDiv.innerHTML =
    '<div class="spinner"><div class="loading"></div></div>';

  try {
    const outlet = document.getElementById("dataOutlet").value;
    const startDate = document.getElementById("dataStartDate").value;
    const endDate = document.getElementById("dataEndDate").value;

    const params = new URLSearchParams({
      start_date: startDate,
      end_date: endDate,
    });
    if (outlet) params.append("outlet", outlet);

    const response = await fetch(API_URL + `/data/daily?${params}`);
    const data = await response.json();

    let html = `<p><strong>${data.count} records found</strong></p><table><thead><tr><th>Date</th><th>Outlet</th><th>Sales</th></tr></thead><tbody>`;
    data.records.slice(0, 50).forEach((row) => {
      html += `<tr><td>${row.date}</td><td>${row.outlet}</td><td>$${Math.round(row.sales).toLocaleString()}</td></tr>`;
    });
    html += "</tbody></table>";
    resultsDiv.innerHTML = html;
  } catch (error) {
    resultsDiv.innerHTML = `<div class="alert alert-error">Error loading data: ${error.message}</div>`;
  }
}

function predictWithModel(modelName) {
  document.getElementById("modelSelect").value = modelName;
  switchTab("predictions");
}

function switchTab(tabName) {
  document
    .querySelectorAll(".tab-content")
    .forEach((el) => el.classList.remove("active"));
  document
    .querySelectorAll(".tab-button")
    .forEach((el) => el.classList.remove("active"));

  document.getElementById(tabName + "-tab").classList.add("active");
  event.target.classList.add("active");
}
