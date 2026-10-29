import React, { useEffect, useMemo, useRef, useState } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Polyline } from 'react-leaflet';
import L from 'leaflet';
import {
  Activity,
  AlertTriangle,
  ArrowUpRight,
  Bell,
  BellOff,
  CheckCircle2,
  ChevronRight,
  Clock3,
  Cpu,
  Database,
  ExternalLink,
  Info,
  Layers,
  MapPin,
  Menu,
  Play,
  RefreshCw,
  RotateCcw,
  ShieldAlert,
  ShieldCheck,
  Square,
  TrainFront,
  TriangleAlert,
  Wifi,
  Workflow
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Cell
} from 'recharts';
import api from './services/api';

// Map Leaflet Pin Helper
const createPinIcon = (color) =>
  L.divIcon({
    className: 'custom-pin',
    html: `<div style="background:${color};width:16px;height:16px;border:3px solid white;border-radius:50%;box-shadow:0 2px 8px rgba(0,0,0,0.25);"></div>`,
    iconSize: [20, 20],
    iconAnchor: [10, 10]
  });

// Status Color Helper based on exact threshold rules:
// 0%–25%: No Error (Green)
// > 25% and <= 75%: Normal Alert (Amber)
// > 75%: Warning Alert (Red)
function getStatusFromPercentage(pct) {
  if (pct <= 25.0) return 'No Error';
  if (pct <= 75.0) return 'Normal Alert';
  return 'Warning Alert';
}

function getStatusColor(status) {
  switch (status) {
    case 'No Error':
    case 'COMPLETED':
      return '#16a34a'; // Green
    case 'Normal Alert':
    case 'RUNNING':
      return '#d97706'; // Amber
    case 'Warning Alert':
    case 'FAILED':
      return '#dc2626'; // Red
    default:
      return '#64748b'; // Slate neutral
  }
}

// Predefined 3 Simulation Scenarios
const SIMULATION_SCENARIOS = [
  {
    id: 1,
    name: 'Scenario 1 — Low Error',
    percentage: 10,
    status: 'No Error',
    alertType: null,
    badgeClass: 'green',
    description: 'Track measurements nominal. Acoustic and vibration signals within safety baseline.'
  },
  {
    id: 2,
    name: 'Scenario 2 — Moderate Error',
    percentage: 50,
    status: 'Normal Alert',
    alertType: 'Normal Alert',
    badgeClass: 'amber',
    description: 'Moderate dynamic rail excitation / deviation detected. Scheduled inspection alert triggered.'
  },
  {
    id: 3,
    name: 'Scenario 3 — High Error',
    percentage: 90,
    status: 'Warning Alert',
    alertType: 'Warning Alert',
    badgeClass: 'red',
    description: 'High error percentage / critical rail stress. Immediate warning alert triggered.'
  }
];

export default function App() {
  // Navigation & UI state
  const [activeTab, setActiveTab] = useState('Overview');
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [notice, setNotice] = useState('');
  const [notificationsAllowed, setNotificationsAllowed] = useState(
    typeof Notification !== 'undefined' && Notification.permission === 'granted'
  );

  // Backend Data
  const [health, setHealth] = useState(null);
  const [model, setModel] = useState(null);
  const [mapData, setMapData] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [datasetPreview, setDatasetPreview] = useState(null);
  const [sampleRows, setSampleRows] = useState([]);

  // Primary Monitoring & Error Percentage State
  const [operatingMode, setOperatingMode] = useState('Real Prediction Mode'); // 'Simulation Mode' | 'Real Prediction Mode'
  const [errorPercentage, setErrorPercentage] = useState(4.4); // Initial value from row 0
  const [currentStatus, setCurrentStatus] = useState('No Error');
  const [latestUpdateTime, setLatestUpdateTime] = useState(new Date().toLocaleTimeString());
  const [latestAlert, setLatestAlert] = useState(null);

  // Simulation State
  const [isSimulating, setIsSimulating] = useState(false);
  const [currentScenarioIdx, setCurrentScenarioIdx] = useState(0);
  const [countdownSeconds, setCountdownSeconds] = useState(10);
  const simulationTimerRef = useRef(null);
  const countdownTimerRef = useRef(null);
  const scenarioIndexRef = useRef(0);

  // Real Prediction Mode State
  const [selectedRowIndex, setSelectedRowIndex] = useState(0);
  const [isPredicting, setIsPredicting] = useState(false);
  const [predictionResult, setPredictionResult] = useState(null);

  // 3-Node Sequential Simulation State (Preserved existing feature)
  const [nodeSimulation, setNodeSimulation] = useState(null);
  const [nodeSimBusy, setNodeSimBusy] = useState(false);
  const [confirmResetModal, setConfirmResetModal] = useState(false);

  // Alert Filter
  const [alertFilter, setAlertFilter] = useState('ALL');

  // Request Notification Permission
  const requestNotificationPermission = async () => {
    if (typeof Notification === 'undefined') {
      setNotice('Browser notifications are not supported in this browser environment.');
      return;
    }
    try {
      const permission = await Notification.requestPermission();
      setNotificationsAllowed(permission === 'granted');
      if (permission === 'granted') {
        setNotice('Browser notifications enabled for Normal and Warning alerts.');
      } else {
        setNotice('Browser notifications were denied. Alerts will continue to be displayed on this dashboard.');
      }
    } catch {
      setNotice('Unable to request notification permission.');
    }
  };

  const sendBrowserNotification = (title, body) => {
    if (notificationsAllowed && typeof Notification !== 'undefined') {
      try {
        new Notification(title, {
          body,
          icon: '/favicon.ico'
        });
      } catch {
        // Fallback silently if blocked by environment
      }
    }
  };

  // Initial load and polling
  const fetchAllData = async () => {
    try {
      const [h, m, mp, a, samples, preview] = await Promise.all([
        api.getHealth(),
        api.getModelInfo(),
        api.getMapData(),
        api.getAlerts(),
        api.getDatasetSample().catch(() => []),
        api.getDatasetPreview().catch(() => null)
      ]);
      setHealth(h);
      setModel(m);
      setMapData(mp);
      setAlerts(a);
      setSampleRows(samples);
      setDatasetPreview(preview);

      // Set latest alert from loaded alerts
      if (a && a.length > 0 && !latestAlert) {
        setLatestAlert(a[0]);
      }
    } catch (err) {
      setNotice(err.message || 'Connecting to backend API...');
    }
  };

  useEffect(() => {
    fetchAllData();
    const interval = setInterval(() => {
      api.getAlerts().then(setAlerts).catch(() => {});
      api.getHealth().then(setHealth).catch(() => {});
    }, 6000);
    return () => clearInterval(interval);
  }, []);

  // Cleanup simulation timers when component unmounts
  useEffect(() => {
    return () => {
      if (simulationTimerRef.current) clearInterval(simulationTimerRef.current);
      if (countdownTimerRef.current) clearInterval(countdownTimerRef.current);
    };
  }, []);

  // SIMULATION CONTROLS (Section 4)
  const applyScenario = async (idx) => {
    const scenario = SIMULATION_SCENARIOS[idx];
    setErrorPercentage(scenario.percentage);
    setCurrentStatus(scenario.status);
    setCurrentScenarioIdx(idx);
    setOperatingMode('Simulation Mode');
    const timeStr = new Date().toLocaleTimeString();
    setLatestUpdateTime(timeStr);

    if (scenario.alertType) {
      const alertPayload = {
        severity: scenario.alertType,
        error_percentage: scenario.percentage,
        source: 'Simulation',
        reason: `${scenario.name} generated ${scenario.alertType} (${scenario.percentage}% error).`
      };

      setLatestAlert({
        id: 'sim-' + Date.now(),
        ...alertPayload,
        created_at: new Date().toISOString()
      });

      // Notify backend to record simulation alert
      try {
        const saved = await api.createAlert(alertPayload);
        setAlerts((prev) => [saved, ...prev]);
      } catch {
        // Fallback locally if backend call fails
        setAlerts((prev) => [
          {
            id: 'sim-' + Date.now(),
            ...alertPayload,
            created_at: new Date().toISOString(),
            status: 'OPEN'
          },
          ...prev
        ]);
      }

      sendBrowserNotification(
        `TrackX ${scenario.alertType}`,
        `Simulated error level at ${scenario.percentage}%.`
      );
    } else {
      // Scenario 1: Low Error (No Alert)
      setLatestAlert(null);
    }
  };

  const startSimulation = () => {
    // Prevent duplicate timers
    if (isSimulating) return;

    setIsSimulating(true);
    setOperatingMode('Simulation Mode');
    setNotice('');

    // Immediately run Scenario 1 (index 0)
    scenarioIndexRef.current = 0;
    applyScenario(0);
    setCountdownSeconds(10);

    // 1-second countdown timer for UI feedback
    if (countdownTimerRef.current) clearInterval(countdownTimerRef.current);
    countdownTimerRef.current = setInterval(() => {
      setCountdownSeconds((sec) => (sec <= 1 ? 10 : sec - 1));
    }, 1000);

    // Exactly 10-second interval between successive scenarios
    if (simulationTimerRef.current) clearInterval(simulationTimerRef.current);
    simulationTimerRef.current = setInterval(() => {
      const nextIdx = (scenarioIndexRef.current + 1) % 3;
      scenarioIndexRef.current = nextIdx;
      setCountdownSeconds(10);
      applyScenario(nextIdx);
    }, 10000);
  };

  const stopSimulation = () => {
    if (simulationTimerRef.current) {
      clearInterval(simulationTimerRef.current);
      simulationTimerRef.current = null;
    }
    if (countdownTimerRef.current) {
      clearInterval(countdownTimerRef.current);
      countdownTimerRef.current = null;
    }
    setIsSimulating(false);
    setNotice('Simulation stopped. You can resume anytime or switch to Real Prediction Mode.');
  };

  // REAL PREDICTION CONTROLS (Section 5)
  const runRealPrediction = async (rowIdx = selectedRowIndex) => {
    // If simulation was running, pause it when user manually triggers a real prediction
    if (isSimulating) {
      stopSimulation();
    }

    setIsPredicting(true);
    setNotice('');
    try {
      const res = await api.predict({ row_index: Number(rowIdx) });
      setPredictionResult(res);
      setOperatingMode('Real Prediction Mode');
      setErrorPercentage(res.error_percentage);
      setCurrentStatus(res.status);
      setLatestUpdateTime(new Date(res.timestamp).toLocaleTimeString());

      if (res.alert_type) {
        setLatestAlert({
          id: 'real-' + Date.now(),
          severity: res.alert_type,
          error_percentage: res.error_percentage,
          source: 'Real Prediction',
          reason: `Model predicted '${res.prediction}' resulting in ${res.alert_type} (${res.error_percentage}% error).`,
          created_at: res.timestamp
        });
        sendBrowserNotification(
          `TrackX ${res.alert_type} (Real ML Prediction)`,
          `Prediction: ${res.prediction} — ${res.error_percentage}% error.`
        );
      } else {
        setLatestAlert(null);
      }

      // Refresh alerts list
      api.getAlerts().then(setAlerts).catch(() => {});
    } catch (err) {
      setNotice(`Prediction error: ${err.message}`);
    } finally {
      setIsPredicting(false);
    }
  };

  // 3-NODE SIMULATION CONTROLS (Preserved existing functionality)
  const start3NodeSimulation = async () => {
    setNodeSimBusy(true);
    setNotice('');
    try {
      const sim = await api.createSimulation({ row_indices: [0, 1000, 2000] });
      setNodeSimulation(sim);
      setNotice('3-Node simulation created. Grant permission sequentially to run each node.');
    } catch (err) {
      setNotice(`Failed to create 3-node simulation: ${err.message}`);
    } finally {
      setNodeSimBusy(false);
    }
  };

  const runSingleNode = async (nodeNumber) => {
    if (!nodeSimulation) return;
    setNodeSimBusy(true);
    setNotice('');
    try {
      const res = await api.runNode(nodeSimulation.simulation_id, nodeNumber);
      setNotice(`Node ${nodeNumber} completed. Prediction: ${res.node.prediction} (${res.error_percentage}% error).`);
      const updated = await api.getSimulation(nodeSimulation.simulation_id);
      setNodeSimulation(updated);
      api.getAlerts().then(setAlerts).catch(() => {});
    } catch (err) {
      setNotice(err.message);
    } finally {
      setNodeSimBusy(false);
    }
  };

  const reset3NodeSimulation = async () => {
    if (!nodeSimulation) return;
    setNodeSimBusy(true);
    try {
      const resetSim = await api.resetSimulation(nodeSimulation.simulation_id);
      setNodeSimulation(resetSim);
      setConfirmResetModal(false);
      setNotice('Simulation reset. All nodes awaiting permission.');
    } catch (err) {
      setNotice(err.message);
    } finally {
      setNodeSimBusy(false);
    }
  };

  // Filtered alerts
  const filteredAlerts = useMemo(() => {
    if (alertFilter === 'REAL') {
      return alerts.filter((a) => a.source === 'Real Prediction');
    }
    if (alertFilter === 'SIMULATION') {
      return alerts.filter((a) => a.source === 'Simulation');
    }
    return alerts;
  }, [alerts, alertFilter]);

  // Nodes for map
  const activeNodes = nodeSimulation?.nodes || [];
  const completedNodesCount = activeNodes.filter((n) => n.status === 'COMPLETED').length;
  const canRunNode = (n) => {
    const node = activeNodes.find((x) => x.node_number === n);
    return (
      node &&
      (node.status === 'WAITING_FOR_PERMISSION' || node.status === 'FAILED') &&
      activeNodes.filter((x) => x.node_number < n).every((x) => x.status === 'COMPLETED')
    );
  };

  const mapNodes = (mapData?.nodes || []).map((n) => ({
    ...n,
    ...(activeNodes.find((x) => x.node_number === n.id)
      ? {
          status: activeNodes.find((x) => x.node_number === n.id).status,
          prediction: activeNodes.find((x) => x.node_number === n.id).prediction
        }
      : {})
  }));

  // Feature importance formatting
  const importanceData = useMemo(() => {
    return (model?.feature_importance || []).slice(0, 7).map((x) => ({
      name: x.feature.length > 22 ? x.feature.slice(0, 20) + '…' : x.feature,
      value: x.importance
    }));
  }, [model]);

  // CSS class for status indicator
  const statusClass =
    currentStatus === 'No Error'
      ? 'no-error'
      : currentStatus === 'Normal Alert'
      ? 'normal-alert'
      : 'warning-alert';

  return (
    <div className="app-shell">
      {/* Sidebar - Clean Modern Light Theme */}
      <aside className={`sidebar ${mobileMenuOpen ? 'show' : ''}`}>
        <div className="brand">
          <div className="brand-mark">
            <TrainFront size={22} />
          </div>
          <div className="brand-title">
            <b>TrackX</b>
            <small>RAIL MONITORING & ML</small>
          </div>
        </div>

        <div className="workspace-badge">
          <span className="workspace-dot" />
          <span>Continuous Health Console</span>
          <ChevronRight size={14} style={{ marginLeft: 'auto', color: '#94a3b8' }} />
        </div>

        <div className="nav-section-title">Navigation</div>
        {[
          ['Overview', Activity],
          ['Real ML Inference', Cpu],
          ['3-Node Simulation', Workflow],
          ['Live Map', MapPin],
          ['Alert History', AlertTriangle],
          ['Model Insights', Database]
        ].map(([name, Icon]) => (
          <button
            key={name}
            className={`nav-item ${activeTab === name ? 'selected' : ''}`}
            onClick={() => {
              setActiveTab(name);
              setMobileMenuOpen(false);
            }}
          >
            <Icon size={18} />
            <span>{name}</span>
            {name === 'Alert History' && alerts.length > 0 && (
              <span className="nav-count">{alerts.length}</span>
            )}
          </button>
        ))}

        <div className="sidebar-footer">
          <div className="env-status-card">
            <b>
              <span className={`live-dot ${health?.model_available ? '' : 'amber'}`} />
              {health?.model_available ? 'ML Engine Ready' : 'ML Connecting'}
            </b>
            <p>
              {health?.model_available
                ? `RandomForest loaded (${health.dataset_rows} records)`
                : 'Waiting for trained model'}
            </p>
          </div>

          <div className="user-profile">
            <div className="avatar">TX</div>
            <div className="user-meta">
              <b>Railway Operator</b>
              <small>System Engineer</small>
            </div>
            <span
              className="live-dot"
              style={{ marginLeft: 'auto' }}
              title="Online"
            />
          </div>
        </div>
      </aside>

      {/* Main Container */}
      <main className="main">
        {/* Topbar */}
        <header className="topbar">
          <div className="topbar-left">
            <button
              className="mobile-menu-btn"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              aria-label="Toggle menu"
            >
              <Menu size={20} />
            </button>
            <div className="breadcrumb">
              <span>TrackX System</span>
              <ChevronRight size={14} />
              <b>{activeTab}</b>
            </div>
          </div>

          <div className="topbar-actions">
            <span
              className={`mode-badge ${
                operatingMode === 'Simulation Mode' ? 'simulation' : 'real'
              }`}
            >
              <span
                className={`live-dot ${
                  operatingMode === 'Simulation Mode' ? 'amber' : ''
                }`}
              />
              {operatingMode}
            </span>

            <button
              className="icon-btn"
              title={
                notificationsAllowed
                  ? 'Notifications active'
                  : 'Enable browser notifications'
              }
              onClick={requestNotificationPermission}
            >
              {notificationsAllowed ? (
                <Bell size={18} color="#2563eb" />
              ) : (
                <BellOff size={18} />
              )}
            </button>

            <button
              className="icon-btn"
              title="Refresh backend status"
              onClick={fetchAllData}
            >
              <RefreshCw size={17} />
            </button>

            <div className="avatar small">TX</div>
          </div>
        </header>

        {/* Page Content */}
        <div className="page-content">
          <div className="page-header">
            <div>
              <div className="eyebrow">
                <span className="eyebrow-line" />
                Track Health Intelligence & Simulation
              </div>
              <h1>
                {activeTab === 'Overview'
                  ? 'Railway Monitoring Dashboard'
                  : activeTab}
              </h1>
              <p>
                Continuous error percentage assessment, automated scenario testing, and real ML inference.
              </p>
            </div>

            <div className="header-buttons">
              <button className="btn btn-secondary" onClick={fetchAllData}>
                <RefreshCw size={15} />
                Refresh
              </button>
              {isSimulating ? (
                <button className="btn btn-danger" onClick={stopSimulation}>
                  <Square size={15} fill="currentColor" />
                  Stop Simulation
                </button>
              ) : (
                <button className="btn btn-primary" onClick={startSimulation}>
                  <Play size={15} fill="currentColor" />
                  Start Simulation
                </button>
              )}
            </div>
          </div>

          {/* User Feedback / Notice Banner */}
          {notice && (
            <div className="notice-box">
              <span>{notice}</span>
              <button
                onClick={() => setNotice('')}
                style={{
                  border: 0,
                  background: 'none',
                  fontSize: 18,
                  cursor: 'pointer',
                  color: 'inherit'
                }}
              >
                ×
              </button>
            </div>
          )}

          {/* Backend / Model Warnings */}
          {health && !health.model_available && (
            <div className="warning-callout">
              <TriangleAlert size={20} color="#d97706" />
              <div>
                <b>Trained Model Offline</b>
                <p>
                  {health.model_error ||
                    'Run python trackXModel/train_model.py to generate the trained model artifact.'}
                </p>
              </div>
            </div>
          )}

          {/* Prominent Active Alert Banner (Section 6) */}
          {latestAlert && latestAlert.severity && (
            <div
              className={`alert-banner ${
                latestAlert.severity === 'Warning Alert' ? 'warning' : 'normal'
              }`}
            >
              <div className="alert-banner-left">
                <div className="alert-icon-box">
                  {latestAlert.severity === 'Warning Alert' ? (
                    <ShieldAlert size={22} />
                  ) : (
                    <AlertTriangle size={22} />
                  )}
                </div>
                <div className="alert-banner-text">
                  <b>
                    {latestAlert.severity} Active ({latestAlert.error_percentage}%)
                  </b>
                  <p>
                    {latestAlert.reason} · Source: {latestAlert.source} ·{' '}
                    {new Date(latestAlert.created_at).toLocaleTimeString()}
                  </p>
                </div>
              </div>

              <div style={{ display: 'flex', gap: 8 }}>
                <button
                  className="btn btn-secondary btn-sm"
                  onClick={() => setActiveTab('Alert History')}
                >
                  View Alert Center
                </button>
              </div>
            </div>
          )}

          {/* SECTION 2 & 3: STATS GRID */}
          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-header">
                <span className="stat-label">Current Error %</span>
                <span
                  className={`stat-icon ${
                    statusClass === 'no-error'
                      ? 'green'
                      : statusClass === 'normal-alert'
                      ? 'amber'
                      : 'red'
                  }`}
                >
                  <Activity size={18} />
                </span>
              </div>
              <div className="stat-value">{errorPercentage.toFixed(1)}%</div>
              <div className="stat-sub">Status: {currentStatus}</div>
            </div>

            <div className="stat-card">
              <div className="stat-header">
                <span className="stat-label">Operating Mode</span>
                <span className="stat-icon blue">
                  <Layers size={18} />
                </span>
              </div>
              <div className="stat-value" style={{ fontSize: 20 }}>
                {operatingMode}
              </div>
              <div className="stat-sub">
                {isSimulating ? `Scenario ${currentScenarioIdx + 1} active` : 'Interactive'}
              </div>
            </div>

            <div className="stat-card">
              <div className="stat-header">
                <span className="stat-label">ML Model Engine</span>
                <span className="stat-icon green">
                  <ShieldCheck size={18} />
                </span>
              </div>
              <div className="stat-value">
                {health?.model_available ? '96.3% Acc' : 'Unavailable'}
              </div>
              <div className="stat-sub">
                {health?.model_features?.length || 20} Input features validated
              </div>
            </div>

            <div className="stat-card">
              <div className="stat-header">
                <span className="stat-label">Logged Alerts</span>
                <span className="stat-icon red">
                  <AlertTriangle size={18} />
                </span>
              </div>
              <div className="stat-value">{alerts.length}</div>
              <div className="stat-sub">
                {alerts.filter((a) => a.status === 'OPEN').length} Unresolved
              </div>
            </div>
          </div>

          {/* MAIN TAB 1: OVERVIEW */}
          {activeTab === 'Overview' && (
            <>
              {/* PRIMARY PROMINENT ERROR PERCENTAGE CARD (Section 3 & 4) */}
              <section className="monitoring-card">
                <div className="monitoring-top">
                  <div>
                    <span className="monitoring-kicker">
                      Core Track Condition Indicator
                    </span>
                    <h2 className="monitoring-title">Track Error Assessment</h2>
                    <p style={{ fontSize: 13, color: '#64748b' }}>
                      Replaced 3-condition UI with continuous percentage-based defect and disturbance scoring.
                    </p>
                  </div>

                  <div className="monitoring-meta">
                    <span
                      className={`mode-badge ${
                        operatingMode === 'Simulation Mode' ? 'simulation' : 'real'
                      }`}
                    >
                      {operatingMode}
                    </span>
                    <span
                      style={{
                        fontSize: 12,
                        color: '#64748b',
                        display: 'flex',
                        alignItems: 'center',
                        gap: 4
                      }}
                    >
                      <Clock3 size={14} /> Updated: {latestUpdateTime}
                    </span>
                  </div>
                </div>

                {/* Hero Error Percentage & Visual Gauge (Section 3) */}
                <div className="error-hero-container">
                  <div className="error-stat-main">
                    <span className="error-title-label">Primary Monitoring Metric</span>
                    <div className={`error-hero-value ${statusClass}`}>
                      Error Percentage: {errorPercentage.toFixed(1)}%
                    </div>
                    <div className={`status-pill ${statusClass}`}>
                      <span className="status-indicator-dot" />
                      <span>Current Status: {currentStatus}</span>
                    </div>
                  </div>

                  <div className="gauge-container">
                    <div className="gauge-markers">
                      <span>0% (Nominal)</span>
                      <span>25% (Threshold 1)</span>
                      <span>75% (Threshold 2)</span>
                      <span>100% (Critical)</span>
                    </div>

                    <div className="gauge-track">
                      <div
                        className={`gauge-fill ${statusClass}`}
                        style={{ width: `${Math.min(Math.max(errorPercentage, 3), 100)}%` }}
                      />
                    </div>

                    <div className="gauge-zone-labels">
                      <div className="zone-label green">0%–25%: No Error</div>
                      <div className="zone-label amber">&gt;25%–75%: Normal Alert</div>
                      <div className="zone-label red">&gt;75%: Warning Alert</div>
                    </div>
                  </div>
                </div>

                {/* PROMINENT SIMULATION BUTTON & AUTOMATIC CONTROLS (Section 4) */}
                <div className="simulation-controls-box">
                  <div className="controls-header">
                    <div className="controls-header-text">
                      <h3>Automated 3-Scenario Simulation</h3>
                      <p>
                        Cycles through 10% (Low Error), 50% (Moderate Error), and 90% (High Error) with exactly 10s between transitions.
                      </p>
                    </div>

                    <div className="controls-actions">
                      {isSimulating ? (
                        <button className="btn btn-danger btn-lg" onClick={stopSimulation}>
                          <Square size={16} fill="currentColor" />
                          Stop Simulation
                        </button>
                      ) : (
                        <button className="btn btn-primary btn-lg" onClick={startSimulation}>
                          <Play size={16} fill="currentColor" />
                          Start Simulation
                        </button>
                      )}
                    </div>
                  </div>

                  {/* 3 Simulation Scenarios Display */}
                  <div className="scenario-stepper">
                    {SIMULATION_SCENARIOS.map((sc, idx) => (
                      <div
                        key={sc.id}
                        className={`scenario-card ${
                          isSimulating && currentScenarioIdx === idx ? 'active' : ''
                        }`}
                      >
                        <div className="scenario-number">Scenario {sc.id}</div>
                        <div className="scenario-title">
                          <span>{sc.name.split('—')[1]}</span>
                          <span className={`scenario-pct ${sc.badgeClass}`}>
                            {sc.percentage}%
                          </span>
                        </div>
                        <div className="scenario-status-sub">
                          Status: <b>{sc.status}</b> · {sc.alertType ? 'Alert Sent' : 'No Alert'}
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Countdown indicator during active simulation */}
                  {isSimulating && (
                    <div className="timer-countdown-bar">
                      <span>
                        Cycling Scenario {currentScenarioIdx + 1} of 3 ({SIMULATION_SCENARIOS[currentScenarioIdx].name})
                      </span>
                      <span>Next scenario in {countdownSeconds}s</span>
                    </div>
                  )}
                </div>
              </section>

              {/* Lower 2-Column Grid */}
              <div className="overview-grid">
                {/* Real Prediction Quick Test Panel (Section 5) */}
                <div className="panel">
                  <div className="panel-header">
                    <div>
                      <h2>Real Prediction Mode Quick Test</h2>
                      <p>Select actual dataset rows to test real ML model inference</p>
                    </div>
                    <button
                      className="btn btn-secondary btn-sm"
                      onClick={() => setActiveTab('Real ML Inference')}
                    >
                      Full Console <ArrowUpRight size={14} />
                    </button>
                  </div>

                  <div className="panel-body">
                    <div className="row-selector-bar">
                      <label style={{ fontSize: 13, fontWeight: 600 }}>
                        Select Measurement Row:
                      </label>
                      <select
                        className="select-input"
                        value={selectedRowIndex}
                        onChange={(e) => setSelectedRowIndex(Number(e.target.value))}
                      >
                        <option value={0}>Row #0 — No train (Baseline, Expect &le; 25%)</option>
                        <option value={1000}>Row #1000 — Passenger train (Moderate, Expect ~50-65%)</option>
                        <option value={2000}>Row #2000 — Freight train (High load, Expect &gt; 75%)</option>
                      </select>

                      <button
                        className="btn btn-primary"
                        onClick={() => runRealPrediction(selectedRowIndex)}
                        disabled={isPredicting || !health?.model_available}
                      >
                        {isPredicting ? <RefreshCw size={14} className="spin" /> : <Cpu size={14} />}
                        Run Real ML Prediction
                      </button>
                    </div>

                    {predictionResult && (
                      <div style={{ marginTop: 20 }}>
                        <div
                          style={{
                            background: '#f8fafc',
                            border: '1px solid #e2e8f0',
                            borderRadius: 8,
                            padding: 16
                          }}
                        >
                          <div
                            style={{
                              display: 'flex',
                              justifyContent: 'space-between',
                              marginBottom: 8
                            }}
                          >
                            <b>Prediction: {predictionResult.prediction}</b>
                            <span
                              className={`badge-pill ${
                                predictionResult.status === 'No Error'
                                  ? 'no-error'
                                  : predictionResult.status === 'Normal Alert'
                                  ? 'normal'
                                  : 'warning'
                              }`}
                            >
                              {predictionResult.status} ({predictionResult.error_percentage}%)
                            </span>
                          </div>
                          <p style={{ fontSize: 12, color: '#64748b' }}>
                            Calculation: {predictionResult.calculation_method}
                          </p>

                          <div className="probabilities-grid">
                            {Object.entries(predictionResult.probabilities || {}).map(
                              ([cls, prob]) => (
                                <div className="prob-card" key={cls}>
                                  <div className="prob-label">{cls}</div>
                                  <div className="prob-val">{(prob * 100).toFixed(1)}%</div>
                                </div>
                              )
                            )}
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                {/* Recent Alerts Feed (Section 6) */}
                <div className="panel">
                  <div className="panel-header">
                    <div>
                      <h2>Recent Alert Activity</h2>
                      <p>Real-time audit log of simulation & real model alerts</p>
                    </div>
                    <button
                      className="btn btn-secondary btn-sm"
                      onClick={() => setActiveTab('Alert History')}
                    >
                      View All ({alerts.length})
                    </button>
                  </div>

                  <div className="panel-body">
                    {alerts.length === 0 ? (
                      <div
                        style={{
                          textAlign: 'center',
                          padding: '30px 16px',
                          color: '#64748b'
                        }}
                      >
                        <ShieldCheck size={36} color="#16a34a" style={{ margin: '0 auto 10px' }} />
                        <b>No Alerts Logged</b>
                        <p style={{ fontSize: 12, marginTop: 4 }}>
                          Track operation within standard error tolerance (0%–25%).
                        </p>
                      </div>
                    ) : (
                      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                        {alerts.slice(0, 4).map((al) => (
                          <div
                            key={al.id}
                            style={{
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'space-between',
                              padding: '10px 14px',
                              background: '#f8fafc',
                              border: '1px solid #e2e8f0',
                              borderRadius: 8
                            }}
                          >
                            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                              <span
                                className={`badge-pill ${
                                  al.severity === 'Warning Alert' ? 'warning' : 'normal'
                                }`}
                              >
                                {al.severity}
                              </span>
                              <div>
                                <b style={{ fontSize: 13 }}>{al.error_percentage}% Error</b>
                                <p style={{ fontSize: 11, color: '#64748b' }}>
                                  {al.reason?.slice(0, 50)}…
                                </p>
                              </div>
                            </div>

                            <div style={{ textAlign: 'right' }}>
                              <span
                                className={`badge-pill ${
                                  al.source === 'Simulation' ? 'source-sim' : 'source-real'
                                }`}
                              >
                                {al.source}
                              </span>
                              <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 3 }}>
                                {al.created_at
                                  ? new Date(al.created_at).toLocaleTimeString()
                                  : 'Just now'}
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </>
          )}

          {/* MAIN TAB 2: REAL ML PREDICTION CONSOLE (Section 1 & 5) */}
          {activeTab === 'Real ML Inference' && (
            <div className="panel">
              <div className="panel-header">
                <div>
                  <h2>Real Machine Learning Inference Console</h2>
                  <p>
                    Pass validated sensor features to the trained Random Forest model and compute calibrated defect error percentages.
                  </p>
                </div>
                <span className="badge-pill source-real">REAL PREDICTION MODE</span>
              </div>

              <div className="panel-body">
                <div className="prediction-box">
                  <h3 style={{ fontSize: 15, marginBottom: 8 }}>Select Measurement Row from Dataset</h3>
                  <p style={{ fontSize: 12, color: '#64748b', marginBottom: 14 }}>
                    Loads genuine physical / synthetic sensor measurements from data/railway_1km_measurements.csv.
                  </p>

                  <div className="row-selector-bar">
                    <select
                      className="select-input"
                      value={selectedRowIndex}
                      onChange={(e) => setSelectedRowIndex(Number(e.target.value))}
                      style={{ minWidth: 320 }}
                    >
                      <option value={0}>Row #0 — No train (Baseline conditions)</option>
                      <option value={500}>Row #500 — No train (Midway sensor 500m)</option>
                      <option value={1000}>Row #1000 — Passenger train (Moderate load)</option>
                      <option value={1500}>Row #1500 — Passenger train (1500m)</option>
                      <option value={2000}>Row #2000 — Freight train (Heavy load)</option>
                      <option value={2500}>Row #2500 — Freight train (2500m)</option>
                    </select>

                    <button
                      className="btn btn-primary"
                      onClick={() => runRealPrediction(selectedRowIndex)}
                      disabled={isPredicting || !health?.model_available}
                    >
                      {isPredicting ? <RefreshCw size={15} /> : <Cpu size={15} />}
                      Execute Model Prediction
                    </button>
                  </div>
                </div>

                {predictionResult && (
                  <div style={{ marginTop: 24 }}>
                    <div
                      style={{
                        background: '#ffffff',
                        border: '1px solid #e2e8f0',
                        borderRadius: 10,
                        padding: 24,
                        boxShadow: 'var(--shadow-sm)'
                      }}
                    >
                      <div
                        style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center',
                          flexWrap: 'wrap',
                          gap: 12,
                          marginBottom: 16
                        }}
                      >
                        <div>
                          <span style={{ fontSize: 11, color: '#64748b', fontWeight: 700 }}>
                            MODEL CLASSIFICATION OUTPUT
                          </span>
                          <h3 style={{ fontSize: 22, color: '#0f172a', margin: '4px 0' }}>
                            {predictionResult.prediction}
                          </h3>
                        </div>

                        <div style={{ textAlign: 'right' }}>
                          <span className={`status-pill ${statusClass}`}>
                            <span className="status-indicator-dot" />
                            {predictionResult.status} ({predictionResult.error_percentage}%)
                          </span>
                        </div>
                      </div>

                      <div
                        style={{
                          background: '#f8fafc',
                          padding: 12,
                          borderRadius: 6,
                          fontSize: 12,
                          color: '#475569',
                          marginBottom: 20
                        }}
                      >
                        <b>Calculation Methodology:</b> {predictionResult.calculation_method}
                      </div>

                      <h4 style={{ fontSize: 13, marginBottom: 10, color: '#334155' }}>
                        Class Probabilities Breakdown
                      </h4>
                      <div className="probabilities-grid" style={{ marginBottom: 24 }}>
                        {Object.entries(predictionResult.probabilities || {}).map(([cls, p]) => (
                          <div className="prob-card" key={cls}>
                            <div className="prob-label">{cls}</div>
                            <div className="prob-val">{(p * 100).toFixed(2)}%</div>
                          </div>
                        ))}
                      </div>

                      <h4 style={{ fontSize: 13, marginBottom: 10, color: '#334155' }}>
                        Input Feature Vector ({Object.keys(predictionResult.features || {}).length} variables)
                      </h4>
                      <div className="data-table-container">
                        <table className="clean-table">
                          <thead>
                            <tr>
                              <th>Feature Name</th>
                              <th>Input Value</th>
                            </tr>
                          </thead>
                          <tbody>
                            {Object.entries(predictionResult.features || {}).map(([key, val]) => (
                              <tr key={key}>
                                <td style={{ fontWeight: 600 }}>{key}</td>
                                <td>{val !== null ? String(val) : '—'}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* MAIN TAB 3: 3-NODE SEQUENTIAL SIMULATION (Preserved existing feature) */}
          {activeTab === '3-Node Simulation' && (
            <div className="panel">
              <div className="panel-header">
                <div>
                  <h2>3-Node Sequential Authorization Simulation</h2>
                  <p>
                    Sequential permission gate: Node 1 &rarr; Node 2 &rarr; Node 3. Enforced server-side.
                  </p>
                </div>
                <div style={{ display: 'flex', gap: 8 }}>
                  {!nodeSimulation ? (
                    <button
                      className="btn btn-primary"
                      onClick={start3NodeSimulation}
                      disabled={nodeSimBusy}
                    >
                      <Play size={14} /> Initialize Session
                    </button>
                  ) : (
                    <button
                      className="btn btn-secondary"
                      onClick={() => setConfirmResetModal(true)}
                      disabled={nodeSimBusy}
                    >
                      <RotateCcw size={14} /> Reset Session
                    </button>
                  )}
                </div>
              </div>

              <div className="panel-body">
                {!nodeSimulation ? (
                  <div style={{ textAlign: 'center', padding: '40px 16px', color: '#64748b' }}>
                    <Workflow size={40} color="#2563eb" style={{ margin: '0 auto 12px' }} />
                    <h3 style={{ fontSize: 16, color: '#0f172a' }}>Ready to Start 3-Node Session</h3>
                    <p style={{ fontSize: 13, maxWidth: 440, margin: '8px auto 20px' }}>
                      Creates a dedicated session across Node 1, Node 2, and Node 3 with independent manual permission gates.
                    </p>
                    <button
                      className="btn btn-primary"
                      onClick={start3NodeSimulation}
                      disabled={nodeSimBusy}
                    >
                      <Play size={15} /> Create 3-Node Simulation
                    </button>
                  </div>
                ) : (
                  <div>
                    <div
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        padding: '12px 16px',
                        background: '#f8fafc',
                        border: '1px solid #e2e8f0',
                        borderRadius: 8,
                        marginBottom: 16,
                        fontSize: 12
                      }}
                    >
                      <span>
                        Session ID:{' '}
                        <b>{nodeSimulation.simulation_id.slice(0, 8).toUpperCase()}</b>
                      </span>
                      <span>Progress: {completedNodesCount} of 3 Nodes Completed</span>
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                      {(nodeSimulation.nodes || []).map((node) => (
                        <div
                          key={node.node_number}
                          style={{
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'space-between',
                            padding: '16px 20px',
                            background: '#ffffff',
                            border: '1px solid #e2e8f0',
                            borderRadius: 10,
                            gap: 16,
                            flexWrap: 'wrap'
                          }}
                        >
                          <div>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                              <b style={{ fontSize: 15 }}>Rail Node {node.node_number}</b>
                              <span
                                className={`badge-pill ${
                                  node.status === 'COMPLETED'
                                    ? 'no-error'
                                    : node.status === 'RUNNING'
                                    ? 'normal'
                                    : 'source-sim'
                                }`}
                              >
                                {node.status.replaceAll('_', ' ')}
                              </span>
                            </div>
                            <p style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>
                              Assigned dataset row #{node.row_index}
                            </p>
                            {node.prediction && (
                              <div style={{ fontSize: 12, marginTop: 6 }}>
                                Model Output: <b>{node.prediction}</b> ({node.error_percentage}% error)
                              </div>
                            )}
                          </div>

                          <button
                            className="btn btn-primary btn-sm"
                            disabled={
                              nodeSimBusy ||
                              !canRunNode(node.node_number) ||
                              !health?.model_available
                            }
                            onClick={() => runSingleNode(node.node_number)}
                          >
                            {node.status === 'COMPLETED' ? (
                              <>
                                <CheckCircle2 size={14} /> Completed
                              </>
                            ) : (
                              <>
                                <ShieldCheck size={14} /> Authorize &amp; Run
                              </>
                            )}
                          </button>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* MAIN TAB 4: LIVE MAP */}
          {activeTab === 'Live Map' && (
            <div className="panel">
              <div className="panel-header">
                <div>
                  <h2>Railway Corridor Interactive Map</h2>
                  <p>{mapData?.city || 'Delhi Corridor Network'} · OpenStreetMap base tiles</p>
                </div>
              </div>

              <div className="panel-body">
                <div className="map-wrap large">
                  <MapContainer
                    center={[28.625, 77.22]}
                    zoom={12}
                    scrollWheelZoom={true}
                    style={{ height: '100%', width: '100%' }}
                  >
                    <TileLayer
                      attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                      url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                    />
                    {mapNodes.map((n) => (
                      <Marker
                        key={n.id}
                        position={[n.lat, n.lng]}
                        icon={createPinIcon(getStatusColor(n.status))}
                      >
                        <Popup>
                          <div style={{ padding: 4 }}>
                            <b>{n.name}</b>
                            <div style={{ fontSize: 12, marginTop: 4 }}>
                              Status: {n.status?.replaceAll('_', ' ')}
                            </div>
                            <div style={{ fontSize: 12 }}>
                              Prediction: {n.prediction || 'Not run'}
                            </div>
                          </div>
                        </Popup>
                      </Marker>
                    ))}
                    {(mapData?.stations || []).map((s) => (
                      <Marker
                        key={s.id}
                        position={[s.lat, s.lng]}
                        icon={createPinIcon('#2563eb')}
                      >
                        <Popup>
                          <div style={{ padding: 4 }}>
                            <b>{s.name}</b>
                            <div style={{ fontSize: 12, color: '#64748b' }}>
                              Track Junction / Station
                            </div>
                          </div>
                        </Popup>
                      </Marker>
                    ))}
                    {mapNodes.length > 1 && (
                      <Polyline
                        positions={mapNodes.map((n) => [n.lat, n.lng])}
                        pathOptions={{ color: '#2563eb', weight: 3, dashArray: '6 8' }}
                      />
                    )}
                  </MapContainer>
                </div>
                <div className="map-disclaimer">
                  Schematic demo coordinates for visualization purposes. Nodes represent sensor checkpoints along track corridor.
                </div>
              </div>
            </div>
          )}

          {/* MAIN TAB 5: ALERT HISTORY (Section 6) */}
          {activeTab === 'Alert History' && (
            <div className="panel">
              <div className="panel-header">
                <div>
                  <h2>Alert History &amp; Incident Log</h2>
                  <p>Comprehensive audit log tracking timestamps, error percentages, and sources.</p>
                </div>
                <div style={{ display: 'flex', gap: 8 }}>
                  <select
                    className="select-input"
                    value={alertFilter}
                    onChange={(e) => setAlertFilter(e.target.value)}
                  >
                    <option value="ALL">All Alerts ({alerts.length})</option>
                    <option value="REAL">Real Prediction Alerts</option>
                    <option value="SIMULATION">Simulation Alerts</option>
                  </select>
                  <button
                    className="btn btn-secondary btn-sm"
                    onClick={() => {
                      if (window.confirm('Clear all recorded alert logs?')) {
                        api.clearAlerts().then(() => setAlerts([]));
                      }
                    }}
                  >
                    Clear History
                  </button>
                </div>
              </div>

              <div className="panel-body data-table-container">
                {filteredAlerts.length === 0 ? (
                  <div style={{ textAlign: 'center', padding: '40px 16px', color: '#64748b' }}>
                    <ShieldCheck size={40} color="#16a34a" style={{ margin: '0 auto 12px' }} />
                    <h3>No Alerts Found</h3>
                    <p style={{ fontSize: 13, marginTop: 4 }}>
                      No incidents have been triggered under the current filter.
                    </p>
                  </div>
                ) : (
                  <table className="clean-table">
                    <thead>
                      <tr>
                        <th>ID</th>
                        <th>Timestamp</th>
                        <th>Error %</th>
                        <th>Severity</th>
                        <th>Source</th>
                        <th>Reason / Details</th>
                        <th>Status</th>
                        <th>Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredAlerts.map((a) => (
                        <tr key={a.id}>
                          <td>#{a.id}</td>
                          <td>
                            {a.created_at
                              ? new Date(a.created_at).toLocaleString()
                              : '—'}
                          </td>
                          <td style={{ fontWeight: 800 }}>{a.error_percentage}%</td>
                          <td>
                            <span
                              className={`badge-pill ${
                                a.severity === 'Warning Alert' ? 'warning' : 'normal'
                              }`}
                            >
                              {a.severity}
                            </span>
                          </td>
                          <td>
                            <span
                              className={`badge-pill ${
                                a.source === 'Simulation' ? 'source-sim' : 'source-real'
                              }`}
                            >
                              {a.source}
                            </span>
                          </td>
                          <td>{a.reason}</td>
                          <td>
                            <span
                              style={{
                                fontSize: 11,
                                fontWeight: 700,
                                color: a.status === 'RESOLVED' ? '#16a34a' : '#d97706'
                              }}
                            >
                              {a.status}
                            </span>
                          </td>
                          <td>
                            {a.status !== 'RESOLVED' && (
                              <button
                                className="btn btn-secondary btn-sm"
                                onClick={() =>
                                  api.updateAlert(a.id, 'RESOLVED').then(fetchAllData)
                                }
                              >
                                Resolve
                              </button>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            </div>
          )}

          {/* MAIN TAB 6: MODEL INSIGHTS */}
          {activeTab === 'Model Insights' && (
            <div className="panel">
              <div className="panel-header">
                <div>
                  <h2>Model Architecture &amp; Dataset Insights</h2>
                  <p>Trained Random Forest estimator metadata and feature importances.</p>
                </div>
              </div>

              <div className="panel-body">
                <div className="stats-grid" style={{ marginBottom: 24 }}>
                  <div className="stat-card">
                    <span className="stat-label">Model Status</span>
                    <div className="stat-value">{model?.available ? 'Loaded' : 'Offline'}</div>
                    <div className="stat-sub">train_condition_model.joblib</div>
                  </div>
                  <div className="stat-card">
                    <span className="stat-label">Dataset Records</span>
                    <div className="stat-value">{health?.dataset_rows || 3000}</div>
                    <div className="stat-sub">1km Track Measurements</div>
                  </div>
                  <div className="stat-card">
                    <span className="stat-label">Model Target Classes</span>
                    <div className="stat-value" style={{ fontSize: 16 }}>
                      {model?.classes?.join(', ') || 'Freight, No train, Passenger'}
                    </div>
                    <div className="stat-sub">3 discrete conditions</div>
                  </div>
                  <div className="stat-card">
                    <span className="stat-label">Features Expected</span>
                    <div className="stat-value">{model?.features?.length || 20}</div>
                    <div className="stat-sub">Numerical sensor inputs</div>
                  </div>
                </div>

                <h3 style={{ fontSize: 15, marginBottom: 12 }}>Top Feature Importances</h3>
                {importanceData.length > 0 ? (
                  <div style={{ height: 260, marginBottom: 28 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={importanceData}
                        layout="vertical"
                        margin={{ left: 20, right: 30, top: 10, bottom: 10 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#e2e8f0" />
                        <XAxis type="number" tick={{ fontSize: 11, fill: '#64748b' }} />
                        <YAxis
                          type="category"
                          dataKey="name"
                          width={140}
                          tick={{ fontSize: 11, fill: '#334155' }}
                        />
                        <Tooltip />
                        <Bar dataKey="value" fill="#2563eb" radius={[0, 4, 4, 0]} barSize={16} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
                    Feature importances will display once the model is active.
                  </p>
                )}

                <h3 style={{ fontSize: 15, marginBottom: 12 }}>Expected Feature Manifest</h3>
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 24 }}>
                  {(model?.features || []).map((feat) => (
                    <span
                      key={feat}
                      style={{
                        fontSize: 11,
                        background: '#f1f5f9',
                        padding: '6px 10px',
                        borderRadius: 6,
                        color: '#334155',
                        fontWeight: 500
                      }}
                    >
                      {feat}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <footer className="footer">
          <span>TRACKX RAILWAY HEALTH &amp; DEFECT INTELLIGENCE PLATFORM</span>
          <span>
            API Status:{' '}
            <b style={{ color: health?.status === 'ok' ? '#16a34a' : '#d97706' }}>
              {health?.status === 'ok' ? 'Connected (FastAPI)' : 'Connecting...'}
            </b>
          </span>
          <span>Light Theme UI v2.0 · Demonstrational &amp; Analytical Console</span>
        </footer>
      </main>

      {/* Confirmation Modal for 3-Node Reset */}
      {confirmResetModal && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(15, 23, 42, 0.4)',
            zIndex: 100,
            display: 'grid',
            placeItems: 'center',
            padding: 20
          }}
        >
          <div
            style={{
              background: '#ffffff',
              borderRadius: 12,
              padding: 28,
              maxWidth: 420,
              width: '100%',
              boxShadow: 'var(--shadow-lg)'
            }}
          >
            <h3 style={{ fontSize: 18, marginBottom: 8 }}>Reset 3-Node Simulation?</h3>
            <p style={{ fontSize: 13, color: '#64748b', lineHeight: 1.5 }}>
              This will clear execution states and predictions across all 3 nodes. You will need to grant sequential permissions again.
            </p>
            <div
              style={{
                display: 'flex',
                justifyContent: 'flex-end',
                gap: 10,
                marginTop: 24
              }}
            >
              <button
                className="btn btn-secondary"
                onClick={() => setConfirmResetModal(false)}
              >
                Cancel
              </button>
              <button className="btn btn-danger" onClick={reset3NodeSimulation}>
                Reset Session
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
