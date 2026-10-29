/**
 * TrackX API Service
 * Handles communication with the FastAPI backend.
 * Uses VITE_API_BASE_URL environment variable with fallback to local development URL.
 */

const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000/api';

async function request(path, options = {}) {
  const url = `${BASE_URL}${path}`;
  const headers = {
    'Content-Type': 'application/json',
    ...(options.headers || {})
  };

  try {
    const response = await fetch(url, { ...options, headers });
    let data;
    try {
      data = await response.json();
    } catch {
      data = null;
    }

    if (!response.ok) {
      const msg = data?.detail || data?.message || `Request failed with status ${response.status}`;
      const err = new Error(msg);
      err.status = response.status;
      err.data = data;
      throw err;
    }
    return data;
  } catch (err) {
    if (err.name === 'TypeError' && err.message.includes('fetch')) {
      const connErr = new Error(`Cannot connect to backend at ${BASE_URL}. Ensure FastAPI is running.`);
      connErr.status = 503;
      throw connErr;
    }
    throw err;
  }
}

export const api = {
  baseUrl: BASE_URL,

  getHealth: () => request('/health'),
  getModelInfo: () => request('/model'),
  getMapData: () => request('/map'),

  // Prediction
  predict: (params) => request('/predict', {
    method: 'POST',
    body: JSON.stringify(params)
  }),

  // Alerts
  getAlerts: () => request('/alerts'),
  createAlert: (alert) => request('/alerts', {
    method: 'POST',
    body: JSON.stringify(alert)
  }),
  updateAlert: (id, status) => request(`/alerts/${id}`, {
    method: 'PATCH',
    body: JSON.stringify({ status })
  }),
  clearAlerts: () => request('/alerts', { method: 'DELETE' }),

  // Dataset
  getDatasetPreview: () => request('/dataset/preview'),
  getDatasetSample: () => request('/dataset/sample'),

  // 3-Node simulation
  createSimulation: (params = {}) => request('/simulations', {
    method: 'POST',
    body: JSON.stringify(params)
  }),
  getSimulation: (id) => request(`/simulations/${id}`),
  runNode: (id, nodeNumber) => request(`/simulations/${id}/nodes/${nodeNumber}/run`, {
    method: 'POST'
  }),
  resetSimulation: (id) => request(`/simulations/${id}/reset`, {
    method: 'POST'
  })
};

export default api;
