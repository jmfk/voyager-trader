import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8001/api';

class ApiService {
  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 10000,
    });

    // Add auth token to requests
    this.client.interceptors.request.use((config) => {
      const token = localStorage.getItem('authToken');
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
      return config;
    });

    // Handle auth errors
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        if (error.response?.status === 401) {
          localStorage.removeItem('authToken');
          window.location.href = '/login';
        }
        return Promise.reject(error);
      }
    );
  }

  // Authentication
  async login(username, password) {
    const response = await this.client.post('/auth/login', {
      username,
      password,
    });
    const { access_token } = response.data;
    localStorage.setItem('authToken', access_token);
    return response.data;
  }

  logout() {
    localStorage.removeItem('authToken');
  }

  isAuthenticated() {
    return !!localStorage.getItem('authToken');
  }

  // System status
  async getSystemStatus() {
    const response = await this.client.get('/status');
    return response.data;
  }

  // System control
  async controlSystem(action) {
    const response = await this.client.post('/system/control', { action });
    return response.data;
  }

  // Skills
  async getSkills() {
    const response = await this.client.get('/skills');
    return response.data;
  }

  // Performance metrics
  async getPerformanceMetrics() {
    const response = await this.client.get('/performance');
    return response.data;
  }

  // System logs
  async getSystemLogs(limit = 100) {
    const response = await this.client.get(`/logs?limit=${limit}`);
    return response.data;
  }

  // Health check
  async healthCheck() {
    const response = await this.client.get('/health');
    return response.data;
  }
}

export default new ApiService();