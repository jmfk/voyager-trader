import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  PlayIcon,
  StopIcon,
  ArrowPathIcon,
  ChartBarIcon,
  CpuChipIcon,
  DocumentTextIcon,
  PowerIcon,
} from '@heroicons/react/24/outline';
import apiService from '../services/api';
import { createLogger } from '../utils/logger';
import SystemStatus from './SystemStatus';
import PerformanceChart from './PerformanceChart';
import SkillsTable from './SkillsTable';
import LogsPanel from './LogsPanel';

const Dashboard = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const [systemStatus, setSystemStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();
  const logger = createLogger('Dashboard');

  useEffect(() => {
    loadSystemStatus();
    // Refresh status every 10 seconds
    const interval = setInterval(loadSystemStatus, 10000);
    return () => clearInterval(interval);
  }, []);

  const loadSystemStatus = async () => {
    try {
      const status = await apiService.getSystemStatus();
      setSystemStatus(status);
      logger.debug('System status loaded successfully', { status });
    } catch (error) {
      logger.apiError('/api/status', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSystemControl = async (action) => {
    try {
      logger.userAction(`system-control-${action}`, { action });
      await apiService.controlSystem(action);
      await loadSystemStatus();
      logger.info(`System ${action} completed successfully`);
    } catch (error) {
      logger.apiError('/api/system/control', error, { action });
    }
  };

  const handleLogout = () => {
    logger.userAction('logout');
    apiService.logout();
    navigate('/login');
  };

  const tabs = [
    { id: 'overview', name: 'Overview', icon: ChartBarIcon },
    { id: 'performance', name: 'Performance', icon: ChartBarIcon },
    { id: 'skills', name: 'Skills', icon: CpuChipIcon },
    { id: 'logs', name: 'Logs', icon: DocumentTextIcon },
  ];

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div className="flex items-center">
              <h1 className="text-3xl font-bold text-gray-900">
                VoyagerTrader Admin
              </h1>
              {systemStatus && (
                <div className="ml-6 flex items-center">
                  <div className={`h-3 w-3 rounded-full mr-2 ${
                    systemStatus.is_running ? 'bg-green-500' : 'bg-red-500'
                  }`}></div>
                  <span className="text-sm text-gray-600">
                    {systemStatus.is_running ? 'Running' : 'Stopped'}
                  </span>
                </div>
              )}
            </div>
            
            <div className="flex items-center space-x-4">
              {/* System Control Buttons */}
              <div className="flex space-x-2">
                <button
                  onClick={() => handleSystemControl('start')}
                  disabled={systemStatus?.is_running}
                  className="inline-flex items-center px-3 py-2 border border-transparent text-sm leading-4 font-medium rounded-md text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 disabled:opacity-50"
                >
                  <PlayIcon className="h-4 w-4 mr-1" />
                  Start
                </button>
                <button
                  onClick={() => handleSystemControl('stop')}
                  disabled={!systemStatus?.is_running}
                  className="inline-flex items-center px-3 py-2 border border-transparent text-sm leading-4 font-medium rounded-md text-white bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500 disabled:opacity-50"
                >
                  <StopIcon className="h-4 w-4 mr-1" />
                  Stop
                </button>
                <button
                  onClick={() => handleSystemControl('restart')}
                  className="inline-flex items-center px-3 py-2 border border-transparent text-sm leading-4 font-medium rounded-md text-white bg-yellow-600 hover:bg-yellow-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-yellow-500"
                >
                  <ArrowPathIcon className="h-4 w-4 mr-1" />
                  Restart
                </button>
              </div>
              
              <button
                onClick={handleLogout}
                className="inline-flex items-center px-3 py-2 border border-gray-300 text-sm leading-4 font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500"
              >
                <PowerIcon className="h-4 w-4 mr-1" />
                Logout
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Tab Navigation */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="border-b border-gray-200">
          <nav className="-mb-px flex space-x-8">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`${
                    activeTab === tab.id
                      ? 'border-primary-500 text-primary-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm flex items-center`}
                >
                  <Icon className="h-5 w-5 mr-2" />
                  {tab.name}
                </button>
              );
            })}
          </nav>
        </div>
      </div>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'overview' && <SystemStatus status={systemStatus} />}
        {activeTab === 'performance' && <PerformanceChart />}
        {activeTab === 'skills' && <SkillsTable />}
        {activeTab === 'logs' && <LogsPanel />}
      </main>
    </div>
  );
};

export default Dashboard;