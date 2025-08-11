import React, { useState, useEffect } from 'react';
import { DocumentTextIcon, ExclamationTriangleIcon, InformationCircleIcon } from '@heroicons/react/24/outline';
import apiService from '../services/api';

const LogsPanel = () => {
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('all');
  const [autoRefresh, setAutoRefresh] = useState(true);

  useEffect(() => {
    loadLogs();
    
    let interval;
    if (autoRefresh) {
      interval = setInterval(loadLogs, 5000); // Refresh every 5 seconds
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [autoRefresh]);

  const loadLogs = async () => {
    try {
      const logsData = await apiService.getSystemLogs(200);
      setLogs(logsData.logs || []);
    } catch (error) {
      console.error('Failed to load logs:', error);
    } finally {
      setLoading(false);
    }
  };

  const getLogIcon = (level) => {
    switch (level.toLowerCase()) {
      case 'error':
        return <ExclamationTriangleIcon className="h-5 w-5 text-red-500" />;
      case 'warning':
        return <ExclamationTriangleIcon className="h-5 w-5 text-yellow-500" />;
      case 'info':
        return <InformationCircleIcon className="h-5 w-5 text-blue-500" />;
      default:
        return <DocumentTextIcon className="h-5 w-5 text-gray-500" />;
    }
  };

  const getLogColor = (level) => {
    switch (level.toLowerCase()) {
      case 'error':
        return 'text-red-700 bg-red-50';
      case 'warning':
        return 'text-yellow-700 bg-yellow-50';
      case 'info':
        return 'text-blue-700 bg-blue-50';
      default:
        return 'text-gray-700 bg-gray-50';
    }
  };

  const filteredLogs = logs.filter(log => {
    if (filter === 'all') return true;
    return log.level.toLowerCase() === filter.toLowerCase();
  });

  const logCounts = {
    all: logs.length,
    info: logs.filter(log => log.level.toLowerCase() === 'info').length,
    warning: logs.filter(log => log.level.toLowerCase() === 'warning').length,
    error: logs.filter(log => log.level.toLowerCase() === 'error').length,
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-900">System Logs</h2>
        <div className="flex items-center space-x-4">
          <label className="flex items-center">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
            />
            <span className="ml-2 text-sm text-gray-700">Auto-refresh</span>
          </label>
          <button
            onClick={loadLogs}
            className="inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm leading-4 font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500"
          >
            Refresh
          </button>
        </div>
      </div>

      {/* Log Level Filters */}
      <div className="flex space-x-4">
        {['all', 'info', 'warning', 'error'].map((level) => (
          <button
            key={level}
            onClick={() => setFilter(level)}
            className={`px-4 py-2 text-sm font-medium rounded-md ${
              filter === level
                ? 'bg-primary-100 text-primary-800'
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            {level.charAt(0).toUpperCase() + level.slice(1)} ({logCounts[level]})
          </button>
        ))}
      </div>

      {/* Logs Display */}
      <div className="bg-white shadow rounded-lg">
        <div className="max-h-96 overflow-y-auto">
          {filteredLogs.length === 0 ? (
            <div className="p-6 text-center">
              <DocumentTextIcon className="mx-auto h-12 w-12 text-gray-400" />
              <h3 className="mt-2 text-sm font-medium text-gray-900">No logs found</h3>
              <p className="mt-1 text-sm text-gray-500">
                {filter === 'all' 
                  ? 'No logs available at this time.'
                  : `No ${filter} logs found.`
                }
              </p>
            </div>
          ) : (
            <div className="divide-y divide-gray-200">
              {filteredLogs.map((log, index) => (
                <div key={index} className={`p-4 ${getLogColor(log.level)}`}>
                  <div className="flex items-start">
                    <div className="flex-shrink-0 mr-3">
                      {getLogIcon(log.level)}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between">
                        <p className="text-sm font-medium">
                          {log.message}
                        </p>
                        <div className="flex items-center text-xs text-gray-500">
                          <span className="mr-2 px-2 py-1 bg-white rounded">
                            {log.level.toUpperCase()}
                          </span>
                          <span>
                            {new Date(log.timestamp).toLocaleString()}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Log Statistics */}
      <div className="bg-white shadow rounded-lg p-6">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Log Statistics</h3>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-4">
          <div className="text-center">
            <div className="text-2xl font-semibold text-gray-900">{logCounts.all}</div>
            <div className="text-sm text-gray-500">Total Logs</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-semibold text-blue-600">{logCounts.info}</div>
            <div className="text-sm text-gray-500">Info</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-semibold text-yellow-600">{logCounts.warning}</div>
            <div className="text-sm text-gray-500">Warnings</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-semibold text-red-600">{logCounts.error}</div>
            <div className="text-sm text-gray-500">Errors</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LogsPanel;