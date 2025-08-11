import React from 'react';
import {
  ClockIcon,
  CpuChipIcon,
  ChartBarIcon,
  CheckCircleIcon,
} from '@heroicons/react/24/outline';

const SystemStatus = ({ status }) => {
  if (!status) return null;

  const cards = [
    {
      name: 'System Status',
      value: status.is_running ? 'Running' : 'Stopped',
      icon: CheckCircleIcon,
      color: status.is_running ? 'text-green-600' : 'text-red-600',
      bgColor: status.is_running ? 'bg-green-100' : 'bg-red-100',
    },
    {
      name: 'Uptime',
      value: status.uptime,
      icon: ClockIcon,
      color: 'text-blue-600',
      bgColor: 'bg-blue-100',
    },
    {
      name: 'Skills Learned',
      value: status.skills_learned,
      icon: CpuChipIcon,
      color: 'text-purple-600',
      bgColor: 'bg-purple-100',
    },
    {
      name: 'Current Task',
      value: status.current_task || 'Idle',
      icon: ChartBarIcon,
      color: 'text-yellow-600',
      bgColor: 'bg-yellow-100',
    },
  ];

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-gray-900">System Overview</h2>
      
      {/* Status Cards */}
      <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
        {cards.map((card) => {
          const Icon = card.icon;
          return (
            <div
              key={card.name}
              className="bg-white overflow-hidden shadow rounded-lg"
            >
              <div className="p-5">
                <div className="flex items-center">
                  <div className="flex-shrink-0">
                    <div className={`p-3 rounded-md ${card.bgColor}`}>
                      <Icon className={`h-6 w-6 ${card.color}`} />
                    </div>
                  </div>
                  <div className="ml-5 w-0 flex-1">
                    <dl>
                      <dt className="text-sm font-medium text-gray-500 truncate">
                        {card.name}
                      </dt>
                      <dd className="text-lg font-medium text-gray-900">
                        {card.value}
                      </dd>
                    </dl>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Performance Metrics */}
      {status.performance && Object.keys(status.performance).length > 0 && (
        <div className="bg-white shadow rounded-lg">
          <div className="p-6">
            <h3 className="text-lg leading-6 font-medium text-gray-900">
              Performance Summary
            </h3>
            <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {Object.entries(status.performance).map(([key, value]) => (
                <div key={key} className="border-l-4 border-primary-500 pl-4">
                  <p className="text-sm font-medium text-gray-500">
                    {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </p>
                  <p className="text-xl font-semibold text-gray-900">
                    {typeof value === 'number' ? value.toFixed(2) : value}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default SystemStatus;