import React, { useState, useEffect } from 'react';
import { CpuChipIcon } from '@heroicons/react/24/outline';
import apiService from '../services/api';

const SkillsTable = () => {
  const [skills, setSkills] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadSkills();
  }, []);

  const loadSkills = async () => {
    try {
      const skillsData = await apiService.getSkills();
      setSkills(skillsData);
    } catch (error) {
      console.error('Failed to load skills:', error);
    } finally {
      setLoading(false);
    }
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
        <h2 className="text-2xl font-bold text-gray-900">Skill Library</h2>
        <div className="text-sm text-gray-500">
          {skills.length} skills available
        </div>
      </div>

      {skills.length === 0 ? (
        <div className="text-center py-12">
          <CpuChipIcon className="mx-auto h-12 w-12 text-gray-400" />
          <h3 className="mt-2 text-sm font-medium text-gray-900">No skills found</h3>
          <p className="mt-1 text-sm text-gray-500">
            Skills will appear here as the system learns and develops trading strategies.
          </p>
        </div>
      ) : (
        <div className="bg-white shadow rounded-lg overflow-hidden">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Skill Name
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Description
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Success Rate
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Usage Count
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Created
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {skills.map((skill, index) => (
                <tr key={skill.name || index} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <div className="flex-shrink-0 h-10 w-10">
                        <div className="h-10 w-10 rounded-full bg-primary-100 flex items-center justify-center">
                          <CpuChipIcon className="h-6 w-6 text-primary-600" />
                        </div>
                      </div>
                      <div className="ml-4">
                        <div className="text-sm font-medium text-gray-900">
                          {skill.name}
                        </div>
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    <div className="text-sm text-gray-900 max-w-xs truncate">
                      {skill.description}
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <div className="text-sm text-gray-900">
                        {(skill.success_rate * 100).toFixed(1)}%
                      </div>
                      <div className="ml-2 w-16 bg-gray-200 rounded-full h-2">
                        <div 
                          className="bg-green-500 h-2 rounded-full"
                          style={{ width: `${skill.success_rate * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {skill.usage_count}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {new Date(skill.created_at).toLocaleDateString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Skills Summary */}
      {skills.length > 0 && (
        <div className="bg-white shadow rounded-lg p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Skills Summary</h3>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            <div className="text-center">
              <div className="text-2xl font-semibold text-gray-900">{skills.length}</div>
              <div className="text-sm text-gray-500">Total Skills</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-semibold text-green-600">
                {((skills.reduce((sum, skill) => sum + skill.success_rate, 0) / skills.length) * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-gray-500">Average Success Rate</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-semibold text-blue-600">
                {skills.reduce((sum, skill) => sum + skill.usage_count, 0)}
              </div>
              <div className="text-sm text-gray-500">Total Usage</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default SkillsTable;