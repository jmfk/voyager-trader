import React from 'react';
import { Navigate } from 'react-router-dom';
import apiService from '../services/api';

const ProtectedRoute = ({ children }) => {
  if (!apiService.isAuthenticated()) {
    return <Navigate to="/login" replace />;
  }

  return children;
};

export default ProtectedRoute;