import axios from 'axios';
import type { AppointmentFeatures, PredictionResponse } from '../types';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
    baseURL: API_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

export const predictNoShow = async (data: AppointmentFeatures): Promise<PredictionResponse> => {
    const response = await api.post<PredictionResponse>('/api/v1/predict', data);
    return response.data;
};

export const batchPredict = async (data: { appointments: AppointmentFeatures[] }) => {
    const response = await api.post('/api/v1/predict/batch', data);
    return response.data;
};

export const chatWithAssistant = async (data: { message: string; session_id?: string; context?: any }) => {
    const response = await api.post('/api/v1/llm/chat', data);
    return response.data;
};

export const getSmartFill = async (scenario: string = 'high') => {
    const response = await api.post('/api/v1/llm/smart-fill', scenario, {
        headers: { 'Content-Type': 'application/json' }
    });
    return response.data;
};

export const getModelInfo = async () => {
    const response = await api.get('/api/v1/model');
    return response.data;
};

export const getModelMetrics = async () => {
    const response = await api.get('/api/v1/model/metrics');
    return response.data;
};

export const getPredictionHistory = async () => {
    const response = await api.get('/api/v1/model/history');
    return response.data;
};

export const getSettings = async () => {
    const response = await api.get('/api/v1/settings/');
    return response.data;
};

export const updateSettings = async (settings: any) => {
    const response = await api.put('/api/v1/settings/', settings);
    return response.data;
};

export const getUserProfile = async () => {
    const response = await api.get('/api/v1/users/profile');
    return response.data;
};

export default api;
