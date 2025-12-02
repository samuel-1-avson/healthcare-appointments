import axios from 'axios';
import type { AppointmentFeatures, PredictionResponse } from '../types';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001';

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

export const chatWithAssistant = async (data: { message: string; session_id?: string }) => {
    const response = await api.post('/api/v1/llm/chat', data);
    return response.data;
};

export const getModelInfo = async () => {
    const response = await api.get('/api/v1/model/info');
    return response.data;
};

export const getModelMetrics = async () => {
    const response = await api.get('/api/v1/model/metrics');
    return response.data;
};

export default api;
