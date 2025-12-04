import { useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import ModelDashboard from './components/ModelDashboard';
import PredictionForm from './components/PredictionForm';
import BatchUpload from './components/BatchUpload';
import ChatAssistant from './components/ChatAssistant';
import Settings from './components/Settings';
import PatientList from './components/PatientList';
import { SettingsProvider } from './context/SettingsContext';
import { predictNoShow } from './services/api';
import type { AppointmentFeatures } from './types';

import PredictionResult from './components/PredictionResult';
import type { PredictionResponse } from './types';

const PredictionPage = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);

  const handlePredictionSubmit = async (data: AppointmentFeatures) => {
    setIsLoading(true);
    try {
      const response = await predictNoShow(data);
      setResult(response);
    } catch (error) {
      console.error("Prediction failed", error);
      // You might want to add error state handling here too
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setResult(null);
  };

  if (result) {
    return <PredictionResult result={result} onReset={handleReset} />;
  }

  return <PredictionForm onSubmit={handlePredictionSubmit} isLoading={isLoading} />;
};

function App() {
  return (
    <SettingsProvider>
      <Router>
        <Layout>
          <Routes>
            <Route path="/" element={<ModelDashboard />} />
            <Route path="/predict" element={<PredictionPage />} />
            <Route path="/batch" element={<BatchUpload />} />
            <Route path="/patients" element={<PatientList />} />
            <Route path="/chat" element={<ChatAssistant />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </Layout>
      </Router>
    </SettingsProvider>
  );
}

export default App;
