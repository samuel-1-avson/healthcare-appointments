import { useState } from 'react';
import Layout from './components/Layout';
import PredictionForm from './components/PredictionForm';
import PredictionResult from './components/PredictionResult';
import BatchUpload from './components/BatchUpload';
import ChatAssistant from './components/ChatAssistant';
import ModelDashboard from './components/ModelDashboard';
import { predictNoShow } from './services/api';
import type { AppointmentFeatures, PredictionResponse } from './types';

type Tab = 'predict' | 'batch' | 'chat' | 'dashboard';

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('predict');
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [darkMode, setDarkMode] = useState(true);

  const handlePrediction = async (data: AppointmentFeatures) => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await predictNoShow(data);
      setResult(response);
    } catch (err) {
      setError('Failed to get prediction. Please try again.');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setResult(null);
    setError(null);
  };

  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
    if (darkMode) {
      document.documentElement.classList.remove('dark');
    } else {
      document.documentElement.classList.add('dark');
    }
  };

  return (
    <Layout
      activeTab={activeTab}
      setActiveTab={setActiveTab}
      darkMode={darkMode}
      toggleDarkMode={toggleDarkMode}
    >
      {activeTab === 'predict' && (
        <div className="w-full">
          <div className="text-center mb-12">
            <h1 className="text-5xl md:text-7xl font-black text-gray-900 dark:text-white mb-6 tracking-tight">
              Predict Patient{' '}
              <span className="bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 dark:from-blue-400 dark:via-purple-400 dark:to-pink-400 bg-clip-text text-transparent">
                Attendance
              </span>
            </h1>
            <p className="text-xl text-gray-600 dark:text-gray-300 max-w-2xl mx-auto leading-relaxed">
              Leverage cutting-edge AI to identify high-risk appointments and optimize your schedule efficiency.
            </p>
          </div>

          {error && (
            <div className="max-w-4xl mx-auto mb-8 bg-red-500/10 border border-red-500/20 text-red-600 dark:text-red-300 px-6 py-4 rounded-2xl flex items-center gap-3">
              <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
              {error}
            </div>
          )}

          <div className="max-w-7xl mx-auto">
            {result ? (
              <PredictionResult result={result} onReset={handleReset} />
            ) : (
              <PredictionForm onSubmit={handlePrediction} isLoading={isLoading} />
            )}
          </div>
        </div>
      )}

      {activeTab === 'batch' && <BatchUpload />}
      {activeTab === 'chat' && <ChatAssistant />}
      {activeTab === 'dashboard' && <ModelDashboard />}
    </Layout>
  );
}

export default App;
