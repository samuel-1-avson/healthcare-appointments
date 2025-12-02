import React from 'react';
import { motion } from 'framer-motion';
import type { PredictionResponse } from '../types';
import { CheckCircle, AlertTriangle, XCircle, Info, Activity, ArrowRight } from 'lucide-react';

interface PredictionResultProps {
    result: PredictionResponse;
    onReset: () => void;
}

const PredictionResult: React.FC<PredictionResultProps> = ({ result, onReset }) => {
    const { risk, intervention, probability } = result;

    const getRiskIcon = () => {
        switch (risk.tier) {
            case 'CRITICAL':
            case 'HIGH':
                return <XCircle className="w-20 h-20 text-red-500" />;
            case 'MEDIUM':
                return <AlertTriangle className="w-20 h-20 text-yellow-500" />;
            default:
                return <CheckCircle className="w-20 h-20 text-green-500" />;
        }
    };

    const getRiskColor = () => {
        switch (risk.tier) {
            case 'CRITICAL':
            case 'HIGH':
                return 'red';
            case 'MEDIUM':
                return 'yellow';
            default:
                return 'green';
        }
    };

    const color = getRiskColor();

    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5 }}
            className="relative overflow-hidden rounded-3xl backdrop-blur-2xl bg-white/50 dark:bg-white/5 border border-white/20 dark:border-white/10 shadow-2xl"
        >
            {/* Background Glow */}
            <div className={`absolute top-0 left-1/2 -translate-x-1/2 w-full h-64 bg-${color}-500/20 blur-3xl rounded-full pointer-events-none`} />

            {/* Header */}
            <div className="relative z-10 p-8 text-center border-b border-gray-100 dark:border-white/5">
                <motion.div
                    initial={{ scale: 0, rotate: -180 }}
                    animate={{ scale: 1, rotate: 0 }}
                    transition={{ type: "spring", stiffness: 200, damping: 20, delay: 0.2 }}
                    className="flex justify-center mb-6"
                >
                    <div className={`p-4 rounded-full bg-${color}-500/10 backdrop-blur-xl border border-${color}-500/20 shadow-lg shadow-${color}-500/20`}>
                        {getRiskIcon()}
                    </div>
                </motion.div>

                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3 }}
                >
                    <h2 className="text-4xl font-black text-gray-900 dark:text-white mb-2 tracking-tight">
                        {risk.tier} RISK
                    </h2>
                    <div className="flex items-center justify-center gap-2 text-xl text-gray-600 dark:text-gray-300">
                        <span>No-Show Probability:</span>
                        <span className={`font-bold text-${color}-600 dark:text-${color}-400`}>
                            {(probability * 100).toFixed(1)}%
                        </span>
                    </div>
                </motion.div>
            </div>

            {/* Content */}
            <div className="relative z-10 p-8 space-y-8">
                {/* Risk Assessment */}
                <motion.div
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.4 }}
                    className="bg-blue-50 dark:bg-blue-500/10 p-6 rounded-2xl border border-blue-100 dark:border-blue-500/20"
                >
                    <h3 className="text-sm font-bold text-blue-800 dark:text-blue-300 mb-2 flex items-center gap-2 uppercase tracking-wider">
                        <Activity className="w-4 h-4" />
                        AI Confidence
                    </h3>
                    <p className="text-blue-900 dark:text-blue-100 text-lg">
                        The model is <span className="font-bold">{risk.confidence}</span> in this prediction based on patient history and demographics.
                    </p>
                </motion.div>

                {/* Intervention */}
                <motion.div
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.5 }}
                    className="space-y-4"
                >
                    <h3 className="text-xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
                        <Info className="w-6 h-6 text-indigo-500" />
                        Recommended Action Plan
                    </h3>

                    <div className="bg-white/50 dark:bg-white/5 backdrop-blur-sm rounded-2xl border border-gray-200 dark:border-white/10 overflow-hidden">
                        <div className="p-5 border-b border-gray-200 dark:border-white/10 flex justify-between items-center group hover:bg-white/50 dark:hover:bg-white/5 transition-colors">
                            <span className="text-gray-600 dark:text-gray-400 font-medium">Primary Action</span>
                            <span className="font-bold text-gray-900 dark:text-white text-right">{intervention.action}</span>
                        </div>

                        <div className="p-5 border-b border-gray-200 dark:border-white/10 flex justify-between items-center group hover:bg-white/50 dark:hover:bg-white/5 transition-colors">
                            <span className="text-gray-600 dark:text-gray-400 font-medium">SMS Strategy</span>
                            <span className="font-bold text-gray-900 dark:text-white">{intervention.sms_reminders}</span>
                        </div>

                        <div className="p-5 border-b border-gray-200 dark:border-white/10 flex justify-between items-center group hover:bg-white/50 dark:hover:bg-white/5 transition-colors">
                            <span className="text-gray-600 dark:text-gray-400 font-medium">Phone Call</span>
                            <span className={`font-bold px-3 py-1 rounded-lg text-sm ${intervention.phone_call
                                ? 'bg-red-100 text-red-700 dark:bg-red-500/20 dark:text-red-300'
                                : 'bg-green-100 text-green-700 dark:bg-green-500/20 dark:text-green-300'
                                }`}>
                                {intervention.phone_call ? 'Required' : 'Not Required'}
                            </span>
                        </div>

                        {intervention.notes && (
                            <div className="p-5 bg-gray-50/50 dark:bg-black/20">
                                <span className="text-xs font-bold text-gray-500 uppercase tracking-wider block mb-2">Clinical Notes</span>
                                <p className="text-gray-700 dark:text-gray-300 italic leading-relaxed">
                                    "{intervention.notes}"
                                </p>
                            </div>
                        )}
                    </div>
                </motion.div>

                <motion.button
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.6 }}
                    onClick={onReset}
                    className="w-full py-4 px-6 rounded-2xl border-2 border-gray-200 dark:border-white/10 text-gray-700 dark:text-gray-300 font-bold hover:bg-gray-50 dark:hover:bg-white/10 hover:border-gray-300 dark:hover:border-white/20 transition-all flex items-center justify-center gap-2 group"
                >
                    Run Another Prediction
                    <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                </motion.button>
            </div>
        </motion.div>
    );
};

export default PredictionResult;
