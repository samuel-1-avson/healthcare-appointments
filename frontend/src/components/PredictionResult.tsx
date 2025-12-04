import React from 'react';
import { motion } from 'framer-motion';
import type { PredictionResponse } from '../types';
import { CheckCircle, AlertTriangle, XCircle, Info, Activity, ArrowRight, MessageSquare } from 'lucide-react';
import { Link } from 'react-router-dom';

interface PredictionResultProps {
    result: PredictionResponse;
    onReset: () => void;
}

const PredictionResult: React.FC<PredictionResultProps> = ({ result, onReset }) => {
    const { risk, intervention, probability } = result;

    const getRiskConfig = () => {
        switch (risk.tier) {
            case 'CRITICAL':
            case 'HIGH':
                return { icon: XCircle, color: 'text-error', bg: 'bg-error', border: 'border-error' };
            case 'MEDIUM':
                return { icon: AlertTriangle, color: 'text-warning', bg: 'bg-warning', border: 'border-warning' };
            default:
                return { icon: CheckCircle, color: 'text-success', bg: 'bg-success', border: 'border-success' };
        }
    };

    const config = getRiskConfig();
    const Icon = config.icon;

    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5 }}
            className="glass-card rounded-3xl overflow-hidden max-w-4xl mx-auto"
        >
            {/* Header */}
            <div className="relative z-10 p-8 text-center border-b border-surface-hover bg-surface/30">
                <motion.div
                    initial={{ scale: 0, rotate: -180 }}
                    animate={{ scale: 1, rotate: 0 }}
                    transition={{ type: "spring", stiffness: 200, damping: 20, delay: 0.2 }}
                    className="flex justify-center mb-6"
                >
                    <div className={`p-4 rounded-full ${config.bg}/10 backdrop-blur-xl border ${config.border}/20 shadow-lg shadow-${config.bg}/20`}>
                        <Icon className={`w-20 h-20 ${config.color}`} />
                    </div>
                </motion.div>

                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3 }}
                >
                    <h2 className="text-4xl font-black text-text-primary mb-2 tracking-tight">
                        {risk.tier} RISK
                    </h2>
                    <div className="flex items-center justify-center gap-2 text-xl text-text-secondary">
                        <span>No-Show Probability:</span>
                        <span className={`font-bold ${config.color}`}>
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
                    className="bg-primary/5 p-6 rounded-2xl border border-primary/10"
                >
                    <h3 className="text-sm font-bold text-primary mb-2 flex items-center gap-2 uppercase tracking-wider">
                        <Activity className="w-4 h-4" />
                        AI Confidence
                    </h3>
                    <p className="text-text-primary text-lg">
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
                    <h3 className="text-xl font-bold text-text-primary flex items-center gap-2">
                        <Info className="w-6 h-6 text-accent" />
                        Recommended Action Plan
                    </h3>

                    <div className="bg-surface rounded-2xl border border-surface-hover overflow-hidden">
                        <div className="p-5 border-b border-surface-hover flex justify-between items-center hover:bg-surface-hover/50 transition-colors">
                            <span className="text-text-secondary font-medium">Primary Action</span>
                            <span className="font-bold text-text-primary text-right">{intervention.action}</span>
                        </div>

                        <div className="p-5 border-b border-surface-hover flex justify-between items-center hover:bg-surface-hover/50 transition-colors">
                            <span className="text-text-secondary font-medium">SMS Strategy</span>
                            <span className="font-bold text-text-primary">{intervention.sms_reminders}</span>
                        </div>

                        <div className="p-5 border-b border-surface-hover flex justify-between items-center hover:bg-surface-hover/50 transition-colors">
                            <span className="text-text-secondary font-medium">Phone Call</span>
                            <span className={`font-bold px-3 py-1 rounded-lg text-sm ${intervention.phone_call
                                ? 'bg-error/10 text-error'
                                : 'bg-success/10 text-success'
                                }`}>
                                {intervention.phone_call ? 'Required' : 'Not Required'}
                            </span>
                        </div>

                        {intervention.notes && (
                            <div className="p-5 bg-surface-hover/30">
                                <span className="text-xs font-bold text-text-secondary uppercase tracking-wider block mb-2">Clinical Notes</span>
                                <p className="text-text-primary italic leading-relaxed">
                                    "{intervention.notes}"
                                </p>
                            </div>
                        )}
                    </div>
                </motion.div>

                {/* Risk Factors Analysis (SHAP) */}
                {result.explanation && (
                    <motion.div
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.55 }}
                        className="space-y-4"
                    >
                        <h3 className="text-xl font-bold text-text-primary flex items-center gap-2">
                            <Activity className="w-6 h-6 text-primary" />
                            Risk Factors Analysis
                        </h3>

                        <div className="bg-surface rounded-2xl border border-surface-hover p-6 space-y-6">
                            <p className="text-text-secondary text-sm">
                                Factors contributing to this prediction. <span className="text-error font-bold">Red</span> increases risk, <span className="text-success font-bold">Green</span> decreases it.
                            </p>

                            {/* Risk Factors */}
                            {result.explanation.top_risk_factors.length > 0 && (
                                <div className="space-y-3">
                                    <h4 className="text-xs font-bold text-text-secondary uppercase tracking-wider">Top Risk Drivers</h4>
                                    {result.explanation.top_risk_factors.map((factor, idx) => (
                                        <div key={idx} className="relative">
                                            <div className="flex justify-between text-sm mb-1">
                                                <span className="font-medium text-text-primary">{factor.feature}</span>
                                                <span className="font-bold text-error">+{factor.contribution.toFixed(3)}</span>
                                            </div>
                                            <div className="h-2 bg-surface-hover rounded-full overflow-hidden">
                                                <motion.div
                                                    initial={{ width: 0 }}
                                                    animate={{ width: `${Math.min(Math.abs(factor.contribution) * 100, 100)}%` }}
                                                    transition={{ duration: 1, delay: 0.6 + (idx * 0.1) }}
                                                    className="h-full bg-error rounded-full"
                                                />
                                            </div>
                                            <div className="text-xs text-text-secondary mt-0.5">Value: {factor.value}</div>
                                        </div>
                                    ))}
                                </div>
                            )}

                            {/* Protective Factors */}
                            {result.explanation.top_protective_factors.length > 0 && (
                                <div className="space-y-3 pt-2 border-t border-surface-hover">
                                    <h4 className="text-xs font-bold text-text-secondary uppercase tracking-wider mt-2">Protective Factors</h4>
                                    {result.explanation.top_protective_factors.map((factor, idx) => (
                                        <div key={idx} className="relative">
                                            <div className="flex justify-between text-sm mb-1">
                                                <span className="font-medium text-text-primary">{factor.feature}</span>
                                                <span className="font-bold text-success">{factor.contribution.toFixed(3)}</span>
                                            </div>
                                            <div className="h-2 bg-surface-hover rounded-full overflow-hidden">
                                                <motion.div
                                                    initial={{ width: 0 }}
                                                    animate={{ width: `${Math.min(Math.abs(factor.contribution) * 100, 100)}%` }}
                                                    transition={{ duration: 1, delay: 0.8 + (idx * 0.1) }}
                                                    className="h-full bg-success rounded-full"
                                                />
                                            </div>
                                            <div className="text-xs text-text-secondary mt-0.5">Value: {factor.value}</div>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    </motion.div>
                )}

                <div className="flex flex-col sm:flex-row gap-4">
                    <motion.button
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.6 }}
                        onClick={onReset}
                        className="flex-1 py-4 px-6 rounded-2xl border-2 border-surface-hover text-text-secondary font-bold hover:bg-surface-hover hover:text-text-primary transition-all flex items-center justify-center gap-2 group"
                    >
                        Run Another Prediction
                        <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                    </motion.button>

                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.7 }}
                        className="flex-1"
                    >
                        <Link
                            to="/chat"
                            className="w-full h-full py-4 px-6 rounded-2xl bg-primary/10 text-primary font-bold hover:bg-primary/20 transition-all flex items-center justify-center gap-2 group"
                        >
                            <MessageSquare className="w-5 h-5" />
                            Explain with AI
                        </Link>
                    </motion.div>
                </div>
            </div>
        </motion.div>
    );
};

export default PredictionResult;
