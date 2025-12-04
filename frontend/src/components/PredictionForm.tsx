import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { AppointmentFeatures } from '../types';
import { getSmartFill } from '../services/api';
import {
    Calendar,
    User,
    Activity,
    Zap,
    Heart,
    Shield,
    Sparkles,
    Info,
    Check,
    ChevronDown,
    Loader
} from 'lucide-react';

interface PredictionFormProps {
    onSubmit: (data: AppointmentFeatures) => void;
    isLoading: boolean;
}

const Toggle = ({ label, checked, onChange, icon: Icon }: any) => (
    <div
        onClick={() => onChange(!checked)}
        className={`relative flex items-center justify-between p-4 rounded-xl border cursor-pointer transition-all duration-200 group ${checked
            ? 'bg-primary/10 border-primary/30 shadow-[0_0_15px_rgba(59,130,246,0.15)]'
            : 'bg-surface border-border hover:border-primary/30 hover:bg-surface-hover'
            }`}
    >
        <div className="flex items-center gap-3">
            <div className={`p-2 rounded-lg transition-colors ${checked ? 'bg-primary text-white' : 'bg-surface-hover text-text-secondary'}`}>
                <Icon className="w-4 h-4" />
            </div>
            <span className={`font-medium transition-colors ${checked ? 'text-text-primary' : 'text-text-secondary'}`}>
                {label}
            </span>
        </div>
        <div className={`w-12 h-6 rounded-full p-1 transition-colors ${checked ? 'bg-primary' : 'bg-surface-hover'}`}>
            <div className={`w-4 h-4 rounded-full bg-white shadow-sm transition-transform ${checked ? 'translate-x-6' : 'translate-x-0'}`} />
        </div>
    </div>
);

const PredictionForm: React.FC<PredictionFormProps> = ({ onSubmit, isLoading }) => {
    const [formData, setFormData] = useState<AppointmentFeatures>({
        age: 30,
        gender: 'F',
        scholarship: 0,
        hypertension: 0,
        diabetes: 0,
        alcoholism: 0,
        handicap: 0,
        sms_received: 1,
        lead_days: 7,
    });
    const [smartFillLoading, setSmartFillLoading] = useState(false);
    const [showSmartFillMenu, setShowSmartFillMenu] = useState(false);

    const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
        const { name, value, type } = e.target;
        setFormData(prev => ({
            ...prev,
            [name]: type === 'number' ? Number(value) : value,
        }));
    };

    const handleToggle = (key: keyof AppointmentFeatures) => {
        setFormData(prev => ({
            ...prev,
            [key]: prev[key] ? 0 : 1
        }));
    };

    const handleSmartFill = async (scenario: string) => {
        setSmartFillLoading(true);
        setShowSmartFillMenu(false);
        try {
            const response = await getSmartFill(scenario);
            if (response?.data) {
                setFormData(prev => ({
                    ...prev,
                    ...response.data
                }));
            }
        } catch (error) {
            console.error('Smart fill failed:', error);
            // Fallback to hardcoded high-risk profile
            if (scenario === 'high') {
                setFormData({
                    age: 65,
                    gender: 'M',
                    scholarship: 0,
                    hypertension: 1,
                    diabetes: 1,
                    alcoholism: 0,
                    handicap: 0,
                    sms_received: 0,
                    lead_days: 14,
                });
            }
        } finally {
            setSmartFillLoading(false);
        }
    };

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        onSubmit(formData);
    };

    const containerVariants = {
        hidden: { opacity: 0, y: 20 },
        visible: {
            opacity: 1,
            y: 0,
            transition: { duration: 0.6, staggerChildren: 0.1 }
        }
    };

    const itemVariants = {
        hidden: { opacity: 0, x: -20 },
        visible: { opacity: 1, x: 0 }
    };

    return (
        <motion.div
            className="w-full max-w-5xl mx-auto"
            variants={containerVariants}
            initial="hidden"
            animate="visible"
        >
            <div className="flex items-center justify-between mb-8">
                <div>
                    <h2 className="text-3xl font-bold text-text-primary tracking-tight">New Prediction</h2>
                    <p className="text-text-secondary mt-1">Enter patient details to assess no-show risk.</p>
                </div>
                <div className="relative">
                    <button
                        type="button"
                        onClick={() => setShowSmartFillMenu(!showSmartFillMenu)}
                        disabled={smartFillLoading}
                        className="flex items-center gap-2 px-4 py-2 rounded-xl bg-surface border border-border hover:bg-surface-hover text-sm font-medium text-primary transition-colors disabled:opacity-50"
                    >
                        {smartFillLoading ? (
                            <Loader className="w-4 h-4 animate-spin" />
                        ) : (
                            <Sparkles className="w-4 h-4" />
                        )}
                        Smart Fill
                        <ChevronDown className={`w-4 h-4 transition-transform ${showSmartFillMenu ? 'rotate-180' : ''}`} />
                    </button>
                    <AnimatePresence>
                        {showSmartFillMenu && (
                            <motion.div
                                initial={{ opacity: 0, y: -10 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0, y: -10 }}
                                className="absolute right-0 mt-2 w-48 bg-surface border border-border rounded-xl shadow-xl z-10 overflow-hidden"
                            >
                                {[
                                    { value: 'high', label: 'High Risk Patient', color: 'text-error' },
                                    { value: 'medium', label: 'Medium Risk Patient', color: 'text-warning' },
                                    { value: 'low', label: 'Low Risk Patient', color: 'text-success' },
                                    { value: 'random', label: 'Random Profile', color: 'text-primary' }
                                ].map((option) => (
                                    <button
                                        key={option.value}
                                        type="button"
                                        onClick={() => handleSmartFill(option.value)}
                                        className="w-full px-4 py-3 text-left text-sm hover:bg-surface-hover transition-colors flex items-center gap-2"
                                    >
                                        <span className={`w-2 h-2 rounded-full ${option.color.replace('text-', 'bg-')}`} />
                                        <span className="text-text-primary">{option.label}</span>
                                    </button>
                                ))}
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>
            </div>

            <form onSubmit={handleSubmit} className="grid grid-cols-1 lg:grid-cols-12 gap-8">
                {/* Left Column - Demographics & Basic Info */}
                <div className="lg:col-span-7 space-y-6">
                    <motion.div variants={itemVariants} className="glass-card p-6 rounded-2xl">
                        <h3 className="text-lg font-bold text-text-primary mb-6 flex items-center gap-2">
                            <User className="w-5 h-5 text-primary" />
                            Patient Demographics
                        </h3>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <div className="space-y-2">
                                <label className="text-sm font-medium text-text-secondary ml-1">Age</label>
                                <div className="relative group">
                                    <input
                                        type="number"
                                        name="age"
                                        value={formData.age}
                                        onChange={handleChange}
                                        className="w-full pl-4 pr-4 py-3 bg-surface/50 border border-border rounded-xl text-text-primary focus:outline-none focus:border-primary focus:ring-2 focus:ring-primary/20 transition-all font-mono text-lg"
                                        required
                                    />
                                    <div className="absolute right-4 top-1/2 -translate-y-1/2 text-text-tertiary text-sm">years</div>
                                </div>
                            </div>

                            <div className="space-y-2">
                                <label className="text-sm font-medium text-text-secondary ml-1">Gender</label>
                                <div className="grid grid-cols-2 gap-3">
                                    {['F', 'M'].map((g) => (
                                        <div
                                            key={g}
                                            onClick={() => setFormData(prev => ({ ...prev, gender: g }))}
                                            className={`cursor-pointer py-3 rounded-xl border text-center font-medium transition-all ${formData.gender === g
                                                ? 'bg-primary text-white border-primary shadow-lg shadow-primary/20'
                                                : 'bg-surface/50 border-border text-text-secondary hover:bg-surface-hover'
                                                }`}
                                        >
                                            {g === 'F' ? 'Female' : 'Male'}
                                        </div>
                                    ))}
                                </div>
                            </div>

                            <div className="space-y-2 md:col-span-2">
                                <label className="text-sm font-medium text-text-secondary ml-1">Lead Time (Days in Advance)</label>
                                <div className="relative group">
                                    <Calendar className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-text-tertiary group-focus-within:text-primary transition-colors" />
                                    <input
                                        type="number"
                                        name="lead_days"
                                        value={formData.lead_days}
                                        onChange={handleChange}
                                        className="w-full pl-12 pr-4 py-3 bg-surface/50 border border-border rounded-xl text-text-primary focus:outline-none focus:border-primary focus:ring-2 focus:ring-primary/20 transition-all font-mono text-lg"
                                        required
                                    />
                                </div>
                                <p className="text-xs text-text-tertiary ml-1">Number of days between scheduling and the appointment.</p>
                            </div>
                        </div>
                    </motion.div>

                    <motion.div variants={itemVariants} className="glass-card p-6 rounded-2xl">
                        <h3 className="text-lg font-bold text-text-primary mb-6 flex items-center gap-2">
                            <Shield className="w-5 h-5 text-accent" />
                            Social Factors
                        </h3>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <Toggle
                                label="Welfare Scholarship"
                                checked={!!formData.scholarship}
                                onChange={() => handleToggle('scholarship')}
                                icon={Sparkles}
                            />
                            <Toggle
                                label="SMS Reminder Sent"
                                checked={!!formData.sms_received}
                                onChange={() => handleToggle('sms_received')}
                                icon={Zap}
                            />
                        </div>
                    </motion.div>
                </div>

                {/* Right Column - Medical History */}
                <div className="lg:col-span-5 space-y-6">
                    <motion.div variants={itemVariants} className="glass-card p-6 rounded-2xl h-full flex flex-col">
                        <h3 className="text-lg font-bold text-text-primary mb-6 flex items-center gap-2">
                            <Heart className="w-5 h-5 text-error" />
                            Medical History
                        </h3>

                        <div className="space-y-4 flex-1">
                            <Toggle
                                label="Hypertension"
                                checked={!!formData.hypertension}
                                onChange={() => handleToggle('hypertension')}
                                icon={Activity}
                            />
                            <Toggle
                                label="Diabetes"
                                checked={!!formData.diabetes}
                                onChange={() => handleToggle('diabetes')}
                                icon={Activity}
                            />
                            <Toggle
                                label="Alcoholism"
                                checked={!!formData.alcoholism}
                                onChange={() => handleToggle('alcoholism')}
                                icon={Activity}
                            />
                            <div className="pt-4 border-t border-border">
                                <label className="text-sm font-medium text-text-secondary ml-1 mb-2 block">Disability Level (0-4)</label>
                                <div className="flex items-center gap-4">
                                    <input
                                        type="range"
                                        min="0"
                                        max="4"
                                        value={formData.handicap}
                                        onChange={(e) => setFormData(prev => ({ ...prev, handicap: Number(e.target.value) }))}
                                        className="w-full h-2 bg-surface-hover rounded-lg appearance-none cursor-pointer accent-primary"
                                    />
                                    <div className="w-10 h-10 rounded-xl bg-surface border border-border flex items-center justify-center font-bold text-text-primary">
                                        {formData.handicap}
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div className="mt-8 pt-6 border-t border-border">
                            <button
                                type="submit"
                                disabled={isLoading}
                                className="w-full py-4 bg-gradient-to-r from-primary to-accent hover:from-primary-hover hover:to-accent-hover text-white font-bold rounded-xl shadow-lg shadow-primary/25 transition-all transform hover:scale-[1.02] active:scale-[0.98] disabled:opacity-70 disabled:cursor-not-allowed flex items-center justify-center gap-3 group"
                            >
                                {isLoading ? (
                                    <>
                                        <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                                        <span>Analyzing Risk Profile...</span>
                                    </>
                                ) : (
                                    <>
                                        <Zap className="w-5 h-5 fill-current" />
                                        <span>Generate Prediction</span>
                                    </>
                                )}
                            </button>
                            <p className="text-center text-xs text-text-tertiary mt-4 flex items-center justify-center gap-1">
                                <Info className="w-3 h-3" />
                                AI model v2.1.0 • 92% Accuracy
                            </p>
                        </div>
                    </motion.div>
                </div>
            </form>
        </motion.div>
    );
};

export default PredictionForm;
