import React, { useState } from 'react';
import { motion } from 'framer-motion';
import type { AppointmentFeatures } from '../types';
import { Calendar, User, Activity, Zap, Heart, Shield, Sparkles } from 'lucide-react';

interface PredictionFormProps {
    onSubmit: (data: AppointmentFeatures) => void;
    isLoading: boolean;
}

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

    const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
        const { name, value, type } = e.target;
        setFormData(prev => ({
            ...prev,
            [name]: type === 'number' ? Number(value) : value,
        }));
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
            transition: {
                duration: 0.6,
                staggerChildren: 0.1
            }
        }
    };

    const itemVariants = {
        hidden: { opacity: 0, x: -20 },
        visible: { opacity: 1, x: 0 }
    };

    return (
        <motion.form
            onSubmit={handleSubmit}
            className="w-full max-w-4xl mx-auto"
            variants={containerVariants}
            initial="hidden"
            animate="visible"
        >
            <div className="glass-card p-8 rounded-3xl relative overflow-hidden">
                {/* Decorative background elements */}
                <div className="absolute top-0 right-0 w-64 h-64 bg-primary/10 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2"></div>
                <div className="absolute bottom-0 left-0 w-64 h-64 bg-accent/10 rounded-full blur-3xl translate-y-1/2 -translate-x-1/2"></div>

                <div className="relative z-10">
                    <motion.div className="mb-8 flex items-center gap-4" variants={itemVariants}>
                        <div className="p-3 bg-primary/10 rounded-2xl">
                            <Sparkles className="w-8 h-8 text-primary" />
                        </div>
                        <div>
                            <h2 className="text-2xl font-bold text-text-primary">New Prediction</h2>
                            <p className="text-text-secondary">Enter patient details to assess no-show risk.</p>
                        </div>
                    </motion.div>

                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                        {/* Left Column */}
                        <div className="space-y-6">
                            <motion.div variants={itemVariants}>
                                <label className="block text-sm font-medium text-text-secondary mb-2">Patient Age</label>
                                <div className="relative group">
                                    <User className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-text-secondary group-focus-within:text-primary transition-colors" />
                                    <input
                                        type="number"
                                        name="age"
                                        value={formData.age}
                                        onChange={handleChange}
                                        className="w-full pl-12 pr-4 py-3 bg-surface border border-surface-hover rounded-xl text-text-primary focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary transition-all"
                                        required
                                    />
                                </div>
                            </motion.div>

                            <motion.div variants={itemVariants}>
                                <label className="block text-sm font-medium text-text-secondary mb-2">Gender</label>
                                <div className="relative group">
                                    <Activity className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-text-secondary group-focus-within:text-primary transition-colors" />
                                    <select
                                        name="gender"
                                        value={formData.gender}
                                        onChange={handleChange}
                                        className="w-full pl-12 pr-4 py-3 bg-surface border border-surface-hover rounded-xl text-text-primary focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary transition-all appearance-none"
                                    >
                                        <option value="F">Female</option>
                                        <option value="M">Male</option>
                                    </select>
                                </div>
                            </motion.div>

                            <motion.div variants={itemVariants}>
                                <label className="block text-sm font-medium text-text-secondary mb-2">Lead Days</label>
                                <div className="relative group">
                                    <Calendar className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-text-secondary group-focus-within:text-primary transition-colors" />
                                    <input
                                        type="number"
                                        name="lead_days"
                                        value={formData.lead_days}
                                        onChange={handleChange}
                                        className="w-full pl-12 pr-4 py-3 bg-surface border border-surface-hover rounded-xl text-text-primary focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary transition-all"
                                        required
                                    />
                                </div>
                            </motion.div>
                        </div>

                        {/* Right Column */}
                        <div className="space-y-6">
                            <motion.div variants={itemVariants} className="bg-surface/50 p-6 rounded-2xl border border-surface-hover">
                                <h3 className="text-lg font-medium text-text-primary mb-4 flex items-center gap-2">
                                    <Heart className="w-5 h-5 text-error" />
                                    Medical History
                                </h3>
                                <div className="space-y-3">
                                    {[
                                        { key: 'hypertension', label: 'Hypertension' },
                                        { key: 'diabetes', label: 'Diabetes' },
                                        { key: 'alcoholism', label: 'Alcoholism' },
                                        { key: 'handicap', label: 'Handicap' }
                                    ].map(({ key, label }) => (
                                        <label key={key} className="flex items-center justify-between p-3 hover:bg-surface rounded-lg cursor-pointer transition-colors group">
                                            <span className="text-text-secondary group-hover:text-text-primary transition-colors">{label}</span>
                                            <div className="relative">
                                                <input
                                                    type="checkbox"
                                                    checked={!!formData[key as keyof AppointmentFeatures]}
                                                    onChange={(e) => setFormData(prev => ({ ...prev, [key]: e.target.checked ? 1 : 0 }))}
                                                    className="sr-only peer"
                                                />
                                                <div className="w-11 h-6 bg-surface-hover peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary"></div>
                                            </div>
                                        </label>
                                    ))}
                                </div>
                            </motion.div>

                            <motion.div variants={itemVariants} className="bg-surface/50 p-6 rounded-2xl border border-surface-hover">
                                <h3 className="text-lg font-medium text-text-primary mb-4 flex items-center gap-2">
                                    <Shield className="w-5 h-5 text-accent" />
                                    Social & Contact
                                </h3>
                                <div className="space-y-3">
                                    <label className="flex items-center justify-between p-3 hover:bg-surface rounded-lg cursor-pointer transition-colors group">
                                        <span className="text-text-secondary group-hover:text-text-primary transition-colors">Welfare Scholarship</span>
                                        <div className="relative">
                                            <input
                                                type="checkbox"
                                                checked={!!formData.scholarship}
                                                onChange={(e) => setFormData(prev => ({ ...prev, scholarship: e.target.checked ? 1 : 0 }))}
                                                className="sr-only peer"
                                            />
                                            <div className="w-11 h-6 bg-surface-hover peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary"></div>
                                        </div>
                                    </label>
                                    <label className="flex items-center justify-between p-3 hover:bg-surface rounded-lg cursor-pointer transition-colors group">
                                        <span className="text-text-secondary group-hover:text-text-primary transition-colors">SMS Received</span>
                                        <div className="relative">
                                            <input
                                                type="checkbox"
                                                checked={!!formData.sms_received}
                                                onChange={(e) => setFormData(prev => ({ ...prev, sms_received: e.target.checked ? 1 : 0 }))}
                                                className="sr-only peer"
                                            />
                                            <div className="w-11 h-6 bg-surface-hover peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary"></div>
                                        </div>
                                    </label>
                                </div>
                            </motion.div>
                        </div>
                    </div>

                    <motion.div className="mt-8" variants={itemVariants}>
                        <button
                            type="submit"
                            disabled={isLoading}
                            className="w-full py-4 bg-gradient-to-r from-primary to-accent hover:from-primary-hover hover:to-accent-hover text-white font-bold rounded-xl shadow-lg shadow-primary/25 transition-all transform hover:scale-[1.02] active:scale-[0.98] disabled:opacity-70 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                        >
                            {isLoading ? (
                                <>
                                    <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                                    <span>Analyzing...</span>
                                </>
                            ) : (
                                <>
                                    <Zap className="w-5 h-5" />
                                    <span>Generate Prediction</span>
                                </>
                            )}
                        </button>
                    </motion.div>
                </div>
            </div>
        </motion.form>
    );
};

export default PredictionForm;
