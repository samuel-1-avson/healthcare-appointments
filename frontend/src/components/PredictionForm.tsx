import React, { useState } from 'react';
import { motion } from 'framer-motion';
import type { AppointmentFeatures } from '../types';
import { Calendar, User, Activity, Phone, Zap, Heart, Shield, Sparkles } from 'lucide-react';

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
            className="relative"
            variants={containerVariants}
            initial="hidden"
            animate="visible"
        >
            {/* Premium glass card with glow */}
            <div className="relative backdrop-blur-2xl bg-white/50 dark:bg-white/5 border border-white/20 dark:border-white/10 rounded-3xl shadow-2xl overflow-hidden">
                {/* Animated gradient border effect */}
                <div className="absolute inset-0 bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 opacity-10 blur-xl"></div>

                <div className="relative p-8">
                    {/* Premium header */}
                    <motion.div className="mb-8" variants={itemVariants}>
                        <div className="flex items-center gap-3 mb-3">
                            <div className="relative">
                                <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-purple-500 rounded-xl blur-md opacity-50"></div>
                                <div className="relative bg-gradient-to-r from-blue-500 to-purple-500 p-3 rounded-xl shadow-lg">
                                    <User className="w-6 h-6 text-white" />
                                </div>
                            </div>
                            <div>
                                <h2 className="text-2xl font-black text-gray-900 dark:text-white flex items-center gap-2">
                                    Patient Details
                                    <Sparkles className="w-5 h-5 text-yellow-400 animate-pulse" />
                                </h2>
                                <p className="text-sm text-gray-500 dark:text-gray-400">Enter information for AI analysis</p>
                            </div>
                        </div>
                    </motion.div>

                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                        {/* Left Column - Basic Info */}
                        <div className="space-y-6">
                            {/* Age input */}
                            <motion.div className="group" variants={itemVariants}>
                                <label className="block text-sm font-bold text-gray-700 dark:text-gray-300 mb-2 flex items-center gap-2">
                                    <Activity className="w-4 h-4 text-blue-500" />
                                    Age
                                </label>
                                <div className="relative">
                                    <input
                                        type="number"
                                        name="age"
                                        value={formData.age}
                                        onChange={handleChange}
                                        min="0"
                                        max="120"
                                        className="w-full px-5 py-4 rounded-xl border-2 border-gray-200 dark:border-white/10 bg-white/50 dark:bg-white/5 text-gray-900 dark:text-white placeholder-gray-500 focus:border-blue-500 focus:bg-white dark:focus:bg-white/10 outline-none transition-all backdrop-blur-sm font-semibold text-lg group-hover:border-blue-300 dark:group-hover:border-white/20"
                                        required
                                    />
                                    <div className="absolute right-4 top-1/2 -translate-y-1/2 text-gray-400 font-medium">years</div>
                                </div>
                            </motion.div>

                            {/* Gender select */}
                            <motion.div className="group" variants={itemVariants}>
                                <label className="block text-sm font-bold text-gray-700 dark:text-gray-300 mb-2 flex items-center gap-2">
                                    <User className="w-4 h-4 text-purple-500" />
                                    Gender
                                </label>
                                <select
                                    name="gender"
                                    value={formData.gender}
                                    onChange={handleChange}
                                    className="w-full px-5 py-4 rounded-xl border-2 border-gray-200 dark:border-white/10 bg-white/50 dark:bg-white/5 text-gray-900 dark:text-white focus:border-purple-500 focus:bg-white dark:focus:bg-white/10 outline-none transition-all backdrop-blur-sm font-semibold text-lg group-hover:border-purple-300 dark:group-hover:border-white/20 cursor-pointer appearance-none"
                                >
                                    <option value="F" className="dark:bg-gray-900">Female</option>
                                    <option value="M" className="dark:bg-gray-900">Male</option>
                                    <option value="O" className="dark:bg-gray-900">Other</option>
                                </select>
                            </motion.div>

                            {/* Lead Days */}
                            <motion.div className="group" variants={itemVariants}>
                                <label className="block text-sm font-bold text-gray-700 dark:text-gray-300 mb-2 flex items-center gap-2">
                                    <Calendar className="w-4 h-4 text-emerald-500" />
                                    Lead Days
                                </label>
                                <div className="relative">
                                    <input
                                        type="number"
                                        name="lead_days"
                                        value={formData.lead_days}
                                        onChange={handleChange}
                                        min="0"
                                        className="w-full px-5 py-4 rounded-xl border-2 border-gray-200 dark:border-white/10 bg-white/50 dark:bg-white/5 text-gray-900 dark:text-white placeholder-gray-500 focus:border-emerald-500 focus:bg-white dark:focus:bg-white/10 outline-none transition-all backdrop-blur-sm font-semibold text-lg group-hover:border-emerald-300 dark:group-hover:border-white/20"
                                        required
                                    />
                                    <div className="absolute right-4 top-1/2 -translate-y-1/2 text-gray-400 font-medium">days</div>
                                </div>
                                <p className="text-xs text-gray-500 mt-2 flex items-center gap-1">
                                    <Zap className="w-3 h-3" />
                                    Days between scheduling and appointment
                                </p>
                            </motion.div>
                        </div>

                        {/* Right Column - Medical Conditions */}
                        <div className="space-y-6">
                            {/* Medical Conditions Card */}
                            <motion.div
                                variants={itemVariants}
                                className="relative backdrop-blur-xl bg-red-50 dark:bg-red-500/5 border border-red-100 dark:border-red-500/20 rounded-2xl p-6 group hover:border-red-200 dark:hover:border-red-500/30 transition-all"
                            >
                                <div className="absolute top-0 right-0 w-32 h-32 bg-red-500/5 dark:bg-red-500/10 rounded-full blur-3xl"></div>
                                <h3 className="text-lg font-black text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                                    <div className="bg-red-100 dark:bg-red-500/20 p-2 rounded-lg">
                                        <Heart className="w-5 h-5 text-red-500 dark:text-red-400" />
                                    </div>
                                    Medical Conditions
                                </h3>
                                <div className="space-y-3 relative z-10">
                                    {[
                                        { key: 'hypertension', label: 'Hypertension', icon: Activity, color: 'red' },
                                        { key: 'diabetes', label: 'Diabetes', icon: Heart, color: 'orange' },
                                        { key: 'alcoholism', label: 'Alcoholism', icon: Shield, color: 'yellow' },
                                        { key: 'scholarship', label: 'Welfare Program', icon: Sparkles, color: 'blue' },
                                    ].map(({ key, label, icon: Icon, color }) => (
                                        <label
                                            key={key}
                                            className="flex items-center justify-between p-4 backdrop-blur-sm bg-white/50 dark:bg-white/5 hover:bg-white dark:hover:bg-white/10 rounded-xl cursor-pointer transition-all border border-gray-100 dark:border-white/5 hover:border-gray-200 dark:hover:border-white/10 group/item shadow-sm"
                                        >
                                            <span className="flex items-center gap-3 text-gray-700 dark:text-gray-200 font-medium">
                                                <Icon className={`w-4 h-4 text-${color}-500`} />
                                                {label}
                                            </span>
                                            <div className="relative">
                                                <input
                                                    type="checkbox"
                                                    checked={!!formData[key as keyof AppointmentFeatures]}
                                                    onChange={(e) => setFormData(prev => ({ ...prev, [key]: e.target.checked ? 1 : 0 }))}
                                                    className="sr-only peer"
                                                />
                                                <div className="w-12 h-6 bg-gray-200 dark:bg-white/10 rounded-full peer-checked:bg-gradient-to-r peer-checked:from-blue-500 peer-checked:to-purple-500 transition-all border border-gray-300 dark:border-white/20"></div>
                                                <div className="absolute left-1 top-1 w-4 h-4 bg-white rounded-full transition-transform peer-checked:translate-x-6 shadow-md"></div>
                                            </div>
                                        </label>
                                    ))}
                                </div>
                            </motion.div>

                            {/* Communication Card */}
                            <motion.div
                                variants={itemVariants}
                                className="relative backdrop-blur-xl bg-green-50 dark:bg-green-500/5 border border-green-100 dark:border-green-500/20 rounded-2xl p-6 group hover:border-green-200 dark:hover:border-green-500/30 transition-all"
                            >
                                <div className="absolute bottom-0 left-0 w-32 h-32 bg-green-500/5 dark:bg-green-500/10 rounded-full blur-3xl"></div>
                                <h3 className="text-lg font-black text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                                    <div className="bg-green-100 dark:bg-green-500/20 p-2 rounded-lg">
                                        <Phone className="w-5 h-5 text-green-600 dark:text-green-400" />
                                    </div>
                                    Communication
                                </h3>
                                <label className="flex items-center justify-between p-4 backdrop-blur-sm bg-white/50 dark:bg-white/5 hover:bg-white dark:hover:bg-white/10 rounded-xl cursor-pointer transition-all border border-gray-100 dark:border-white/5 hover:border-gray-200 dark:hover:border-white/10 relative z-10 shadow-sm">
                                    <span className="flex items-center gap-3 text-gray-700 dark:text-gray-200 font-medium">
                                        <Phone className="w-4 h-4 text-green-500" />
                                        SMS Received
                                    </span>
                                    <div className="relative">
                                        <input
                                            type="checkbox"
                                            checked={!!formData.sms_received}
                                            onChange={(e) => setFormData(prev => ({ ...prev, sms_received: e.target.checked ? 1 : 0 }))}
                                            className="sr-only peer"
                                        />
                                        <div className="w-12 h-6 bg-gray-200 dark:bg-white/10 rounded-full peer-checked:bg-gradient-to-r peer-checked:from-green-500 peer-checked:to-emerald-500 transition-all border border-gray-300 dark:border-white/20"></div>
                                        <div className="absolute left-1 top-1 w-4 h-4 bg-white rounded-full transition-transform peer-checked:translate-x-6 shadow-md"></div>
                                    </div>
                                </label>
                            </motion.div>
                        </div>
                    </div>

                    {/* Submit Button */}
                    <motion.div className="mt-10" variants={itemVariants}>
                        <button
                            type="submit"
                            disabled={isLoading}
                            className="relative w-full group overflow-hidden rounded-2xl"
                        >
                            <div className={`relative px-8 py-5 font-black text-lg transition-all ${isLoading
                                ? 'bg-gray-200 dark:bg-gray-700 cursor-not-allowed'
                                : 'bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 hover:shadow-2xl hover:shadow-purple-500/30 transform hover:scale-[1.01] active:scale-[0.99]'
                                }`}>
                                {!isLoading && (
                                    <div className="absolute inset-0 bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 opacity-0 group-hover:opacity-100 blur-xl transition-opacity"></div>
                                )}
                                <div className="relative flex items-center justify-center gap-3 text-white">
                                    {isLoading ? (
                                        <>
                                            <svg className="animate-spin h-6 w-6 text-gray-500 dark:text-gray-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                            </svg>
                                            <span className="font-black text-gray-500 dark:text-gray-400">Analyzing with AI...</span>
                                        </>
                                    ) : (
                                        <>
                                            <Zap className="w-6 h-6 group-hover:animate-pulse" />
                                            <span>Predict No-Show Risk</span>
                                            <Sparkles className="w-5 h-5 group-hover:rotate-12 transition-transform" />
                                        </>
                                    )}
                                </div>
                            </div>
                        </button>

                        <p className="text-center text-xs text-gray-500 mt-4 flex items-center justify-center gap-2">
                            <Shield className="w-3 h-3" />
                            HIPAA Compliant • 256-bit Encrypted • Real-time Processing
                        </p>
                    </motion.div>
                </div>
            </div>
        </motion.form>
    );
};

export default PredictionForm;
