import React, { useState } from 'react';
import { useSettings } from '../context/SettingsContext';
import {
    Save,
    RefreshCw,
    DollarSign,
    Bell,
    Moon,
    Sun,
    AlertTriangle,
    Check,
    Shield,
    Sliders
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const Toggle = ({ checked, onChange }: { checked: boolean; onChange: () => void }) => (
    <button
        onClick={onChange}
        className={`w-14 h-7 rounded-full p-1 transition-colors relative ${checked ? 'bg-primary' : 'bg-surface-hover border border-border'}`}
    >
        <motion.div
            animate={{ x: checked ? 28 : 0 }}
            transition={{ type: "spring", stiffness: 500, damping: 30 }}
            className="w-5 h-5 bg-white rounded-full shadow-sm"
        />
    </button>
);

const Settings = () => {
    const { settings, updateSettings, resetSettings } = useSettings();
    const [saved, setSaved] = useState(false);

    const handleCostChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const val = parseInt(e.target.value);
        if (!isNaN(val) && val >= 0) {
            updateSettings({ costPerNoShow: val });
        }
    };

    const handleThresholdChange = (type: 'high' | 'medium', val: string) => {
        const numVal = parseFloat(val);
        if (!isNaN(numVal) && numVal >= 0 && numVal <= 1) {
            updateSettings({
                riskThresholds: {
                    ...settings.riskThresholds,
                    [type]: numVal
                }
            });
        }
    };

    const handleSave = () => {
        // In a real app, this might persist to backend
        setSaved(true);
        setTimeout(() => setSaved(false), 2000);
    };

    return (
        <div className="max-w-5xl mx-auto space-y-8 pb-12">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
                <div>
                    <h1 className="text-3xl font-bold text-text-primary tracking-tight">System Configuration</h1>
                    <p className="text-text-secondary mt-1 text-lg">Manage global parameters and user preferences.</p>
                </div>
                <div className="flex gap-3">
                    <button
                        onClick={resetSettings}
                        className="flex items-center gap-2 px-5 py-2.5 text-text-secondary hover:text-text-primary font-medium transition-colors bg-surface hover:bg-surface-hover border border-border rounded-xl"
                    >
                        <RefreshCw className="w-4 h-4" />
                        Reset Defaults
                    </button>
                    <button
                        onClick={handleSave}
                        className="flex items-center gap-2 px-6 py-2.5 bg-primary hover:bg-primary-hover text-white rounded-xl font-bold shadow-lg shadow-primary/25 transition-all btn-hover"
                    >
                        {saved ? <Check className="w-4 h-4" /> : <Save className="w-4 h-4" />}
                        {saved ? 'Saved!' : 'Save Changes'}
                    </button>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Financial Settings */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="glass-card p-8 rounded-3xl border border-surface-hover"
                >
                    <div className="flex items-center gap-4 mb-8">
                        <div className="p-3 bg-success/10 rounded-2xl border border-success/10">
                            <DollarSign className="w-6 h-6 text-success" />
                        </div>
                        <div>
                            <h2 className="text-xl font-bold text-text-primary">Financial Impact</h2>
                            <p className="text-sm text-text-secondary">ROI calculation parameters</p>
                        </div>
                    </div>

                    <div className="space-y-6">
                        <div>
                            <label className="block text-sm font-bold text-text-secondary mb-3">
                                Average Revenue per Appointment
                            </label>
                            <div className="relative group">
                                <span className="absolute left-4 top-1/2 -translate-y-1/2 text-text-tertiary font-bold">$</span>
                                <input
                                    type="number"
                                    value={settings.costPerNoShow}
                                    onChange={handleCostChange}
                                    className="w-full bg-surface border border-surface-hover text-text-primary rounded-xl pl-8 pr-4 py-4 text-lg font-mono focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary transition-all"
                                />
                            </div>
                            <p className="text-xs text-text-tertiary mt-3 flex items-center gap-1">
                                <CheckCircle className="w-3 h-3 text-success" />
                                Used to estimate monthly savings from prevented no-shows.
                            </p>
                        </div>
                    </div>
                </motion.div>

                {/* Preferences */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 }}
                    className="glass-card p-8 rounded-3xl border border-surface-hover"
                >
                    <div className="flex items-center gap-4 mb-8">
                        <div className="p-3 bg-primary/10 rounded-2xl border border-primary/10">
                            <Sliders className="w-6 h-6 text-primary" />
                        </div>
                        <div>
                            <h2 className="text-xl font-bold text-text-primary">App Preferences</h2>
                            <p className="text-sm text-text-secondary">Customize your experience</p>
                        </div>
                    </div>

                    <div className="space-y-6">
                        <div className="flex items-center justify-between p-4 bg-surface/50 rounded-xl border border-surface-hover">
                            <div className="flex items-center gap-3">
                                <div className="p-2 bg-surface rounded-lg">
                                    <Bell className="w-5 h-5 text-text-secondary" />
                                </div>
                                <div>
                                    <p className="font-bold text-text-primary">System Alerts</p>
                                    <p className="text-xs text-text-secondary">Push notifications for high risk</p>
                                </div>
                            </div>
                            <Toggle
                                checked={settings.notificationsEnabled}
                                onChange={() => updateSettings({ notificationsEnabled: !settings.notificationsEnabled })}
                            />
                        </div>

                        <div className="flex items-center justify-between p-4 bg-surface/50 rounded-xl border border-surface-hover">
                            <div className="flex items-center gap-3">
                                <div className="p-2 bg-surface rounded-lg">
                                    {settings.theme === 'dark' ? <Moon className="w-5 h-5 text-text-secondary" /> : <Sun className="w-5 h-5 text-text-secondary" />}
                                </div>
                                <div>
                                    <p className="font-bold text-text-primary">Dark Mode</p>
                                    <p className="text-xs text-text-secondary">Toggle application theme</p>
                                </div>
                            </div>
                            <Toggle
                                checked={settings.theme === 'dark'}
                                onChange={() => updateSettings({ theme: settings.theme === 'dark' ? 'light' : 'dark' })}
                            />
                        </div>
                    </div>
                </motion.div>
            </div>

            {/* Risk Thresholds - Full Width */}
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="glass-card p-8 rounded-3xl border border-surface-hover"
            >
                <div className="flex items-center gap-4 mb-8">
                    <div className="p-3 bg-error/10 rounded-2xl border border-error/10">
                        <Shield className="w-6 h-6 text-error" />
                    </div>
                    <div>
                        <h2 className="text-xl font-bold text-text-primary">Risk Sensitivity</h2>
                        <p className="text-sm text-text-secondary">Adjust model classification thresholds</p>
                    </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-12">
                    {/* High Risk Slider */}
                    <div className="space-y-4">
                        <div className="flex justify-between items-end">
                            <label className="font-bold text-text-primary flex items-center gap-2">
                                <span className="w-2 h-2 rounded-full bg-error"></span>
                                Critical Risk Threshold
                            </label>
                            <span className="text-2xl font-black text-error">{(settings.riskThresholds.high * 100).toFixed(0)}%</span>
                        </div>
                        <div className="relative h-2 bg-surface-hover rounded-full">
                            <div
                                className="absolute top-0 left-0 h-full bg-error rounded-full opacity-50"
                                style={{ width: `${settings.riskThresholds.high * 100}%` }}
                            />
                            <input
                                type="range"
                                min="0"
                                max="1"
                                step="0.05"
                                value={settings.riskThresholds.high}
                                onChange={(e) => handleThresholdChange('high', e.target.value)}
                                className="absolute top-0 left-0 w-full h-full opacity-0 cursor-pointer"
                            />
                            <div
                                className="absolute top-1/2 -translate-y-1/2 w-6 h-6 bg-white border-4 border-error rounded-full shadow-lg pointer-events-none transition-all"
                                style={{ left: `calc(${settings.riskThresholds.high * 100}% - 12px)` }}
                            />
                        </div>
                        <p className="text-xs text-text-secondary">
                            Patients with a probability score above this level will be flagged as <strong>CRITICAL</strong>.
                        </p>
                    </div>

                    {/* Medium Risk Slider */}
                    <div className="space-y-4">
                        <div className="flex justify-between items-end">
                            <label className="font-bold text-text-primary flex items-center gap-2">
                                <span className="w-2 h-2 rounded-full bg-warning"></span>
                                High Risk Threshold
                            </label>
                            <span className="text-2xl font-black text-warning">{(settings.riskThresholds.medium * 100).toFixed(0)}%</span>
                        </div>
                        <div className="relative h-2 bg-surface-hover rounded-full">
                            <div
                                className="absolute top-0 left-0 h-full bg-warning rounded-full opacity-50"
                                style={{ width: `${settings.riskThresholds.medium * 100}%` }}
                            />
                            <input
                                type="range"
                                min="0"
                                max="1"
                                step="0.05"
                                value={settings.riskThresholds.medium}
                                onChange={(e) => handleThresholdChange('medium', e.target.value)}
                                className="absolute top-0 left-0 w-full h-full opacity-0 cursor-pointer"
                            />
                            <div
                                className="absolute top-1/2 -translate-y-1/2 w-6 h-6 bg-white border-4 border-warning rounded-full shadow-lg pointer-events-none transition-all"
                                style={{ left: `calc(${settings.riskThresholds.medium * 100}% - 12px)` }}
                            />
                        </div>
                        <p className="text-xs text-text-secondary">
                            Patients with a probability score above this level will be flagged as <strong>HIGH</strong>.
                        </p>
                    </div>
                </div>
            </motion.div>
        </div>
    );
};

// Helper component for icon
const CheckCircle = ({ className }: { className?: string }) => (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
        <polyline points="22 4 12 14.01 9 11.01"></polyline>
    </svg>
);

export default Settings;
