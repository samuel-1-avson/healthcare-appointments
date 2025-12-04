import React from 'react';
import { useSettings } from '../context/SettingsContext';
import { Save, RefreshCw, DollarSign, Bell, Moon, Sun, AlertTriangle } from 'lucide-react';

const Settings = () => {
    const { settings, updateSettings, resetSettings } = useSettings();

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

    return (
        <div className="space-y-6 max-w-4xl mx-auto">
            <div>
                <h1 className="text-3xl font-bold text-text-primary">Settings</h1>
                <p className="text-text-secondary mt-1">Configure system parameters and preferences.</p>
            </div>

            {/* Financial Settings */}
            <div className="glass-card p-6 rounded-2xl">
                <div className="flex items-center gap-3 mb-6">
                    <div className="p-2 bg-success/10 rounded-lg">
                        <DollarSign className="w-6 h-6 text-success" />
                    </div>
                    <h2 className="text-xl font-bold text-text-primary">Financial Configuration</h2>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                        <label className="block text-sm font-medium text-text-secondary mb-2">
                            Average Cost per No-Show ($)
                        </label>
                        <input
                            type="number"
                            value={settings.costPerNoShow}
                            onChange={handleCostChange}
                            className="w-full bg-surface border border-surface-hover text-text-primary rounded-xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-primary transition-all"
                        />
                        <p className="text-xs text-text-secondary mt-2">
                            Used to calculate estimated ROI and financial impact of prevented no-shows.
                        </p>
                    </div>
                </div>
            </div>

            {/* Risk Thresholds */}
            <div className="glass-card p-6 rounded-2xl">
                <div className="flex items-center gap-3 mb-6">
                    <div className="p-2 bg-error/10 rounded-lg">
                        <AlertTriangle className="w-6 h-6 text-error" />
                    </div>
                    <h2 className="text-xl font-bold text-text-primary">Risk Analysis Thresholds</h2>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                    <div>
                        <div className="flex justify-between mb-2">
                            <label className="text-sm font-medium text-text-secondary">High Risk Threshold</label>
                            <span className="text-sm font-bold text-error">{(settings.riskThresholds.high * 100).toFixed(0)}%</span>
                        </div>
                        <input
                            type="range"
                            min="0"
                            max="1"
                            step="0.05"
                            value={settings.riskThresholds.high}
                            onChange={(e) => handleThresholdChange('high', e.target.value)}
                            className="w-full h-2 bg-surface-hover rounded-lg appearance-none cursor-pointer accent-error"
                        />
                        <p className="text-xs text-text-secondary mt-2">
                            Appointments with probability above this value are flagged as <strong>CRITICAL</strong>.
                        </p>
                    </div>

                    <div>
                        <div className="flex justify-between mb-2">
                            <label className="text-sm font-medium text-text-secondary">Medium Risk Threshold</label>
                            <span className="text-sm font-bold text-warning">{(settings.riskThresholds.medium * 100).toFixed(0)}%</span>
                        </div>
                        <input
                            type="range"
                            min="0"
                            max="1"
                            step="0.05"
                            value={settings.riskThresholds.medium}
                            onChange={(e) => handleThresholdChange('medium', e.target.value)}
                            className="w-full h-2 bg-surface-hover rounded-lg appearance-none cursor-pointer accent-warning"
                        />
                        <p className="text-xs text-text-secondary mt-2">
                            Appointments with probability above this value are flagged as <strong>HIGH/MEDIUM</strong>.
                        </p>
                    </div>
                </div>
            </div>

            {/* Preferences */}
            <div className="glass-card p-6 rounded-2xl">
                <div className="flex items-center gap-3 mb-6">
                    <div className="p-2 bg-primary/10 rounded-lg">
                        <Moon className="w-6 h-6 text-primary" />
                    </div>
                    <h2 className="text-xl font-bold text-text-primary">Preferences</h2>
                </div>

                <div className="space-y-4">
                    <div className="flex items-center justify-between p-4 bg-surface rounded-xl border border-surface-hover">
                        <div className="flex items-center gap-3">
                            <Bell className="w-5 h-5 text-text-secondary" />
                            <div>
                                <p className="font-medium text-text-primary">Notifications</p>
                                <p className="text-sm text-text-secondary">Enable system alerts and sounds</p>
                            </div>
                        </div>
                        <button
                            onClick={() => updateSettings({ notificationsEnabled: !settings.notificationsEnabled })}
                            className={`w-12 h-6 rounded-full transition-colors relative ${settings.notificationsEnabled ? 'bg-primary' : 'bg-surface-hover'}`}
                        >
                            <div className={`absolute top-1 left-1 w-4 h-4 bg-white rounded-full transition-transform ${settings.notificationsEnabled ? 'translate-x-6' : ''}`} />
                        </button>
                    </div>

                    <div className="flex items-center justify-between p-4 bg-surface rounded-xl border border-surface-hover">
                        <div className="flex items-center gap-3">
                            {settings.theme === 'dark' ? <Moon className="w-5 h-5 text-text-secondary" /> : <Sun className="w-5 h-5 text-text-secondary" />}
                            <div>
                                <p className="font-medium text-text-primary">Theme</p>
                                <p className="text-sm text-text-secondary">Switch between dark and light mode</p>
                            </div>
                        </div>
                        <button
                            onClick={() => updateSettings({ theme: settings.theme === 'dark' ? 'light' : 'dark' })}
                            className="px-4 py-2 bg-surface-hover hover:bg-surface-hover/80 text-text-primary rounded-lg text-sm font-medium transition-colors"
                        >
                            {settings.theme === 'dark' ? 'Switch to Light' : 'Switch to Dark'}
                        </button>
                    </div>
                </div>
            </div>

            {/* Actions */}
            <div className="flex justify-end gap-4">
                <button
                    onClick={resetSettings}
                    className="flex items-center gap-2 px-6 py-3 text-text-secondary hover:text-text-primary font-medium transition-colors"
                >
                    <RefreshCw className="w-4 h-4" />
                    Reset to Defaults
                </button>
                <button
                    className="flex items-center gap-2 px-8 py-3 bg-primary hover:bg-primary-hover text-white rounded-xl font-bold shadow-lg shadow-primary/20 transition-all"
                >
                    <Save className="w-4 h-4" />
                    Save Changes
                </button>
            </div>
        </div>
    );
};

export default Settings;
