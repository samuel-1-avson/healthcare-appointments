import React, { createContext, useContext, useState, useEffect } from 'react';
import { getSettings, updateSettings as updateSettingsApi } from '../services/api';

interface Settings {
    costPerNoShow: number;
    riskThresholds: {
        high: number;
        medium: number;
    };
    theme: 'dark' | 'light';
    notificationsEnabled: boolean;
}

interface SettingsContextType {
    settings: Settings;
    updateSettings: (newSettings: Partial<Settings>) => Promise<void>;
    resetSettings: () => void;
    isLoading: boolean;
}

const defaultSettings: Settings = {
    costPerNoShow: 50,
    riskThresholds: {
        high: 0.8,
        medium: 0.5,
    },
    theme: 'dark',
    notificationsEnabled: true,
};

const SettingsContext = createContext<SettingsContextType | undefined>(undefined);

export const SettingsProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const [settings, setSettings] = useState<Settings>(defaultSettings);
    const [isLoading, setIsLoading] = useState(true);

    // Fetch settings from API
    useEffect(() => {
        const fetchSettings = async () => {
            try {
                const data = await getSettings();
                // Map backend snake_case to frontend camelCase
                setSettings({
                    costPerNoShow: data.cost_per_no_show,
                    riskThresholds: {
                        high: data.risk_threshold_high,
                        medium: data.risk_threshold_medium
                    },
                    theme: data.theme as 'dark' | 'light',
                    notificationsEnabled: data.notifications_enabled
                });
            } catch (error) {
                console.error("Failed to fetch settings, using defaults", error);
            } finally {
                setIsLoading(false);
            }
        };
        fetchSettings();
    }, []);

    // Apply theme side-effect
    useEffect(() => {
        if (settings.theme === 'dark') {
            document.documentElement.classList.add('dark');
        } else {
            document.documentElement.classList.remove('dark');
        }
    }, [settings.theme]);

    const updateSettings = async (newSettings: Partial<Settings>) => {
        // Optimistic update
        const updated = { ...settings, ...newSettings };

        // Handle nested updates for riskThresholds
        if (newSettings.riskThresholds) {
            updated.riskThresholds = { ...settings.riskThresholds, ...newSettings.riskThresholds };
        }

        setSettings(updated);

        // Sync with backend
        try {
            const backendPayload = {
                cost_per_no_show: updated.costPerNoShow,
                risk_threshold_high: updated.riskThresholds.high,
                risk_threshold_medium: updated.riskThresholds.medium,
                notifications_enabled: updated.notificationsEnabled,
                theme: updated.theme
            };
            await updateSettingsApi(backendPayload);
        } catch (error) {
            console.error("Failed to save settings", error);
            // Revert on error? For now, just log.
        }
    };

    const resetSettings = async () => {
        setSettings(defaultSettings);
        try {
            const backendPayload = {
                cost_per_no_show: defaultSettings.costPerNoShow,
                risk_threshold_high: defaultSettings.riskThresholds.high,
                risk_threshold_medium: defaultSettings.riskThresholds.medium,
                notifications_enabled: defaultSettings.notificationsEnabled,
                theme: defaultSettings.theme
            };
            await updateSettingsApi(backendPayload);
        } catch (error) {
            console.error("Failed to reset settings", error);
        }
    };

    return (
        <SettingsContext.Provider value={{ settings, updateSettings, resetSettings, isLoading }}>
            {children}
        </SettingsContext.Provider>
    );
};

export const useSettings = () => {
    const context = useContext(SettingsContext);
    if (context === undefined) {
        throw new Error('useSettings must be used within a SettingsProvider');
    }
    return context;
};
