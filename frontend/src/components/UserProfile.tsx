import React, { useEffect, useState } from 'react';
import {
    Mail,
    Shield,
    Activity,
    Clock,
    Calendar,
    MapPin,
    Edit,
    CheckCircle,
    Lock
} from 'lucide-react';
import { motion } from 'framer-motion';
import { getUserProfile } from '../services/api';

const UserProfile = () => {
    const [user, setUser] = useState<any>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchProfile = async () => {
            try {
                const data = await getUserProfile();
                setUser(data);
            } catch (err) {
                console.error("Failed to fetch profile", err);
                setError("Failed to load profile data. Please ensure the backend is running.");
            } finally {
                setLoading(false);
            }
        };
        fetchProfile();
    }, []);

    if (loading) {
        return (
            <div className="flex items-center justify-center h-96">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="flex flex-col items-center justify-center h-96 text-center">
                <div className="w-16 h-16 bg-error/10 rounded-full flex items-center justify-center mb-4">
                    <Shield className="w-8 h-8 text-error" />
                </div>
                <h2 className="text-xl font-bold text-text-primary mb-2">Something went wrong</h2>
                <p className="text-text-secondary mb-6">{error}</p>
                <button
                    onClick={() => window.location.reload()}
                    className="px-5 py-2 bg-primary text-white rounded-xl font-bold hover:bg-primary-hover transition-colors"
                >
                    Retry
                </button>
            </div>
        );
    }

    if (!user) return null;

    return (
        <div className="max-w-5xl mx-auto space-y-8 pb-12">
            {/* Header / Hero */}
            <div className="relative glass-card rounded-3xl overflow-hidden">
                <div className="h-32 bg-gradient-to-r from-primary/20 to-accent/20"></div>
                <div className="px-8 pb-8">
                    <div className="relative flex flex-col md:flex-row items-end -mt-12 mb-6 gap-6">
                        <div className="w-24 h-24 rounded-full bg-gradient-to-tr from-primary to-accent flex items-center justify-center text-white text-3xl font-bold shadow-xl ring-4 ring-surface">
                            {user.avatar}
                        </div>
                        <div className="flex-1 mb-2">
                            <h1 className="text-3xl font-bold text-text-primary">{user.name}</h1>
                            <p className="text-text-secondary font-medium">{user.role}</p>
                        </div>
                        <div className="mb-2">
                            <button className="px-5 py-2.5 bg-primary hover:bg-primary-hover text-white rounded-xl font-bold shadow-lg shadow-primary/25 transition-all flex items-center gap-2">
                                <Edit className="w-4 h-4" />
                                Edit Profile
                            </button>
                        </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-6 border-t border-surface-hover">
                        <div className="flex items-center gap-3 text-text-secondary">
                            <Mail className="w-5 h-5 text-primary" />
                            <span>{user.email}</span>
                        </div>
                        <div className="flex items-center gap-3 text-text-secondary">
                            <MapPin className="w-5 h-5 text-primary" />
                            <span>{user.location}</span>
                        </div>
                        <div className="flex items-center gap-3 text-text-secondary">
                            <Calendar className="w-5 h-5 text-primary" />
                            <span>Joined {user.join_date}</span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Stats Grid */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 }}
                    className="glass-card p-6 rounded-2xl border border-surface-hover"
                >
                    <div className="flex items-center gap-4">
                        <div className="p-3 bg-primary/10 rounded-xl">
                            <Activity className="w-6 h-6 text-primary" />
                        </div>
                        <div>
                            <p className="text-sm text-text-secondary font-medium">Total Predictions</p>
                            <h3 className="text-2xl font-bold text-text-primary">{user.stats.predictions.toLocaleString()}</h3>
                        </div>
                    </div>
                </motion.div>

                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2 }}
                    className="glass-card p-6 rounded-2xl border border-surface-hover"
                >
                    <div className="flex items-center gap-4">
                        <div className="p-3 bg-success/10 rounded-xl">
                            <CheckCircle className="w-6 h-6 text-success" />
                        </div>
                        <div>
                            <p className="text-sm text-text-secondary font-medium">Alerts Resolved</p>
                            <h3 className="text-2xl font-bold text-text-primary">{user.stats.alerts_resolved}</h3>
                        </div>
                    </div>
                </motion.div>

                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3 }}
                    className="glass-card p-6 rounded-2xl border border-surface-hover"
                >
                    <div className="flex items-center gap-4">
                        <div className="p-3 bg-accent/10 rounded-xl">
                            <Clock className="w-6 h-6 text-accent" />
                        </div>
                        <div>
                            <p className="text-sm text-text-secondary font-medium">System Uptime</p>
                            <h3 className="text-2xl font-bold text-text-primary">{user.stats.uptime}</h3>
                        </div>
                    </div>
                </motion.div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* Activity Log */}
                <div className="lg:col-span-2 glass-card p-8 rounded-3xl border border-surface-hover">
                    <h2 className="text-xl font-bold text-text-primary mb-6 flex items-center gap-2">
                        <Activity className="w-5 h-5 text-primary" />
                        Recent Activity
                    </h2>
                    <div className="space-y-6">
                        {user.activities.map((activity: any, idx: number) => (
                            <div key={idx} className="flex items-start gap-4 group">
                                <div className="relative">
                                    <div className="w-10 h-10 rounded-full bg-surface-hover flex items-center justify-center border border-border group-hover:border-primary/50 transition-colors">
                                        <Activity className="w-5 h-5 text-text-secondary group-hover:text-primary transition-colors" />
                                    </div>
                                    {idx !== user.activities.length - 1 && (
                                        <div className="absolute top-10 left-1/2 -translate-x-1/2 w-px h-full bg-border group-hover:bg-primary/20 transition-colors" />
                                    )}
                                </div>
                                <div className="pt-2">
                                    <p className="text-text-primary font-medium">{activity.action}</p>
                                    <p className="text-sm text-text-secondary">{activity.time}</p>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Security */}
                <div className="glass-card p-8 rounded-3xl border border-surface-hover">
                    <h2 className="text-xl font-bold text-text-primary mb-6 flex items-center gap-2">
                        <Shield className="w-5 h-5 text-primary" />
                        Security
                    </h2>
                    <div className="space-y-4">
                        <button className="w-full p-4 bg-surface hover:bg-surface-hover border border-surface-hover rounded-xl text-left transition-all group">
                            <div className="flex items-center justify-between mb-1">
                                <span className="font-bold text-text-primary group-hover:text-primary transition-colors">Change Password</span>
                                <Lock className="w-4 h-4 text-text-tertiary group-hover:text-primary transition-colors" />
                            </div>
                            <p className="text-xs text-text-secondary">Last changed 3 months ago</p>
                        </button>

                        <button className="w-full p-4 bg-surface hover:bg-surface-hover border border-surface-hover rounded-xl text-left transition-all group">
                            <div className="flex items-center justify-between mb-1">
                                <span className="font-bold text-text-primary group-hover:text-primary transition-colors">Two-Factor Auth</span>
                                <div className="px-2 py-0.5 rounded-full bg-success/10 text-success text-[10px] font-bold uppercase">Enabled</div>
                            </div>
                            <p className="text-xs text-text-secondary">Authenticator App</p>
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default UserProfile;
