import React from 'react';
import { motion } from 'framer-motion';
import type { PredictionResponse } from '../types';
import {
    CheckCircle,
    AlertTriangle,
    XCircle,
    Info,
    Activity,
    ArrowRight,
    MessageSquare,
    Copy,
    Check,
    Phone,
    Calendar,
    Sparkles
} from 'lucide-react';
import { Link } from 'react-router-dom';
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    Cell,
    ReferenceLine
} from 'recharts';

interface PredictionResultProps {
    result: PredictionResponse;
    onReset: () => void;
}

const RadialGauge = ({ value, color }: { value: number, color: string }) => {
    const radius = 80;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - (value * circumference);

    return (
        <div className="relative flex items-center justify-center w-64 h-64">
            <svg className="transform -rotate-90 w-full h-full">
                <circle
                    className="text-surface-hover"
                    strokeWidth="12"
                    stroke="currentColor"
                    fill="transparent"
                    r={radius}
                    cx="128"
                    cy="128"
                />
                <motion.circle
                    initial={{ strokeDashoffset: circumference }}
                    animate={{ strokeDashoffset: offset }}
                    transition={{ duration: 1.5, ease: "easeOut" }}
                    className={color}
                    strokeWidth="12"
                    strokeDasharray={circumference}
                    strokeLinecap="round"
                    stroke="currentColor"
                    fill="transparent"
                    r={radius}
                    cx="128"
                    cy="128"
                />
            </svg>
            <div className="absolute flex flex-col items-center justify-center text-center">
                <span className="text-5xl font-black text-text-primary tracking-tighter">
                    {(value * 100).toFixed(0)}%
                </span>
                <span className="text-sm font-medium text-text-secondary uppercase tracking-widest mt-1">Probability</span>
            </div>
        </div>
    );
};

const PredictionResult: React.FC<PredictionResultProps> = ({ result, onReset }) => {
    const { risk, intervention, probability } = result;
    const [copied, setCopied] = React.useState(false);

    const handleCopy = () => {
        if (intervention.notes) {
            navigator.clipboard.writeText(intervention.notes);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        }
    };

    const getRiskConfig = () => {
        switch (risk.tier) {
            case 'CRITICAL':
            case 'HIGH':
                return { icon: XCircle, color: 'text-error', bg: 'bg-error', border: 'border-error', hex: '#EF4444' };
            case 'MEDIUM':
                return { icon: AlertTriangle, color: 'text-warning', bg: 'bg-warning', border: 'border-warning', hex: '#F59E0B' };
            default:
                return { icon: CheckCircle, color: 'text-success', bg: 'bg-success', border: 'border-success', hex: '#10B981' };
        }
    };

    const config = getRiskConfig();
    const [chartsReady, setChartsReady] = React.useState(false);

    React.useEffect(() => {
        const timer = setTimeout(() => setChartsReady(true), 500); // Slight delay for animation
        return () => clearTimeout(timer);
    }, []);

    // Prepare SHAP data for Tornado Chart
    const shapData = result.explanation ? [
        ...result.explanation.top_risk_factors.map(f => ({ ...f, type: 'risk' })),
        ...result.explanation.top_protective_factors.map(f => ({ ...f, type: 'protective' }))
    ].sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution)) : [];

    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5 }}
            className="max-w-6xl mx-auto space-y-8"
        >
            {/* Hero Section */}
            <div className="glass-card rounded-3xl overflow-hidden relative">
                <div className={`absolute top-0 left-0 w-full h-2 ${config.bg}`} />
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 p-8 lg:p-12">

                    {/* Left: Gauge & Score */}
                    <div className="lg:col-span-5 flex flex-col items-center justify-center border-b lg:border-b-0 lg:border-r border-border pb-8 lg:pb-0 lg:pr-8">
                        <motion.div
                            initial={{ scale: 0.8, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            transition={{ delay: 0.2 }}
                        >
                            <RadialGauge value={probability} color={config.color} />
                        </motion.div>
                        <motion.div
                            initial={{ y: 20, opacity: 0 }}
                            animate={{ y: 0, opacity: 1 }}
                            transition={{ delay: 0.4 }}
                            className="text-center mt-6"
                        >
                            <h2 className={`text-3xl font-black tracking-tight mb-2 ${config.color}`}>
                                {risk.tier} RISK
                            </h2>
                            <p className="text-text-secondary">
                                Confidence Score: <span className="text-text-primary font-bold">{risk.confidence}</span>
                            </p>
                        </motion.div>
                    </div>

                    {/* Right: Action Plan */}
                    <div className="lg:col-span-7 flex flex-col justify-center space-y-6">
                        <div className="flex items-center gap-3 mb-2">
                            <div className="p-2 bg-accent/10 rounded-lg">
                                <Activity className="w-5 h-5 text-accent" />
                            </div>
                            <h3 className="text-xl font-bold text-text-primary">Recommended Intervention</h3>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div className="p-4 bg-surface rounded-xl border border-border">
                                <span className="text-xs font-bold text-text-tertiary uppercase tracking-wider block mb-1">Primary Action</span>
                                <span className="text-lg font-bold text-text-primary">{intervention.action}</span>
                            </div>
                            <div className="p-4 bg-surface rounded-xl border border-border">
                                <span className="text-xs font-bold text-text-tertiary uppercase tracking-wider block mb-1">SMS Strategy</span>
                                <span className="text-lg font-bold text-text-primary">{intervention.sms_reminders}</span>
                            </div>
                        </div>

                        {intervention.notes && (
                            <div className="relative p-5 bg-surface-hover/30 rounded-xl border border-border/50 group">
                                <button
                                    onClick={handleCopy}
                                    className="absolute top-3 right-3 p-2 text-text-tertiary hover:text-primary transition-colors rounded-lg hover:bg-surface-hover"
                                    title="Copy notes"
                                >
                                    {copied ? <Check className="w-4 h-4 text-success" /> : <Copy className="w-4 h-4" />}
                                </button>
                                <span className="text-xs font-bold text-text-secondary uppercase tracking-wider block mb-2 flex items-center gap-2">
                                    <MessageSquare className="w-3 h-3" /> Clinical Notes
                                </span>
                                <p className="text-text-primary italic leading-relaxed pr-8">
                                    "{intervention.notes}"
                                </p>
                            </div>
                        )}

                        <div className="flex gap-3 pt-2">
                            {intervention.phone_call && (
                                <button className="flex-1 py-3 bg-primary/10 hover:bg-primary/20 text-primary font-bold rounded-xl transition-colors flex items-center justify-center gap-2">
                                    <Phone className="w-4 h-4" /> Call Patient
                                </button>
                            )}
                            <button className="flex-1 py-3 bg-surface hover:bg-surface-hover text-text-primary font-bold rounded-xl border border-border transition-colors flex items-center justify-center gap-2">
                                <Calendar className="w-4 h-4" /> Reschedule
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            {/* Explainability Section */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                <div className="lg:col-span-2 glass-card p-8 rounded-3xl">
                    <h3 className="text-xl font-bold text-text-primary mb-6 flex items-center gap-2">
                        <Info className="w-5 h-5 text-primary" />
                        Risk Factor Analysis
                    </h3>
                    <div className="h-[400px] w-full">
                        {chartsReady && (
                            <ResponsiveContainer width="100%" height={400} minHeight={400}>
                                <BarChart
                                    layout="vertical"
                                    data={shapData}
                                    margin={{ top: 5, right: 30, left: 40, bottom: 5 }}
                                >
                                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={false} />
                                    <XAxis type="number" stroke="#64748B" tick={{ fontSize: 12 }} />
                                    <YAxis
                                        dataKey="feature"
                                        type="category"
                                        stroke="#94A3B8"
                                        width={120}
                                        tick={{ fontSize: 11, fontWeight: 500 }}
                                    />
                                    <Tooltip
                                        cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                                        contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', borderRadius: '8px' }}
                                    />
                                    <ReferenceLine x={0} stroke="#64748B" />
                                    <Bar dataKey="contribution" name="Impact" radius={[0, 4, 4, 0]}>
                                        {shapData.map((entry, index) => (
                                            <Cell key={`cell-${index}`} fill={entry.contribution > 0 ? '#EF4444' : '#10B981'} />
                                        ))}
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                        )}
                    </div>
                </div>

                <div className="glass-card p-8 rounded-3xl flex flex-col justify-center items-center text-center space-y-6">
                    <div className="p-4 bg-surface rounded-full border border-border">
                        <Sparkles className="w-8 h-8 text-accent" />
                    </div>
                    <div>
                        <h3 className="text-xl font-bold text-text-primary mb-2">Next Steps</h3>
                        <p className="text-text-secondary text-sm">
                            Use these insights to optimize your schedule and reduce revenue loss.
                        </p>
                    </div>

                    <div className="w-full space-y-3">
                        <button
                            onClick={onReset}
                            className="w-full py-4 bg-primary hover:bg-primary-hover text-white font-bold rounded-xl shadow-lg shadow-primary/25 transition-all flex items-center justify-center gap-2 group"
                        >
                            New Prediction
                            <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                        </button>

                        <Link
                            to="/chat"
                            state={{
                                predictionContext: {
                                    risk: result.risk,
                                    probability: result.probability,
                                    intervention: result.intervention,
                                    explanation: result.explanation
                                }
                            }}
                            className="w-full py-4 bg-surface hover:bg-surface-hover text-text-primary font-bold rounded-xl border border-border transition-all flex items-center justify-center gap-2"
                        >
                            <MessageSquare className="w-5 h-5" />
                            Ask Assistant
                        </Link>
                    </div>
                </div>
            </div>
        </motion.div>
    );
};

export default PredictionResult;
