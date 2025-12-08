import { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    AreaChart,
    Area,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    PieChart,
    Pie,
    Cell,
    BarChart,
    Bar
} from 'recharts';
import {
    Calendar,
    AlertTriangle,
    CheckCircle,
    TrendingUp,
    TrendingDown,
    X,
    Phone,
    Calendar as CalendarIcon,
    DollarSign,
    Download,
    ArrowRight
} from 'lucide-react';
import { getModelMetrics, getPredictionHistory, getModelInfo } from '../services/api';
import { useSettings } from '../context/SettingsContext';
import { useNavigate } from 'react-router-dom';

// --- Components ---

const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
        return (
            <div className="bg-surface/90 backdrop-blur-md border border-border p-4 rounded-xl shadow-xl">
                <p className="text-text-secondary text-xs font-semibold mb-2 uppercase tracking-wider">{label}</p>
                {payload.map((entry: any, index: number) => (
                    <div key={index} className="flex items-center gap-2 mb-1 last:mb-0">
                        <div className="w-2 h-2 rounded-full" style={{ backgroundColor: entry.color }} />
                        <span className="text-text-primary font-medium text-sm">
                            {entry.name}: <span className="font-bold">{entry.value.toLocaleString()}</span>
                        </span>
                    </div>
                ))}
            </div>
        );
    }
    return null;
};

const StatCard = ({ title, value, change, icon: Icon, trend, data, ready = true }: any) => (
    <div className="glass-card p-6 rounded-2xl relative overflow-hidden group hover:bg-surface-hover/50 transition-all duration-300">
        <div className="absolute -right-6 -top-6 p-8 bg-primary/5 rounded-full blur-2xl group-hover:bg-primary/10 transition-colors" />

        <div className="relative z-10 flex flex-col h-full justify-between">
            <div>
                <div className="flex items-center justify-between mb-4">
                    <div className="p-3 bg-surface border border-border rounded-xl shadow-sm group-hover:border-primary/30 transition-colors">
                        <Icon className="w-5 h-5 text-primary" />
                    </div>
                    {change && (
                        <div className={`flex items-center text-xs font-bold px-2 py-1 rounded-lg ${trend === 'up' ? 'bg-success/10 text-success' : 'bg-error/10 text-error'
                            }`}>
                            {trend === 'up' ? <TrendingUp className="w-3 h-3 mr-1" /> : <TrendingDown className="w-3 h-3 mr-1" />}
                            {change}
                        </div>
                    )}
                </div>
                <h3 className="text-text-secondary text-sm font-medium mb-1">{title}</h3>
                <p className="text-3xl font-bold text-text-primary tracking-tight">{value}</p>
            </div>

            {/* Mini Sparkline Area */}
            {data && (
                <div className="h-12 mt-4 -mx-2" style={{ minHeight: '48px', width: '100%' }}>
                    {ready && (
                        <ResponsiveContainer width="100%" height={48} minHeight={48}>
                            <AreaChart data={data}>
                                <defs>
                                    <linearGradient id={`gradient-${title}`} x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="0%" stopColor={trend === 'up' ? '#10B981' : '#EF4444'} stopOpacity={0.2} />
                                        <stop offset="100%" stopColor={trend === 'up' ? '#10B981' : '#EF4444'} stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <Area
                                    type="monotone"
                                    dataKey="value"
                                    stroke={trend === 'up' ? '#10B981' : '#EF4444'}
                                    strokeWidth={2}
                                    fill={`url(#gradient-${title})`}
                                />
                            </AreaChart>
                        </ResponsiveContainer>
                    )}
                </div>
            )}
        </div>
    </div>
);

const ReviewModal = ({ alertData, onClose }: { alertData: any; onClose: () => void }) => {
    if (!alertData) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
            <motion.div
                initial={{ opacity: 0, scale: 0.95, y: 20 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.95, y: 20 }}
                className="bg-surface border border-border rounded-2xl shadow-2xl w-full max-w-lg overflow-hidden"
            >
                <div className="p-6 border-b border-border flex justify-between items-center bg-surface-hover/30">
                    <h3 className="text-xl font-bold text-text-primary flex items-center gap-3">
                        <div className="p-2 bg-error/10 rounded-lg">
                            <AlertTriangle className="w-5 h-5 text-error" />
                        </div>
                        High Risk Alert
                    </h3>
                    <button onClick={onClose} className="text-text-secondary hover:text-text-primary transition-colors p-2 hover:bg-surface-hover rounded-lg">
                        <X className="w-5 h-5" />
                    </button>
                </div>

                <div className="p-6 space-y-6">
                    <div className="flex items-center gap-5 p-5 bg-surface-hover/30 rounded-2xl border border-border">
                        <div className="w-16 h-16 bg-gradient-to-br from-primary to-accent rounded-full flex items-center justify-center text-white font-bold text-2xl shadow-lg">
                            {alertData.patientid.toString().slice(-2)}
                        </div>
                        <div>
                            <p className="text-sm text-text-secondary font-medium uppercase tracking-wider">Patient ID</p>
                            <p className="text-2xl font-bold text-text-primary">#{alertData.patientid}</p>
                            <div className="flex items-center gap-2 mt-1">
                                <span className="w-2 h-2 rounded-full bg-success"></span>
                                <span className="text-xs text-text-secondary">Active Patient</span>
                            </div>
                        </div>
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                        <div className="p-4 bg-surface-hover/30 rounded-xl border border-border">
                            <p className="text-sm text-text-secondary mb-2">Risk Tier</p>
                            <span className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-bold tracking-wide ${alertData.risk_tier === 'CRITICAL' ? 'bg-error/10 text-error border border-error/20' :
                                alertData.risk_tier === 'HIGH' ? 'bg-warning/10 text-warning border border-warning/20' :
                                    'bg-success/10 text-success border border-success/20'
                                }`}>
                                {alertData.risk_tier}
                            </span>
                        </div>
                        <div className="p-4 bg-surface-hover/30 rounded-xl border border-border">
                            <p className="text-sm text-text-secondary mb-2">Risk Score</p>
                            <div className="flex items-baseline gap-1">
                                <p className="text-2xl font-bold text-text-primary">{alertData.composite_risk_score.toFixed(2)}</p>
                                <span className="text-xs text-text-secondary">/ 1.0</span>
                            </div>
                        </div>
                    </div>

                    <div className="space-y-3">
                        <h4 className="text-xs font-bold text-text-tertiary uppercase tracking-wider mb-2">Recommended Actions</h4>
                        <div className="grid grid-cols-2 gap-3">
                            <a href={`tel:${alertData.patientid}`} className="py-3 px-4 bg-primary/10 hover:bg-primary/20 text-primary border border-primary/20 rounded-xl font-medium transition-all flex items-center justify-center gap-2 btn-hover">
                                <Phone className="w-4 h-4" />
                                Call Patient
                            </a>
                            <a href="mailto:scheduling@clinic.com" className="py-3 px-4 bg-surface-hover hover:bg-surface-hover/80 text-text-primary border border-border rounded-xl font-medium transition-all flex items-center justify-center gap-2 btn-hover">
                                <CalendarIcon className="w-4 h-4" />
                                Reschedule
                            </a>
                        </div>
                    </div>
                </div>

                <div className="p-6 border-t border-border bg-surface-hover/10 flex justify-end gap-3">
                    <button
                        onClick={onClose}
                        className="px-5 py-2.5 text-text-secondary hover:text-text-primary font-medium transition-colors"
                    >
                        Dismiss
                    </button>
                    <button
                        onClick={() => {
                            alert("Marked as reviewed!");
                            onClose();
                        }}
                        className="px-6 py-2.5 bg-primary hover:bg-primary-hover text-white rounded-xl font-bold shadow-lg shadow-primary/25 transition-all btn-hover flex items-center gap-2"
                    >
                        <CheckCircle className="w-4 h-4" />
                        Mark as Reviewed
                    </button>
                </div>
            </motion.div>
        </div>
    );
};

const ModelDashboard = () => {
    const [metrics, setMetrics] = useState<any>(null);
    const [history, setHistory] = useState<any>(null);
    const [modelInfo, setModelInfo] = useState<any>(null);
    const [loading, setLoading] = useState(true);
    const [selectedAlert, setSelectedAlert] = useState<any>(null);
    const { settings } = useSettings();
    const [chartsReady, setChartsReady] = useState(false);
    const navigate = useNavigate();

    useEffect(() => {
        if (!loading) {
            const timer = setTimeout(() => setChartsReady(true), 300);
            return () => clearTimeout(timer);
        }
    }, [loading]);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const [metricsData, historyData, infoData] = await Promise.all([
                    getModelMetrics(),
                    getPredictionHistory(),
                    getModelInfo()
                ]);
                setMetrics(metricsData);
                setHistory(historyData);
                setModelInfo(infoData);
            } catch (error) {
                console.error("Failed to fetch dashboard data", error);
            } finally {
                setLoading(false);
            }
        };
        fetchData();
    }, []);

    const handleExport = () => {
        if (!history?.daily_trends) return;
        const csvContent = "data:text/csv;charset=utf-8,"
            + "Date,Total Appointments,No Shows,No Show Rate\n"
            + history.daily_trends.map((row: any) =>
                `${row.date},${row.total_appointments},${row.no_shows},${(row.no_shows / row.total_appointments).toFixed(2)}`
            ).join("\n");
        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", "attendance_report.csv");
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    // Calculate ROI - use daily_trends data for total appointments
    const totalAppointments = history?.daily_trends?.reduce((acc: number, curr: any) => acc + curr.total_appointments, 0) || 0;
    const preventedNoShows = totalAppointments > 0 ? Math.round(totalAppointments * 0.15) : 0;
    const estimatedSavings = preventedNoShows * settings.costPerNoShow;

    // Extract accuracy from nested metrics object
    const modelAccuracy = metrics?.metrics?.accuracy ?? metrics?.accuracy;

    // Transform Data
    const trendData = history?.daily_trends?.map((item: any) => ({
        name: new Date(item.date).toLocaleDateString('en-US', { weekday: 'short' }),
        fullDate: item.date,
        total: item.total_appointments,
        noShows: item.no_shows,
        rate: (item.no_shows / item.total_appointments) * 100
    })) || [];

    // Generate real sparkline data from daily_trends (last 7 days)
    const appointmentSparkline = trendData.slice(-7).map((d: any) => ({ value: d.total }));
    const noShowRateSparkline = trendData.slice(-7).map((d: any) => ({ value: d.rate }));
    const savingsSparkline = trendData.slice(-7).map((d: any) => ({
        value: Math.round(d.total * 0.15 * settings.costPerNoShow / 1000)
    }));

    // Calculate week-over-week percentage changes
    const calculateChange = (data: any[], key: string) => {
        if (!data || data.length < 14) return null;
        const thisWeek = data.slice(-7).reduce((acc, curr) => acc + (curr[key] || 0), 0);
        const lastWeek = data.slice(-14, -7).reduce((acc, curr) => acc + (curr[key] || 0), 0);
        if (lastWeek === 0) return null;
        const change = ((thisWeek - lastWeek) / lastWeek) * 100;
        return change;
    };

    const appointmentChange = calculateChange(history?.daily_trends, 'total_appointments');
    const noShowRates = history?.daily_trends?.map((d: any) => ({ rate: (d.no_shows / d.total_appointments) * 100 })) || [];
    const thisWeekRate = noShowRates.slice(-7).reduce((acc: number, curr: any) => acc + curr.rate, 0) / 7;
    const lastWeekRate = noShowRates.slice(-14, -7).reduce((acc: number, curr: any) => acc + curr.rate, 0) / 7;
    const noShowRateChange = lastWeekRate > 0 ? ((thisWeekRate - lastWeekRate) / lastWeekRate) * 100 : null;

    // Risk distribution data
    const riskData = history?.risk_distribution?.map((item: any) => ({
        name: item.risk_tier === 'MINIMAL' ? 'Low' :
            item.risk_tier === 'LOW' ? 'Low' :
                item.risk_tier === 'MEDIUM' ? 'Medium' : 'High',
        value: item.total_appointments,
        color: item.risk_tier === 'CRITICAL' || item.risk_tier === 'HIGH' ? '#EF4444' :
            item.risk_tier === 'MEDIUM' ? '#F59E0B' : '#10B981'
    })).reduce((acc: any[], curr: any) => {
        const existing = acc.find(i => i.name === curr.name);
        if (existing) {
            existing.value += curr.value;
        } else {
            acc.push(curr);
        }
        return acc;
    }, []) || [];

    // Feature importance data
    const featureImportanceData = modelInfo?.feature_importance?.slice(0, 8).map((item: any) => ({
        name: item.feature.replace(/_/g, ' '),
        importance: item.importance
    })) || [];

    return (
        <div className="space-y-8 pb-8">
            <AnimatePresence>
                {selectedAlert && (
                    <ReviewModal alertData={selectedAlert} onClose={() => setSelectedAlert(null)} />
                )}
            </AnimatePresence>

            {/* Header Section */}
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
                <div>
                    <h1 className="text-3xl font-bold text-text-primary tracking-tight">Dashboard Overview</h1>
                    <p className="text-text-secondary mt-1 text-lg">Real-time insights into appointment attendance.</p>
                </div>
                <div className="flex items-center gap-3">
                    <div className="relative">
                        <select className="appearance-none bg-surface border border-border text-text-primary rounded-xl pl-4 pr-10 py-2.5 text-sm font-medium focus:outline-none focus:ring-2 focus:ring-primary/50 hover:border-primary/50 transition-colors cursor-pointer">
                            <option>Last 30 Days</option>
                            <option>Last 7 Days</option>
                            <option>This Month</option>
                        </select>
                        <div className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none text-text-secondary">
                            <TrendingDown className="w-4 h-4" />
                        </div>
                    </div>
                    <button
                        onClick={handleExport}
                        className="bg-primary hover:bg-primary-hover text-white px-5 py-2.5 rounded-xl text-sm font-bold shadow-lg shadow-primary/25 transition-all flex items-center gap-2 btn-hover"
                    >
                        <Download className="w-4 h-4" />
                        Export Report
                    </button>
                </div>
            </div>

            {/* Stats Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                {loading ? (
                    [...Array(4)].map((_, i) => (
                        <div key={i} className="glass-card p-6 rounded-2xl animate-pulse h-48">
                            <div className="flex justify-between items-start mb-4">
                                <div className="w-12 h-12 bg-surface-hover rounded-xl"></div>
                            </div>
                            <div className="h-4 w-24 bg-surface-hover rounded mb-2"></div>
                            <div className="h-8 w-16 bg-surface-hover rounded"></div>
                        </div>
                    ))
                ) : (
                    <>
                        <StatCard
                            title="Total Appointments"
                            value={totalAppointments.toLocaleString()}
                            change={appointmentChange !== null ? `${appointmentChange >= 0 ? '+' : ''}${appointmentChange.toFixed(1)}%` : null}
                            trend={appointmentChange !== null ? (appointmentChange >= 0 ? "up" : "down") : "up"}
                            icon={Calendar}
                            data={appointmentSparkline.length > 0 ? appointmentSparkline : undefined}
                            ready={chartsReady}
                        />
                        <StatCard
                            title="No-Show Rate"
                            value={`${thisWeekRate.toFixed(1)}%`}
                            change={noShowRateChange !== null ? `${noShowRateChange >= 0 ? '+' : ''}${noShowRateChange.toFixed(1)}%` : null}
                            trend={noShowRateChange !== null ? (noShowRateChange <= 0 ? "up" : "down") : "down"}
                            icon={AlertTriangle}
                            data={noShowRateSparkline.length > 0 ? noShowRateSparkline : undefined}
                            ready={chartsReady}
                        />
                        <StatCard
                            title="Est. Savings"
                            value={`$${estimatedSavings.toLocaleString()}`}
                            change="ROI"
                            trend="up"
                            icon={DollarSign}
                            data={savingsSparkline.length > 0 ? savingsSparkline : undefined}
                            ready={chartsReady}
                        />
                        <StatCard
                            title="Model Accuracy"
                            value={modelAccuracy ? `${(modelAccuracy * 100).toFixed(1)}%` : "N/A"}
                            change={modelAccuracy ? null : null}
                            trend="up"
                            icon={CheckCircle}
                            data={undefined}
                            ready={chartsReady}
                        />
                    </>
                )}
            </div>

            {/* Main Charts Row */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Attendance Trends */}
                <div className="lg:col-span-2 glass-card p-6 rounded-2xl flex flex-col">
                    <div className="flex items-center justify-between mb-8">
                        <div>
                            <h3 className="text-lg font-bold text-text-primary">Attendance Trends</h3>
                            <p className="text-sm text-text-secondary">Daily appointment volume vs. no-shows</p>
                        </div>
                        <div className="flex items-center gap-2">
                            <span className="flex items-center text-xs text-text-secondary">
                                <span className="w-2 h-2 rounded-full bg-primary mr-1"></span> Total
                            </span>
                            <span className="flex items-center text-xs text-text-secondary">
                                <span className="w-2 h-2 rounded-full bg-error mr-1"></span> No-Show
                            </span>
                        </div>
                    </div>
                    <div className="flex-1" style={{ minHeight: '300px', height: '300px', width: '100%' }}>
                        {chartsReady && (
                            <ResponsiveContainer width="100%" height={300} minHeight={300}>
                                <AreaChart data={trendData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                                    <defs>
                                        <linearGradient id="colorTotal" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.3} />
                                            <stop offset="95%" stopColor="#3B82F6" stopOpacity={0} />
                                        </linearGradient>
                                        <linearGradient id="colorNoShow" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#EF4444" stopOpacity={0.3} />
                                            <stop offset="95%" stopColor="#EF4444" stopOpacity={0} />
                                        </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                                    <XAxis dataKey="name" stroke="#64748B" axisLine={false} tickLine={false} tick={{ fontSize: 12 }} dy={10} />
                                    <YAxis stroke="#64748B" axisLine={false} tickLine={false} tick={{ fontSize: 12 }} />
                                    <Tooltip content={<CustomTooltip />} cursor={{ stroke: 'rgba(255,255,255,0.1)', strokeWidth: 2 }} />
                                    <Area type="monotone" dataKey="total" stroke="#3B82F6" strokeWidth={3} fillOpacity={1} fill="url(#colorTotal)" name="Total Appointments" />
                                    <Area type="monotone" dataKey="noShows" stroke="#EF4444" strokeWidth={3} fillOpacity={1} fill="url(#colorNoShow)" name="No-Shows" />
                                </AreaChart>
                            </ResponsiveContainer>
                        )}
                    </div>
                </div>

                {/* Risk Distribution Donut */}
                <div className="glass-card p-6 rounded-2xl flex flex-col">
                    <h3 className="text-lg font-bold text-text-primary mb-2">Risk Distribution</h3>
                    <p className="text-sm text-text-secondary mb-6">Patient risk categorization</p>

                    <div className="flex-1 relative" style={{ minHeight: '250px', height: '250px', width: '100%' }}>
                        {chartsReady && (
                            <ResponsiveContainer width="100%" height={250} minHeight={250}>
                                <PieChart>
                                    <Pie
                                        data={riskData}
                                        cx="50%"
                                        cy="50%"
                                        innerRadius={80}
                                        outerRadius={100}
                                        paddingAngle={5}
                                        dataKey="value"
                                        stroke="none"
                                    >
                                        {riskData.map((entry: any, index: number) => (
                                            <Cell key={`cell-${index}`} fill={entry.color} />
                                        ))}
                                    </Pie>
                                    <Tooltip content={<CustomTooltip />} />
                                </PieChart>
                            </ResponsiveContainer>
                        )}
                        {/* Center Text */}
                        <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
                            <span className="text-3xl font-bold text-text-primary">
                                {metrics?.total_predictions?.toLocaleString() || "0"}
                            </span>
                            <span className="text-xs text-text-secondary uppercase tracking-wider">Patients</span>
                        </div>
                    </div>

                    <div className="mt-6 space-y-3">
                        {riskData.map((item: any, i: number) => (
                            <div key={i} className="flex items-center justify-between text-sm">
                                <span className="flex items-center text-text-secondary">
                                    <div className="w-3 h-3 rounded-full mr-3" style={{ backgroundColor: item.color }}></div>
                                    {item.name} Risk
                                </span>
                                <span className="font-bold text-text-primary">{((item.value / (metrics?.total_predictions || 1)) * 100).toFixed(1)}%</span>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Secondary Charts Row */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Feature Importance */}
                <div className="glass-card p-6 rounded-2xl">
                    <h3 className="text-lg font-bold text-text-primary mb-6">Top Risk Drivers</h3>
                    <div style={{ height: '256px', width: '100%', minHeight: '256px' }}>
                        {chartsReady && (
                            <ResponsiveContainer width="100%" height={256} minHeight={256}>
                                <BarChart data={featureImportanceData} layout="vertical" margin={{ left: 0, right: 20 }}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={true} vertical={false} />
                                    <XAxis type="number" stroke="#64748B" hide />
                                    <YAxis dataKey="name" type="category" stroke="#94A3B8" width={120} tick={{ fontSize: 11 }} axisLine={false} tickLine={false} />
                                    <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.05)' }} />
                                    <Bar dataKey="importance" fill="#8B5CF6" radius={[0, 4, 4, 0]} barSize={20} name="Importance">
                                        {featureImportanceData.map((_: any, index: number) => (
                                            <Cell key={`cell-${index}`} fill={`rgba(139, 92, 246, ${1 - (index * 0.1)})`} />
                                        ))}
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                        )}
                    </div>
                </div>

                {/* Recent Activity Table */}
                <div className="glass-card p-6 rounded-2xl flex flex-col">
                    <div className="flex items-center justify-between mb-6">
                        <h3 className="text-lg font-bold text-text-primary">Recent Alerts</h3>
                        <button
                            onClick={() => navigate('/patients')}
                            className="text-sm text-primary hover:text-primary-hover font-medium flex items-center gap-1 transition-colors"
                        >
                            View All <ArrowRight className="w-4 h-4" />
                        </button>
                    </div>

                    <div className="flex-1 overflow-x-auto">
                        <table className="w-full text-left border-collapse">
                            <thead>
                                <tr className="text-xs text-text-tertiary uppercase tracking-wider border-b border-border">
                                    <th className="pb-3 font-semibold pl-2">Patient</th>
                                    <th className="pb-3 font-semibold">Risk Level</th>
                                    <th className="pb-3 font-semibold">Score</th>
                                    <th className="pb-3 font-semibold text-right pr-2">Action</th>
                                </tr>
                            </thead>
                            <tbody className="text-sm">
                                {history?.recent_alerts?.length > 0 ? (
                                    history.recent_alerts.slice(0, 5).map((alert: any, i: number) => (
                                        <tr key={i} className="group hover:bg-surface-hover/30 transition-colors border-b border-border last:border-0">
                                            <td className="py-3 pl-2">
                                                <div className="font-medium text-text-primary">#{alert.patientid}</div>
                                            </td>
                                            <td className="py-3">
                                                <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-bold ${alert.risk_tier === 'CRITICAL' ? 'bg-error/10 text-error' :
                                                    alert.risk_tier === 'HIGH' ? 'bg-warning/10 text-warning' : 'bg-success/10 text-success'
                                                    }`}>
                                                    {alert.risk_tier}
                                                </span>
                                            </td>
                                            <td className="py-3 font-mono text-text-secondary">
                                                {alert.composite_risk_score.toFixed(2)}
                                            </td>
                                            <td className="py-3 text-right pr-2">
                                                <button
                                                    onClick={() => setSelectedAlert(alert)}
                                                    className="text-primary hover:text-primary-hover font-medium text-xs bg-primary/10 hover:bg-primary/20 px-3 py-1.5 rounded-lg transition-colors"
                                                >
                                                    Review
                                                </button>
                                            </td>
                                        </tr>
                                    ))
                                ) : (
                                    <tr>
                                        <td colSpan={4} className="py-8 text-center text-text-secondary">
                                            No recent alerts found.
                                        </td>
                                    </tr>
                                )}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ModelDashboard;
