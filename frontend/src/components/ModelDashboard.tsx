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
    Legend,
    BarChart,
    Bar,
    LineChart,
    Line
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
    Download
} from 'lucide-react';
import { getModelMetrics, getPredictionHistory, getModelInfo } from '../services/api';
import { useSettings } from '../context/SettingsContext';
import { useNavigate } from 'react-router-dom';

const StatCard = ({ title, value, change, icon: Icon, trend }: any) => (
    <div className="glass-card p-6 rounded-2xl relative overflow-hidden group hover:bg-surface-hover/50 transition-all duration-300">
        <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
            <Icon className="w-24 h-24 text-primary" />
        </div>
        <div className="relative z-10">
            <div className="flex items-center justify-between mb-4">
                <div className="p-3 bg-primary/10 rounded-xl">
                    <Icon className="w-6 h-6 text-primary" />
                </div>
                {change && (
                    <div className={`flex items-center text-sm font-medium ${trend === 'up' ? 'text-success' : 'text-error'}`}>
                        {trend === 'up' ? <TrendingUp className="w-4 h-4 mr-1" /> : <TrendingDown className="w-4 h-4 mr-1" />}
                        {change}
                    </div>
                )}
            </div>
            <h3 className="text-text-secondary text-sm font-medium mb-1">{title}</h3>
            <p className="text-3xl font-bold text-text-primary">{value}</p>
        </div>
    </div>
);

const ReviewModal = ({ alertData, onClose }: { alertData: any; onClose: () => void }) => {
    if (!alertData) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm">
            <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                className="bg-surface border border-surface-hover rounded-2xl shadow-2xl w-full max-w-md overflow-hidden"
            >
                <div className="p-6 border-b border-surface-hover flex justify-between items-center">
                    <h3 className="text-xl font-bold text-text-primary flex items-center gap-2">
                        <AlertTriangle className="w-5 h-5 text-error" />
                        High Risk Alert
                    </h3>
                    <button onClick={onClose} className="text-text-secondary hover:text-text-primary transition-colors btn-hover">
                        <X className="w-6 h-6" />
                    </button>
                </div>

                <div className="p-6 space-y-6">
                    <div className="flex items-center gap-4 p-4 bg-surface-hover/30 rounded-xl">
                        <div className="w-12 h-12 bg-primary/10 rounded-full flex items-center justify-center text-primary font-bold text-lg">
                            {alertData.patientid.toString().slice(-2)}
                        </div>
                        <div>
                            <p className="text-sm text-text-secondary">Patient ID</p>
                            <p className="text-lg font-bold text-text-primary">#{alertData.patientid}</p>
                        </div>
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                        <div className="p-4 bg-surface-hover/30 rounded-xl">
                            <p className="text-sm text-text-secondary mb-1">Risk Tier</p>
                            <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${alertData.risk_tier === 'CRITICAL' ? 'bg-error/10 text-error' :
                                alertData.risk_tier === 'HIGH' ? 'bg-warning/10 text-warning' :
                                    'bg-success/10 text-success'
                                }`}>
                                {alertData.risk_tier}
                            </span>
                        </div>
                        <div className="p-4 bg-surface-hover/30 rounded-xl">
                            <p className="text-sm text-text-secondary mb-1">Risk Score</p>
                            <p className="text-xl font-bold text-text-primary">{alertData.composite_risk_score.toFixed(2)}</p>
                        </div>
                    </div>

                    <div className="space-y-3">
                        <h4 className="text-sm font-bold text-text-secondary uppercase tracking-wider">Recommended Actions</h4>
                        <a href={`tel:${alertData.patientid}`} className="w-full py-3 px-4 bg-primary/10 hover:bg-primary/20 text-primary rounded-xl font-medium transition-colors flex items-center justify-center gap-2 btn-hover">
                            <Phone className="w-4 h-4" />
                            Call Patient
                        </a>
                        <a href="mailto:scheduling@clinic.com" className="w-full py-3 px-4 bg-surface-hover hover:bg-surface-hover/80 text-text-primary rounded-xl font-medium transition-colors flex items-center justify-center gap-2 btn-hover">
                            <CalendarIcon className="w-4 h-4" />
                            Reschedule Appointment
                        </a>
                    </div>
                </div>

                <div className="p-6 border-t border-surface-hover bg-surface-hover/10 flex justify-end gap-3">
                    <button
                        onClick={onClose}
                        className="px-4 py-2 text-text-secondary hover:text-text-primary font-medium transition-colors btn-hover"
                    >
                        Close
                    </button>
                    <button
                        onClick={() => {
                            alert("Marked as reviewed!");
                            onClose();
                        }}
                        className="px-6 py-2 bg-primary hover:bg-primary-hover text-white rounded-xl font-bold shadow-lg shadow-primary/20 transition-all btn-hover"
                    >
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

    useEffect(() => {
        if (!loading) {
            const timer = setTimeout(() => setChartsReady(true), 200);
            return () => clearTimeout(timer);
        }
    }, [loading]);
    const navigate = useNavigate();

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

    // Calculate ROI
    const preventedNoShows = metrics?.total_predictions ? Math.round(metrics.total_predictions * 0.15) : 0; // Assuming 15% prevention rate for demo
    const estimatedSavings = preventedNoShows * settings.costPerNoShow;

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

    // Transform history data for charts
    const trendData = history?.daily_trends?.map((item: any) => ({
        name: new Date(item.date).toLocaleDateString('en-US', { weekday: 'short' }),
        fullDate: item.date,
        total: item.total_appointments,
        noShows: item.no_shows
    })) || [];

    const riskData = history?.risk_distribution?.map((item: any) => ({
        name: item.risk_tier === 'MINIMAL' ? 'Low Risk' : // Group minimal with low for simpler chart
            item.risk_tier === 'LOW' ? 'Low Risk' :
                item.risk_tier === 'MEDIUM' ? 'Medium Risk' : 'High Risk',
        value: item.total_appointments,
        color: item.risk_tier === 'CRITICAL' || item.risk_tier === 'HIGH' ? '#ef4444' :
            item.risk_tier === 'MEDIUM' ? '#eab308' : '#22c55e'
    })).reduce((acc: any[], curr: any) => {
        const existing = acc.find(i => i.name === curr.name);
        if (existing) {
            existing.value += curr.value;
        } else {
            acc.push(curr);
        }
        return acc;
    }, []) || [];

    const segmentData = history?.patient_segments?.map((item: any) => ({
        name: item.patient_segment,
        value: item.patient_count,
        percentage: item.percentage
    })) || [];

    const behaviorData = history?.behavior_evolution?.map((item: any) => ({
        visitNumber: item.visit_number,
        noShowRate: item.noshow_rate
    })) || [];

    // Transform feature importance data
    const featureImportanceData = modelInfo?.feature_importance?.slice(0, 10).map((item: any) => ({
        name: item.feature,
        importance: item.importance
    })) || [];

    return (
        <div className="space-y-6">
            <AnimatePresence>
                {selectedAlert && (
                    <ReviewModal alertData={selectedAlert} onClose={() => setSelectedAlert(null)} />
                )}
            </AnimatePresence>

            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                    <h1 className="text-3xl font-bold text-text-primary">Dashboard Overview</h1>
                    <p className="text-text-secondary mt-1">Real-time insights into appointment attendance and model performance.</p>
                </div>
                <div className="flex items-center gap-3">
                    <select className="bg-surface border border-surface-hover text-text-primary rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary">
                        <option>Last 30 Days</option>
                        <option>Last 7 Days</option>
                        <option>This Month</option>
                    </select>
                    <button
                        onClick={handleExport}
                        className="bg-primary hover:bg-primary-hover text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2 btn-hover"
                    >
                        <Download className="w-4 h-4" />
                        Export Report
                    </button>
                </div>
            </div>

            {/* Stats Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                {loading ? (
                    // Skeleton Loading State
                    [...Array(4)].map((_, i) => (
                        <div key={i} className="glass-card p-6 rounded-2xl animate-pulse">
                            <div className="flex justify-between items-start mb-4">
                                <div className="w-12 h-12 bg-surface-hover rounded-xl"></div>
                                <div className="w-16 h-6 bg-surface-hover rounded-md"></div>
                            </div>
                            <div className="h-4 w-24 bg-surface-hover rounded mb-2"></div>
                            <div className="h-8 w-16 bg-surface-hover rounded"></div>
                        </div>
                    ))
                ) : (
                    <>
                        <StatCard
                            title="Total Appointments"
                            value={metrics?.total_predictions?.toLocaleString() || history?.daily_trends?.reduce((acc: number, curr: any) => acc + curr.total_appointments, 0).toLocaleString() || "0"}
                            change="+12.5%"
                            trend="up"
                            icon={Calendar}
                        />
                        <StatCard
                            title="No-Show Rate"
                            value={`${(history?.daily_trends?.reduce((acc: number, curr: any) => acc + (curr.no_shows / curr.total_appointments), 0) / (history?.daily_trends?.length || 1) * 100).toFixed(1)}%`}
                            change="-2.1%"
                            trend="down"
                            icon={AlertTriangle}
                        />
                        <StatCard
                            title="Est. Savings"
                            value={`$${estimatedSavings.toLocaleString()}`}
                            change="ROI"
                            trend="up"
                            icon={DollarSign}
                        />
                        <StatCard
                            title="Model Accuracy"
                            value={metrics?.accuracy ? `${(metrics.accuracy * 100).toFixed(1)}%` : "N/A"}
                            change="+0.3%"
                            trend="up"
                            icon={CheckCircle}
                        />
                    </>
                )}
            </div>

            {/* Charts Section */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {loading ? (
                    <>
                        <div className="lg:col-span-2 glass-card p-6 rounded-2xl animate-pulse">
                            <div className="h-8 w-48 bg-surface-hover rounded mb-6"></div>
                            <div className="h-80 w-full bg-surface-hover rounded-xl"></div>
                        </div>
                        <div className="glass-card p-6 rounded-2xl animate-pulse">
                            <div className="h-8 w-48 bg-surface-hover rounded mb-6"></div>
                            <div className="h-64 w-full bg-surface-hover rounded-full mx-auto"></div>
                            <div className="mt-4 space-y-3">
                                <div className="h-4 w-full bg-surface-hover rounded"></div>
                                <div className="h-4 w-full bg-surface-hover rounded"></div>
                                <div className="h-4 w-full bg-surface-hover rounded"></div>
                            </div>
                        </div>
                    </>
                ) : (
                    <>
                        {/* Main Trend Chart */}
                        <div className="lg:col-span-2 glass-card p-6 rounded-2xl">
                            <h3 className="text-lg font-bold text-text-primary mb-6">Attendance Trends</h3>
                            <div className="h-80 w-full min-w-0">
                                {chartsReady && (
                                    <ResponsiveContainer width="100%" height="100%" debounce={200} minWidth={0} minHeight={0}>
                                        <AreaChart data={trendData.length > 0 ? trendData : []}>
                                            <defs>
                                                <linearGradient id="colorTotal" x1="0" y1="0" x2="0" y2="1">
                                                    <stop offset="5%" stopColor="#0ea5e9" stopOpacity={0.3} />
                                                    <stop offset="95%" stopColor="#0ea5e9" stopOpacity={0} />
                                                </linearGradient>
                                                <linearGradient id="colorNoShow" x1="0" y1="0" x2="0" y2="1">
                                                    <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3} />
                                                    <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
                                                </linearGradient>
                                            </defs>
                                            <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                                            <XAxis dataKey="name" stroke="#94a3b8" axisLine={false} tickLine={false} />
                                            <YAxis stroke="#94a3b8" axisLine={false} tickLine={false} />
                                            <Tooltip
                                                contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', borderRadius: '8px' }}
                                                itemStyle={{ color: '#f8fafc' }}
                                            />
                                            <Legend />
                                            <Area type="monotone" dataKey="total" stroke="#0ea5e9" strokeWidth={3} fillOpacity={1} fill="url(#colorTotal)" name="Total Appointments" />
                                            <Area type="monotone" dataKey="noShows" stroke="#ef4444" strokeWidth={3} fillOpacity={1} fill="url(#colorNoShow)" name="No-Shows" />
                                        </AreaChart>
                                    </ResponsiveContainer>
                                )}
                            </div>
                        </div>

                        {/* Risk Distribution */}
                        <div className="glass-card p-6 rounded-2xl">
                            <h3 className="text-lg font-bold text-text-primary mb-6">Risk Distribution</h3>
                            <div className="h-64 w-full min-w-0">
                                {chartsReady && (
                                    <ResponsiveContainer width="100%" height="100%" debounce={200} minWidth={0} minHeight={0}>
                                        <PieChart>
                                            <Pie
                                                data={riskData.length > 0 ? riskData : []}
                                                cx="50%"
                                                cy="50%"
                                                innerRadius={60}
                                                outerRadius={80}
                                                paddingAngle={5}
                                                dataKey="value"
                                            >
                                                {riskData.map((entry: any, index: number) => (
                                                    <Cell key={`cell-${index}`} fill={entry.color} />
                                                ))}
                                            </Pie>
                                            <Tooltip
                                                contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', borderRadius: '8px' }}
                                                itemStyle={{ color: '#f8fafc' }}
                                            />
                                            <Legend verticalAlign="bottom" height={36} />
                                        </PieChart>
                                    </ResponsiveContainer>
                                )}
                            </div>
                            <div className="mt-4 space-y-3">
                                <div className="flex items-center justify-between text-sm">
                                    <span className="flex items-center text-text-secondary">
                                        <div className="w-3 h-3 rounded-full bg-success mr-2"></div>
                                        Low Risk
                                    </span>
                                    <span className="font-bold text-text-primary">50%</span>
                                </div>
                                <div className="flex items-center justify-between text-sm">
                                    <span className="flex items-center text-text-secondary">
                                        <div className="w-3 h-3 rounded-full bg-warning mr-2"></div>
                                        Medium Risk
                                    </span>
                                    <span className="font-bold text-text-primary">37.5%</span>
                                </div>
                                <div className="flex items-center justify-between text-sm">
                                    <span className="flex items-center text-text-secondary">
                                        <div className="w-3 h-3 rounded-full bg-error mr-2"></div>
                                        High Risk
                                    </span>
                                    <span className="font-bold text-text-primary">12.5%</span>
                                </div>
                            </div>
                        </div>
                    </>
                )
                }
            </div>

            {/* Model Explainability */}
            <div className="grid grid-cols-1 lg:grid-cols-1 gap-6">
                <div className="glass-card p-6 rounded-2xl">
                    <h3 className="text-lg font-bold text-text-primary mb-6">Top Risk Factors (Feature Importance)</h3>
                    <div className="h-64 w-full min-w-0">
                        {chartsReady && (
                            <ResponsiveContainer width="100%" height="100%" debounce={200} minWidth={0} minHeight={0}>
                                <BarChart data={featureImportanceData} layout="vertical" margin={{ left: 40, right: 20 }}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={true} vertical={false} />
                                    <XAxis type="number" stroke="#94a3b8" />
                                    <YAxis dataKey="name" type="category" stroke="#94a3b8" width={150} tick={{ fontSize: 11 }} />
                                    <Tooltip
                                        cursor={{ fill: '#334155', opacity: 0.2 }}
                                        contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', borderRadius: '8px' }}
                                        itemStyle={{ color: '#f8fafc' }}
                                    />
                                    <Bar dataKey="importance" fill="#3b82f6" radius={[0, 4, 4, 0]} name="Importance" />
                                </BarChart>
                            </ResponsiveContainer>
                        )}
                    </div>
                </div>
            </div>

            {/* Patient Insights */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {loading ? (
                    // Skeletons
                    [...Array(2)].map((_, i) => (
                        <div key={i} className="glass-card p-6 rounded-2xl animate-pulse">
                            <div className="h-8 w-48 bg-surface-hover rounded mb-6"></div>
                            <div className="h-64 w-full bg-surface-hover rounded-xl"></div>
                        </div>
                    ))
                ) : (
                    <>
                        {/* Patient Segments */}
                        <div className="glass-card p-6 rounded-2xl">
                            <h3 className="text-lg font-bold text-text-primary mb-6">Patient Segments</h3>
                            <div className="h-64 w-full min-w-0">
                                {chartsReady && (
                                    <ResponsiveContainer width="100%" height="100%" debounce={200} minWidth={0} minHeight={0}>
                                        <BarChart data={segmentData} layout="vertical" margin={{ left: 20 }}>
                                            <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={true} vertical={false} />
                                            <XAxis type="number" stroke="#94a3b8" hide />
                                            <YAxis dataKey="name" type="category" stroke="#94a3b8" width={110} tick={{ fontSize: 11 }} />
                                            <Tooltip
                                                cursor={{ fill: '#334155', opacity: 0.2 }}
                                                contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', borderRadius: '8px' }}
                                                itemStyle={{ color: '#f8fafc' }}
                                            />
                                            <Bar dataKey="value" fill="#8b5cf6" radius={[0, 4, 4, 0]} name="Patients" />
                                        </BarChart>
                                    </ResponsiveContainer>
                                )}
                            </div>
                        </div>

                        {/* Behavior Evolution */}
                        <div className="glass-card p-6 rounded-2xl">
                            <h3 className="text-lg font-bold text-text-primary mb-6">Behavior Evolution</h3>
                            <div className="h-64 w-full min-w-0">
                                {chartsReady && (
                                    <ResponsiveContainer width="100%" height="100%" debounce={200} minWidth={0} minHeight={0}>
                                        <LineChart data={behaviorData}>
                                            <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                                            <XAxis dataKey="visitNumber" stroke="#94a3b8" label={{ value: 'Visit Number', position: 'insideBottom', offset: -5, fill: '#94a3b8', fontSize: 12 }} />
                                            <YAxis stroke="#94a3b8" label={{ value: 'No-Show Rate (%)', angle: -90, position: 'insideLeft', fill: '#94a3b8', fontSize: 12 }} />
                                            <Tooltip
                                                contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', borderRadius: '8px' }}
                                                itemStyle={{ color: '#f8fafc' }}
                                            />
                                            <Line type="monotone" dataKey="noShowRate" stroke="#10b981" strokeWidth={3} dot={{ fill: '#10b981' }} name="No-Show Rate" />
                                        </LineChart>
                                    </ResponsiveContainer>
                                )}
                            </div>
                        </div>
                    </>
                )}
            </div>

            {/* Recent Activity / Alerts */}
            <div className="glass-card p-6 rounded-2xl">
                <div className="flex items-center justify-between mb-6">
                    <h3 className="text-lg font-bold text-text-primary">Recent High-Risk Alerts</h3>
                    <button
                        onClick={() => navigate('/patients')}
                        className="text-primary hover:text-primary-hover text-sm font-medium btn-hover"
                    >
                        View All
                    </button>
                </div>
                <div className="space-y-4">
                    {history?.recent_alerts?.length > 0 ? (
                        history.recent_alerts.map((alert: any, i: number) => (
                            <div key={i} className="flex items-center justify-between p-4 bg-surface rounded-xl border border-surface-hover hover:bg-surface-hover/50 transition-colors">
                                <div className="flex items-center gap-4">
                                    <div className="p-2 bg-error/10 rounded-lg">
                                        <AlertTriangle className="w-5 h-5 text-error" />
                                    </div>
                                    <div>
                                        <p className="font-bold text-text-primary">Patient #{alert.patientid}</p>
                                        <p className="text-sm text-text-secondary">
                                            {alert.risk_tier} Risk • Score: {alert.composite_risk_score.toFixed(2)}
                                        </p>
                                    </div>
                                </div>
                                <button
                                    onClick={() => setSelectedAlert(alert)}
                                    className="px-4 py-2 text-sm font-medium text-primary bg-primary/10 rounded-lg hover:bg-primary/20 transition-colors btn-hover"
                                >
                                    Review
                                </button>
                            </div>
                        ))
                    ) : (
                        <p className="text-text-secondary text-center py-4">No high-risk alerts found.</p>
                    )}
                </div>
            </div>
        </div>
    );
};

export default ModelDashboard;
