import { motion } from 'framer-motion';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
    PieChart, Pie, Cell, AreaChart, Area
} from 'recharts';
import { TrendingUp, Users, AlertTriangle, CheckCircle, DollarSign, Clock } from 'lucide-react';

// Mock Data based on SQL Analytics Report
const neighborhoodData = [
    { name: 'ILHAS OCEÂNICAS', rate: 34.0 },
    { name: 'SANTOS DUMONT', rate: 28.9 },
    { name: 'SANTA CECÍLIA', rate: 27.5 },
    { name: 'SANTA CLARA', rate: 26.5 },
    { name: 'ITARARÉ', rate: 26.3 },
];

const smsData = [
    { name: 'Received SMS', value: 72.4, color: '#8b5cf6' }, // Show rate (100 - 27.6)
    { name: 'No SMS', value: 66.7, color: '#ef4444' },       // Show rate (100 - 33.3)
];

const leadTimeData = [
    { days: 'Same Day', rate: 15.9 },
    { days: '1-3 Days', rate: 23.1 },
    { days: '4-7 Days', rate: 25.8 },
    { days: '8-14 Days', rate: 30.2 },
    { days: '15-30 Days', rate: 32.5 },
    { days: '>30 Days', rate: 33.1 },
];

const kpiData = [
    {
        title: 'Total Appointments',
        value: '110,527',
        change: '+12.5%',
        trend: 'up',
        icon: Users,
        color: 'blue'
    },
    {
        title: 'Overall No-Show Rate',
        value: '20.2%',
        change: '-1.4%',
        trend: 'down',
        icon: AlertTriangle,
        color: 'purple'
    },
    {
        title: 'Est. Revenue Saved',
        value: '$860K',
        change: 'Potential',
        trend: 'up',
        icon: DollarSign,
        color: 'green'
    },
    {
        title: 'Avg. Lead Time',
        value: '10.2 Days',
        change: '-0.5 Days',
        trend: 'down',
        icon: Clock,
        color: 'orange'
    }
];

export default function ModelDashboard() {
    return (
        <div className="space-y-8">
            {/* Header */}
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                <div>
                    <h2 className="text-3xl font-bold text-gray-900 dark:text-white">Analytics Dashboard</h2>
                    <p className="text-gray-500 dark:text-gray-400">Real-time insights from SQL Analytics Module</p>
                </div>
                <div className="flex gap-2">
                    <select className="bg-white dark:bg-white/5 border border-gray-200 dark:border-white/10 rounded-lg px-4 py-2 text-sm">
                        <option>Last 30 Days</option>
                        <option>Last Quarter</option>
                        <option>Year to Date</option>
                    </select>
                    <button className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors">
                        Export Report
                    </button>
                </div>
            </div>

            {/* KPI Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                {kpiData.map((kpi, index) => {
                    const Icon = kpi.icon;
                    return (
                        <motion.div
                            key={index}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: index * 0.1 }}
                            className="bg-white dark:bg-white/5 backdrop-blur-xl border border-gray-200 dark:border-white/10 p-6 rounded-2xl shadow-sm"
                        >
                            <div className="flex justify-between items-start mb-4">
                                <div className={`p-3 rounded-xl bg-${kpi.color}-500/10`}>
                                    <Icon className={`w-6 h-6 text-${kpi.color}-500`} />
                                </div>
                                <div className={`flex items-center gap-1 text-sm font-medium ${kpi.trend === 'up' ? 'text-green-500' : 'text-red-500'
                                    }`}>
                                    {kpi.change}
                                    <TrendingUp className={`w-3 h-3 ${kpi.trend === 'down' ? 'rotate-180' : ''}`} />
                                </div>
                            </div>
                            <h3 className="text-gray-500 dark:text-gray-400 text-sm font-medium">{kpi.title}</h3>
                            <p className="text-2xl font-bold text-gray-900 dark:text-white mt-1">{kpi.value}</p>
                        </motion.div>
                    );
                })}
            </div>

            {/* Charts Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Neighborhood Risk */}
                <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.2 }}
                    className="bg-white dark:bg-white/5 backdrop-blur-xl border border-gray-200 dark:border-white/10 p-6 rounded-2xl shadow-sm"
                >
                    <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-6">High Risk Neighborhoods</h3>
                    <div className="h-80 relative" style={{ width: '100%', height: 320 }}>
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={neighborhoodData} layout="vertical">
                                <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.1} />
                                <XAxis type="number" stroke="#9ca3af" />
                                <YAxis dataKey="name" type="category" width={120} stroke="#9ca3af" fontSize={12} />
                                <Tooltip
                                    contentStyle={{ backgroundColor: '#1f2937', border: 'none', borderRadius: '8px', color: '#fff' }}
                                />
                                <Bar dataKey="rate" fill="#8b5cf6" radius={[0, 4, 4, 0]} name="No-Show Rate (%)" />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </motion.div>

                {/* Lead Time Trend */}
                <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.3 }}
                    className="bg-white dark:bg-white/5 backdrop-blur-xl border border-gray-200 dark:border-white/10 p-6 rounded-2xl shadow-sm"
                >
                    <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-6">No-Show Rate by Lead Time</h3>
                    <div className="h-80 relative" style={{ width: '100%', height: 320 }}>
                        <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={leadTimeData}>
                                <defs>
                                    <linearGradient id="colorRate" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#ec4899" stopOpacity={0.3} />
                                        <stop offset="95%" stopColor="#ec4899" stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.1} />
                                <XAxis dataKey="days" stroke="#9ca3af" fontSize={12} />
                                <YAxis stroke="#9ca3af" />
                                <Tooltip
                                    contentStyle={{ backgroundColor: '#1f2937', border: 'none', borderRadius: '8px', color: '#fff' }}
                                />
                                <Area type="monotone" dataKey="rate" stroke="#ec4899" fillOpacity={1} fill="url(#colorRate)" name="No-Show Rate (%)" />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                </motion.div>

                {/* SMS Impact */}
                <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.4 }}
                    className="bg-white dark:bg-white/5 backdrop-blur-xl border border-gray-200 dark:border-white/10 p-6 rounded-2xl shadow-sm"
                >
                    <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-6">SMS Reminder Impact</h3>
                    <div className="flex flex-col md:flex-row items-center justify-center gap-8">
                        <div className="h-64 w-64 relative" style={{ width: 256, height: 256 }}>
                            <ResponsiveContainer width="100%" height="100%">
                                <PieChart>
                                    <Pie
                                        data={smsData}
                                        cx="50%"
                                        cy="50%"
                                        innerRadius={60}
                                        outerRadius={80}
                                        paddingAngle={5}
                                        dataKey="value"
                                    >
                                        {smsData.map((entry, index) => (
                                            <Cell key={`cell-${index}`} fill={entry.color} />
                                        ))}
                                    </Pie>
                                    <Tooltip />
                                    <Legend />
                                </PieChart>
                            </ResponsiveContainer>
                        </div>
                        <div className="space-y-4">
                            <div className="flex items-center gap-3">
                                <div className="w-3 h-3 rounded-full bg-purple-500" />
                                <div>
                                    <p className="text-sm text-gray-500 dark:text-gray-400">With SMS</p>
                                    <p className="text-xl font-bold text-gray-900 dark:text-white">72.4% Attendance</p>
                                </div>
                            </div>
                            <div className="flex items-center gap-3">
                                <div className="w-3 h-3 rounded-full bg-red-500" />
                                <div>
                                    <p className="text-sm text-gray-500 dark:text-gray-400">Without SMS</p>
                                    <p className="text-xl font-bold text-gray-900 dark:text-white">66.7% Attendance</p>
                                </div>
                            </div>
                            <div className="pt-4 border-t border-gray-200 dark:border-white/10">
                                <p className="text-sm font-medium text-green-500">
                                    +5.7% Improvement with SMS
                                </p>
                            </div>
                        </div>
                    </div>
                </motion.div>

                {/* Recent Activity */}
                <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.5 }}
                    className="bg-white dark:bg-white/5 backdrop-blur-xl border border-gray-200 dark:border-white/10 p-6 rounded-2xl shadow-sm"
                >
                    <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-6">Recent System Activity</h3>
                    <div className="space-y-6">
                        {[1, 2, 3, 4].map((_, i) => (
                            <div key={i} className="flex items-start gap-4">
                                <div className="p-2 rounded-full bg-blue-500/10 mt-1">
                                    <CheckCircle className="w-4 h-4 text-blue-500" />
                                </div>
                                <div>
                                    <p className="text-sm font-medium text-gray-900 dark:text-white">
                                        Batch prediction completed
                                    </p>
                                    <p className="text-xs text-gray-500 dark:text-gray-400">
                                        Processed 150 appointments • {i * 15 + 5} mins ago
                                    </p>
                                </div>
                            </div>
                        ))}
                    </div>
                </motion.div>
            </div>
        </div>
    );
}
