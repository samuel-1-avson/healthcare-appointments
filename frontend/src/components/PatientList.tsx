import { useEffect, useState } from 'react';
import { getPredictionHistory } from '../services/api';
import {
    Search,
    Filter,
    MoreVertical,
    Phone,
    Calendar,
    AlertTriangle,
    Mail,
    ChevronLeft,
    ChevronRight,
    Download,
    User,
    CheckCircle,
    Clock
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const PatientList = () => {
    const [patients, setPatients] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);
    const [searchTerm, setSearchTerm] = useState('');
    const [filter, setFilter] = useState('all');
    const [currentPage, setCurrentPage] = useState(1);
    const itemsPerPage = 10;

    useEffect(() => {
        const fetchPatients = async () => {
            try {
                const history = await getPredictionHistory();
                // Use recent_alerts as the base
                const data = history?.recent_alerts || [];

                setPatients(data);
            } catch (error) {
                console.error("Failed to fetch patient history", error);
            } finally {
                setLoading(false);
            }
        };
        fetchPatients();
    }, []);

    const filteredPatients = patients.filter(p => {
        const matchesSearch = p.patientid.toString().includes(searchTerm);
        const matchesFilter = filter === 'all' || p.risk_tier.toLowerCase() === filter;
        return matchesSearch && matchesFilter;
    });

    // Pagination logic
    const totalPages = Math.ceil(filteredPatients.length / itemsPerPage);
    const paginatedPatients = filteredPatients.slice(
        (currentPage - 1) * itemsPerPage,
        currentPage * itemsPerPage
    );

    const getRiskBadge = (tier: string) => {
        switch (tier) {
            case 'CRITICAL':
                return { bg: 'bg-error/10', text: 'text-error', border: 'border-error/20', icon: AlertTriangle };
            case 'HIGH':
                return { bg: 'bg-warning/10', text: 'text-warning', border: 'border-warning/20', icon: AlertTriangle };
            case 'MEDIUM':
                return { bg: 'bg-yellow-500/10', text: 'text-yellow-500', border: 'border-yellow-500/20', icon: Clock };
            default:
                return { bg: 'bg-success/10', text: 'text-success', border: 'border-success/20', icon: CheckCircle };
        }
    };

    return (
        <div className="space-y-8">
            {/* Header Section */}
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
                <div>
                    <h1 className="text-3xl font-bold text-text-primary tracking-tight">Patient Management</h1>
                    <p className="text-text-secondary mt-1 text-lg">Monitor high-risk patients and intervene early.</p>
                </div>
                <div className="flex gap-3">
                    <button className="bg-surface hover:bg-surface-hover text-text-primary border border-border px-4 py-2.5 rounded-xl text-sm font-bold transition-colors flex items-center gap-2">
                        <Download className="w-4 h-4" />
                        Export CSV
                    </button>
                    <button className="bg-primary hover:bg-primary-hover text-white px-5 py-2.5 rounded-xl text-sm font-bold shadow-lg shadow-primary/25 transition-all flex items-center gap-2 btn-hover">
                        <User className="w-4 h-4" />
                        Add Patient
                    </button>
                </div>
            </div>

            {/* Filters Bar */}
            <div className="glass-card p-5 rounded-2xl flex flex-col md:flex-row gap-4 items-center justify-between">
                <div className="relative w-full md:w-96 group">
                    <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-text-secondary w-5 h-5 group-focus-within:text-primary transition-colors" />
                    <input
                        type="text"
                        placeholder="Search by Patient ID..."
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        className="w-full bg-surface border border-surface-hover text-text-primary rounded-xl pl-12 pr-4 py-3 focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary transition-all"
                    />
                </div>
                <div className="flex items-center gap-3 w-full md:w-auto overflow-x-auto pb-2 md:pb-0">
                    <Filter className="text-text-secondary w-5 h-5 flex-shrink-0" />
                    <div className="flex bg-surface border border-surface-hover rounded-xl p-1">
                        {['all', 'critical', 'high', 'medium', 'low'].map((f) => (
                            <button
                                key={f}
                                onClick={() => setFilter(f)}
                                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all capitalize ${filter === f
                                    ? 'bg-primary text-white shadow-md'
                                    : 'text-text-secondary hover:text-text-primary hover:bg-surface-hover'
                                    }`}
                            >
                                {f}
                            </button>
                        ))}
                    </div>
                </div>
            </div>

            {/* Table */}
            <div className="glass-card rounded-2xl overflow-hidden border border-surface-hover shadow-xl">
                <div className="overflow-x-auto">
                    <table className="w-full">
                        <thead>
                            <tr className="bg-surface/50 border-b border-surface-hover backdrop-blur-sm">
                                <th className="px-8 py-5 text-left text-xs font-bold text-text-tertiary uppercase tracking-wider">Patient</th>
                                <th className="px-6 py-5 text-left text-xs font-bold text-text-tertiary uppercase tracking-wider">Risk Assessment</th>
                                <th className="px-6 py-5 text-left text-xs font-bold text-text-tertiary uppercase tracking-wider">Score</th>
                                <th className="px-6 py-5 text-left text-xs font-bold text-text-tertiary uppercase tracking-wider">Last Prediction</th>
                                <th className="px-8 py-5 text-right text-xs font-bold text-text-tertiary uppercase tracking-wider">Quick Actions</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-surface-hover">
                            {loading ? (
                                [...Array(5)].map((_, i) => (
                                    <tr key={i} className="animate-pulse">
                                        <td className="px-8 py-6"><div className="h-10 w-10 bg-surface-hover rounded-full inline-block mr-3 align-middle"></div><div className="h-4 w-24 bg-surface-hover rounded inline-block align-middle"></div></td>
                                        <td className="px-6 py-6"><div className="h-6 w-20 bg-surface-hover rounded-full"></div></td>
                                        <td className="px-6 py-6"><div className="h-4 w-12 bg-surface-hover rounded"></div></td>
                                        <td className="px-6 py-6"><div className="h-4 w-32 bg-surface-hover rounded"></div></td>
                                        <td className="px-8 py-6"></td>
                                    </tr>
                                ))
                            ) : paginatedPatients.length > 0 ? (
                                <AnimatePresence>
                                    {paginatedPatients.map((patient, i) => {
                                        const badge = getRiskBadge(patient.risk_tier);
                                        const BadgeIcon = badge.icon;
                                        return (
                                            <motion.tr
                                                key={i}
                                                initial={{ opacity: 0, y: 10 }}
                                                animate={{ opacity: 1, y: 0 }}
                                                exit={{ opacity: 0, y: -10 }}
                                                transition={{ delay: i * 0.05 }}
                                                className="group hover:bg-surface-hover/30 transition-colors"
                                            >
                                                <td className="px-8 py-5 whitespace-nowrap">
                                                    <div className="flex items-center">
                                                        <div className="w-10 h-10 rounded-full bg-gradient-to-br from-primary/20 to-accent/20 flex items-center justify-center text-primary font-bold text-sm mr-4 border border-white/5">
                                                            {patient.patientid.toString().slice(-2)}
                                                        </div>
                                                        <div>
                                                            <div className="font-bold text-text-primary">#{patient.patientid}</div>
                                                            <div className="text-xs text-text-secondary">ID: {patient.patientid}</div>
                                                        </div>
                                                    </div>
                                                </td>
                                                <td className="px-6 py-5 whitespace-nowrap">
                                                    <span className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-bold border ${badge.bg} ${badge.text} ${badge.border}`}>
                                                        <BadgeIcon className="w-3 h-3 mr-1.5" />
                                                        {patient.risk_tier}
                                                    </span>
                                                </td>
                                                <td className="px-6 py-5 whitespace-nowrap">
                                                    <div className="flex items-center gap-2">
                                                        <div className="w-16 h-2 bg-surface-hover rounded-full overflow-hidden">
                                                            <div
                                                                className={`h-full rounded-full ${patient.composite_risk_score > 0.7 ? 'bg-error' :
                                                                    patient.composite_risk_score > 0.4 ? 'bg-warning' : 'bg-success'
                                                                    }`}
                                                                style={{ width: `${patient.composite_risk_score * 100}%` }}
                                                            />
                                                        </div>
                                                        <span className="text-sm font-mono font-medium text-text-secondary">
                                                            {(patient.composite_risk_score * 100).toFixed(0)}%
                                                        </span>
                                                    </div>
                                                </td>
                                                <td className="px-6 py-5 whitespace-nowrap text-text-secondary text-sm">
                                                    {patient.prediction_date ? new Date(patient.prediction_date).toLocaleDateString() : 'Just now'}
                                                </td>
                                                <td className="px-8 py-5 whitespace-nowrap text-right">
                                                    <div className="flex items-center justify-end gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                                                        <button className="p-2 text-text-secondary hover:text-primary hover:bg-primary/10 rounded-lg transition-colors" title="Call Patient">
                                                            <Phone className="w-4 h-4" />
                                                        </button>
                                                        <button className="p-2 text-text-secondary hover:text-primary hover:bg-primary/10 rounded-lg transition-colors" title="Send Email">
                                                            <Mail className="w-4 h-4" />
                                                        </button>
                                                        <button className="p-2 text-text-secondary hover:text-primary hover:bg-primary/10 rounded-lg transition-colors" title="Reschedule">
                                                            <Calendar className="w-4 h-4" />
                                                        </button>
                                                        <div className="w-px h-4 bg-border mx-1"></div>
                                                        <button className="p-2 text-text-secondary hover:text-text-primary hover:bg-surface-hover rounded-lg transition-colors">
                                                            <MoreVertical className="w-4 h-4" />
                                                        </button>
                                                    </div>
                                                </td>
                                            </motion.tr>
                                        );
                                    })}
                                </AnimatePresence>
                            ) : (
                                <tr>
                                    <td colSpan={5} className="px-6 py-16 text-center text-text-secondary">
                                        <div className="flex flex-col items-center justify-center">
                                            <div className="w-16 h-16 bg-surface-hover rounded-full flex items-center justify-center mb-4">
                                                <Search className="w-8 h-8 text-text-tertiary" />
                                            </div>
                                            <p className="text-lg font-medium text-text-primary">No patients found</p>
                                            <p className="text-sm">Try adjusting your search or filters.</p>
                                        </div>
                                    </td>
                                </tr>
                            )}
                        </tbody>
                    </table>
                </div>

                {/* Pagination */}
                <div className="p-4 border-t border-surface-hover flex items-center justify-between bg-surface/30">
                    <p className="text-sm text-text-secondary">
                        Showing <span className="font-bold text-text-primary">{(currentPage - 1) * itemsPerPage + 1}</span> to <span className="font-bold text-text-primary">{Math.min(currentPage * itemsPerPage, filteredPatients.length)}</span> of <span className="font-bold text-text-primary">{filteredPatients.length}</span> results
                    </p>
                    <div className="flex items-center gap-2">
                        <button
                            onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                            disabled={currentPage === 1}
                            className="p-2 rounded-lg border border-surface-hover hover:bg-surface-hover disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                        >
                            <ChevronLeft className="w-4 h-4" />
                        </button>
                        <button
                            onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                            disabled={currentPage === totalPages}
                            className="p-2 rounded-lg border border-surface-hover hover:bg-surface-hover disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                        >
                            <ChevronRight className="w-4 h-4" />
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default PatientList;
