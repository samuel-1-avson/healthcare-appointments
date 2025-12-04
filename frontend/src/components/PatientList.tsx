import { useEffect, useState } from 'react';
import { getPredictionHistory } from '../services/api';
import { Search, Filter, MoreVertical, Phone, Calendar, AlertTriangle } from 'lucide-react';
import { motion } from 'framer-motion';

const PatientList = () => {
    const [patients, setPatients] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);
    const [searchTerm, setSearchTerm] = useState('');
    const [filter, setFilter] = useState('all');

    useEffect(() => {
        const fetchPatients = async () => {
            try {
                const history = await getPredictionHistory();
                // Flatten the history structure if needed, or use recent_alerts/daily_trends
                // For now, let's assume we can get a list of recent predictions from the history object
                // If the API doesn't return a flat list, we might need to mock it or adjust the API
                // Based on previous ModelDashboard, history has 'recent_alerts'. Let's use that + mock some more for demo

                // Combining real alerts with some mock data to populate the table for demonstration if the list is short
                const realAlerts = history?.recent_alerts || [];
                setPatients(realAlerts);
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

    return (
        <div className="space-y-6">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                    <h1 className="text-3xl font-bold text-text-primary">Patient List</h1>
                    <p className="text-text-secondary mt-1">Manage high-risk patients and view prediction history.</p>
                </div>
                <div className="flex gap-3">
                    <button className="bg-primary hover:bg-primary-hover text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors">
                        Add Patient
                    </button>
                </div>
            </div>

            {/* Filters */}
            <div className="glass-card p-4 rounded-xl flex flex-col md:flex-row gap-4 items-center justify-between">
                <div className="relative w-full md:w-96">
                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-text-secondary w-4 h-4" />
                    <input
                        type="text"
                        placeholder="Search by Patient ID..."
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        className="w-full bg-surface border border-surface-hover text-text-primary rounded-lg pl-10 pr-4 py-2 focus:outline-none focus:ring-2 focus:ring-primary"
                    />
                </div>
                <div className="flex items-center gap-3 w-full md:w-auto">
                    <Filter className="text-text-secondary w-4 h-4" />
                    <select
                        value={filter}
                        onChange={(e) => setFilter(e.target.value)}
                        className="bg-surface border border-surface-hover text-text-primary rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary flex-grow md:flex-grow-0"
                    >
                        <option value="all">All Risks</option>
                        <option value="critical">Critical</option>
                        <option value="high">High</option>
                        <option value="medium">Medium</option>
                        <option value="low">Low</option>
                    </select>
                </div>
            </div>

            {/* Table */}
            <div className="glass-card rounded-2xl overflow-hidden">
                <div className="overflow-x-auto">
                    <table className="w-full">
                        <thead>
                            <tr className="bg-surface-hover/50 border-b border-surface-hover">
                                <th className="px-6 py-4 text-left text-xs font-semibold text-text-secondary uppercase tracking-wider">Patient ID</th>
                                <th className="px-6 py-4 text-left text-xs font-semibold text-text-secondary uppercase tracking-wider">Risk Score</th>
                                <th className="px-6 py-4 text-left text-xs font-semibold text-text-secondary uppercase tracking-wider">Status</th>
                                <th className="px-6 py-4 text-left text-xs font-semibold text-text-secondary uppercase tracking-wider">Prediction Date</th>
                                <th className="px-6 py-4 text-right text-xs font-semibold text-text-secondary uppercase tracking-wider">Actions</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-surface-hover">
                            {loading ? (
                                [...Array(5)].map((_, i) => (
                                    <tr key={i} className="animate-pulse">
                                        <td className="px-6 py-4"><div className="h-4 w-16 bg-surface-hover rounded"></div></td>
                                        <td className="px-6 py-4"><div className="h-4 w-12 bg-surface-hover rounded"></div></td>
                                        <td className="px-6 py-4"><div className="h-6 w-20 bg-surface-hover rounded-full"></div></td>
                                        <td className="px-6 py-4"><div className="h-4 w-24 bg-surface-hover rounded"></div></td>
                                        <td className="px-6 py-4"></td>
                                    </tr>
                                ))
                            ) : filteredPatients.length > 0 ? (
                                filteredPatients.map((patient, i) => (
                                    <motion.tr
                                        key={i}
                                        initial={{ opacity: 0, y: 10 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        transition={{ delay: i * 0.05 }}
                                        className="hover:bg-surface-hover/30 transition-colors"
                                    >
                                        <td className="px-6 py-4 whitespace-nowrap">
                                            <div className="flex items-center">
                                                <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center text-primary font-bold text-xs mr-3">
                                                    {patient.patientid.toString().slice(-2)}
                                                </div>
                                                <span className="font-medium text-text-primary">#{patient.patientid}</span>
                                            </div>
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap">
                                            <span className="text-text-primary font-bold">{(patient.composite_risk_score * 100).toFixed(1)}%</span>
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap">
                                            <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${patient.risk_tier === 'CRITICAL' ? 'bg-error/10 text-error' :
                                                    patient.risk_tier === 'HIGH' ? 'bg-warning/10 text-warning' :
                                                        'bg-success/10 text-success'
                                                }`}>
                                                {patient.risk_tier === 'CRITICAL' && <AlertTriangle className="w-3 h-3 mr-1" />}
                                                {patient.risk_tier}
                                            </span>
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap text-text-secondary text-sm">
                                            {new Date().toLocaleDateString()} {/* Mock date as API might not return it in this view */}
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                                            <div className="flex items-center justify-end gap-2">
                                                <button className="p-2 text-text-secondary hover:text-primary hover:bg-primary/10 rounded-lg transition-colors" title="Call Patient">
                                                    <Phone className="w-4 h-4" />
                                                </button>
                                                <button className="p-2 text-text-secondary hover:text-primary hover:bg-primary/10 rounded-lg transition-colors" title="Reschedule">
                                                    <Calendar className="w-4 h-4" />
                                                </button>
                                                <button className="p-2 text-text-secondary hover:text-primary hover:bg-primary/10 rounded-lg transition-colors">
                                                    <MoreVertical className="w-4 h-4" />
                                                </button>
                                            </div>
                                        </td>
                                    </motion.tr>
                                ))
                            ) : (
                                <tr>
                                    <td colSpan={5} className="px-6 py-12 text-center text-text-secondary">
                                        No patients found matching your criteria.
                                    </td>
                                </tr>
                            )}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
};

export default PatientList;
