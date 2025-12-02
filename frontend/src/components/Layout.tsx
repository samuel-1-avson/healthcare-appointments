import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { HeartPulse, Upload, MessageSquare, BarChart3, Moon, Sun, Menu, X } from 'lucide-react';

interface LayoutProps {
    children: React.ReactNode;
    activeTab: string;
    setActiveTab: (tab: any) => void;
    darkMode: boolean;
    toggleDarkMode: () => void;
}

const tabs = [
    { id: 'predict', label: 'Predict', icon: HeartPulse },
    { id: 'batch', label: 'Batch Upload', icon: Upload },
    { id: 'chat', label: 'AI Assistant', icon: MessageSquare },
    { id: 'dashboard', label: 'Dashboard', icon: BarChart3 },
];

export default function Layout({ children, activeTab, setActiveTab, darkMode, toggleDarkMode }: LayoutProps) {
    const [isMobileMenuOpen, setIsMobileMenuOpen] = React.useState(false);

    return (
        <div className={`min-h-screen transition-colors duration-300 ${darkMode ? 'dark bg-slate-900' : 'bg-slate-50'} flex`}>
            {/* Background Effects */}
            <div className="fixed inset-0 z-0 pointer-events-none">
                <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 via-purple-500/5 to-pink-500/5" />
                <div className="absolute top-0 left-1/4 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl opacity-20 animate-pulse" />
                <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl opacity-20 animate-pulse delay-1000" />
            </div>

            {/* Desktop Sidebar */}
            <aside className="hidden md:flex flex-col w-72 fixed inset-y-0 left-0 z-50 bg-white/80 dark:bg-black/40 backdrop-blur-xl border-r border-gray-200 dark:border-white/10">
                <div className="p-6">
                    <div className="flex items-center gap-3 mb-8">
                        <div className="bg-gradient-to-r from-blue-600 to-purple-600 p-2.5 rounded-xl shadow-lg shadow-blue-500/20">
                            <HeartPulse className="w-6 h-6 text-white" />
                        </div>
                        <span className="font-black text-xl bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 dark:from-blue-400 dark:via-purple-400 dark:to-pink-400 bg-clip-text text-transparent">
                            NoShowPredict
                        </span>
                    </div>

                    <nav className="space-y-2">
                        {tabs.map((tab) => {
                            const Icon = tab.icon;
                            const isActive = activeTab === tab.id;
                            return (
                                <button
                                    key={tab.id}
                                    onClick={() => setActiveTab(tab.id)}
                                    className={`w-full relative flex items-center gap-3 px-4 py-3.5 rounded-xl transition-all duration-300 font-medium text-sm group ${isActive
                                        ? 'text-white shadow-lg shadow-blue-500/25'
                                        : 'text-gray-600 dark:text-gray-400 hover:bg-black/5 dark:hover:bg-white/5'
                                        }`}
                                >
                                    {isActive && (
                                        <motion.div
                                            layoutId="activeTabSidebar"
                                            className="absolute inset-0 bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl"
                                            initial={false}
                                            transition={{ type: "spring", stiffness: 500, damping: 30 }}
                                        />
                                    )}
                                    <Icon className={`w-5 h-5 relative z-10 ${isActive ? 'text-white' : 'text-gray-500 dark:text-gray-400 group-hover:text-gray-900 dark:group-hover:text-white'}`} />
                                    <span className={`relative z-10 ${isActive ? 'text-white' : ''}`}>{tab.label}</span>
                                </button>
                            );
                        })}
                    </nav>
                </div>

                <div className="mt-auto p-6 border-t border-gray-200 dark:border-white/10 space-y-2">
                    <button
                        onClick={toggleDarkMode}
                        className="w-full flex items-center gap-3 px-4 py-3 rounded-xl hover:bg-black/5 dark:hover:bg-white/5 text-gray-600 dark:text-gray-400 transition-all"
                    >
                        {darkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
                        <span className="font-medium text-sm">Toggle Theme</span>
                    </button>
                    <div className="flex items-center gap-3 px-4 py-3 text-xs text-gray-400">
                        <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                        System Operational
                    </div>
                </div>
            </aside>

            {/* Mobile Header */}
            <div className="md:hidden fixed top-0 left-0 right-0 z-50 bg-white/80 dark:bg-black/80 backdrop-blur-xl border-b border-gray-200 dark:border-white/10 px-4 h-16 flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <div className="bg-gradient-to-r from-blue-600 to-purple-600 p-1.5 rounded-lg">
                        <HeartPulse className="w-5 h-5 text-white" />
                    </div>
                    <span className="font-bold text-lg text-gray-900 dark:text-white">NSP</span>
                </div>
                <div className="flex items-center gap-2">
                    <button onClick={toggleDarkMode} className="p-2 text-gray-600 dark:text-gray-300">
                        {darkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
                    </button>
                    <button onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)} className="p-2 text-gray-600 dark:text-gray-300">
                        {isMobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
                    </button>
                </div>
            </div>

            {/* Mobile Menu Overlay */}
            <AnimatePresence>
                {isMobileMenuOpen && (
                    <motion.div
                        initial={{ opacity: 0, y: -20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -20 }}
                        className="md:hidden fixed inset-0 z-40 bg-white dark:bg-slate-900 pt-20 px-4"
                    >
                        <nav className="space-y-2">
                            {tabs.map((tab) => {
                                const Icon = tab.icon;
                                const isActive = activeTab === tab.id;
                                return (
                                    <button
                                        key={tab.id}
                                        onClick={() => {
                                            setActiveTab(tab.id);
                                            setIsMobileMenuOpen(false);
                                        }}
                                        className={`w-full flex items-center gap-4 px-4 py-4 rounded-xl transition-all ${isActive
                                            ? 'bg-blue-600 text-white'
                                            : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-white/5'
                                            }`}
                                    >
                                        <Icon className="w-6 h-6" />
                                        <span className="font-medium text-lg">{tab.label}</span>
                                    </button>
                                );
                            })}
                        </nav>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Main Content Area */}
            <main className="flex-1 md:ml-72 min-h-screen relative z-10">
                <div className="max-w-7xl mx-auto px-4 sm:px-8 py-8 md:py-12 mt-16 md:mt-0">
                    <AnimatePresence mode="wait">
                        <motion.div
                            key={activeTab}
                            initial={{ opacity: 0, x: 20 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0, x: -20 }}
                            transition={{ duration: 0.3 }}
                        >
                            {children}
                        </motion.div>
                    </AnimatePresence>
                </div>
            </main>
        </div>
    );
}
