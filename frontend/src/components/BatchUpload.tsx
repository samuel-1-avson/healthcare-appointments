import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileText, CheckCircle, X, Download, CloudUpload } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import Papa from 'papaparse';
import { batchPredict } from '../services/api';
import type { AppointmentFeatures, PredictionResponse } from '../types';

const BatchUpload = () => {
    const [files, setFiles] = useState<File[]>([]);
    const [uploading, setUploading] = useState(false);
    const [progress, setProgress] = useState(0);
    const [completed, setCompleted] = useState(false);
    const [results, setResults] = useState<PredictionResponse[]>([]);
    const [error, setError] = useState<string | null>(null);

    const onDrop = useCallback((acceptedFiles: File[]) => {
        setFiles(acceptedFiles);
        setCompleted(false);
        setProgress(0);
        setResults([]);
        setError(null);
    }, []);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: {
            'text/csv': ['.csv']
        },
        maxFiles: 1
    });

    const handleUpload = async () => {
        if (files.length === 0) return;

        setUploading(true);
        setError(null);
        setProgress(10); // Start progress

        const file = files[0];

        Papa.parse(file, {
            header: true,
            dynamicTyping: true,
            skipEmptyLines: true,
            complete: async (results) => {
                try {
                    setProgress(50); // Parsed
                    const appointments = results.data as AppointmentFeatures[];
                    console.log('Parsed appointments:', appointments);

                    // Basic validation to ensure required fields exist
                    if (!appointments.length || appointments[0].age === undefined || appointments[0].lead_days === undefined) {
                        throw new Error("Invalid CSV format. Please ensure headers match required fields (age, gender, lead_days, etc.)");
                    }

                    const response = await batchPredict({ appointments });
                    setResults(response.predictions);
                    setProgress(100);
                    setCompleted(true);
                } catch (err: any) {
                    console.error("Batch prediction failed", err);
                    setError(err.message || "Failed to process batch predictions");
                    setProgress(0);
                } finally {
                    setUploading(false);
                }
            },
            error: (err) => {
                console.error("CSV Parse Error", err);
                setError("Failed to parse CSV file");
                setUploading(false);
            }
        });
    };

    const removeFile = () => {
        setFiles([]);
        setCompleted(false);
        setProgress(0);
        setResults([]);
        setError(null);
    };

    return (
        <div className="max-w-5xl mx-auto space-y-8">
            <div className="text-center">
                <h1 className="text-3xl font-bold text-text-primary mb-2">Batch Prediction</h1>
                <p className="text-text-secondary">Upload a CSV file to generate predictions for multiple appointments at once.</p>
            </div>

            <div className="glass-card p-8 rounded-3xl">
                <div
                    {...getRootProps()}
                    className={`border-2 border-dashed rounded-2xl p-12 text-center transition-all cursor-pointer ${isDragActive
                        ? 'border-primary bg-primary/5'
                        : 'border-surface-hover hover:border-primary/50 hover:bg-surface-hover/30'
                        }`}
                >
                    <input {...getInputProps()} />
                    <div className="flex flex-col items-center gap-4">
                        <div className={`p-4 rounded-full ${isDragActive ? 'bg-primary/20' : 'bg-surface border border-surface-hover'}`}>
                            <CloudUpload className={`w-8 h-8 ${isDragActive ? 'text-primary' : 'text-text-secondary'}`} />
                        </div>
                        <div>
                            <p className="text-lg font-medium text-text-primary mb-1">
                                {isDragActive ? 'Drop the file here' : 'Drag & drop your CSV file here'}
                            </p>
                            <p className="text-sm text-text-secondary">or click to browse</p>
                        </div>
                        <p className="text-xs text-text-secondary mt-2">Supported format: .csv (headers: age, gender, lead_days, etc.)</p>
                    </div>
                </div>

                <AnimatePresence>
                    {error && (
                        <motion.div
                            key="error-message"
                            initial={{ opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: 'auto' }}
                            exit={{ opacity: 0, height: 0 }}
                            className="mt-4 p-4 bg-error/10 text-error rounded-xl text-sm font-medium text-center"
                        >
                            {error}
                        </motion.div>
                    )}

                    {files.length > 0 && (
                        <motion.div
                            key="file-upload-progress"
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -20 }}
                            className="mt-6 bg-surface border border-surface-hover rounded-xl p-4"
                        >
                            <div className="flex items-center justify-between mb-4">
                                <div className="flex items-center gap-3">
                                    <div className="p-2 bg-primary/10 rounded-lg">
                                        <FileText className="w-5 h-5 text-primary" />
                                    </div>
                                    <div>
                                        <p className="font-medium text-text-primary">{files[0].name}</p>
                                        <p className="text-xs text-text-secondary">{(files[0].size / 1024).toFixed(2)} KB</p>
                                    </div>
                                </div>
                                {!uploading && !completed && (
                                    <button
                                        onClick={(e) => { e.stopPropagation(); removeFile(); }}
                                        className="p-1 hover:bg-surface-hover rounded-full text-text-secondary hover:text-error transition-colors"
                                    >
                                        <X className="w-5 h-5" />
                                    </button>
                                )}
                            </div>

                            {(uploading || completed) && (
                                <div className="space-y-2">
                                    <div className="flex justify-between text-xs font-medium">
                                        <span className={completed ? 'text-success' : 'text-primary'}>
                                            {completed ? 'Processing Complete' : 'Processing...'}
                                        </span>
                                        <span className="text-text-secondary">{progress}%</span>
                                    </div>
                                    <div className="h-2 bg-surface-hover rounded-full overflow-hidden">
                                        <div
                                            className={`h-full transition-all duration-300 ${completed ? 'bg-success' : 'bg-primary'}`}
                                            style={{ width: `${progress}%` }}
                                        />
                                    </div>
                                </div>
                            )}
                        </motion.div>
                    )}
                </AnimatePresence>

                <div className="mt-8 flex justify-end gap-4">
                    <button
                        onClick={() => alert("Template download coming soon!")}
                        className="px-6 py-3 text-text-secondary font-medium hover:text-text-primary transition-colors"
                    >
                        Download Template
                    </button>
                    <button
                        onClick={handleUpload}
                        disabled={files.length === 0 || uploading || completed}
                        className="px-6 py-3 bg-primary hover:bg-primary-hover disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-xl font-medium transition-all shadow-lg shadow-primary/25 flex items-center gap-2"
                    >
                        {completed ? (
                            <>
                                <CheckCircle className="w-5 h-5" />
                                Processed
                            </>
                        ) : (
                            <>
                                <Upload className="w-5 h-5" />
                                {uploading ? 'Processing...' : 'Upload & Predict'}
                            </>
                        )}
                    </button>
                </div>
            </div>

            {completed && results.length > 0 && (
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="glass-card p-8 rounded-3xl"
                >
                    <div className="flex items-center justify-between mb-6">
                        <h3 className="text-lg font-bold text-text-primary">Prediction Results</h3>
                        <button className="flex items-center gap-2 text-primary hover:text-primary-hover font-medium text-sm">
                            <Download className="w-4 h-4" />
                            Export Results
                        </button>
                    </div>

                    <div className="overflow-x-auto">
                        <table className="w-full text-left border-collapse">
                            <thead>
                                <tr className="border-b border-surface-hover">
                                    <th className="py-3 px-4 text-sm font-medium text-text-secondary">Patient ID</th>
                                    <th className="py-3 px-4 text-sm font-medium text-text-secondary">Risk Level</th>
                                    <th className="py-3 px-4 text-sm font-medium text-text-secondary">Probability</th>
                                    <th className="py-3 px-4 text-sm font-medium text-text-secondary">Action</th>
                                </tr>
                            </thead>
                            <tbody className="text-sm">
                                {results.map((result, i) => (
                                    <tr key={i} className="border-b border-surface-hover hover:bg-surface-hover/30 transition-colors">
                                        <td className="py-3 px-4 text-text-primary">#{i + 1}</td>
                                        <td className="py-3 px-4">
                                            <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${result.risk.tier === 'HIGH' || result.risk.tier === 'CRITICAL' ? 'bg-error/10 text-error' :
                                                result.risk.tier === 'MEDIUM' ? 'bg-warning/10 text-warning' :
                                                    'bg-success/10 text-success'
                                                }`}>
                                                {result.risk.tier}
                                            </span>
                                        </td>
                                        <td className="py-3 px-4 text-text-primary">
                                            {(result.probability * 100).toFixed(1)}%
                                        </td>
                                        <td className="py-3 px-4 text-text-primary">
                                            {result.intervention.action}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </motion.div>
            )}
        </div>
    );
};

export default BatchUpload;
