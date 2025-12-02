import { useState } from 'react';
import { Upload } from 'lucide-react';

const BatchUpload = () => {
    const [file, setFile] = useState<File | null>(null);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const selectedFile = e.target.files?.[0];
        if (selectedFile) {
            setFile(selectedFile);
        }
    };

    return (
        <div className="max-w-6xl mx-auto">
            <div className="mb-8 text-center">
                <h1 className="text-4xl font-extrabold text-gray-900 dark:text-white mb-4">
                    Batch <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-indigo-600">Predictions</span>
                </h1>
                <p className="text-xl text-gray-500 dark:text-gray-400">
                    Upload a CSV file (Coming Soon)
                </p>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-8">
                <div className="border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg p-12 text-center">
                    <div className="flex flex-col items-center gap-4">
                        <div className="bg-blue-100 dark:bg-blue-900/30 p-4 rounded-full">
                            <Upload className="w-8 h-8 text-blue-600 dark:text-blue-400" />
                        </div>
                        <div>
                            <p className="text-lg font-semibold text-gray-900 dark:text-white mb-1">
                                {file ? file.name : 'Drop your CSV file here'}
                            </p>
                            <p className="text-sm text-gray-500 dark:text-gray-400">or click to browse</p>
                        </div>
                        <input
                            type="file"
                            accept=".csv"
                            onChange={handleFileChange}
                            className="hidden"
                            id="file-upload"
                        />
                        <label
                            htmlFor="file-upload"
                            className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 cursor-pointer transition-colors"
                        >
                            Select File
                        </label>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default BatchUpload;
