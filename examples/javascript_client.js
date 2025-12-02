/**
 * No-Show Prediction API - JavaScript Client Example
 * ===================================================
 * 
 * Usage (Node.js):
 *   node examples/javascript_client.js
 * 
 * Usage (Browser):
 *   Include in HTML and call the functions
 */

const API_URL = process.env.API_URL || 'http://localhost:8000';
const API_PREFIX = '/api/v1';

/**
 * Make a single prediction
 */
async function predict(appointment) {
    const response = await fetch(`${API_URL}${API_PREFIX}/predict`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(appointment),
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(`Prediction failed: ${JSON.stringify(error)}`);
    }

    return response.json();
}

/**
 * Make batch predictions
 */
async function predictBatch(appointments, options = {}) {
    const response = await fetch(`${API_URL}${API_PREFIX}/predict/batch`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            appointments,
            include_explanations: options.includeExplanations || false,
            threshold: options.threshold || null,
        }),
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(`Batch prediction failed: ${JSON.stringify(error)}`);
    }

    return response.json();
}

/**
 * Check API health
 */
async function healthCheck() {
    const response = await fetch(`${API_URL}/health`);
    return response.json();
}

/**
 * Get model information
 */
async function getModelInfo() {
    const response = await fetch(`${API_URL}${API_PREFIX}/model`);
    return response.json();
}

// ==================== Example Usage ====================

async function main() {
    console.log('='.repeat(50));
    console.log('No-Show Prediction API - JavaScript Client');
    console.log('='.repeat(50));
    console.log(`API URL: ${API_URL}\n`);

    try {
        // Health check
        console.log('1. Health Check:');
        const health = await healthCheck();
        console.log(`   Status: ${health.status}`);
        console.log(`   Model Loaded: ${health.model_loaded}\n`);

        // Model info
        console.log('2. Model Info:');
        const modelInfo = await getModelInfo();
        console.log(`   Name: ${modelInfo.name}`);
        console.log(`   Version: ${modelInfo.version}\n`);

        // Single prediction
        console.log('3. Single Prediction:');
        const prediction = await predict({
            age: 35,
            gender: 'F',
            lead_days: 7,
            sms_received: 1,
        });
        console.log(`   Probability: ${(prediction.probability * 100).toFixed(1)}%`);
        console.log(`   Risk Tier: ${prediction.risk.tier}`);
        console.log(`   Intervention: ${prediction.intervention.action}\n`);

        // Batch prediction
        console.log('4. Batch Prediction:');
        const batchResult = await predictBatch([
            { age: 25, gender: 'M', lead_days: 3, sms_received: 1 },
            { age: 45, gender: 'F', lead_days: 14, sms_received: 0 },
            { age: 65, gender: 'M', lead_days: 21, sms_received: 1 },
        ]);
        console.log(`   Total: ${batchResult.summary.total}`);
        console.log(`   Predicted No-Shows: ${batchResult.summary.predicted_noshows}`);
        console.log(`   Average Probability: ${(batchResult.summary.avg_probability * 100).toFixed(1)}%\n`);

        console.log('✅ All examples completed successfully!');

    } catch (error) {
        console.error('❌ Error:', error.message);
        process.exit(1);
    }
}

// Run if executed directly (Node.js)
if (typeof require !== 'undefined' && require.main === module) {
    main();
}

// Export for module usage
if (typeof module !== 'undefined') {
    module.exports = {
        predict,
        predictBatch,
        healthCheck,
        getModelInfo,
    };
}