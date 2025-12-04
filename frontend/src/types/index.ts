// Existing types
export interface AppointmentFeatures {
    age: number;
    gender: 'M' | 'F' | 'O' | string;
    scholarship?: number;
    hypertension?: number;
    diabetes?: number;
    alcoholism?: number;
    handicap?: number;
    sms_received?: number;
    lead_days: number;
    neighbourhood?: string;
    appointment_weekday?: string;
    appointment_month?: string;
    patient_historical_noshow_rate?: number;
    patient_total_appointments?: number;
    is_first_appointment?: number;
}

export interface RiskInfo {
    tier: string;
    color: string;
    emoji: string;
    priority: number;
    confidence?: string;
}

export interface InterventionRecommendation {
    action: string;
    priority: string;
    estimated_cost?: number;
    expected_impact?: number;
    sms_reminders?: number;
    phone_call?: boolean;
    notes?: string;
}

export interface FeatureContribution {
    feature: string;
    value: number | string;
    contribution: number;
    direction: 'positive' | 'negative';
}

export interface PredictionExplanation {
    top_risk_factors: FeatureContribution[];
    top_protective_factors: FeatureContribution[];
    summary: string;
}

export interface PredictionResponse {
    prediction: number;
    probability: number;
    risk: RiskInfo;
    intervention: InterventionRecommendation;
    explanation?: PredictionExplanation;
    model_version?: string;
    prediction_id?: string;
    timestamp?: string;
}

export interface ChatMessage {
    role: 'user' | 'assistant' | 'system';
    content: string;
    timestamp?: string;
}

export interface ModelInfo {
    model_name: string;
    model_version: string;
    training_date: string;
    features: string[];
    performance_metrics: {
        accuracy?: number;
        precision?: number;
        recall?: number;
        f1_score?: number;
    };
}

export interface ModelMetrics {
    total_predictions: number;
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
}
