import json
import numpy as np
import os
import itertools
import re
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Any, Tuple
from gensim.models import Word2Vec
from agents.parser import ResumeParser as AgentA
from agents.technical_scorer import TechnicalSkillScorer as AgentB
from agents.experience_scorer import ExperienceRelevanceScorer as AgentC
from agents.soft_skills_scorer import SoftSkillsScorer as AgentD
from utils.data_generator import generate_sample_job_description, generate_candidate_resumes
from collections import defaultdict
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut, KFold
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.signal import find_peaks
import seaborn as sns

# Configuration for number of agents in each category
n_A = 1  # Number of Agent A instances
n_combinations = 1000  # Number of agent combinations to generate

# Weights and embedding configuration
WEIGHTS_FILE = "weights.json"
EMBEDDING_DIM = 100  # Dimension for Word2Vec embeddings
WEIGHT_MIN = 1  # Minimum value for uniform distribution
WEIGHT_MAX = 10.0   # Maximum value for uniform distribution
WORD2VEC_WINDOW = 5  # Window size for Word2Vec model
WORD2VEC_MIN_COUNT = 1  # Minimum count for Word2Vec model

# Analysis output configuration
OUTPUT_DIR = "analysis_output"
VISUALIZATIONS_DIR = os.path.join(OUTPUT_DIR, "visualizations")
CSV_DIR = os.path.join(OUTPUT_DIR, "csv")

# Sigmoid transformation and filtering parameters
NORMALIZATION_CONSTANT_L = 1  # Normalization factor for accuracy scores
FILTERING_INTERVAL_WIDTH_A = 0.2  # Width of exclusion interval around 0.5

# Scatter plot visualization parameters
SCATTER_ALPHA = 0.8  # Transparency for scatter plot points
SCATTER_SIZE = 10   # Size of scatter plot points

# Neural Network configuration parameters
NN_OUTPUT_MODE = 3  # 1: Regression, 2: Binary Classification, 3: Three-Class Classification
NN_HIDDEN_LAYERS = [27, 9, 3]  # Hidden layer sizes
NN_EPOCHS = 100  # Number of training epochs
NN_BATCH_SIZE = 32  # Batch size for training
NN_VALIDATION_SPLIT = 0.2  # Fraction of data to use for validation
NN_LEARNING_RATE = 0.001  # Learning rate for Adam optimizer
NN_RANDOM_STATE = 42  # Random state for reproducibility

# Create a logger for storing processing details
class Logger:
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        # Initialize or clear the log file
        with open(log_file_path, 'w', encoding='utf-8') as f:
            f.write("===== RESUME SCORING SYSTEM PROCESSING DETAILS =====\n\n")
    
    def log(self, message):
        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            f.write(f"{message}\n")
    
    def log_json(self, data, title=None):
        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            if title:
                f.write(f"\n{title}:\n")
            f.write(json.dumps(data, indent=2) + "\n")

# Initialize logger
logger = Logger(os.path.join("logs", "processing_details.log"))


def generate_random_weights(n_weights: int, mean: float = 1.0, variance: float = 0.05) -> List[float]:
    """
    Generate random weights from a normal distribution
    
    Args:
        n_weights (int): Number of weights to generate
        mean (float): Mean of the normal distribution
        variance (float): Variance of the normal distribution
        
    Returns:
        List[float]: List of randomly generated weights
    """
    std_dev = np.sqrt(variance)
    weights = np.random.normal(mean, std_dev, n_weights)
    
    # Ensure all weights are positive
    weights = np.maximum(weights, 0.1)
    
    return weights.tolist()

def extract_technical_skills_features():
    """
    Extract technical skills features to use as keys in Agent B weights
    
    Returns:
        List[str]: List of technical skills
    """
    return [
        "python", "java", "javascript", "typescript", "c++", "c#", "ruby", "php", "swift",
        "kotlin", "go", "rust", "scala", "perl", "html", "css", "sql", "r", "matlab",
        "react", "angular", "vue", "node.js", "django", "flask", "spring", "express",
        "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy", "kubernetes", "docker",
        "aws", "azure", "gcp", "git", "jenkins", "ci/cd", "agile", "scrum", "jira",
        "mongodb", "postgresql", "mysql", "oracle", "redis", "elasticsearch", "nosql"
    ]

def extract_experience_features():
    """
    Extract experience features to use as keys in Agent C weights
    
    Returns:
        List[str]: List of experience features
    """
    return [
        "years_experience", "relevant_role", "industry_match", "responsibilities_match",
        "senior_leadership", "team_management", "project_management", "technical_expertise",
        "domain_knowledge", "achievements", "career_progression", "company_tier"
    ]

def extract_soft_skills_features():
    """
    Extract soft skills features to use as keys in Agent D weights
    
    Returns:
        List[str]: List of soft skills
    """
    return [
        "communication", "teamwork", "leadership", "problem-solving", "adaptability",
        "time_management", "conflict_resolution", "creativity", "emotional_intelligence",
        "critical_thinking", "decision_making", "negotiation", "presentation", "interpersonal",
        "organization", "flexibility", "collaboration", "customer_service", "mentoring",
        "initiative", "analytical"
    ]

def create_word2vec_model(job_description: str) -> Tuple[Word2Vec, List[float]]:
    """
    Create a Word2Vec model from job description
    
    Args:
        job_description (str): The job description text
        
    Returns:
        Tuple[Word2Vec, List[float]]: Word2Vec model and embedding of the job description
    """
    # Tokenize the job description
    sentences = [re.findall(r'\w+', job_description.lower())]
    
    # Train Word2Vec model
    model = Word2Vec(sentences, vector_size=EMBEDDING_DIM, window=WORD2VEC_WINDOW, 
                    min_count=WORD2VEC_MIN_COUNT, workers=4)
    
    # Get job description embedding
    words = job_description.lower().split()
    word_vectors = []
    for word in words:
        if word in model.wv:
            word_vectors.append(model.wv[word])
    
    # If no word vectors found, return a zero vector
    if not word_vectors:
        return model, np.zeros(EMBEDDING_DIM).tolist()
    
    # Calculate average embedding vector
    jd_embedding = np.mean(word_vectors, axis=0).tolist()
    
    return model, jd_embedding

def generate_combination_weights(n_combinations, b_features, c_features, d_features):
    """
    Generate weights for agent combinations directly
    
    Args:
        n_combinations (int): Number of combinations to generate
        b_features (int): Number of Agent B features
        c_features (int): Number of Agent C features
        d_features (int): Number of Agent D features
        
    Returns:
        List[List[float]]: List of combination weight vectors
    """
    combination_weights = []
    total_features = b_features + c_features + d_features
    
    for i in range(n_combinations):
        # Generate weights from uniform distribution in range [-10, 10]
        weights = np.random.uniform(WEIGHT_MIN, WEIGHT_MAX, total_features)
        combination_weights.append(weights.tolist())
    
    return combination_weights

def load_or_generate_weights(job_description: str):
    """
    Load weights from JSON file if it exists, otherwise generate new weights and save to file
    
    Args:
        job_description (str): The job description text to generate embeddings for
        
    Returns:
        dict: Dictionary containing weights for agent combinations and job description embedding
    """
    if os.path.exists(WEIGHTS_FILE):
        # Load existing weights
        with open(WEIGHTS_FILE, 'r') as f:
            data = json.load(f)
        logger.log(f"Loaded existing weights and embedding from {WEIGHTS_FILE}")
        return data
    else:
        # Generate Word2Vec model and job description embedding
        word2vec_model, jd_embedding = create_word2vec_model(job_description)
        
        # Get feature lists for each agent type
        tech_features = extract_technical_skills_features()
        exp_features = extract_experience_features()
        soft_features = extract_soft_skills_features()
        
        # Get feature counts
        b_features = len(tech_features)
        c_features = len(exp_features)
        d_features = len(soft_features)
        
        # Generate combination-level weights
        combination_weights = generate_combination_weights(n_combinations, b_features, c_features, d_features)
        
        # Create data structure
        data = {
            "jd_embedding": jd_embedding,
            "combination_weights": combination_weights,
            "feature_counts": {
                "b_features": b_features,
                "c_features": c_features,
                "d_features": d_features
            },
            "feature_names": {
                "b_features": tech_features,
                "c_features": exp_features,
                "d_features": soft_features
            }
        }
        
        # Save weights to file
        with open(WEIGHTS_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.log(f"Generated {n_combinations} combination weight vectors, saved to {WEIGHTS_FILE}")
        logger.log(f"Feature counts: B={b_features}, C={c_features}, D={d_features}")
        return data

def create_agents(weights_data, word2vec_model, jd_embedding):
    """
    Create agent instances for evaluation
    
    Args:
        weights_data (dict): Dictionary of weights and feature information
        word2vec_model (Word2Vec): Word2Vec model for embedding
        jd_embedding (List[float]): Job description embedding
    
    Returns:
        tuple: (agent_A_list, feature_counts, feature_names)
    """
    agent_A_list = []
    
    # Create Agent A instances (no weights needed)
    for i in range(n_A):
        agent_id = f"A{i+1}"
        agent_A_list.append((agent_id, AgentA(word2vec_model=word2vec_model)))
    
    # Get feature information
    feature_counts = weights_data["feature_counts"]
    feature_names = weights_data["feature_names"]
    
    return agent_A_list, feature_counts, feature_names

def score_single_resume_with_combination(
    resume_text: str, 
    job_description: str, 
    combination_weights: List[float],
    feature_counts: dict,
    feature_names: dict,
    agent_A: Tuple[str, AgentA], 
    word2vec_model,
    jd_embedding: List[float],
    candidate_index: int = None,
    combination_index: int = None
) -> Tuple[float, Dict[str, Any]]:
    """
    Score a single resume against a job description using the specified combination weights
    
    Args:
        resume_text (str): The resume text
        job_description (str): The job description text
        combination_weights (List[float]): Combined weight vector for B, C, D agents
        feature_counts (dict): Dictionary with feature counts for each agent
        feature_names (dict): Dictionary with feature names for each agent
        agent_A (tuple): (agent_id, agent_instance) for Agent A
        word2vec_model: Word2Vec model
        jd_embedding (List[float]): Job description embedding
        candidate_index (int): Index of the candidate for logging purposes
        combination_index (int): Index of the combination for logging purposes
        
    Returns:
        tuple: (final_score, parsed_resume)
    """
    agent_A_id, agent_A_instance = agent_A
    
    # Extract feature counts
    b_features = feature_counts["b_features"]
    c_features = feature_counts["c_features"]
    d_features = feature_counts["d_features"]
    
    # Split the combination weights for each agent
    b_weights = combination_weights[:b_features]
    c_weights = combination_weights[b_features:b_features + c_features]
    d_weights = combination_weights[b_features + c_features:]
    
    # Create structured weights dictionaries
    b_structured_weights = dict(zip(feature_names["b_features"], b_weights))
    c_structured_weights = dict(zip(feature_names["c_features"], c_weights))
    d_structured_weights = dict(zip(feature_names["d_features"], d_weights))
    
    # Create agent instances with the split weights
    agent_B_instance = AgentB(
        weights=b_weights,
        word2vec_model=word2vec_model,
        jd_embedding=jd_embedding,
        structured_weights=b_structured_weights
    )
    
    agent_C_instance = AgentC(
        weights=c_weights,
        word2vec_model=word2vec_model,
        jd_embedding=jd_embedding,
        structured_weights=c_structured_weights
    )
    
    agent_D_instance = AgentD(
        weights=d_weights,
        word2vec_model=word2vec_model,
        jd_embedding=jd_embedding,
        structured_weights=d_structured_weights
    )
    
    # Log separator and candidate info
    candidate_name = resume_text.strip().split("\n")[0]
    logger.log(f"\n{'='*50}")
    logger.log(f"CANDIDATE {candidate_index}: {candidate_name}")
    logger.log(f"COMBINATION {combination_index}: {agent_A_id}")
    logger.log(f"{'='*50}\n")
    
    # Log resume excerpt
    logger.log("RESUME EXCERPT:")
    logger.log(resume_text[:200] + "...\n")
    
    # Agent A: Parse the resume
    logger.log(f"Running Agent {agent_A_id}: Resume Parser...")
    parsed_resume = agent_A_instance.process(resume_text)
    logger.log_json(parsed_resume, "Parsed Resume Data")
    
    # Agent B: Technical Skill Scorer
    logger.log(f"\nRunning Agent B: Technical Skill Scorer...")
    s1, s1_technical_match_details = agent_B_instance.process(parsed_resume["technical_skills"], job_description, resume_text)
    logger.log(f"technical match details: {s1_technical_match_details}")
    logger.log(f"Technical Skill Match Score (S1): {s1:.2f}")
    
    # Agent C: Experience Relevance Scorer
    logger.log(f"\nRunning Agent C: Experience Relevance Scorer...")
    s2, s2_justification = agent_C_instance.process(parsed_resume["job_experience"], job_description)
    logger.log(f"Experience Relevance Score (S2): {s2:.2f}")
    logger.log(f"Justification: {s2_justification}")
    
    # Agent D: Soft Skills Scorer
    logger.log(f"\nRunning Agent D: Soft Skills Scorer...")
    s3, s3_justification = agent_D_instance.process(resume_text, job_description)
    logger.log(f"Soft Skills Score (S3): {s3:.2f}")
    logger.log(f"Justification: {s3_justification}")
    
    # Aggregator
    final_score = s1 + s2 + s3
    
    logger.log("\nFINAL RESULTS:")
    logger.log(f"Technical Skill Match (S1): {s1:.2f}")
    logger.log(f"Experience Relevance (S2): {s2:.2f}")
    logger.log(f"Soft Skills (S3): {s3:.2f}")
    logger.log(f"Final Score: {final_score:.2f}/3.00")
    
    return final_score, parsed_resume

def evaluate_ranking_accuracy(human_ranking, scores):
    """
    Calculate the accuracy of the system ranking compared to human ranking using linear regression,
    then apply normalization and sigmoid transformation
    
    Args:
        human_ranking (list): Human's ranking of candidates
        scores (list): System-generated scores for each resume
        
    Returns:
        dict: Accuracy metrics with both raw and transformed values
    """
    # Get the number of resumes
    n = len(human_ranking)
    
    # Create x values as position numbers (1 to n)
    x = [1, 2, 3, 4, 5]
    
    # Create y values as scores corresponding to human ranking
    # For each position in human ranking, get the corresponding resume's score
    y = []
    score_map = {candidate: score for candidate, score in enumerate(scores, 1)}
    for candidate in human_ranking:
        y.append(score_map[candidate])
    
    # Fit linear regression
    slope, intercept = np.polyfit(x, y, 1)
    
    # Calculate raw accuracy as negative slope
    raw_accuracy = -slope
    
    # # Apply normalization and sigmoid transformation
    # normalized_accuracy = raw_accuracy / NORMALIZATION_CONSTANT_L
    
    # # Apply sigmoid transformation
    # def sigmoid(x):
    #     return 1 / (1 + np.exp(-x))
    
    # transformed_accuracy = sigmoid(normalized_accuracy)
    
    return {
        "accuracy_raw": raw_accuracy,
        # "accuracy_normalized": normalized_accuracy,
        # "accuracy": transformed_accuracy  # This is the final transformed accuracy
    }

def run_evaluation_for_combination(combination_weights, combination_index, resumes, job_description, human_ranking, feature_counts, feature_names, agent_A_list, word2vec_model, jd_embedding):
    """
    Run the performance evaluation pipeline for a specific agent combination
    
    Args:
        combination_weights (List[float]): Weight vector for the combination
        combination_index (int): Index of the combination
        resumes (list): List of resume texts
        job_description (str): Job description
        human_ranking (list): Human expert ranking
        feature_counts (dict): Feature counts for each agent type
        feature_names (dict): Feature names for each agent type
        agent_A_list: List of Agent A instances
        word2vec_model: Word2Vec model
        jd_embedding: Job description embedding
        
    Returns:
        dict: Evaluation results
    """
    agent_A = agent_A_list[0]  # Use the first (and only) Agent A instance
    agent_A_id = agent_A[0]
    
    combination_name = f"Combination_{combination_index+1}"

    logger.log(f"\n\n{'='*30} EVALUATING COMBINATION: {combination_name} {'='*30}\n")
    
    # Score each resume and collect results
    scores = []
    candidate_names = []
    
    for i, resume in enumerate(resumes):
        candidate_name = resume.strip().split("\n")[0]
        candidate_names.append(candidate_name)
        
        score, _ = score_single_resume_with_combination(
            resume, 
            job_description, 
            combination_weights,
            feature_counts,
            feature_names,
            agent_A,
            word2vec_model,
            jd_embedding,
            candidate_index=i+1,
            combination_index=combination_index+1
        )
        
        # Print only the scores to terminal (no processing details)
        logger.log(f"Final Score: {score:.2f}/3.00")
        
        scores.append(score)
    
    # Create ranking based on scores
    # Sort candidate indices by score in descending order
    ranked_indices = [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]
    # Convert to 1-indexed for display
    system_ranking = [i+1 for i in ranked_indices]
    
    # Calculate accuracy
    accuracy_metrics = evaluate_ranking_accuracy(human_ranking, scores)
    
    # Log the rankings and accuracy to the log file
    logger.log(f"\nSystem Ranking: {system_ranking}")
    logger.log(f"Human Ranking:  {human_ranking}")
    logger.log(f"\nRaw Accuracy (negative slope): {accuracy_metrics['accuracy_raw']:.4f}")
    # logger.log(f"Normalized Accuracy: {accuracy_metrics['accuracy_normalized']:.4f}")
    # logger.log(f"Transformed Accuracy (sigmoid): {accuracy_metrics['accuracy']:.4f}")
    
    return {
        "combination": combination_name,
        "combination_index": combination_index,
        "combination_weights": combination_weights,
        "scores": scores,
        "system_ranking": system_ranking,
        "human_ranking": human_ranking,
        "accuracy_raw": accuracy_metrics['accuracy_raw'],
        # "accuracy_normalized": accuracy_metrics['accuracy_normalized'],
        # "accuracy": accuracy_metrics['accuracy']  # This is the transformed accuracy
    }

def compute_comprehensive_pls_metrics(X, y, agent_name, feature_names=None, max_components=None, cv_folds=5, n_permutations=1000):
    """
    Compute comprehensive PLS evaluation metrics for a given dataset
    
    Args:
        X (np.array): Input features (n_samples x n_features)
        y (np.array): Target variable (n_samples,)
        agent_name (str): Name of the agent/model for logging
        feature_names (list): Names of input features for VIP analysis
        max_components (int): Maximum number of components to test
        cv_folds (int): Number of cross-validation folds
        n_permutations (int): Number of permutations for significance test
        
    Returns:
        dict: Comprehensive metrics dictionary
    """
    logger.log(f"\n{'='*20} PLS EVALUATION: {agent_name} {'='*20}")
    
    n_samples, n_features = X.shape
    if max_components is None:
        max_components = min(n_samples - 1, n_features, 10)  # Reasonable upper bound
    
    # Standardize the data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    
    # Initialize storage for component-wise metrics
    r2x_cum = []
    r2y_cum = []
    q2_scores = []
    press_scores = []
    rmsecv_scores = []
    
    # Test different numbers of components
    component_range = range(1, max_components + 1)
    
    # Split data for final test set evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )
    
    best_q2 = -np.inf
    optimal_components = 1
    
    for n_comp in component_range:
        # Fit PLS model
        pls = PLSRegression(n_components=n_comp)
        pls.fit(X_train, y_train)
        
        # 1. Goodness of Fit Metrics
        
        # R²X (cumulative): Variance explained in X
        X_scores = pls.transform(X_train)
        X_reconstructed = X_scores @ pls.x_loadings_.T
        ss_res_x = np.sum((X_train - X_reconstructed) ** 2)
        ss_tot_x = np.sum((X_train - np.mean(X_train, axis=0)) ** 2)
        r2x = 1 - (ss_res_x / ss_tot_x)
        r2x_cum.append(r2x)
        
        # R²Y (cumulative): Variance explained in Y
        y_pred_train = pls.predict(X_train).ravel()
        r2y = r2_score(y_train, y_pred_train)
        r2y_cum.append(r2y)
        
        # 2. Predictive Power Metrics
        
        # Cross-validation Q² and RMSECV
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        y_pred_cv = np.zeros_like(y_train)
        
        for train_idx, val_idx in cv.split(X_train):
            pls_cv = PLSRegression(n_components=n_comp)
            pls_cv.fit(X_train[train_idx], y_train[train_idx])
            y_pred_cv[val_idx] = pls_cv.predict(X_train[val_idx]).ravel()
        
        # Q² calculation
        press = np.sum((y_train - y_pred_cv) ** 2)
        ss_tot_y = np.sum((y_train - np.mean(y_train)) ** 2)
        q2 = 1 - (press / ss_tot_y)
        
        q2_scores.append(q2)
        press_scores.append(press)
        rmsecv_scores.append(np.sqrt(mean_squared_error(y_train, y_pred_cv)))
        
        # Track best Q² for optimal component selection
        if q2 > best_q2:
            best_q2 = q2
            optimal_components = n_comp
    
    # Fit final model with optimal components
    pls_final = PLSRegression(n_components=optimal_components)
    pls_final.fit(X_train, y_train)
    
    # RMSEP on test set
    y_pred_test = pls_final.predict(X_test).ravel()
    rmsep = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    # 3. Diagnostic Correlation Check
    X_scores_first = pls_final.transform(X_train)[:, 0]
    correlation_coeff, correlation_p = pearsonr(X_scores_first, y_train)
    
    # 4. Component Selection and Elbow Identification
    
    # Find elbow point in PRESS curve
    press_array = np.array(press_scores)
    # Use negative PRESS for peak finding (to find minimum)
    peaks, _ = find_peaks(-press_array)
    if len(peaks) > 0:
        elbow_component = peaks[0] + 1  # Convert to 1-indexed
    else:
        # Fallback: find point where PRESS stops decreasing significantly
        press_diff = np.diff(press_array)
        elbow_idx = np.where(press_diff > -0.01 * press_array[0])[0]
        elbow_component = elbow_idx[0] + 1 if len(elbow_idx) > 0 else optimal_components
    
    # 5. Variable Relevance (VIP Scores)
    
    def compute_vip_scores(pls_model, X_data):
        """Compute VIP scores for PLS model"""
        T = pls_model.x_scores_  # X scores
        W = pls_model.x_weights_  # X weights
        Q = pls_model.y_loadings_  # Y loadings
        
        # Number of variables and components
        p = X_data.shape[1]
        h = pls_model.n_components
        
        # Calculate VIP scores
        vip_scores = np.zeros(p)
        
        for j in range(p):
            wj = W[j, :]  # Weights for variable j
            qh = Q[:, :h].T  # Y loadings for h components
            
            # Sum of squares of Y loadings weighted by X weights
            numerator = np.sum((wj ** 2) * np.sum(qh ** 2, axis=0))
            denominator = np.sum(qh ** 2)
            
            vip_scores[j] = np.sqrt(p * numerator / denominator)
        
        return vip_scores
    
    vip_scores = compute_vip_scores(pls_final, X_train)
    important_features = vip_scores > 1.0
    
    # 6. Whole-Model Significance (Permutation Test)
    
    logger.log(f"Running permutation test with {n_permutations} permutations...")
    permuted_q2_scores = []
    
    for _ in range(n_permutations):
        # Shuffle target variable
        y_permuted = np.random.permutation(y_train)
        
        # Fit PLS with optimal components
        pls_perm = PLSRegression(n_components=optimal_components)
        pls_perm.fit(X_train, y_permuted)
        
        # Calculate Q² for permuted data
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=None)
        y_pred_cv_perm = np.zeros_like(y_permuted)
        
        for train_idx, val_idx in cv.split(X_train):
            pls_cv_perm = PLSRegression(n_components=optimal_components)
            pls_cv_perm.fit(X_train[train_idx], y_permuted[train_idx])
            y_pred_cv_perm[val_idx] = pls_cv_perm.predict(X_train[val_idx]).ravel()
        
        press_perm = np.sum((y_permuted - y_pred_cv_perm) ** 2)
        ss_tot_perm = np.sum((y_permuted - np.mean(y_permuted)) ** 2)
        q2_perm = 1 - (press_perm / ss_tot_perm)
        permuted_q2_scores.append(q2_perm)
    
    # Calculate p-value
    original_q2 = q2_scores[optimal_components - 1]
    p_value = np.sum(np.array(permuted_q2_scores) >= original_q2) / n_permutations
    
    # Compile results
    results = {
        'agent_name': agent_name,
        'n_samples': n_samples,
        'n_features': n_features,
        'optimal_components': optimal_components,
        'elbow_component': elbow_component,
        
        # Goodness of fit
        'r2x_cum': r2x_cum,
        'r2y_cum': r2y_cum,
        'r2x_final': r2x_cum[optimal_components - 1],
        'r2y_final': r2y_cum[optimal_components - 1],
        
        # Predictive power
        'q2_scores': q2_scores,
        'q2_final': original_q2,
        'press_scores': press_scores,
        'rmsecv_scores': rmsecv_scores,
        'rmsecv_final': rmsecv_scores[optimal_components - 1],
        'rmsep': rmsep,
        
        # Diagnostic correlation
        'correlation_coeff': correlation_coeff,
        'correlation_p': correlation_p,
        
        # Variable relevance
        'vip_scores': vip_scores,
        'important_features': important_features,
        'n_important_features': np.sum(important_features),
        
        # Significance test
        'permutation_p_value': p_value,
        'is_significant': p_value < 0.05,
        
        # For plotting
        'component_range': list(component_range),
        'feature_names': feature_names if feature_names else [f'Feature_{i}' for i in range(n_features)],
        
        # Model objects (for further analysis)
        'final_model': pls_final,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y
    }
    
    # Log key results
    logger.log(f"Optimal components: {optimal_components}")
    logger.log(f"Elbow point: {elbow_component}")
    logger.log(f"Final R²X: {results['r2x_final']:.4f}")
    logger.log(f"Final R²Y: {results['r2y_final']:.4f}")
    logger.log(f"Final Q²: {results['q2_final']:.4f}")
    logger.log(f"RMSECV: {results['rmsecv_final']:.4f}")
    logger.log(f"RMSEP: {results['rmsep']:.4f}")
    logger.log(f"First component correlation: {correlation_coeff:.4f} (p={correlation_p:.4f})")
    logger.log(f"Important features (VIP > 1): {results['n_important_features']}/{n_features}")
    logger.log(f"Permutation test p-value: {p_value:.4f}")
    logger.log(f"Model is significant: {results['is_significant']}")
    
    return results

def plot_pls_diagnostics(pls_results, save_dir):
    """
    Create comprehensive diagnostic plots for PLS results
    
    Args:
        pls_results (dict): Results from compute_comprehensive_pls_metrics
        save_dir (str): Directory to save plots
    """
    agent_name = pls_results['agent_name']
    os.makedirs(save_dir, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Component Selection Plot (R²X, R²Y, Q² vs Components)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    components = pls_results['component_range']
    ax.plot(components, pls_results['r2x_cum'], 'o-', label='R²X (cumulative)', linewidth=2, markersize=6)
    ax.plot(components, pls_results['r2y_cum'], 's-', label='R²Y (cumulative)', linewidth=2, markersize=6)
    ax.plot(components, pls_results['q2_scores'], '^-', label='Q² (cross-validation)', linewidth=2, markersize=6)
    
    # Mark optimal and elbow points
    ax.axvline(x=pls_results['optimal_components'], color='red', linestyle='--', 
               label=f'Optimal (Q²): {pls_results["optimal_components"]}', alpha=0.7)
    ax.axvline(x=pls_results['elbow_component'], color='orange', linestyle='--', 
               label=f'Elbow (PRESS): {pls_results["elbow_component"]}', alpha=0.7)
    
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Variance Explained / Predictive Ability')
    ax.set_title(f'{agent_name} - PLS Component Selection')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{agent_name}_component_selection.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # 2. PRESS Curve
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(components, pls_results['press_scores'], 'o-', linewidth=2, markersize=6, color='darkred')
    ax.axvline(x=pls_results['elbow_component'], color='orange', linestyle='--', 
               label=f'Elbow: {pls_results["elbow_component"]}', alpha=0.7)
    
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('PRESS (Predicted Residual Sum of Squares)')
    ax.set_title(f'{agent_name} - PRESS Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{agent_name}_press_curve.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # 3. VIP Scores
    if len(pls_results['feature_names']) <= 50:  # Only plot if not too many features
        fig, ax = plt.subplots(figsize=(12, 8))
        
        vip_scores = pls_results['vip_scores']
        feature_names = pls_results['feature_names']
        colors = ['red' if vip > 1.0 else 'blue' for vip in vip_scores]
        
        bars = ax.bar(range(len(vip_scores)), vip_scores, color=colors, alpha=0.7)
        ax.axhline(y=1.0, color='red', linestyle='--', label='VIP = 1.0 threshold')
        
        ax.set_xlabel('Features')
        ax.set_ylabel('VIP Scores')
        ax.set_title(f'{agent_name} - Variable Importance in Projection (VIP)')
        ax.set_xticks(range(len(feature_names)))
        ax.set_xticklabels(feature_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{agent_name}_vip_scores.png'), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    
    # 4. Actual vs Predicted (if we have the data)
    # Note: This would require storing predictions in the results
    # For now, we'll create a summary table plot instead
    
    # Summary statistics table
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('tight')
    ax.axis('off')
    
    summary_data = [
        ['Metric', 'Value'],
        ['Optimal Components', f"{pls_results['optimal_components']}"],
        ['Elbow Point', f"{pls_results['elbow_component']}"],
        ['R²X (final)', f"{pls_results['r2x_final']:.4f}"],
        ['R²Y (final)', f"{pls_results['r2y_final']:.4f}"],
        ['Q² (final)', f"{pls_results['q2_final']:.4f}"],
        ['RMSECV', f"{pls_results['rmsecv_final']:.4f}"],
        ['RMSEP', f"{pls_results['rmsep']:.4f}"],
        ['First Component Correlation', f"{pls_results['correlation_coeff']:.4f}"],
        ['Correlation p-value', f"{pls_results['correlation_p']:.4f}"],
        ['Important Features (VIP > 1)', f"{pls_results['n_important_features']}/{pls_results['n_features']}"],
        ['Permutation p-value', f"{pls_results['permutation_p_value']:.4f}"],
        ['Model Significant', f"{pls_results['is_significant']}"]
    ]
    
    table = ax.table(cellText=summary_data, cellLoc='left', loc='center', colWidths=[0.4, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    # Style the header row
    for i in range(len(summary_data[0])):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title(f'{agent_name} - PLS Evaluation Summary', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{agent_name}_summary.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def visualize_agent_combinations(results, feature_counts, feature_names, is_3d=True):
    """
    Visualize agent combinations using a multi-stage PCA approach with sigmoid transformation and filtering
    PLUS comprehensive PLS evaluation for all models
    
    Args:
        results (list): List of evaluation results for each agent combination
        feature_counts (dict): Dictionary with feature counts for each agent
        feature_names (dict): Dictionary with feature names for each agent
        is_3d (bool): Whether to create 3D visualizations (True) or 2D (False)
        
    Returns:
        dict: Dictionary containing PLS evaluation results for each model
    """
    # Create directory for visualizations if it doesn't exist
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
    
    # Create subdirectory for PLS diagnostics
    pls_diagnostics_dir = os.path.join(VISUALIZATIONS_DIR, "pls_diagnostics")
    os.makedirs(pls_diagnostics_dir, exist_ok=True)
    
    # Extract performance metrics and weights for each agent combination
    combination_names = []
    combination_accuracies_raw = []
    combination_accuracies_transformed = []
    all_combination_weights = []
    
    # Extract feature counts
    b_features = feature_counts["b_features"]
    c_features = feature_counts["c_features"]
    d_features = feature_counts["d_features"]
    
    # Collect data from results
    for result in results:
        combination_names.append(result["combination"])
        combination_accuracies_raw.append(result["accuracy_raw"])
        # combination_accuracies_transformed.append(result["accuracy"])  # Already sigmoid-transformed
        all_combination_weights.append(result["combination_weights"])
    
    # Convert to numpy arrays
    accuracies_raw = np.array(combination_accuracies_raw)
    # accuracies_transformed = np.array(combination_accuracies_transformed)  # Already sigmoid values 0-1
    all_combination_weights = np.array(all_combination_weights)
    
    # Log the raw feature dimensions
    logger.log(f"\nRaw feature dimensions:")
    logger.log(f"  Agent B features: {b_features}")
    logger.log(f"  Agent C features: {c_features}")
    logger.log(f"  Agent D features: {d_features}")
    logger.log(f"  Total features per combination: {b_features + c_features + d_features}")
    
    # Split combination weights into agent-specific weights (using filtered data)
    b_weights_all = all_combination_weights[:, :b_features]
    c_weights_all = all_combination_weights[:, b_features:b_features + c_features]
    d_weights_all = all_combination_weights[:, b_features + c_features:]
    
    # Define the target dimensionality for per-agent PLS
    pls_dim = 5
    
    # Create scalers and PLS models for each agent type
    b_scaler = StandardScaler()
    c_scaler = StandardScaler()
    d_scaler = StandardScaler()
    
    b_weights_all_scaled = b_scaler.fit_transform(b_weights_all)
    c_weights_all_scaled = c_scaler.fit_transform(c_weights_all)
    d_weights_all_scaled = d_scaler.fit_transform(d_weights_all)
    
    # Normalize accuracies for PLS evaluation
    mean = np.mean(accuracies_raw)
    std = np.std(accuracies_raw)
    accuracies_raw_normalized = (accuracies_raw - mean) / std
    
    # ===== COMPREHENSIVE PLS EVALUATION =====
    logger.log(f"\n{'='*60}")
    logger.log("STARTING COMPREHENSIVE PLS EVALUATION FOR ALL MODELS")
    logger.log(f"{'='*60}")
    
    # Store PLS evaluation results for all models
    pls_evaluation_results = {}
    
    # 1. Evaluate Agent B PLS Model
    logger.log(f"\n{'-'*40}")
    logger.log("EVALUATING AGENT B PLS MODEL")
    logger.log(f"{'-'*40}")
    
    pls_results_b = compute_comprehensive_pls_metrics(
        X=b_weights_all_scaled, 
        y=accuracies_raw_normalized, 
        agent_name="Agent_B",
        feature_names=feature_names["b_features"],
        max_components=min(10, b_features),
        cv_folds=5,
        n_permutations=500  # Reduced for faster execution
    )
    pls_evaluation_results['Agent_B'] = pls_results_b
    plot_pls_diagnostics(pls_results_b, pls_diagnostics_dir)
    
    # 2. Evaluate Agent C PLS Model
    logger.log(f"\n{'-'*40}")
    logger.log("EVALUATING AGENT C PLS MODEL")
    logger.log(f"{'-'*40}")
    
    pls_results_c = compute_comprehensive_pls_metrics(
        X=c_weights_all_scaled, 
        y=accuracies_raw_normalized, 
        agent_name="Agent_C",
        feature_names=feature_names["c_features"],
        max_components=min(10, c_features),
        cv_folds=5,
        n_permutations=500
    )
    pls_evaluation_results['Agent_C'] = pls_results_c
    plot_pls_diagnostics(pls_results_c, pls_diagnostics_dir)
    
    # 3. Evaluate Agent D PLS Model
    logger.log(f"\n{'-'*40}")
    logger.log("EVALUATING AGENT D PLS MODEL")
    logger.log(f"{'-'*40}")
    
    pls_results_d = compute_comprehensive_pls_metrics(
        X=d_weights_all_scaled, 
        y=accuracies_raw_normalized, 
        agent_name="Agent_D",
        feature_names=feature_names["d_features"],
        max_components=min(10, d_features),
        cv_folds=5,
        n_permutations=500
    )
    pls_evaluation_results['Agent_D'] = pls_results_d
    plot_pls_diagnostics(pls_results_d, pls_diagnostics_dir)
    
    # Apply PLS to each agent type for visualization (using optimal components from evaluation)
    b_pls = PLSRegression(n_components=min(pls_results_b['optimal_components'], pls_dim))
    c_pls = PLSRegression(n_components=min(pls_results_c['optimal_components'], pls_dim))
    d_pls = PLSRegression(n_components=min(pls_results_d['optimal_components'], pls_dim))
    
    # Apply PLS to each agent type
    b_pls_features, _ = b_pls.fit_transform(b_weights_all_scaled, accuracies_raw_normalized)
    c_pls_features, _ = c_pls.fit_transform(c_weights_all_scaled, accuracies_raw_normalized)
    d_pls_features, _ = d_pls.fit_transform(d_weights_all_scaled, accuracies_raw_normalized)
    
    # Create color normalization and colormap for sigmoid scores
    # Sigmoid scores range from 0 to 1, with 0.5 as neutral (now filtered out)
    norm = plt.Normalize(-2, 2)
    colors = ["blue", "white", "red"]
    cmap = LinearSegmentedColormap.from_list("BWR", colors)
    
    # Function to visualize agent-specific PLS
    def visualize_agent_pls(agent_embedding, agent_type):
        num_actual_components = agent_embedding.shape[1]

        if num_actual_components == 0:
            logger.log(f"Agent {agent_type} PLS: No components to plot (num_actual_components is 0).")
            return

        dim_label = f"{num_actual_components}d"
        title_suffix = f"({num_actual_components}D)"

        # Create figure
        if is_3d and num_actual_components >= 3:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(
                agent_embedding[:, 0], agent_embedding[:, 1], agent_embedding[:, 2],
                c=accuracies_raw_normalized, cmap=cmap, norm=norm, s=SCATTER_SIZE, alpha=SCATTER_ALPHA
            )
            ax.set_xlabel("PLS Component 1")
            ax.set_ylabel("PLS Component 2")
            ax.set_zlabel("PLS Component 3")
            ax.set_title(f"Agent {agent_type} PLS Projection {title_suffix}")
        elif num_actual_components >= 2: # Handles 2D, or 3D request with only 2 components
            fig, ax = plt.subplots(figsize=(12, 10))
            scatter = ax.scatter(
                agent_embedding[:, 0], agent_embedding[:, 1],
                c=accuracies_raw_normalized, cmap=cmap, norm=norm, s=SCATTER_SIZE, alpha=SCATTER_ALPHA
            )
            ax.set_xlabel("PLS Component 1")
            ax.set_ylabel("PLS Component 2")
            if is_3d and num_actual_components == 2:
                 ax.set_title(f"Agent {agent_type} PLS Projection (3D requested, 2D Fallback)")
            else:
                 ax.set_title(f"Agent {agent_type} PLS Projection {title_suffix}")
        elif num_actual_components == 1:
            fig, ax = plt.subplots(figsize=(12, 8))
            # For 1D, scatter y values with some jitter for better visualization (strip plot like)
            jitter = (np.random.rand(agent_embedding.shape[0]) - 0.5) * 0.1 
            scatter = ax.scatter(
                agent_embedding[:, 0], jitter,
                c=accuracies_raw_normalized, cmap=cmap, norm=norm, s=SCATTER_SIZE, alpha=SCATTER_ALPHA
            )
            ax.set_xlabel("PLS Component 1")
            ax.set_yticks([]) # No meaningful y-axis for 1D PLS
            ax.set_ylabel("") 
            ax.set_title(f"Agent {agent_type} PLS Projection {title_suffix}")
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Normalized Accuracy Score')
        
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(VISUALIZATIONS_DIR, f"agent_{agent_type}_pls_{dim_label}.png"))
        plt.show()
        plt.close()
        
        # Log information
        logger.log(f"\nAgent {agent_type} PLS Visualization:")
        logger.log(f"  PLS components plotted: {num_actual_components} (dim_label: {dim_label})")
        
        # Create a CSV file with the results
        df_data = {
            "combination": combination_names,
            "accuracy_original": accuracies_raw,
            "accuracy_normalized": accuracies_raw_normalized
        }
        if num_actual_components >= 1:
            df_data["pls_x"] = agent_embedding[:, 0]
        if num_actual_components >= 2:
            df_data["pls_y"] = agent_embedding[:, 1]
        if num_actual_components >= 3:
            df_data["pls_z"] = agent_embedding[:, 2]
            
        df = pd.DataFrame(df_data)
        # Save CSV file
        os.makedirs(CSV_DIR, exist_ok=True)
        df.to_csv(os.path.join(CSV_DIR, f"agent_{agent_type}_pls_{dim_label}.csv"), index=False)
    
    # Visualize each agent type separately
    visualize_agent_pls(b_pls_features, "B")
    visualize_agent_pls(c_pls_features, "C")
    visualize_agent_pls(d_pls_features, "D")
    
    # Normalize each agent's PLS features using L2 norm
    concatenated_features = []
    for i in range(len(combination_names)):
        # Get PLS features for this combination
        b_features_i = b_pls_features[i]
        c_features_i = c_pls_features[i]
        d_features_i = d_pls_features[i]
        
        # Normalize each using L2 norm
        b_norm = np.linalg.norm(b_features_i)
        c_norm = np.linalg.norm(c_features_i)
        d_norm = np.linalg.norm(d_features_i)
        
        if b_norm > 0:
            b_features_i = b_features_i / b_norm
        if c_norm > 0:
            c_features_i = c_features_i / c_norm
        if d_norm > 0:
            d_features_i = d_features_i / d_norm
        
        # Concatenate normalized features
        combined_features = np.concatenate([b_features_i, c_features_i, d_features_i])
        concatenated_features.append(combined_features)
    
    # Convert to numpy array
    X = np.array(concatenated_features)
    
    # Skip if we don't have enough data points
    if len(X) < 2:
        logger.log("Not enough filtered combinations for global PCA (need at least 2)")
        return
    
    # Normalize the combined feature vectors
    global_scaler = StandardScaler()
    X_normalized = global_scaler.fit_transform(X)
    
    # 4. Evaluate Global Combination PLS Model
    logger.log(f"\n{'-'*40}")
    logger.log("EVALUATING GLOBAL COMBINATION PLS MODEL")
    logger.log(f"{'-'*40}")
    
    # Create feature names for global model
    global_feature_names = []
    for i in range(b_pls_features.shape[1]):
        global_feature_names.append(f"B_PLS_{i+1}")
    for i in range(c_pls_features.shape[1]):
        global_feature_names.append(f"C_PLS_{i+1}")
    for i in range(d_pls_features.shape[1]):
        global_feature_names.append(f"D_PLS_{i+1}")
    
    pls_results_global = compute_comprehensive_pls_metrics(
        X=X_normalized, 
        y=accuracies_raw_normalized, 
        agent_name="Global_Combination",
        feature_names=global_feature_names,
        max_components=min(10, X_normalized.shape[1]),
        cv_folds=5,
        n_permutations=500
    )
    pls_evaluation_results['Global_Combination'] = pls_results_global
    plot_pls_diagnostics(pls_results_global, pls_diagnostics_dir)
    
    # Apply global PLS using optimal components from evaluation
    n_components_pref = 3 if is_3d else 2 # Preferred number of components based on is_3d
    # Actual components will be min(optimal_from_eval, preferred_for_is_3d, embedding_max_possible)
    # optimal_global_components was already min(pls_results_global['optimal_components'], n_components_pref)
    # So global_pls already fitted with a sensible number of components. embedding.shape[1] is the source of truth.
    
    global_pls = PLSRegression(n_components=pls_results_global['optimal_components'])
    embedding, _ = global_pls.fit_transform(X_normalized, accuracies_raw_normalized)
    
    # Print global PLS explained variance (actual components used)
    logger.log(f"\nGlobal PLS model fitted with {pls_results_global['optimal_components']} components based on evaluation.")
    logger.log(f"Resulting Global PLS embedding dimensions: {embedding.shape[1]}")
    print(f"Global PLS components in embedding: {embedding.shape[1]}")

    # Create the global visualization
    global_num_actual_components = embedding.shape[1]
    global_dim_label = "0d" # Default if no components

    if global_num_actual_components > 0:
        global_dim_label = f"{global_num_actual_components}d"
        title_suffix = f"({global_num_actual_components}D)"

        if is_3d and global_num_actual_components >= 3:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(
                embedding[:, 0], embedding[:, 1], embedding[:, 2],
                c=accuracies_raw_normalized, cmap=cmap, norm=norm, s=SCATTER_SIZE, alpha=SCATTER_ALPHA
            )
            ax.set_title(f"Agent Combination Weight Space (Multi-Stage PLS - {title_suffix})")
            ax.set_xlabel("Global PLS Component 1")
            ax.set_ylabel("Global PLS Component 2")
            ax.set_zlabel("Global PLS Component 3")
        elif global_num_actual_components >= 2: # Handles 2D, or 3D request with only 2 components
            fig, ax = plt.subplots(figsize=(12, 10))
            scatter = ax.scatter(
                embedding[:, 0], embedding[:, 1],
                c=accuracies_raw_normalized, cmap=cmap, norm=norm, s=SCATTER_SIZE, alpha=SCATTER_ALPHA
            )
            if is_3d and global_num_actual_components == 2:
                 ax.set_title(f"Agent Combination Weight Space (Multi-Stage PLS - 3D requested, 2D Fallback)")
            else:
                 ax.set_title(f"Agent Combination Weight Space (Multi-Stage PLS - {title_suffix})")
            ax.set_xlabel("Global PLS Component 1")
            ax.set_ylabel("Global PLS Component 2")
        elif global_num_actual_components == 1:
            fig, ax = plt.subplots(figsize=(12, 8))
            jitter = (np.random.rand(embedding.shape[0]) - 0.5) * 0.1
            scatter = ax.scatter(
                embedding[:, 0], jitter,
                c=accuracies_raw_normalized, cmap=cmap, norm=norm, s=SCATTER_SIZE, alpha=SCATTER_ALPHA
            )
            ax.set_title(f"Agent Combination Weight Space (Multi-Stage PLS - {title_suffix})")
            ax.set_xlabel("Global PLS Component 1")
            ax.set_yticks([])
            ax.set_ylabel("")
        
        cbar = plt.colorbar(scatter)
        cbar.set_label('Normalized Accuracy Score')
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATIONS_DIR, f"agent_combinations_pls_{global_dim_label}.png"))
        plt.show()
        plt.close()
    else:
        logger.log("Global PLS: No components to plot as embedding has 0 components.")

    # ===== CREATE COMPREHENSIVE SUMMARY =====
    logger.log(f"\n{'='*60}")
    logger.log("COMPREHENSIVE PLS EVALUATION SUMMARY")
    logger.log(f"{'='*60}")
    
    # Create comparison table
    comparison_data = []
    comparison_data.append(['Model', 'Optimal Components', 'R²X', 'R²Y', 'Q²', 'RMSECV', 'RMSEP', 'First Comp. Corr.', 'VIP > 1', 'p-value', 'Significant'])
    
    for model_name, pls_result in pls_evaluation_results.items():
        comparison_data.append([
            model_name,
            pls_result['optimal_components'],
            f"{pls_result['r2x_final']:.3f}",
            f"{pls_result['r2y_final']:.3f}",
            f"{pls_result['q2_final']:.3f}",
            f"{pls_result['rmsecv_final']:.3f}",
            f"{pls_result['rmsep']:.3f}",
            f"{pls_result['correlation_coeff']:.3f}",
            f"{pls_result['n_important_features']}/{pls_result['n_features']}",
            f"{pls_result['permutation_p_value']:.3f}",
            "Yes" if pls_result['is_significant'] else "No"
        ])
    
    # Log the comparison table
    logger.log("\nCOMPREHENSIVE PLS MODEL COMPARISON:")
    header = comparison_data[0]
    log_header_str = f"{header[0]:<20} {header[1]:<15} {header[2]:<8} {header[3]:<8} {header[4]:<8} {header[5]:<10} {header[6]:<10} {header[7]:<18} {header[8]:<10} {header[9]:<10} {header[10]:<12}"
    logger.log(log_header_str)
    logger.log("-" * (len(log_header_str) + 5))
    for row_idx in range(1, len(comparison_data)):
        row = comparison_data[row_idx]
        logger.log(f"{row[0]:<20} {str(row[1]):<15} {row[2]:<8} {row[3]:<8} {row[4]:<8} {row[5]:<10} {row[6]:<10} {row[7]:<18} {row[8]:<10} {row[9]:<10} {row[10]:<12}")

    # Save the comprehensive summary table as an image
    fig_table, ax_table = plt.subplots(figsize=(20, 4)) # Adjust figsize as needed
    ax_table.axis('tight')
    ax_table.axis('off')

    # Create the table - colWidths might need tuning
    col_widths = [0.12, 0.08, 0.05, 0.05, 0.05, 0.07, 0.07, 0.1, 0.07, 0.07, 0.07]
    summary_mpl_table = ax_table.table(cellText=comparison_data[1:], 
                                     colLabels=comparison_data[0],
                                     cellLoc='center', 
                                     loc='center', 
                                     colWidths=col_widths)
    
    summary_mpl_table.auto_set_font_size(False)
    summary_mpl_table.set_fontsize(9)
    summary_mpl_table.scale(1.2, 1.2) # Adjust scale as needed

    # Style the header cells
    for i in range(len(comparison_data[0])):
        summary_mpl_table[(0, i)].set_facecolor('#40466e')
        summary_mpl_table[(0, i)].set_text_props(weight='bold', color='white')
        
    # Style data cells - optional, e.g., for alternating row colors or specific column alignment
    for i in range(1, len(comparison_data)):
        for j in range(len(comparison_data[0])):
            summary_mpl_table[(i, j)].set_height(0.1) # Adjust cell height

    ax_table.set_title('Comprehensive PLS Model Evaluation Summary', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout(pad=1.5)
    summary_table_path = os.path.join(pls_diagnostics_dir, 'pls_evaluation_summary_table.png')
    plt.savefig(summary_table_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig_table)
    logger.log(f"Comprehensive PLS summary table image saved to: {summary_table_path}")

    # Create a comprehensive comparison plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    models = list(pls_evaluation_results.keys())
    
    # 1. R² comparison
    r2x_values = [pls_evaluation_results[model]['r2x_final'] for model in models]
    r2y_values = [pls_evaluation_results[model]['r2y_final'] for model in models]
    q2_values = [pls_evaluation_results[model]['q2_final'] for model in models]
    
    x_pos = np.arange(len(models))
    width = 0.25
    
    ax1.bar(x_pos - width, r2x_values, width, label='R²X', alpha=0.7)
    ax1.bar(x_pos, r2y_values, width, label='R²Y', alpha=0.7)
    ax1.bar(x_pos + width, q2_values, width, label='Q²', alpha=0.7)
    
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Variance Explained / Predictive Power')
    ax1.set_title('R² and Q² Comparison Across Models')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(models, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. RMSE comparison
    rmsecv_values = [pls_evaluation_results[model]['rmsecv_final'] for model in models]
    rmsep_values = [pls_evaluation_results[model]['rmsep'] for model in models]
    
    ax2.bar(x_pos - width/2, rmsecv_values, width, label='RMSECV', alpha=0.7)
    ax2.bar(x_pos + width/2, rmsep_values, width, label='RMSEP', alpha=0.7)
    
    ax2.set_xlabel('Models')
    ax2.set_ylabel('Root Mean Square Error')
    ax2.set_title('RMSE Comparison Across Models')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(models, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Correlation and significance
    corr_values = [pls_evaluation_results[model]['correlation_coeff'] for model in models]
    p_values = [pls_evaluation_results[model]['permutation_p_value'] for model in models]
    
    ax3.bar(models, corr_values, alpha=0.7, color='green')
    ax3.set_xlabel('Models')
    ax3.set_ylabel('First Component Correlation')
    ax3.set_title('First Component vs Y Correlation')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # 4. p-values with significance threshold
    colors = ['red' if p < 0.05 else 'blue' for p in p_values]
    ax4.bar(models, p_values, alpha=0.7, color=colors)
    ax4.axhline(y=0.05, color='red', linestyle='--', label='p = 0.05 threshold')
    ax4.set_xlabel('Models')
    ax4.set_ylabel('Permutation Test p-value')
    ax4.set_title('Model Significance (Permutation Test)')
    ax4.tick_params(axis='x', rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(pls_diagnostics_dir, 'comprehensive_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Save comprehensive results to CSV
    comparison_df = pd.DataFrame(comparison_data[1:], columns=comparison_data[0])
    comparison_df.to_csv(os.path.join(CSV_DIR, 'pls_comprehensive_comparison.csv'), index=False)
    
    # Log information for global results
    logger.log("\nAgent Combination PLS Visualization (Global Model):")
    logger.log(f"  Total combinations processed: {len(X)}") # X is from concatenated_features
    logger.log(f"  Dimensions plotted for global model: {global_dim_label}")
    logger.log(f"  Min accuracy (original): {min(accuracies_raw):.4f}")
    logger.log(f"  Max accuracy (original): {max(accuracies_raw):.4f}")
    logger.log(f"  Min normalized score: {min(accuracies_raw_normalized):.4f}")
    logger.log(f"  Max normalized score: {max(accuracies_raw_normalized):.4f}")
    
    # Create a CSV file with the global PLS results
    if global_num_actual_components > 0:
        df_data_global = {
            "combination": combination_names,
            "accuracy_original": accuracies_raw,
            "accuracy_normalized": accuracies_raw_normalized
        }
        if global_num_actual_components >= 1:
            df_data_global["pls_x"] = embedding[:, 0]
        if global_num_actual_components >= 2:
            df_data_global["pls_y"] = embedding[:, 1]
        if global_num_actual_components >= 3:
            df_data_global["pls_z"] = embedding[:, 2]
        
        df_global = pd.DataFrame(df_data_global)
        os.makedirs(CSV_DIR, exist_ok=True)
        df_global.to_csv(os.path.join(CSV_DIR, f"agent_combinations_pls_{global_dim_label}.csv"), index=False)
        logger.log(f"Global PLS CSV saved to: agent_combinations_pls_{global_dim_label}.csv")
    else:
        logger.log(f"Skipping CSV for global PLS as there are {global_num_actual_components} components.")
    
    logger.log(f"\n{'='*60}")
    logger.log("PLS EVALUATION COMPLETE - All results saved to analysis_output/")
    logger.log(f"{'='*60}")
    
    return pls_evaluation_results


def run_performance_evaluation():
    """
    Run the performance evaluation pipeline across all agent combinations
    """
    logger.log("\n" + "="*20 + " RESUME RANKING SYSTEM EVALUATION " + "="*20 + "\n")
    
    # Generate job description
    job_description = generate_sample_job_description()
    
    # Log the full job description
    logger.log("FULL JOB DESCRIPTION:")
    logger.log(job_description)
    logger.log("\n" + "="*50 + "\n")
    
    # Load or generate weights and embeddings
    data = load_or_generate_weights(job_description)
    combination_weights = data["combination_weights"]
    feature_counts = data["feature_counts"]
    feature_names = data["feature_names"]
    jd_embedding = data["jd_embedding"]
    
    # Create Word2Vec model for the job description
    word2vec_model, _ = create_word2vec_model(job_description)
    
    # Generate candidate resumes
    resumes = generate_candidate_resumes()
    
    # Human ranking (predefined ground truth)
    # Corresponds to [resume1, resume2, resume3, resume4, resume5]
    # In order of preference: resume2 (idx 1), resume3 (idx 2), resume5 (idx 4), resume4 (idx 3), resume1 (idx 0)
    human_ranking = [2, 3, 5, 4, 1]  # 1-indexed for human readability
    
    logger.log(f"Human Expert Ranking (Ground Truth): {human_ranking}")
    
    # Create agent instances
    agent_A_list, feature_counts, feature_names = create_agents(data, word2vec_model, jd_embedding)
    
    # Log the number of combinations and features
    logger.log(f"\nNumber of combinations to evaluate: {len(combination_weights)}")
    logger.log(f"Feature counts per combination:")
    logger.log(f"  Agent B features: {feature_counts['b_features']}")
    logger.log(f"  Agent C features: {feature_counts['c_features']}")
    logger.log(f"  Agent D features: {feature_counts['d_features']}")
    logger.log(f"  Total features: {sum(feature_counts.values())}")
    
    print(f"\nEvaluating {len(combination_weights)} agent combinations...")
    
    # Evaluate each combination
    results = []
    for i, weights in enumerate(combination_weights):
        result = run_evaluation_for_combination(
            weights, i, resumes, job_description, human_ranking,
            feature_counts, feature_names, agent_A_list, word2vec_model, jd_embedding
        )
        results.append(result)
    
    # Sort results by accuracy 
    # sorted_results = sorted(results, key=lambda x: x["accuracy"], reverse=True)
    sorted_results = sorted(results, key=lambda x: x["accuracy_raw"], reverse=True)
    
    # Display final comparison
    print("\n" + "="*50)
    print("FINAL COMPARISON OF AGENT COMBINATIONS")
    print("="*50)
    
    logger.log("\n" + "="*50)
    logger.log("FINAL COMPARISON OF AGENT COMBINATIONS")
    logger.log("="*50)
    
    print(f"\n{'Rank':^6}|{'Combination':^15}|{'Raw Acc':^10}|{'Norm Acc':^10}|{'Trans Acc':^10}")
    print("-" * 70)
    
    for rank, result in enumerate(sorted_results):
        print(f"{rank+1:^6}|{result['combination']:^15}|{result['accuracy_raw']:^10.4f}|")    
    
    logger.log(f"\n{'Rank':^6}|{'Combination':^15}|{'Raw Acc':^10}|{'Norm Acc':^10}|{'Trans Acc':^10}")
    logger.log("-" * 70)
    
    for rank, result in enumerate(sorted_results):
        logger.log(f"{rank+1:^6}|{result['combination']:^15}|{result['accuracy_raw']:^10.4f}|")       
    
    # Print the best combination
    best_combo = sorted_results[0]['combination']
    best_accuracy_raw = sorted_results[0]['accuracy_raw']
    
    
    print(f"\nBest agent combination: {best_combo}")
    print(f"Raw accuracy (negative slope): {best_accuracy_raw:.4f}")
   
    
    logger.log(f"\nBest agent combination: {best_combo}")
    logger.log(f"Raw accuracy (negative slope): {best_accuracy_raw:.4f}")

    
    # Visualize agent combinations using a multi-stage PCA approach
    logger.log("\nVisualizing agent combinations using a multi-stage PCA approach...")
    pls_evaluation_results = visualize_agent_combinations(results, feature_counts, feature_names, is_3d=True)

    # Print comprehensive PLS evaluation summary
    print("\n" + "="*80)
    print("COMPREHENSIVE PLS EVALUATION SUMMARY")
    print("="*80)
    
    print(f"\n{'Model':<20} {'Opt. Comp.':<10} {'R²X':<8} {'R²Y':<8} {'Q²':<8} {'RMSECV':<10} {'RMSEP':<10} {'Corr.':<8} {'Significant':<10}")
    print("-" * 95)
    
    for model_name, pls_result in pls_evaluation_results.items():
        significance = "Yes" if pls_result['is_significant'] else "No"
        print(f"{model_name:<20} {pls_result['optimal_components']:<10} {pls_result['r2x_final']:<8.3f} {pls_result['r2y_final']:<8.3f} {pls_result['q2_final']:<8.3f} {pls_result['rmsecv_final']:<10.3f} {pls_result['rmsep']:<10.3f} {pls_result['correlation_coeff']:<8.3f} {significance:<10}")
    
    print(f"\nAll PLS diagnostic plots and comprehensive results saved to: {os.path.join(OUTPUT_DIR, 'visualizations', 'pls_diagnostics')}")
    
    return sorted_results

def main():
    """
    Main function with options to run single resume scoring or performance evaluation
    """
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Performance evaluation pipeline across all agent combinations
    run_performance_evaluation()

if __name__ == "__main__":
    main() 
