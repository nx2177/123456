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

# Configuration for number of agents in each category
n_A = 1  # Number of Agent A instances
n_combinations = 10000  # Number of agent combinations to generate

# Weights and embedding configuration
WEIGHTS_FILE = "weights.json"
EMBEDDING_DIM = 100  # Dimension for Word2Vec embeddings
WEIGHT_MIN = -10.0  # Minimum value for uniform distribution
WEIGHT_MAX = 10.0   # Maximum value for uniform distribution
WORD2VEC_WINDOW = 5  # Window size for Word2Vec model
WORD2VEC_MIN_COUNT = 1  # Minimum count for Word2Vec model

# Analysis output configuration
OUTPUT_DIR = "analysis_output"
VISUALIZATIONS_DIR = os.path.join(OUTPUT_DIR, "visualizations")
CSV_DIR = os.path.join(OUTPUT_DIR, "csv")

# Create a logger for storing processing details
class Logger:
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        # Initialize or clear the log file
        with open(log_file_path, 'w') as f:
            f.write("===== RESUME SCORING SYSTEM PROCESSING DETAILS =====\n\n")
    
    def log(self, message):
        with open(self.log_file_path, 'a') as f:
            f.write(f"{message}\n")
    
    def log_json(self, data, title=None):
        with open(self.log_file_path, 'a') as f:
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
    Calculate the accuracy of the system ranking compared to human ranking using linear regression
    
    Args:
        human_ranking (list): Human's ranking of candidates
        scores (list): System-generated scores for each resume
        
    Returns:
        dict: Accuracy metrics
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
    
    # Calculate accuracy as negative slope
    accuracy = -slope
    
    return {
        "accuracy": accuracy
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
    logger.log(f"\nAccuracy (negative slope): {accuracy_metrics['accuracy']:.4f}")
    
    return {
        "combination": combination_name,
        "combination_index": combination_index,
        "combination_weights": combination_weights,
        "scores": scores,
        "system_ranking": system_ranking,
        "human_ranking": human_ranking,
        "accuracy": accuracy_metrics['accuracy']
    }

def visualize_agent_combinations(results, feature_counts, feature_names, is_3d=True):
    """
    Visualize agent combinations using a multi-stage PCA approach with color gradient based on accuracy
    
    Args:
        results (list): List of evaluation results for each agent combination
        feature_counts (dict): Dictionary with feature counts for each agent
        feature_names (dict): Dictionary with feature names for each agent
        is_3d (bool): Whether to create 3D visualizations (True) or 2D (False)
        
    Returns:
        None
    """
    # Create directory for visualizations if it doesn't exist
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
    
    # Extract performance metrics and weights for each agent combination
    combination_names = []
    combination_accuracies = []
    all_combination_weights = []
    
    # Extract feature counts
    b_features = feature_counts["b_features"]
    c_features = feature_counts["c_features"]
    d_features = feature_counts["d_features"]
    
    # Collect data from results
    for result in results:
        combination_names.append(result["combination"])
        combination_accuracies.append(result["accuracy"])
        all_combination_weights.append(result["combination_weights"])
    
    # Convert to numpy arrays
    accuracies = np.array(combination_accuracies)
    all_combination_weights = np.array(all_combination_weights)
    
    # Log the raw feature dimensions
    logger.log(f"\nRaw feature dimensions:")
    logger.log(f"  Agent B features: {b_features}")
    logger.log(f"  Agent C features: {c_features}")
    logger.log(f"  Agent D features: {d_features}")
    logger.log(f"  Total features per combination: {b_features + c_features + d_features}")
    
    # Split combination weights into agent-specific weights
    b_weights_all = all_combination_weights[:, :b_features]
    c_weights_all = all_combination_weights[:, b_features:b_features + c_features]
    d_weights_all = all_combination_weights[:, b_features + c_features:]
    
    # Define the target dimensionality for per-agent PCA
    pca_dim = 5
    
    # Create scalers and PCA models for each agent type
    b_scaler = StandardScaler()
    c_scaler = StandardScaler()
    d_scaler = StandardScaler()
    
    b_pca = PCA(n_components=min(pca_dim, b_features))
    c_pca = PCA(n_components=min(pca_dim, c_features))
    d_pca = PCA(n_components=min(pca_dim, d_features))
    
    # Fit scalers and PCA models
    b_scaled = b_scaler.fit_transform(b_weights_all)
    c_scaled = c_scaler.fit_transform(c_weights_all)
    d_scaled = d_scaler.fit_transform(d_weights_all)
    
    # Apply PCA to each agent type
    b_pca_features = b_pca.fit_transform(b_scaled)
    c_pca_features = c_pca.fit_transform(c_scaled)
    d_pca_features = d_pca.fit_transform(d_scaled)
    
    # Log explained variance
    logger.log(f"\nExplained variance ratios for per-agent PCA:")
    logger.log(f"  Agent B: {b_pca.explained_variance_ratio_}")
    logger.log(f"  Agent C: {c_pca.explained_variance_ratio_}")
    logger.log(f"  Agent D: {d_pca.explained_variance_ratio_}")
    
    # Normalize accuracy values for coloring
    accuracy_scaler = StandardScaler()
    accuracies_normalized = accuracy_scaler.fit_transform(accuracies.reshape(-1, 1)).flatten()
    
    # Create color normalization and colormap
    max_abs_accuracy = max(abs(accuracies_normalized))
    norm = plt.Normalize(-max_abs_accuracy, max_abs_accuracy)
    colors = ["blue", "white", "red"]
    cmap = LinearSegmentedColormap.from_list("BWR", colors)
    
    # Function to visualize agent-specific PCA
    def visualize_agent_pca(agent_features, agent_type):
        # Apply PCA to the agent features
        n_components = 3 if is_3d else 2
        agent_pca = PCA(n_components=n_components)
        agent_embedding = agent_pca.fit_transform(agent_features)
        
        # Create figure
        if is_3d:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig, ax = plt.subplots(figsize=(12, 10))
            
        # Plot the points
        if is_3d:
            scatter = ax.scatter(
                agent_embedding[:, 0],
                agent_embedding[:, 1],
                agent_embedding[:, 2],
                c=accuracies_normalized,
                cmap=cmap,
                norm=norm,
                s=30
            )
        else:
            scatter = ax.scatter(
                agent_embedding[:, 0],
                agent_embedding[:, 1],
                c=accuracies_normalized,
                cmap=cmap,
                norm=norm,
                s=30
            )
            
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Normalized Accuracy (Negative Slope)')
        
        # Add labels
        ax.set_title(f"Agent {agent_type} PCA Projection")
        ax.set_xlabel(f"PCA Dimension 1 (Explained Variance: {agent_pca.explained_variance_ratio_[0]:.2f})")
        ax.set_ylabel(f"PCA Dimension 2 (Explained Variance: {agent_pca.explained_variance_ratio_[1]:.2f})")
        if is_3d:
            ax.set_zlabel(f"PCA Dimension 3 (Explained Variance: {agent_pca.explained_variance_ratio_[2]:.2f})")
        
        plt.tight_layout()
        
        # Save the plot
        dim_label = "3d" if is_3d else "2d"
        plt.show()
        plt.savefig(os.path.join(VISUALIZATIONS_DIR, f"agent_{agent_type}_pca_{dim_label}.png"))
        plt.close()
        
        # Log information
        logger.log(f"\nAgent {agent_type} PCA Visualization:")
        logger.log(f"  Explained variance: {agent_pca.explained_variance_ratio_}")
        
        # Create a CSV file with the results
        df = pd.DataFrame({
            "combination": combination_names,
            "pca_x": agent_embedding[:, 0],
            "pca_y": agent_embedding[:, 1],
            "accuracy": accuracies,
            "accuracy_normalized": accuracies_normalized
        })
        
        if is_3d:
            df["pca_z"] = agent_embedding[:, 2]
            
        # Save CSV file
        os.makedirs(CSV_DIR, exist_ok=True)
        df.to_csv(os.path.join(CSV_DIR, f"agent_{agent_type}_pca_{dim_label}.csv"), index=False)
    
    # Visualize each agent type separately
    visualize_agent_pca(b_pca_features, "B")
    visualize_agent_pca(c_pca_features, "C")
    visualize_agent_pca(d_pca_features, "D")
    
    # Normalize each agent's PCA features using L2 norm
    concatenated_features = []
    for i in range(len(results)):
        # Get PCA features for this combination
        b_features_i = b_pca_features[i]
        c_features_i = c_pca_features[i]
        d_features_i = d_pca_features[i]
        
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
        logger.log("Not enough agent combinations for global PCA (need at least 2)")
        return
    
    # Normalize the combined feature vectors
    global_scaler = StandardScaler()
    X_normalized = global_scaler.fit_transform(X)
    
    # Apply global PCA
    n_components = 3 if is_3d else 2
    global_pca = PCA(n_components=n_components)
    embedding = global_pca.fit_transform(X_normalized)
    
    # Print global PCA explained variance
    logger.log(f"\nGlobal PCA explained variance: {global_pca.explained_variance_ratio_}")
    print(f"Global PCA explained variance: {global_pca.explained_variance_ratio_}")
    
    # Create the global visualization
    if is_3d:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            embedding[:, 2],
            c=accuracies_normalized,
            cmap=cmap,
            norm=norm,
            s=30,
        )
        
        ax.set_title("Agent Combination Weight Space (Multi-Stage PCA - 3D)")
        ax.set_xlabel(f"Global PCA Dimension 1 (Explained Variance: {global_pca.explained_variance_ratio_[0]:.2f})")
        ax.set_ylabel(f"Global PCA Dimension 2 (Explained Variance: {global_pca.explained_variance_ratio_[1]:.2f})")
        ax.set_zlabel(f"Global PCA Dimension 3 (Explained Variance: {global_pca.explained_variance_ratio_[2]:.2f})")
    else:
        fig, ax = plt.subplots(figsize=(12, 10))
        
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=accuracies_normalized,
            cmap=cmap,
            norm=norm,
            s=30,
        )
        
        ax.set_title("Agent Combination Weight Space (Multi-Stage PCA - 2D)")
        ax.set_xlabel(f"Global PCA Dimension 1 (Explained Variance: {global_pca.explained_variance_ratio_[0]:.2f})")
        ax.set_ylabel(f"Global PCA Dimension 2 (Explained Variance: {global_pca.explained_variance_ratio_[1]:.2f})")
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Normalized Accuracy (Negative Slope)')
    
    plt.tight_layout()
    
    # Save the plot
    dim_label = "3d" if is_3d else "2d"
    plt.show()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, f"agent_combinations_pca_{dim_label}.png"))
    plt.close()
    
    # Log information
    logger.log("\nAgent Combination PCA Visualization:")
    logger.log(f"  Total combinations: {len(X)}")
    logger.log(f"  Dimensions: {dim_label}")
    logger.log(f"  Min accuracy (original): {min(accuracies):.4f}")
    logger.log(f"  Max accuracy (original): {max(accuracies):.4f}")
    
    # Create a CSV file with the results
    df = pd.DataFrame({
        "combination": combination_names,
        "pca_x": embedding[:, 0],
        "pca_y": embedding[:, 1],
        "accuracy": accuracies,
        "accuracy_normalized": accuracies_normalized
    })
    
    # Add z dimension if 3D
    if is_3d:
        df["pca_z"] = embedding[:, 2]
    
    # Save CSV file
    os.makedirs(CSV_DIR, exist_ok=True)
    df.to_csv(os.path.join(CSV_DIR, f"agent_combinations_pca_{dim_label}.csv"), index=False)

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
    sorted_results = sorted(results, key=lambda x: x["accuracy"], reverse=True)
    
    # Display final comparison
    print("\n" + "="*50)
    print("FINAL COMPARISON OF AGENT COMBINATIONS")
    print("="*50)
    
    logger.log("\n" + "="*50)
    logger.log("FINAL COMPARISON OF AGENT COMBINATIONS")
    logger.log("="*50)
    
    print(f"\n{'Rank':^10}|{'Combination':^20}|{'Accuracy':^20}")
    print("-" * 50)
    
    for rank, result in enumerate(sorted_results):
        print(f"{rank+1:^10}|{result['combination']:^20}|{result['accuracy']:^20.4f}")
    
    logger.log(f"\n{'Rank':^10}|{'Combination':^20}|{'Accuracy':^20}")
    logger.log("-" * 50)
    
    for rank, result in enumerate(sorted_results):
        logger.log(f"{rank+1:^10}|{result['combination']:^20}|{result['accuracy']:^20.4f}")
    
    # Print the best combination
    best_combo = sorted_results[0]['combination']
    best_accuracy = sorted_results[0]['accuracy']
    
    print(f"\nBest agent combination: {best_combo}")
    print(f"Accuracy (negative slope): {best_accuracy:.4f}")
    
    logger.log(f"\nBest agent combination: {best_combo}")
    logger.log(f"Accuracy (negative slope): {best_accuracy:.4f}")
    
    # Visualize agent combinations using a multi-stage PCA approach
    logger.log("\nVisualizing agent combinations using a multi-stage PCA approach...")
    visualize_agent_combinations(results, feature_counts, feature_names, is_3d=True)
    
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
