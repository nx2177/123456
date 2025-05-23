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
n_B = 40 # Number of Agent B instances
n_C = 40 # Number of Agent C instances
n_D = 40 # Number of Agent D instances

# Weights and embedding configuration
WEIGHTS_FILE = "weights.json"
EMBEDDING_DIM = 100  # Dimension for Word2Vec embeddings
WEIGHT_MEAN = 1.0    # Mean of Gaussian distribution
WEIGHT_VARIANCE = 10000.0  # Variance of Gaussian distribution
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

def generate_structured_weights(features, count, mean=WEIGHT_MEAN, variance=WEIGHT_VARIANCE):
    """
    Generate structured weights for agent types
    
    Args:
        features (List[str]): List of feature names
        count (int): Number of agent variants to generate
        mean (float): Mean of weight distribution
        variance (float): Variance of weight distribution
        
    Returns:
        List[Dict]: List of dictionaries with agent IDs and structured weights
    """
    result = []
    
    for i in range(1, count + 1):
        weights_dict = {}
        # Generate a weight for each feature
        for feature in features:
            weights_dict[feature] = float(np.random.normal(mean, np.sqrt(variance)))
            weights_dict[feature] = max(0.1, weights_dict[feature])  # Ensure positive weights
            
        result.append({"id": i, "weights": weights_dict})
    
    return result

def load_or_generate_weights(job_description: str):
    """
    Load weights from JSON file if it exists, otherwise generate new weights and save to file
    
    Args:
        job_description (str): The job description text to generate embeddings for
        
    Returns:
        dict: Dictionary containing weights for agent types and job description embedding
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
        
        # Generate structured weights for each agent type
        b_weights = generate_structured_weights(tech_features, n_B)
        c_weights = generate_structured_weights(exp_features, n_C)
        d_weights = generate_structured_weights(soft_features, n_D)
        
        # Create data structure
        data = {
            "jd_embedding": jd_embedding,
            "weights": {
                "B": b_weights,
                "C": c_weights,
                "D": d_weights
            }
        }
        
        # Save weights to file
        with open(WEIGHTS_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.log(f"Generated new structured weights and JD embedding, saved to {WEIGHTS_FILE}")
        return data

def create_agents(weights_data, word2vec_model, jd_embedding):
    """
    Create multiple instances of each agent category
    
    Args:
        weights_data (dict): Dictionary of weights for all agents
        word2vec_model (Word2Vec): Word2Vec model for embedding
        jd_embedding (List[float]): Job description embedding
    
    Returns:
        tuple: (agent_A_list, agent_B_list, agent_C_list, agent_D_list)
    """
    agent_A_list = []
    agent_B_list = []
    agent_C_list = []
    agent_D_list = []
    
    # Create Agent A instances (no weights needed)
    for i in range(n_A):
        agent_id = f"A{i+1}"
        agent_A_list.append((agent_id, AgentA(word2vec_model=word2vec_model)))
    
    # Create Agent B instances with structured weights
    for agent_data in weights_data["B"]:
        agent_id = f"B{agent_data['id']}"
        agent_B_list.append((agent_id, AgentB(
            weights=list(agent_data["weights"].values()),  # Convert dict to list for backward compatibility
            word2vec_model=word2vec_model,
            jd_embedding=jd_embedding,
            structured_weights=agent_data["weights"]  # Pass the structured weights
        )))
    
    # Create Agent C instances with structured weights
    for agent_data in weights_data["C"]:
        agent_id = f"C{agent_data['id']}"
        agent_C_list.append((agent_id, AgentC(
            weights=list(agent_data["weights"].values()),  # Convert dict to list for backward compatibility
            word2vec_model=word2vec_model,
            jd_embedding=jd_embedding,
            structured_weights=agent_data["weights"]  # Pass the structured weights
        )))
    
    # Create Agent D instances with structured weights
    for agent_data in weights_data["D"]:
        agent_id = f"D{agent_data['id']}"
        agent_D_list.append((agent_id, AgentD(
            weights=list(agent_data["weights"].values()),  # Convert dict to list for backward compatibility
            word2vec_model=word2vec_model,
            jd_embedding=jd_embedding,
            structured_weights=agent_data["weights"]  # Pass the structured weights
        )))
    
    return agent_A_list, agent_B_list, agent_C_list, agent_D_list

def score_single_resume_with_agents(
    resume_text: str, 
    job_description: str, 
    agent_A: Tuple[str, AgentA], 
    agent_B: Tuple[str, AgentB],
    agent_C: Tuple[str, AgentC],
    agent_D: Tuple[str, AgentD],
    candidate_index: int = None
) -> Tuple[float, Dict[str, Any]]:
    """
    Score a single resume against a job description using the specified agents
    
    Args:
        resume_text (str): The resume text
        job_description (str): The job description text
        agent_A (tuple): (agent_id, agent_instance) for Agent A
        agent_B (tuple): (agent_id, agent_instance) for Agent B
        agent_C (tuple): (agent_id, agent_instance) for Agent C
        agent_D (tuple): (agent_id, agent_instance) for Agent D
        candidate_index (int): Index of the candidate for logging purposes
        
    Returns:
        tuple: (final_score, parsed_resume)
    """
    agent_A_id, agent_A_instance = agent_A
    agent_B_id, agent_B_instance = agent_B
    agent_C_id, agent_C_instance = agent_C
    agent_D_id, agent_D_instance = agent_D
    
    # Log separator and candidate info
    candidate_name = resume_text.strip().split("\n")[0]
    logger.log(f"\n{'='*50}")
    logger.log(f"CANDIDATE {candidate_index}: {candidate_name}")
    logger.log(f"AGENT COMBINATION: {agent_A_id} + {agent_B_id} + {agent_C_id} + {agent_D_id}")
    logger.log(f"{'='*50}\n")
    
    # Log resume excerpt
    logger.log("RESUME EXCERPT:")
    logger.log(resume_text[:200] + "...\n")
    
    # Agent A: Parse the resume
    logger.log(f"Running Agent {agent_A_id}: Resume Parser...")
    parsed_resume = agent_A_instance.process(resume_text)
    logger.log_json(parsed_resume, "Parsed Resume Data")
    
    # Agent B: Technical Skill Scorer - Using pre-computed JD embedding
    logger.log(f"\nRunning Agent {agent_B_id}: Technical Skill Scorer...")
    s1, s1_technical_match_details = agent_B_instance.process(parsed_resume["technical_skills"], job_description, resume_text)
    logger.log(f"technical match details: {s1_technical_match_details}")
    logger.log(f"Technical Skill Match Score (S1): {s1:.2f}")
    
    # Agent C: Experience Relevance Scorer - Using pre-computed JD embedding
    logger.log(f"\nRunning Agent {agent_C_id}: Experience Relevance Scorer...")
    s2, s2_justification = agent_C_instance.process(parsed_resume["job_experience"], job_description)
    logger.log(f"Experience Relevance Score (S2): {s2:.2f}")
    logger.log(f"Justification: {s2_justification}")
    
    # Agent D: Soft Skills Scorer - Using pre-computed JD embedding
    logger.log(f"\nRunning Agent {agent_D_id}: Soft Skills Scorer...")
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

def evaluate_ranking_accuracy(system_ranking, human_ranking, scores):
    """
    Calculate the accuracy of the system ranking compared to human ranking using linear regression
    
    Args:
        system_ranking (list): System's ranking of candidates
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

def run_evaluation_for_combination(agents_combination, resumes, job_description, human_ranking):
    """
    Run the performance evaluation pipeline for a specific agent combination
    
    Args:
        agents_combination (tuple): Tuple of (agent_A, agent_B, agent_C, agent_D)
        resumes (list): List of resume texts
        job_description (str): Job description
        human_ranking (list): Human expert ranking
        
    Returns:
        dict: Evaluation results
    """
    agent_A, agent_B, agent_C, agent_D = agents_combination
    agent_A_id, _ = agent_A
    agent_B_id, _ = agent_B
    agent_C_id, _ = agent_C
    agent_D_id, _ = agent_D
    
    combination_name = f"{agent_A_id}+{agent_B_id}+{agent_C_id}+{agent_D_id}"

    logger.log(f"\n\n{'='*30} EVALUATING COMBINATION: {combination_name} {'='*30}\n")
    
    # Score each resume and collect results
    scores = []
    candidate_names = []
    
    for i, resume in enumerate(resumes):
        candidate_name = resume.strip().split("\n")[0]
        candidate_names.append(candidate_name)
        
        score, _ = score_single_resume_with_agents(
            resume, 
            job_description, 
            agent_A, 
            agent_B,
            agent_C,
            agent_D,
            candidate_index=i+1
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
    accuracy_metrics = evaluate_ranking_accuracy(system_ranking, human_ranking, scores)
    
    # Log the rankings and accuracy to the log file
    logger.log(f"\nSystem Ranking: {system_ranking}")
    logger.log(f"Human Ranking:  {human_ranking}")
    logger.log(f"\nAccuracy (negative slope): {accuracy_metrics['accuracy']:.4f}")
    
    return {
        "combination": combination_name,
        "scores": scores,
        "system_ranking": system_ranking,
        "human_ranking": human_ranking,
        "accuracy": accuracy_metrics['accuracy'],
        "agents": {
            "A": agent_A_id,
            "B": agent_B_id,
            "C": agent_C_id,
            "D": agent_D_id
        }
    }

def visualize_agent_combinations(results, weights_data, is_3d=True):
    """
    Visualize agent combinations using a multi-stage PCA approach with color gradient based on accuracy
    
    Args:
        results (list): List of evaluation results for each agent combination
        weights_data (dict): Dictionary containing agent weights
        is_3d (bool): Whether to create 3D visualizations (True) or 2D (False)
        
    Returns:
        None
    """
    # Create directory for visualizations if it doesn't exist
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
    
    # Extract performance metrics for each agent combination
    combination_names = []
    combination_accuracies = []
    
    # Collect agent IDs and accuracies
    for result in results:
        # Get agent IDs
        agent_a_id = result["agents"]["A"]
        agent_b_id = result["agents"]["B"]
        agent_c_id = result["agents"]["C"]
        agent_d_id = result["agents"]["D"]
        
        # Store names and accuracies
        combination_names.append(f"{agent_a_id}+{agent_b_id}+{agent_c_id}+{agent_d_id}")
        combination_accuracies.append(result["accuracy"])
    
    # Convert to numpy array
    accuracies = np.array(combination_accuracies)
    
    # Step 0: Prepare data structures for per-agent PCA
    # First, get the dimensionality of each agent type's weight vector
    b_features = len(list(weights_data["B"][0]["weights"].values()))
    c_features = len(list(weights_data["C"][0]["weights"].values()))
    d_features = len(list(weights_data["D"][0]["weights"].values()))
    
    # Log the raw feature dimensions for each agent type
    logger.log(f"\nRaw feature dimensions:")
    logger.log(f"  Agent B features: {b_features}")
    logger.log(f"  Agent C features: {c_features}")
    logger.log(f"  Agent D features: {d_features}")
    
    # Define the target dimensionality for per-agent PCA
    pca_dim = 5
    
    # Create a scaler and PCA model for each agent type
    b_scaler = StandardScaler()
    c_scaler = StandardScaler()
    d_scaler = StandardScaler()
    
    # Create PCA models for each agent type
    b_pca = PCA(n_components=min(pca_dim, b_features))
    c_pca = PCA(n_components=min(pca_dim, c_features))
    d_pca = PCA(n_components=min(pca_dim, d_features))
    
    # Step 1: Collect and process all weights for fitting the scalers and PCAs
    all_b_weights = []
    all_c_weights = []
    all_d_weights = []
    agent_b_ids = []
    agent_c_ids = []
    agent_d_ids = []
    
    for result in results:
        # Get agent IDs
        agent_b_id = result["agents"]["B"]
        agent_c_id = result["agents"]["C"]
        agent_d_id = result["agents"]["D"]
        
        # Store agent IDs
        agent_b_ids.append(agent_b_id)
        agent_c_ids.append(agent_c_id)
        agent_d_ids.append(agent_d_id)
        
        # Get agent indices
        b_idx = int(agent_b_id[1:]) - 1
        c_idx = int(agent_c_id[1:]) - 1
        d_idx = int(agent_d_id[1:]) - 1
        
        # Get raw weights for each agent
        b_weights = list(weights_data["B"][b_idx]["weights"].values())
        c_weights = list(weights_data["C"][c_idx]["weights"].values())
        d_weights = list(weights_data["D"][d_idx]["weights"].values())
        
        # Add to collection
        all_b_weights.append(b_weights)
        all_c_weights.append(c_weights)
        all_d_weights.append(d_weights)
    
    # Convert to numpy arrays
    all_b_weights = np.array(all_b_weights)
    all_c_weights = np.array(all_c_weights)
    all_d_weights = np.array(all_d_weights)
    
    # Fit the scalers to each agent type's weights
    b_scaler.fit(all_b_weights)
    c_scaler.fit(all_c_weights)
    d_scaler.fit(all_d_weights)
    
    # Scale the weights
    scaled_b_weights = b_scaler.transform(all_b_weights)
    scaled_c_weights = c_scaler.transform(all_c_weights)
    scaled_d_weights = d_scaler.transform(all_d_weights)
    
    # Fit the PCA models to the scaled weights
    b_pca.fit(scaled_b_weights)
    c_pca.fit(scaled_c_weights)
    d_pca.fit(scaled_d_weights)
    
    # Log the explained variance for each agent's PCA
    logger.log(f"\nExplained variance ratios for per-agent PCA:")
    logger.log(f"  Agent B: {b_pca.explained_variance_ratio_}")
    logger.log(f"  Agent C: {c_pca.explained_variance_ratio_}")
    logger.log(f"  Agent D: {d_pca.explained_variance_ratio_}")
    
    # Step 2: Apply PCA to each agent's weights and normalize
    # Prepare arrays to store the PCA features for each agent type
    b_pca_features_all = []
    c_pca_features_all = []
    d_pca_features_all = []
    concatenated_features = []
    
    # Create a color normalizer for accuracy values
    accuracy_scaler = StandardScaler()
    accuracies_normalized = accuracy_scaler.fit_transform(accuracies.reshape(-1, 1)).flatten()
    
    # Determine color normalization range
    max_abs_accuracy = max(abs(accuracies_normalized))
    norm = plt.Normalize(-max_abs_accuracy, max_abs_accuracy)
    
    # Create colormap
    colors = ["blue", "white", "red"]
    cmap = LinearSegmentedColormap.from_list("BWR", colors)
    
    for i in range(len(results)):
        # Apply PCA to each agent's scaled weights
        b_pca_features = b_pca.transform(scaled_b_weights[i].reshape(1, -1))[0]
        c_pca_features = c_pca.transform(scaled_c_weights[i].reshape(1, -1))[0]
        d_pca_features = d_pca.transform(scaled_d_weights[i].reshape(1, -1))[0]
        
        # Store PCA features
        b_pca_features_all.append(b_pca_features)
        c_pca_features_all.append(c_pca_features)
        d_pca_features_all.append(d_pca_features)
        
        # Step 3: Normalize each PCA vector using L2 norm
        b_norm = np.linalg.norm(b_pca_features)
        c_norm = np.linalg.norm(c_pca_features)
        d_norm = np.linalg.norm(d_pca_features)
        
        # Avoid division by zero
        if b_norm > 0:
            b_pca_features = b_pca_features / b_norm
        if c_norm > 0:
            c_pca_features = c_pca_features / c_norm
        if d_norm > 0:
            d_pca_features = d_pca_features / d_norm
        
        # Step 4: Concatenate the normalized PCA vectors
        combined_features = np.concatenate([b_pca_features, c_pca_features, d_pca_features])
        concatenated_features.append(combined_features)
    
    # Convert to numpy arrays
    b_pca_features_all = np.array(b_pca_features_all)
    c_pca_features_all = np.array(c_pca_features_all)
    d_pca_features_all = np.array(d_pca_features_all)
    
    # Part 2: Visualize PCA-Reduced Agent Vectors Separately
    # Function to visualize agent-specific PCA
    def visualize_agent_pca(agent_features, agent_type, agent_ids):
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
        plt.savefig(os.path.join(VISUALIZATIONS_DIR, f"agent_{agent_type}_pca_{dim_label}.png"))
        plt.close()
        
        # Log information
        logger.log(f"\nAgent {agent_type} PCA Visualization:")
        logger.log(f"  Explained variance: {agent_pca.explained_variance_ratio_}")
        
        # Create a CSV file with the results
        df = pd.DataFrame({
            "agent_id": agent_ids,
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
    visualize_agent_pca(b_pca_features_all, "B", agent_b_ids)
    visualize_agent_pca(c_pca_features_all, "C", agent_c_ids)
    visualize_agent_pca(d_pca_features_all, "D", agent_d_ids)
    
    # Convert to numpy array for global PCA
    X = np.array(concatenated_features)
    
    # Skip if we don't have enough data points
    if len(X) < 2:
        logger.log("Not enough agent combinations for global PCA (need at least 2)")
        return
    
    # Step 5: Normalize the combined feature vectors
    global_scaler = StandardScaler()
    X_normalized = global_scaler.fit_transform(X)
    
    # Step 6: Apply global PCA to reduce to 2D or 3D
    n_components = 3 if is_3d else 2
    global_pca = PCA(n_components=n_components)
    embedding = global_pca.fit_transform(X_normalized)
    
    # Print global PCA explained variance
    logger.log(f"\nGlobal PCA explained variance: {global_pca.explained_variance_ratio_}")
    print(f"Global PCA explained variance: {global_pca.explained_variance_ratio_}")
    
    # Step 7: Create the visualization
    if is_3d:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create 3D scatter plot
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            embedding[:, 2],
            c=accuracies_normalized,
            cmap=cmap,
            norm=norm,
            s=30,
        )
        
        # Add labels
        ax.set_title("Agent Combination Weight Space (Multi-Stage PCA - 3D)")
        ax.set_xlabel(f"Global PCA Dimension 1 (Explained Variance: {global_pca.explained_variance_ratio_[0]:.2f})")
        ax.set_ylabel(f"Global PCA Dimension 2 (Explained Variance: {global_pca.explained_variance_ratio_[1]:.2f})")
        ax.set_zlabel(f"Global PCA Dimension 3 (Explained Variance: {global_pca.explained_variance_ratio_[2]:.2f})")
    else:
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create 2D scatter plot
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=accuracies_normalized,
            cmap=cmap,
            norm=norm,
            s=30,
        )
        
        # Add labels
        ax.set_title("Agent Combination Weight Space (Multi-Stage PCA - 2D)")
        ax.set_xlabel(f"Global PCA Dimension 1 (Explained Variance: {global_pca.explained_variance_ratio_[0]:.2f})")
        ax.set_ylabel(f"Global PCA Dimension 2 (Explained Variance: {global_pca.explained_variance_ratio_[1]:.2f})")
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Normalized Accuracy (Negative Slope)')
    
    plt.tight_layout()
    
    # Save the plot
    dim_label = "3d" if is_3d else "2d"
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
    weights = data["weights"]
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
    
    # Create agent instances using fixed weights and shared JD embedding
    agent_A_list, agent_B_list, agent_C_list, agent_D_list = create_agents(weights, word2vec_model, jd_embedding)
    
    # Log the number of agents in each category
    logger.log(f"\nNumber of agents per category:")
    logger.log(f"Category A: {n_A} agents")
    logger.log(f"Category B: {n_B} agents")
    logger.log(f"Category C: {n_C} agents")
    logger.log(f"Category D: {n_D} agents")
    logger.log(f"Total combinations to evaluate: {n_A * n_B * n_C * n_D}")
    
    # Generate all combinations of agents
    all_combinations = list(itertools.product(agent_A_list, agent_B_list, agent_C_list, agent_D_list))
    print(f"\nEvaluating {len(all_combinations)} agent combinations...")
    
    # Evaluate each combination
    results = []
    for i, combo in enumerate(all_combinations):
        
        result = run_evaluation_for_combination(combo, resumes, job_description, human_ranking)
        results.append(result)
    
    # Sort results by accuracy (higher value is better)
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
    visualize_agent_combinations(results, weights, is_3d=True)

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
