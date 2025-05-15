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
import umap
import hdbscan
from collections import defaultdict

# Configuration for number of agents in each category
n_A = 1  # Number of Agent A instances
n_B = 20  # Number of Agent B instances
n_C = 20  # Number of Agent C instances
n_D = 20  # Number of Agent D instances

# Weights and embedding configuration
WEIGHTS_FILE = "weights.json"
EMBEDDING_DIM = 100  # Dimension for Word2Vec embeddings
WEIGHT_MEAN = 1.0    # Mean of Gaussian distribution
WEIGHT_VARIANCE = 1.0  # Variance of Gaussian distribution
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

def evaluate_ranking_accuracy(system_ranking, human_ranking):
    """
    Calculate the accuracy of the system ranking compared to human ranking
    
    Args:
        system_ranking (list): System's ranking of candidates
        human_ranking (list): Human's ranking of candidates
        
    Returns:
        dict: Accuracy metrics
    """
    # Calculate the number of correct positions
    correct_positions = sum(1 for s, h in zip(system_ranking, human_ranking) if s == h)
    
    # Calculate accuracy
    accuracy = correct_positions / len(human_ranking)
    
    # Calculate Spearman rank correlation
    system_ranks = {candidate: rank for rank, candidate in enumerate(system_ranking)}
    human_ranks = {candidate: rank for rank, candidate in enumerate(human_ranking)}
    
    n = len(human_ranking)
    
    # Calculate sum of squared differences in ranks
    sum_squared_diff = sum((system_ranks[candidate] - human_ranks[candidate])**2 
                          for candidate in range(1, n+1))
    
    # Calculate Spearman's rank correlation coefficient
    spearman_corr = 1 - (6 * sum_squared_diff) / (n * (n**2 - 1))
    
    return {
        "exact_match_accuracy": accuracy,
        "spearman_correlation": spearman_corr
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
    accuracy_metrics = evaluate_ranking_accuracy(system_ranking, human_ranking)
    
    # Log the rankings and accuracy to the log file
    logger.log(f"\nSystem Ranking: {system_ranking}")
    logger.log(f"Human Ranking:  {human_ranking}")
    logger.log(f"\nExact Position Accuracy: {accuracy_metrics['exact_match_accuracy']:.2f}")
    logger.log(f"Spearman Rank Correlation: {accuracy_metrics['spearman_correlation']:.2f}")
    
    return {
        "combination": combination_name,
        "scores": scores,
        "system_ranking": system_ranking,
        "human_ranking": human_ranking,
        "accuracy": accuracy_metrics,
        "agents": {
            "A": agent_A_id,
            "B": agent_B_id,
            "C": agent_C_id,
            "D": agent_D_id
        },
        "spearman": accuracy_metrics["spearman_correlation"],
        "exact_accuracy": accuracy_metrics["exact_match_accuracy"]
    }

def visualize_agent_weights(agent_type, weights_data, agent_performance):
    """
    Visualize agent weights using UMAP + HDBSCAN
    
    Args:
        agent_type (str): Agent type (B, C or D)
        weights_data (dict): Dictionary containing agent weights
        agent_performance (dict): Dictionary mapping agent ID to performance metrics
        
    Returns:
        tuple: (umap_embedding, clusters, performance_data)
    """
    # Create directory for visualizations if it doesn't exist
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
    
    # Extract features and weights for this agent type
    data = []
    agent_ids = []
    
    for agent_data in weights_data[agent_type]:
        agent_id = f"{agent_type}{agent_data['id']}"
        agent_ids.append(agent_id)
        # Extract weights values as a flat vector
        weights_flat = list(agent_data["weights"].values())
        data.append(weights_flat)
    
    # Convert to numpy array
    X = np.array(data)
    
    # Skip if we don't have enough data points
    if len(X) < 2:
        logger.log(f"Not enough {agent_type} agents for clustering (need at least 2)")
        return None, None, None
    
    # Apply UMAP dimension reduction
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(X)
    
    # Apply HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, gen_min_span_tree=True)
    try:
        cluster_labels = clusterer.fit_predict(embedding)
    except Exception as e:
        logger.log(f"Clustering failed for {agent_type} agents: {str(e)}")
        cluster_labels = np.zeros(len(embedding))  # Default to all zero labels
    
    # Create dataframe with results
    performance_data = []
    
    for i, agent_id in enumerate(agent_ids):
        avg_accuracy = agent_performance.get(agent_id, {}).get("avg_accuracy", 0)
        avg_spearman = agent_performance.get(agent_id, {}).get("avg_spearman", 0)
        
        performance_data.append({
            "agent_id": agent_id,
            "umap_x": embedding[i, 0],
            "umap_y": embedding[i, 1],
            "cluster": int(cluster_labels[i]),
            "avg_accuracy": avg_accuracy,
            "avg_spearman": avg_spearman,
            "weights": X[i].tolist()
        })
    
    # Create dataframe
    df = pd.DataFrame(performance_data)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Get unique clusters
    unique_clusters = np.unique(cluster_labels)
    
    # Plot points
    for cluster in unique_clusters:
        cluster_points = df[df["cluster"] == cluster]
        if cluster == -1:  # Noise points in HDBSCAN
            plt.scatter(
                cluster_points["umap_x"], 
                cluster_points["umap_y"], 
                c="black", 
                marker="x", 
                label=f"Noise", 
                s=100
            )
        else:
            plt.scatter(
                cluster_points["umap_x"], 
                cluster_points["umap_y"], 
                marker="o", 
                label=f"Cluster {cluster}", 
                s=100
            )
    
    # Add labels for each point
    for _, row in df.iterrows():
        plt.annotate(
            row["agent_id"], 
            (row["umap_x"], row["umap_y"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center'
        )
        plt.annotate(
            f"Acc: {row['avg_accuracy']:.2f}",
            (row["umap_x"], row["umap_y"]),
            textcoords="offset points",
            xytext=(0, -10),
            ha='center',
            fontsize=8
        )
    
    plt.title(f"Agent {agent_type} Weight Space Visualization (UMAP + HDBSCAN)")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, f"agent_{agent_type}_weights.png"))
    plt.close()
    
    # Save CSV file
    os.makedirs(CSV_DIR, exist_ok=True)
    df.to_csv(os.path.join(CSV_DIR, f"agent_{agent_type}_analysis.csv"), index=False)
    
    # Print summary
    cluster_counts = df["cluster"].value_counts().to_dict()
    logger.log(f"\nAgent {agent_type} Cluster Summary:")
    for cluster, count in cluster_counts.items():
        cluster_avg_acc = df[df["cluster"] == cluster]["avg_accuracy"].mean()
        logger.log(f"  Cluster {cluster}: {count} agents, Avg Accuracy: {cluster_avg_acc:.2f}")
    
    return embedding, cluster_labels, df

def calculate_agent_performance(results):
    """
    Calculate performance metrics for each agent
    
    Args:
        results (list): List of evaluation results
    
    Returns:
        dict: Dictionary mapping agent ID to performance metrics
    """
    agent_results = defaultdict(list)
    
    # Group results by agent
    for result in results:
        for agent_type, agent_id in result["agents"].items():
            agent_results[agent_id].append({
                "exact_accuracy": result["exact_accuracy"],
                "spearman": result["spearman"]
            })
    
    # Calculate average performance for each agent
    agent_performance = {}
    for agent_id, performances in agent_results.items():
        avg_accuracy = np.mean([p["exact_accuracy"] for p in performances])
        avg_spearman = np.mean([p["spearman"] for p in performances])
        
        agent_performance[agent_id] = {
            "avg_accuracy": avg_accuracy,
            "avg_spearman": avg_spearman,
            "num_combinations": len(performances)
        }
    
    return agent_performance

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
    
    # Sort results by Spearman correlation (higher is better)
    sorted_results = sorted(results, key=lambda x: x["spearman"], reverse=True)
    
    # Display final comparison
    print("\n" + "="*50)
    print("FINAL COMPARISON OF AGENT COMBINATIONS")
    print("="*50)
    
    logger.log("\n" + "="*50)
    logger.log("FINAL COMPARISON OF AGENT COMBINATIONS")
    logger.log("="*50)
    
    print(f"\n{'Rank':^10}|{'Combination':^20}|{'Exact Accuracy':^20}|{'Spearman':^20}")
    print("-" * 70)
    
    for rank, result in enumerate(sorted_results):
        print(f"{rank+1:^10}|{result['combination']:^20}|{result['accuracy']['exact_match_accuracy']:^20.2f}|{result['spearman']:^20.2f}")
    
    logger.log(f"\n{'Rank':^10}|{'Combination':^20}|{'Exact Accuracy':^20}|{'Spearman':^20}")
    logger.log("-" * 70)
    
    for rank, result in enumerate(sorted_results):
        logger.log(f"{rank+1:^10}|{result['combination']:^20}|{result['accuracy']['exact_match_accuracy']:^20.2f}|{result['spearman']:^20.2f}")
    
    # Print the best combination
    best_combo = sorted_results[0]['combination']
    best_accuracy = sorted_results[0]['accuracy']['exact_match_accuracy']
    best_spearman = sorted_results[0]['spearman']
    
    print(f"\nBest agent combination: {best_combo}")
    print(f"Exact position accuracy: {best_accuracy:.2f}")
    print(f"Spearman correlation: {best_spearman:.2f}")
    
    logger.log(f"\nBest agent combination: {best_combo}")
    logger.log(f"Exact position accuracy: {best_accuracy:.2f}")
    logger.log(f"Spearman correlation: {best_spearman:.2f}")
    
    # Calculate agent performance
    agent_performance = calculate_agent_performance(results)
    
    # Visualize agent weights
    logger.log("\nVisualizing agent weights...")
    for agent_type in ["B", "C", "D"]:
        visualize_agent_weights(agent_type, weights, agent_performance)
    
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