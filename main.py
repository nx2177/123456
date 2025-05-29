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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Configuration for number of agents in each category
n_A = 1  # Number of Agent A instances
n_combinations = 5000  # Number of agent combinations to generate

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

# Sigmoid transformation and filtering parameters
NORMALIZATION_CONSTANT_L = 0.1  # Normalization factor for accuracy scores
FILTERING_INTERVAL_WIDTH_A = 0.01  # Width of exclusion interval around 0.5

# Scatter plot visualization parameters
SCATTER_ALPHA = 0.8  # Transparency for scatter plot points
SCATTER_SIZE = 20   # Size of scatter plot points

# Neural Network configuration parameters
NN_OUTPUT_MODE = 1  # 1: Regression, 2: Binary Classification, 3: Three-Class Classification
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

class AccuracyPredictor:
    """
    Neural Network model to predict agent combination accuracy with multiple output modes
    """
    
    def __init__(self, input_dim, output_mode=1, hidden_layers=None):
        """
        Initialize the AccuracyPredictor
        
        Args:
            input_dim (int): Dimension of input weight vectors
            output_mode (int): 1=Regression, 2=Binary Classification, 3=Three-Class Classification
            hidden_layers (list): List of hidden layer sizes
        """
        self.input_dim = input_dim
        self.output_mode = output_mode
        self.hidden_layers = hidden_layers or NN_HIDDEN_LAYERS
        self.model = None
        self.history = None
        
    def _prepare_targets(self, accuracies_raw):
        """
        Prepare target values based on output mode
        
        Args:
            accuracies_raw (np.array): Raw accuracy values
            
        Returns:
            np.array: Prepared target values
        """
        if self.output_mode == 1:  # Regression
            return accuracies_raw
            
        elif self.output_mode == 2:  # Binary Classification
            # Good = accuracy > 0, Bad = accuracy <= 0
            binary_labels = (accuracies_raw > 0).astype(int)
            return tf.keras.utils.to_categorical(binary_labels, num_classes=2)
            
        elif self.output_mode == 3:  # Three-Class Classification
            # Bad if accuracy < -a, Mid if -a <= accuracy <= a, Good if accuracy > a
            a = FILTERING_INTERVAL_WIDTH_A
            labels = np.zeros(len(accuracies_raw), dtype=int)
            labels[accuracies_raw < -a] = 0  # Bad
            labels[(accuracies_raw >= -a) & (accuracies_raw <= a)] = 1  # Mid
            labels[accuracies_raw > a] = 2  # Good
            return tf.keras.utils.to_categorical(labels, num_classes=3)
            
        else:
            raise ValueError(f"Invalid output_mode: {self.output_mode}")
    
    def _build_model(self):
        """
        Build the neural network model based on output mode
        """
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Dense(self.hidden_layers[0], 
                              activation='relu', 
                              input_shape=(self.input_dim,),
                              name='hidden_1'))
        
        # Hidden layers
        for i, units in enumerate(self.hidden_layers[1:], 2):
            model.add(layers.Dense(units, activation='relu', name=f'hidden_{i}'))
        
        # Output layer based on mode
        if self.output_mode == 1:  # Regression
            model.add(layers.Dense(1, activation='linear', name='output'))
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=NN_LEARNING_RATE),
                         loss='mse',
                         metrics=['mae'])
            
        elif self.output_mode == 2:  # Binary Classification
            model.add(layers.Dense(2, activation='softmax', name='output'))
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=NN_LEARNING_RATE),
                         loss='categorical_crossentropy',
                         metrics=['accuracy'])
            
        elif self.output_mode == 3:  # Three-Class Classification
            model.add(layers.Dense(3, activation='softmax', name='output'))
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=NN_LEARNING_RATE),
                         loss='categorical_crossentropy',
                         metrics=['accuracy'])
        
        return model
    
    def train(self, X_train, y_train_raw, X_val=None, y_val_raw=None, verbose=1):
        """
        Train the neural network model
        
        Args:
            X_train (np.array): Training input data (weight vectors)
            y_train_raw (np.array): Training target data (raw accuracies)
            X_val (np.array): Validation input data
            y_val_raw (np.array): Validation target data
            verbose (int): Verbosity level
            
        Returns:
            keras.callbacks.History: Training history
        """
        # Prepare targets based on output mode
        y_train = self._prepare_targets(y_train_raw)
        y_val = self._prepare_targets(y_val_raw) if y_val_raw is not None else None
        
        # Build model
        self.model = self._build_model()
        
        # Print model summary
        if verbose:
            print(f"\nNeural Network Architecture (Mode {self.output_mode}):")
            self.model.summary()
        
        # Prepare validation data
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            epochs=NN_EPOCHS,
            batch_size=NN_BATCH_SIZE,
            validation_data=validation_data,
            verbose=verbose
        )
        
        return self.history
    
    def predict(self, X):
        """
        Make predictions using the trained model
        
        Args:
            X (np.array): Input data
            
        Returns:
            np.array: Predictions
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = self.model.predict(X)
        
        # For classification, convert probabilities to class labels
        if self.output_mode in [2, 3]:
            predictions = np.argmax(predictions, axis=1)
            
        return predictions
    
    def evaluate(self, X_test, y_test_raw, verbose=1):
        """
        Evaluate the model on test data
        
        Args:
            X_test (np.array): Test input data
            y_test_raw (np.array): Test target data (raw accuracies)
            verbose (int): Verbosity level
            
        Returns:
            dict: Evaluation results
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        # Prepare targets
        y_test = self._prepare_targets(y_test_raw)
        
        # Evaluate model
        results = {}
        test_loss, test_metric = self.model.evaluate(X_test, y_test, verbose=0)
        results['test_loss'] = test_loss
        
        if self.output_mode == 1:  # Regression
            results['test_mae'] = test_metric
            if verbose:
                print(f"\nRegression Results:")
                print(f"Test Loss (MSE): {test_loss:.4f}")
                print(f"Test MAE: {test_metric:.4f}")
                
        else:  # Classification
            results['test_accuracy'] = test_metric
            
            # Get predictions for confusion matrix and classification report
            y_pred = self.predict(X_test)
            y_true = np.argmax(y_test, axis=1)
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            results['confusion_matrix'] = cm
            
            # Classification report
            if self.output_mode == 2:
                target_names = ['Bad', 'Good']
            else:  # mode 3
                target_names = ['Bad', 'Mid', 'Good']
            
            class_report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
            results['classification_report'] = class_report
            
            if verbose:
                print(f"\nClassification Results:")
                print(f"Test Loss: {test_loss:.4f}")
                print(f"Test Accuracy: {test_metric:.4f}")
                print(f"\nConfusion Matrix:")
                print(cm)
                print(f"\nClassification Report:")
                print(classification_report(y_true, y_pred, target_names=target_names))
        
        return results
    
    def plot_training_history(self):
        """
        Plot training history
        """
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        axes[0].plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            axes[0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        
        # Plot metric
        metric_name = 'mae' if self.output_mode == 1 else 'accuracy'
        if metric_name in self.history.history:
            axes[1].plot(self.history.history[metric_name], label=f'Training {metric_name.upper()}')
            if f'val_{metric_name}' in self.history.history:
                axes[1].plot(self.history.history[f'val_{metric_name}'], label=f'Validation {metric_name.upper()}')
            axes[1].set_title(f'Model {metric_name.upper()}')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel(metric_name.upper())
            axes[1].legend()
        
        plt.tight_layout()
        
        # Save the plot
        os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
        plt.savefig(os.path.join(VISUALIZATIONS_DIR, f'nn_training_history_mode_{self.output_mode}.png'))
        plt.show()
        plt.close()

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
    
    # Apply normalization and sigmoid transformation
    normalized_accuracy = raw_accuracy / NORMALIZATION_CONSTANT_L
    
    # Apply sigmoid transformation
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    transformed_accuracy = sigmoid(normalized_accuracy)
    
    return {
        "accuracy_raw": raw_accuracy,
        "accuracy_normalized": normalized_accuracy,
        "accuracy": transformed_accuracy  # This is the final transformed accuracy
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
    logger.log(f"Normalized Accuracy: {accuracy_metrics['accuracy_normalized']:.4f}")
    logger.log(f"Transformed Accuracy (sigmoid): {accuracy_metrics['accuracy']:.4f}")
    
    return {
        "combination": combination_name,
        "combination_index": combination_index,
        "combination_weights": combination_weights,
        "scores": scores,
        "system_ranking": system_ranking,
        "human_ranking": human_ranking,
        "accuracy_raw": accuracy_metrics['accuracy_raw'],
        "accuracy_normalized": accuracy_metrics['accuracy_normalized'],
        "accuracy": accuracy_metrics['accuracy']  # This is the transformed accuracy
    }

def visualize_agent_combinations(results, feature_counts, feature_names, is_3d=True):
    """
    Visualize agent combinations using a multi-stage PCA approach with sigmoid transformation and filtering
    
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
        combination_accuracies_transformed.append(result["accuracy"])  # Already sigmoid-transformed
        all_combination_weights.append(result["combination_weights"])
    
    # Convert to numpy arrays
    accuracies_raw = np.array(combination_accuracies_raw)
    accuracies_transformed = np.array(combination_accuracies_transformed)  # Already sigmoid values 0-1
    all_combination_weights = np.array(all_combination_weights)
    
    # Step 3: Filter out combinations with scores in the exclusion interval [0.5-a, 0.5+a]
    # Use the already-transformed sigmoid scores
    exclusion_lower = 0.5 - FILTERING_INTERVAL_WIDTH_A
    exclusion_upper = 0.5 + FILTERING_INTERVAL_WIDTH_A
    
    # Create filter mask for combinations outside the exclusion interval
    filter_mask = (accuracies_transformed < exclusion_lower) | (accuracies_transformed > exclusion_upper)
    
    # Apply filter to all data
    filtered_combination_names = [name for i, name in enumerate(combination_names) if filter_mask[i]]
    filtered_accuracies_raw = accuracies_raw[filter_mask]
    filtered_accuracies_transformed = accuracies_transformed[filter_mask]
    filtered_combination_weights = all_combination_weights[filter_mask]
    
    logger.log(f"\nSigmoid Transformation and Filtering:")
    logger.log(f"  Normalization constant L: {NORMALIZATION_CONSTANT_L}")
    logger.log(f"  Filtering interval width a: {FILTERING_INTERVAL_WIDTH_A}")
    logger.log(f"  Exclusion interval: [{exclusion_lower:.2f}, {exclusion_upper:.2f}]")
    logger.log(f"  Original combinations: {len(combination_names)}")
    logger.log(f"  Filtered combinations: {len(filtered_combination_names)}")
    logger.log(f"  Filtered out: {len(combination_names) - len(filtered_combination_names)}")
    
    # Check if we have enough filtered combinations for visualization
    if len(filtered_combination_names) < 2:
        logger.log("Not enough filtered combinations for PCA visualization (need at least 2)")
        print("Warning: Not enough filtered combinations for PCA visualization")
        return
    
    # Log the raw feature dimensions
    logger.log(f"\nRaw feature dimensions:")
    logger.log(f"  Agent B features: {b_features}")
    logger.log(f"  Agent C features: {c_features}")
    logger.log(f"  Agent D features: {d_features}")
    logger.log(f"  Total features per combination: {b_features + c_features + d_features}")
    
    # Split combination weights into agent-specific weights (using filtered data)
    b_weights_all = filtered_combination_weights[:, :b_features]
    c_weights_all = filtered_combination_weights[:, b_features:b_features + c_features]
    d_weights_all = filtered_combination_weights[:, b_features + c_features:]
    
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
    
    # Create color normalization and colormap for sigmoid scores
    # Sigmoid scores range from 0 to 1, with 0.5 as neutral (now filtered out)
    norm = plt.Normalize(0, 1)
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
            
        # Plot the points using sigmoid scores for coloring
        if is_3d:
            scatter = ax.scatter(
                agent_embedding[:, 0],
                agent_embedding[:, 1],
                agent_embedding[:, 2],
                c=filtered_accuracies_transformed,
                cmap=cmap,
                norm=norm,
                s=SCATTER_SIZE,
                alpha=SCATTER_ALPHA
            )
        else:
            scatter = ax.scatter(
                agent_embedding[:, 0],
                agent_embedding[:, 1],
                c=filtered_accuracies_transformed,
                cmap=cmap,
                norm=norm,
                s=SCATTER_SIZE,
                alpha=SCATTER_ALPHA
            )
            
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Sigmoid-Transformed Accuracy (0=Poor, 1=Excellent)')
        
        # Add labels
        ax.set_title(f"Agent {agent_type} PCA Projection (Filtered)")
        ax.set_xlabel(f"PCA Dimension 1 (Explained Variance: {agent_pca.explained_variance_ratio_[0]:.2f})")
        ax.set_ylabel(f"PCA Dimension 2 (Explained Variance: {agent_pca.explained_variance_ratio_[1]:.2f})")
        if is_3d:
            ax.set_zlabel(f"PCA Dimension 3 (Explained Variance: {agent_pca.explained_variance_ratio_[2]:.2f})")
        
        plt.tight_layout()
        
        # Save the plot
        dim_label = "3d" if is_3d else "2d"
        plt.savefig(os.path.join(VISUALIZATIONS_DIR, f"agent_{agent_type}_pca_{dim_label}_filtered.png"))
        plt.show()
        plt.close()
        
        # Log information
        logger.log(f"\nAgent {agent_type} PCA Visualization (Filtered):")
        logger.log(f"  Explained variance: {agent_pca.explained_variance_ratio_}")
        
        # Create a CSV file with the results
        df = pd.DataFrame({
            "combination": filtered_combination_names,
            "pca_x": agent_embedding[:, 0],
            "pca_y": agent_embedding[:, 1],
            "accuracy_original": filtered_accuracies_raw,
            "accuracy_sigmoid": filtered_accuracies_transformed
        })
        
        if is_3d:
            df["pca_z"] = agent_embedding[:, 2]
            
        # Save CSV file
        os.makedirs(CSV_DIR, exist_ok=True)
        df.to_csv(os.path.join(CSV_DIR, f"agent_{agent_type}_pca_{dim_label}_filtered.csv"), index=False)
    
    # Visualize each agent type separately
    visualize_agent_pca(b_pca_features, "B")
    visualize_agent_pca(c_pca_features, "C")
    visualize_agent_pca(d_pca_features, "D")
    
    # Normalize each agent's PCA features using L2 norm
    concatenated_features = []
    for i in range(len(filtered_combination_names)):
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
        logger.log("Not enough filtered combinations for global PCA (need at least 2)")
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
            c=filtered_accuracies_transformed,
            cmap=cmap,
            norm=norm,
            s=SCATTER_SIZE,
            alpha=SCATTER_ALPHA
        )
        
        ax.set_title("Agent Combination Weight Space (Multi-Stage PCA - 3D, Filtered)")
        ax.set_xlabel(f"Global PCA Dimension 1 (Explained Variance: {global_pca.explained_variance_ratio_[0]:.2f})")
        ax.set_ylabel(f"Global PCA Dimension 2 (Explained Variance: {global_pca.explained_variance_ratio_[1]:.2f})")
        ax.set_zlabel(f"Global PCA Dimension 3 (Explained Variance: {global_pca.explained_variance_ratio_[2]:.2f})")
    else:
        fig, ax = plt.subplots(figsize=(12, 10))
        
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=filtered_accuracies_transformed,
            cmap=cmap,
            norm=norm,
            s=SCATTER_SIZE,
            alpha=SCATTER_ALPHA
        )
        
        ax.set_title("Agent Combination Weight Space (Multi-Stage PCA - 2D, Filtered)")
        ax.set_xlabel(f"Global PCA Dimension 1 (Explained Variance: {global_pca.explained_variance_ratio_[0]:.2f})")
        ax.set_ylabel(f"Global PCA Dimension 2 (Explained Variance: {global_pca.explained_variance_ratio_[1]:.2f})")
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Sigmoid-Transformed Accuracy (0=Poor, 1=Excellent)')
    
    plt.tight_layout()
    
    # Save the plot
    dim_label = "3d" if is_3d else "2d"
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, f"agent_combinations_pca_{dim_label}_filtered.png"))
    plt.show()
    plt.close()
    
    # Log information
    logger.log("\nAgent Combination PCA Visualization (Filtered):")
    logger.log(f"  Total filtered combinations: {len(X)}")
    logger.log(f"  Dimensions: {dim_label}")
    logger.log(f"  Min accuracy (original): {min(filtered_accuracies_raw):.4f}")
    logger.log(f"  Max accuracy (original): {max(filtered_accuracies_raw):.4f}")
    logger.log(f"  Min sigmoid score: {min(filtered_accuracies_transformed):.4f}")
    logger.log(f"  Max sigmoid score: {max(filtered_accuracies_transformed):.4f}")
    
    # Create a CSV file with the results
    df = pd.DataFrame({
        "combination": filtered_combination_names,
        "pca_x": embedding[:, 0],
        "pca_y": embedding[:, 1],
        "accuracy_original": filtered_accuracies_raw,
        "accuracy_sigmoid": filtered_accuracies_transformed
    })
    
    # Add z dimension if 3D
    if is_3d:
        df["pca_z"] = embedding[:, 2]
    
    # Save CSV file
    os.makedirs(CSV_DIR, exist_ok=True)
    df.to_csv(os.path.join(CSV_DIR, f"agent_combinations_pca_{dim_label}_filtered.csv"), index=False)

def train_accuracy_predictor(results, feature_counts, output_mode=None):
    """
    Train and evaluate a neural network to predict agent combination accuracy
    
    Args:
        results (list): List of evaluation results for each agent combination
        feature_counts (dict): Dictionary with feature counts for each agent
        output_mode (int): Neural network output mode (1, 2, or 3). If None, uses NN_OUTPUT_MODE
        
    Returns:
        AccuracyPredictor: Trained neural network model
    """
    if output_mode is None:
        output_mode = NN_OUTPUT_MODE
    
    logger.log(f"\n{'='*50}")
    logger.log(f"TRAINING NEURAL NETWORK PREDICTOR (MODE {output_mode})")
    logger.log(f"{'='*50}")
    
    # Extract data from results
    X = []  # Weight vectors
    y_raw = []  # Raw accuracy values
    combination_names = []
    
    for result in results:
        X.append(result["combination_weights"])
        y_raw.append(result["accuracy_raw"])
        combination_names.append(result["combination"])
    
    # Convert to numpy arrays
    X = np.array(X)
    y_raw = np.array(y_raw)
    
    # Log dataset information
    input_dim = X.shape[1]
    n_samples = X.shape[0]
    
    logger.log(f"\nDataset Information:")
    logger.log(f"  Number of samples: {n_samples}")
    logger.log(f"  Input dimension: {input_dim}")
    logger.log(f"  Feature breakdown:")
    logger.log(f"    Agent B features: {feature_counts['b_features']}")
    logger.log(f"    Agent C features: {feature_counts['c_features']}")
    logger.log(f"    Agent D features: {feature_counts['d_features']}")
    logger.log(f"  Raw accuracy range: [{y_raw.min():.4f}, {y_raw.max():.4f}]")
    
    print(f"\nTraining Neural Network Predictor (Mode {output_mode}):")
    print(f"Dataset: {n_samples} samples, {input_dim} features")
    print(f"Raw accuracy range: [{y_raw.min():.4f}, {y_raw.max():.4f}]")
    
    # Check if we have enough data
    if n_samples < 10:
        logger.log("Warning: Very small dataset, neural network training may not be effective")
        print("Warning: Very small dataset, neural network training may not be effective")
        
    # Split data into train/validation/test sets
    # First split: 80% train+val, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_raw, test_size=0.2, random_state=NN_RANDOM_STATE, stratify=None
    )
    
    # Second split: 75% train, 25% val (of the remaining 80%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=NN_RANDOM_STATE, stratify=None
    )
    
    logger.log(f"\nData Split:")
    logger.log(f"  Training set: {len(X_train)} samples")
    logger.log(f"  Validation set: {len(X_val)} samples")
    logger.log(f"  Test set: {len(X_test)} samples")
    
    print(f"Data split: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test")
    
    # For classification modes, show class distribution
    if output_mode in [2, 3]:
        logger.log(f"\nClass Distribution (based on raw accuracy):")
        
        if output_mode == 2:  # Binary
            train_good = (y_train > 0).sum()
            train_bad = (y_train <= 0).sum()
            val_good = (y_val > 0).sum()
            val_bad = (y_val <= 0).sum()
            test_good = (y_test > 0).sum()
            test_bad = (y_test <= 0).sum()
            
            logger.log(f"  Training: Good={train_good}, Bad={train_bad}")
            logger.log(f"  Validation: Good={val_good}, Bad={val_bad}")
            logger.log(f"  Test: Good={test_good}, Bad={test_bad}")
            
        elif output_mode == 3:  # Three-class
            a = FILTERING_INTERVAL_WIDTH_A
            for split_name, y_split in [("Training", y_train), ("Validation", y_val), ("Test", y_test)]:
                bad = (y_split < -a).sum()
                mid = ((y_split >= -a) & (y_split <= a)).sum()
                good = (y_split > a).sum()
                logger.log(f"  {split_name}: Bad={bad}, Mid={mid}, Good={good}")
    
    # Standardize input features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train the model
    predictor = AccuracyPredictor(input_dim=input_dim, output_mode=output_mode)
    
    print(f"\nTraining neural network...")
    history = predictor.train(X_train_scaled, y_train, X_val_scaled, y_val, verbose=1)
    
    # Evaluate on test set
    print(f"\nEvaluating on test set...")
    test_results = predictor.evaluate(X_test_scaled, y_test, verbose=1)
    
    # Log test results
    logger.log(f"\nTest Results:")
    for key, value in test_results.items():
        if key not in ['confusion_matrix', 'classification_report']:
            logger.log(f"  {key}: {value}")
    
    if 'confusion_matrix' in test_results:
        logger.log(f"  Confusion Matrix:")
        logger.log(f"    {test_results['confusion_matrix']}")
    
    if 'classification_report' in test_results:
        logger.log(f"  Classification Report:")
        for class_name, metrics in test_results['classification_report'].items():
            if isinstance(metrics, dict):
                logger.log(f"    {class_name}: precision={metrics.get('precision', 'N/A'):.3f}, "
                          f"recall={metrics.get('recall', 'N/A'):.3f}, "
                          f"f1-score={metrics.get('f1-score', 'N/A'):.3f}")
    
    # Plot training history
    predictor.plot_training_history()
    
    # Save model summary to CSV
    model_info = {
        'output_mode': output_mode,
        'input_dim': input_dim,
        'hidden_layers': predictor.hidden_layers,
        'n_train_samples': len(X_train),
        'n_val_samples': len(X_val),
        'n_test_samples': len(X_test),
        'epochs': NN_EPOCHS,
        'batch_size': NN_BATCH_SIZE,
        'learning_rate': NN_LEARNING_RATE,
    }
    
    # Add test results to model info
    model_info.update(test_results)
    
    # Convert arrays to lists for JSON serialization
    if 'confusion_matrix' in model_info:
        model_info['confusion_matrix'] = model_info['confusion_matrix'].tolist()
    
    # Save model information
    os.makedirs(CSV_DIR, exist_ok=True)
    model_df = pd.DataFrame([model_info])
    model_df.to_csv(os.path.join(CSV_DIR, f'nn_model_info_mode_{output_mode}.csv'), index=False)
    
    logger.log(f"\nNeural network training completed for mode {output_mode}")
    print(f"Neural network training completed for mode {output_mode}")
    
    return predictor

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
    
    print(f"\n{'Rank':^6}|{'Combination':^15}|{'Raw Acc':^10}|{'Norm Acc':^10}|{'Trans Acc':^10}")
    print("-" * 70)
    
    for rank, result in enumerate(sorted_results):
        print(f"{rank+1:^6}|{result['combination']:^15}|{result['accuracy_raw']:^10.4f}|{result['accuracy_normalized']:^10.4f}|{result['accuracy']:^10.4f}")
    
    logger.log(f"\n{'Rank':^6}|{'Combination':^15}|{'Raw Acc':^10}|{'Norm Acc':^10}|{'Trans Acc':^10}")
    logger.log("-" * 70)
    
    for rank, result in enumerate(sorted_results):
        logger.log(f"{rank+1:^6}|{result['combination']:^15}|{result['accuracy_raw']:^10.4f}|{result['accuracy_normalized']:^10.4f}|{result['accuracy']:^10.4f}")
    
    # Print the best combination
    best_combo = sorted_results[0]['combination']
    best_accuracy_raw = sorted_results[0]['accuracy_raw']
    best_accuracy_normalized = sorted_results[0]['accuracy_normalized']
    best_accuracy_transformed = sorted_results[0]['accuracy']
    
    print(f"\nBest agent combination: {best_combo}")
    print(f"Raw accuracy (negative slope): {best_accuracy_raw:.4f}")
    print(f"Normalized accuracy: {best_accuracy_normalized:.4f}")
    print(f"Transformed accuracy (sigmoid): {best_accuracy_transformed:.4f}")
    
    logger.log(f"\nBest agent combination: {best_combo}")
    logger.log(f"Raw accuracy (negative slope): {best_accuracy_raw:.4f}")
    logger.log(f"Normalized accuracy: {best_accuracy_normalized:.4f}")
    logger.log(f"Transformed accuracy (sigmoid): {best_accuracy_transformed:.4f}")
    
    # Visualize agent combinations using a multi-stage PCA approach
    logger.log("\nVisualizing agent combinations using a multi-stage PCA approach...")
    visualize_agent_combinations(results, feature_counts, feature_names, is_3d=True)
    
    # Train and evaluate neural network to predict agent combination accuracy
    logger.log("\nTraining neural network to predict agent combination accuracy...")
    predictor = train_accuracy_predictor(results, feature_counts, output_mode=NN_OUTPUT_MODE)
    
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
