import json
import numpy as np
import os
from typing import List, Dict, Any, Tuple
from agents.parser import ResumeParser as ResumeParserA1
from agents.parser_2 import ResumeParser as ResumeParserA2
from agents.technical_scorer import TechnicalSkillScorer as TechnicalSkillScorerB1
from agents.technical_scorer_2 import TechnicalSkillScorer as TechnicalSkillScorerB2
from agents.experience_scorer import ExperienceRelevanceScorer
from agents.soft_skills_scorer import SoftSkillsScorer
from utils.data_generator import generate_sample_job_description, generate_candidate_resumes

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

def get_agent_A(variant: str, llm_model: str = "llama3.2:latest"):
    """
    Get the appropriate Agent A variant
    
    Args:
        variant (str): 'A1' or 'A2'
        llm_model (str): LLM model to use
        
    Returns:
        ResumeParser instance (A1 or A2)
    """
    if variant == 'A1':
        return ResumeParserA1(llm_model=llm_model)
    elif variant == 'A2':
        return ResumeParserA2(llm_model=llm_model)
    else:
        raise ValueError(f"Unknown Agent A variant: {variant}")

def get_agent_B(variant: str):
    """
    Get the appropriate Agent B variant
    
    Args:
        variant (str): 'B1' or 'B2'
        
    Returns:
        TechnicalSkillScorer instance (B1 or B2)
    """
    if variant == 'B1':
        # B1 uses llama3.2
        return TechnicalSkillScorerB1(llm_model="llama3.2:latest")
    elif variant == 'B2':
        # B2 uses deepseek-r1:8b (updated to use available model)
        return TechnicalSkillScorerB2(llm_model="deepseek-r1:8b")
    else:
        raise ValueError(f"Unknown Agent B variant: {variant}")

def score_single_resume_with_agents(
    resume_text: str, 
    job_description: str, 
    agent_A_variant: str, 
    agent_B_variant: str,
    candidate_index: int = None
) -> Tuple[float, Dict[str, Any]]:
    """
    Score a single resume against a job description using the specified agent variants
    
    Args:
        resume_text (str): The resume text
        job_description (str): The job description text
        agent_A_variant (str): Which Agent A to use ('A1' or 'A2')
        agent_B_variant (str): Which Agent B to use ('B1' or 'B2')
        candidate_index (int): Index of the candidate for logging purposes
        
    Returns:
        tuple: (final_score, parsed_resume)
    """
    # Log separator and candidate info
    candidate_name = resume_text.strip().split("\n")[0]
    logger.log(f"\n{'='*50}")
    logger.log(f"CANDIDATE {candidate_index}: {candidate_name}")
    logger.log(f"AGENT COMBINATION: {agent_A_variant} + {agent_B_variant} + C + D")
    logger.log(f"{'='*50}\n")
    
    # Log resume excerpt
    logger.log("RESUME EXCERPT:")
    logger.log(resume_text[:200] + "...\n")
    
    # Agent A: Parse the resume (using the specified variant)
    logger.log(f"Running Agent {agent_A_variant}: Resume Parser...")
    parser = get_agent_A(agent_A_variant)
    parsed_resume = parser.process(resume_text)
    logger.log_json(parsed_resume, "Parsed Resume Data")
    
    # Agent B: Technical Skill Scorer (using the specified variant)
    logger.log(f"\nRunning Agent {agent_B_variant}: Technical Skill Scorer...")
    tech_scorer = get_agent_B(agent_B_variant)
    s1, s1_technical_match_details = tech_scorer.process(parsed_resume["technical_skills"], job_description)
    logger.log(f"technical match details: {s1_technical_match_details}")
    logger.log(f"Technical Skill Match Score (S1): {s1:.2f}")
    
    # Agent C: Experience Relevance Scorer (always the same)
    logger.log(f"\nRunning Agent C: Experience Relevance Scorer...")
    exp_scorer = ExperienceRelevanceScorer(llm_model="llama3.2:latest")
    s2, s2_justification = exp_scorer.process(parsed_resume["job_experience"], job_description)
    logger.log(f"Experience Relevance Score (S2): {s2:.2f}")
    logger.log(f"Justification: {s2_justification}")
    
    # Agent D: Soft Skills Scorer (always the same)
    logger.log(f"\nRunning Agent D: Soft Skills Scorer...")
    soft_scorer = SoftSkillsScorer(llm_model="llama3.2:latest")
    s3, s3_justification = soft_scorer.process(resume_text, job_description)
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
        float: Accuracy score between 0 and 1
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

def run_evaluation_for_combination(agent_combination, resumes, job_description, human_ranking):
    """
    Run the performance evaluation pipeline for a specific agent combination
    
    Args:
        agent_combination (list): List of agent variants [A_variant, B_variant, C, D]
        resumes (list): List of resume texts
        job_description (str): Job description
        human_ranking (list): Human expert ranking
        
    Returns:
        dict: Evaluation results
    """
    agent_A_variant, agent_B_variant = agent_combination[0], agent_combination[1]
    combination_name = f"{agent_A_variant}+{agent_B_variant}+C+D"
    
    print(f"\n{'='*20} EVALUATING COMBINATION: {combination_name} {'='*20}\n")
    logger.log(f"\n\n{'='*30} EVALUATING COMBINATION: {combination_name} {'='*30}\n")
    
    # Score each resume and collect results
    scores = []
    candidate_names = []
    
    for i, resume in enumerate(resumes):
        candidate_name = resume.strip().split("\n")[0]
        candidate_names.append(candidate_name)
        
        print(f"\nCandidate {i+1}: {candidate_name}")
        score, _ = score_single_resume_with_agents(
            resume, 
            job_description, 
            agent_A_variant, 
            agent_B_variant,
            candidate_index=i+1
        )
        
        # Print only the scores to terminal (no processing details)
        print(f"Final Score: {score:.2f}/3.00")
        
        scores.append(score)
    
    # Create ranking based on scores
    # Sort candidate indices by score in descending order
    ranked_indices = [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]
    # Convert to 1-indexed for display
    system_ranking = [i+1 for i in ranked_indices]
    
    # Calculate accuracy
    accuracy_metrics = evaluate_ranking_accuracy(system_ranking, human_ranking)
    
    print(f"\nSystem Ranking: {system_ranking}")
    print(f"Human Ranking:  {human_ranking}")
    print(f"\nExact Position Accuracy: {accuracy_metrics['exact_match_accuracy']:.2f}")
    print(f"Spearman Rank Correlation: {accuracy_metrics['spearman_correlation']:.2f}")
    
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
        "spearman": accuracy_metrics["spearman_correlation"]
    }

def run_performance_evaluation():
    """
    Run the performance evaluation pipeline across all agent combinations
    """
    print("\n" + "="*20 + " RESUME RANKING SYSTEM EVALUATION " + "="*20 + "\n")
    logger.log("\n" + "="*20 + " RESUME RANKING SYSTEM EVALUATION " + "="*20 + "\n")
    
    # Generate job description
    job_description = generate_sample_job_description()
    print("Job Description Summary:")
    print(job_description.split("\n\n")[0])
    print(job_description.split("\n\n")[1])
    print("...")
    
    # Log the full job description
    logger.log("FULL JOB DESCRIPTION:")
    logger.log(job_description)
    logger.log("\n" + "="*50 + "\n")
    
    # Generate candidate resumes
    resumes = generate_candidate_resumes()
    
    # Human ranking (predefined ground truth)
    # Corresponds to [resume1, resume2, resume3, resume4, resume5]
    # In order of preference: resume2 (idx 1), resume3 (idx 2), resume5 (idx 4), resume4 (idx 3), resume1 (idx 0)
    human_ranking = [2, 3, 5, 4, 1]  # 1-indexed for human readability
    
    print(f"\nHuman Expert Ranking (Ground Truth): {human_ranking}")
    logger.log(f"Human Expert Ranking (Ground Truth): {human_ranking}")
    
    # Define the agent combinations to evaluate
    combinations = [
        ['A1', 'B1'],  # baseline
        ['A2', 'B1'],  # change in prompt style
        ['A1', 'B2'],  # change in LLM model
        ['A2', 'B2'],  # change in both
    ]
    
    # Evaluate each combination
    results = []
    for combo in combinations:
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
    
    print(sorted_results)
    
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