import requests
import numpy as np
import re
from typing import List, Dict, Tuple

class OllamaLLM:
    """
    LLM handler for Ollama API
    """
    def __init__(self, model="deepseek-r1:8b"):  # Updated to use available model
        self.model = model
        self.api_url = "http://localhost:11434/api/generate"
        self.embedding_url = "http://localhost:11434/api/embeddings"
        
    def query(self, prompt, system_prompt=""):
        """
        Query the Ollama LLM API
        
        Args:
            prompt (str): The prompt to send to the LLM
            system_prompt (str): Optional system prompt
            
        Returns:
            str: The response from the LLM
        """
        headers = {"Content-Type": "application/json"}
        
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        
        if system_prompt:
            data["system"] = system_prompt
            
        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()["response"]
            
        except requests.exceptions.RequestException as e:
            print(f"Error querying LLM: {e}")
            return ""
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding vector for a text using Ollama API
        
        Args:
            text (str): The text to embed
            
        Returns:
            List[float]: Embedding vector or empty list if error
        """
        headers = {"Content-Type": "application/json"}
        
        data = {
            "model": self.model,
            "prompt": text
        }
        
        try:
            response = requests.post(self.embedding_url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()["embedding"]
            
        except requests.exceptions.RequestException as e:
            print(f"Error getting embedding: {e}")
            return []


class TechnicalSkillScorer:
    """
    Agent B2: Score how well a candidate's technical skills match job requirements
    using embedding-based semantic similarity with the deepseek model
    """
    def __init__(self, llm_model="deepseek-r1:8b"):  # Updated to use available model
        """
        Initialize the technical skill scorer agent
        
        Args:
            llm_model (str): The LLM model to use (default is deepseek-r1:8b for B2)
        """
        self.llm = OllamaLLM(model=llm_model)
    
    def preprocess_skill(self, skill: str) -> str:
        """
        Preprocess a skill string by lowercasing and removing punctuation
        
        Args:
            skill (str): The skill to preprocess
            
        Returns:
            str: The preprocessed skill
        """
        # Convert to lowercase
        skill = skill.lower()
        
        # Remove punctuation except hyphens (for compound terms like "object-oriented")
        skill = re.sub(r'[^\w\s-]', '', skill)
        
        # Remove extra whitespace
        skill = re.sub(r'\s+', ' ', skill).strip()
        
        return skill
    
    def extract_required_skills(self, job_description: str) -> List[str]:
        """
        Extract required technical skills from the job description
        
        Args:
            job_description (str): The job description text
            
        Returns:
            list: List of required technical skills
        """
        system_prompt = """
        You are a job description analyzer focusing on technical requirements.
        Extract all technical skills required for the position.
        Technical skills include programming languages, frameworks, tools, platforms, 
        methodologies, and specific technical knowledge areas.
        """
        
        prompt = f"""
        Please analyze the following job description and extract all required technical skills
        as a comma-separated list of skills. Return ONLY the comma-separated list, nothing else.

        JOB DESCRIPTION:
        {job_description}
        """
        
        response = self.llm.query(prompt, system_prompt)
        # Process the response into a list
        skills = [self.preprocess_skill(skill.strip()) for skill in response.split(",") if skill.strip()]
        return skills
    
    def get_embeddings(self, skills: List[str]) -> Dict[str, List[float]]:
        """
        Get embeddings for a list of skills
        
        Args:
            skills (List[str]): List of skills to embed
            
        Returns:
            Dict[str, List[float]]: Dictionary mapping skills to their embedding vectors
        """
        embeddings = {}
        for skill in skills:
            embedding = self.llm.get_embedding(skill)
            if embedding:  # Only add if we got a valid embedding
                embeddings[skill] = embedding
        
        return embeddings
    
    def compute_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Compute cosine similarity between two vectors
        
        Args:
            vec1 (List[float]): First vector
            vec2 (List[float]): Second vector
            
        Returns:
            float: Cosine similarity between 0 and 1
        """
        if not vec1 or not vec2:
            return 0.0
            
        # Convert to numpy arrays
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        # Compute cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        
        if norm_product == 0:
            return 0.0
            
        similarity = dot_product / norm_product
        
        # Cap between 0 and 1
        return max(0.0, min(1.0, similarity))
    
    def compute_skill_match_scores(self, 
                                  required_embeddings: Dict[str, List[float]], 
                                  candidate_embeddings: Dict[str, List[float]]) -> List[Tuple[str, str, float]]:
        """
        Compute match scores between required skills and candidate skills
        
        Args:
            required_embeddings (Dict[str, List[float]]): Embeddings of required skills
            candidate_embeddings (Dict[str, List[float]]): Embeddings of candidate skills
            
        Returns:
            List[Tuple[str, str, float]]: List of (required_skill, best_candidate_skill, similarity_score)
            for each required skill
        """
        match_scores = []
        
        for req_skill, req_embedding in required_embeddings.items():
            best_match_skill = None
            best_similarity = 0.0
            
            # Find the candidate skill with highest similarity for this required skill
            for cand_skill, cand_embedding in candidate_embeddings.items():
                similarity = self.compute_cosine_similarity(req_embedding, cand_embedding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_skill = cand_skill
            
            match_scores.append((req_skill, best_match_skill, best_similarity))
            
        return match_scores
    
    def process(self, candidate_skills: List[str], job_description: str) -> float:
        """
        Score how well candidate's technical skills match job requirements using
        embedding-based semantic similarity
        
        Args:
            candidate_skills (list): The candidate's technical skills
            job_description (str): The job description text
            
        Returns:
            float: Score between 0 and 1
        """
        # Extract required skills from job description
        required_skills = self.extract_required_skills(job_description)
        
        if not required_skills:
            return 0.0
            
        # Preprocess candidate skills
        candidate_skills = [self.preprocess_skill(skill) for skill in candidate_skills]
        
        # Get embeddings for both sets of skills
        required_embeddings = self.get_embeddings(required_skills)
        candidate_embeddings = self.get_embeddings(candidate_skills)
        
        if not required_embeddings or not candidate_embeddings:
            return 0.0
            
        # Compute similarity scores
        match_scores = self.compute_skill_match_scores(required_embeddings, candidate_embeddings)
        
        # Calculate average similarity score
        if not match_scores:
            return 0.0
            
        average_similarity = sum(score for _, _, score in match_scores) / len(match_scores)
        
        technical_match_details = ""
        for req_skill, cand_skill, score in match_scores:
            technical_match_details += f"Required: '{req_skill}' → Best match: '{cand_skill}' (similarity: {score:.3f})"       
        
        return average_similarity, technical_match_details