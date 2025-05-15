import requests
import numpy as np
import re
from typing import List, Dict, Tuple
from gensim.models import Word2Vec
from scipy.stats import norm
import random

class TechnicalSkillScorer:
    """
    Agent B: Score how well a candidate's technical skills match job requirements
    using embedding-based semantic similarity
    """
    def __init__(self, llm_model=None, weights=None, word2vec_model=None, jd_embedding=None, structured_weights=None):
        """
        Initialize the technical skill scorer agent
        
        Args:
            llm_model (str): Unused parameter kept for backward compatibility
            weights (List[float]): Optional pre-defined weights for scoring (flat list format)
            word2vec_model (Word2Vec): Pre-computed Word2Vec model
            jd_embedding (List[float]): Pre-computed job description embedding
            structured_weights (Dict[str, float]): Dictionary mapping features to weights
        """
        self.word2vec_model = word2vec_model
        self.weights = weights
        self.jd_embedding = jd_embedding
        self.structured_weights = structured_weights
        
        # Common technical skills to look for in job descriptions
        self.common_tech_skills = [
            "python", "java", "javascript", "typescript", "c++", "c#", "ruby", "php", "swift",
            "kotlin", "go", "rust", "scala", "perl", "html", "css", "sql", "r", "matlab",
            "react", "angular", "vue", "node.js", "django", "flask", "spring", "express",
            "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy", "kubernetes", "docker",
            "aws", "azure", "gcp", "git", "jenkins", "ci/cd", "agile", "scrum", "jira",
            "mongodb", "postgresql", "mysql", "oracle", "redis", "elasticsearch", "nosql"
        ]
        
    def initialize_word2vec_model(self, job_description, resume):
        """
        Initialize Word2Vec model based on job description and resume if not already provided
        
        Args:
            job_description (str): The job description text
            resume (str): The resume text
            
        Returns:
            None
        """
        # Skip initialization if model was provided in constructor
        if self.word2vec_model is not None:
            return
            
        # Combine both texts
        combined_text = job_description + " " + resume
        
        # Tokenize combined text
        sentences = [re.findall(r'\w+', combined_text.lower())]
        
        # Train Word2Vec model
        self.word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    
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
        Extract required technical skills from the job description using pattern matching
        
        Args:
            job_description (str): The job description text
            
        Returns:
            list: List of required technical skills
        """
        skills = set()
        job_lower = job_description.lower()
        
        # Look for sections commonly containing required skills
        skill_sections = []
        
        # Find potential skill sections using common headers
        sections = job_description.split('\n\n')
        for section in sections:
            if any(keyword in section.lower() for keyword in 
                  ["required skills", "technical requirements", "qualifications", 
                   "requirements", "skills & experience", "skills and experience"]):
                skill_sections.append(section)
        
        # If no clear sections found, use the whole job description
        if not skill_sections:
            skill_sections = [job_description]
        
        # Extract skills from sections
        for section in skill_sections:
            # Extract skills from bullet points
            lines = section.split('\n')
            for line in lines:
                line = line.strip()
                
                # Check if the line contains any of our known technical skills
                for skill in self.common_tech_skills:
                    if skill in line.lower():
                        skills.add(skill)
                
                # Look for bullet points with skills
                if line.startswith('-') or line.startswith('•') or line.startswith('*'):
                    line = line[1:].strip()
                    
                    # Check for multiple skills in a single line (comma or semicolon separated)
                    if ',' in line or ';' in line:
                        parts = re.split(r'[,;]', line)
                        for part in parts:
                            part = part.strip().lower()
                            if part and any(tech in part for tech in self.common_tech_skills):
                                # Extract the specific technology mentioned
                                for tech in self.common_tech_skills:
                                    if tech in part:
                                        skills.add(tech)
                    else:
                        # Check for single skill
                        for tech in self.common_tech_skills:
                            if tech in line.lower():
                                skills.add(tech)
        
        # Also check for direct mentions of technologies in the entire text
        for skill in self.common_tech_skills:
            if skill in job_lower:
                skills.add(skill)
        
        # Check for some common skill pairs
        if "machine learning" in job_lower or "ml" in job_lower.split():
            skills.add("machine learning")
        if "artificial intelligence" in job_lower or "ai" in job_lower.split():
            skills.add("artificial intelligence")
        if "natural language processing" in job_lower or "nlp" in job_lower.split():
            skills.add("nlp")
        
        # Convert to list and preprocess
        skill_list = [self.preprocess_skill(skill) for skill in skills]
        return skill_list
    
    def get_word2vec_embedding(self, skill: str) -> List[float]:
        """
        Get Word2Vec embedding for a skill
        
        Args:
            skill (str): The skill to get embedding for
            
        Returns:
            List[float]: Embedding vector
        """
        # Split the skill into individual words
        words = skill.lower().split()
        
        # Get embeddings for words that are in the model's vocabulary
        word_vectors = []
        for word in words:
            if word in self.word2vec_model.wv:
                word_vectors.append(self.word2vec_model.wv[word])
        
        # If no word vectors were found, return a zero vector
        if not word_vectors:
            return np.zeros(self.word2vec_model.vector_size)
        
        # Average the word vectors to get a skill vector
        skill_vector = np.mean(word_vectors, axis=0)
        return skill_vector.tolist()
    
    def get_embeddings(self, skills: List[str]) -> Dict[str, List[float]]:
        """
        Get Word2Vec embeddings for a list of skills
        
        Args:
            skills (List[str]): List of skills to embed
            
        Returns:
            Dict[str, List[float]]: Dictionary mapping skills to their embedding vectors
        """
        embeddings = {}
        for skill in skills:
            embedding = self.get_word2vec_embedding(skill)
            # Check if embedding exists and is not all zeros
            if isinstance(embedding, list) or (isinstance(embedding, np.ndarray) and np.any(embedding)):
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
    
    def generate_gaussian_weights(self, n: int) -> List[float]:
        """
        Generate Gaussian distributed weights with mean=1 and variance=0.05
        
        Args:
            n (int): Number of weights to generate
            
        Returns:
            List[float]: List of weights
        """
        # Use pre-defined weights if available
        if self.weights is not None:
            return self.weights[:n] if len(self.weights) >= n else self.weights + [1.0] * (n - len(self.weights))
        
        # If structured weights are available, use them
        if self.structured_weights is not None:
            # Create a list that has consistent ordering with match_scores
            return list(self.structured_weights.values())[:n]
        
        mean = 1.0
        variance = 0.05
        std_dev = np.sqrt(variance)
        
        # Generate weights from Gaussian distribution
        weights = np.random.normal(mean, std_dev, n)
        
        # Ensure all weights are positive (as they're used for importance weighting)
        weights = np.maximum(weights, 0.1)
        
        return weights.tolist()
    
    def get_feature_weight(self, feature_name, default_weight=1.0):
        """
        Get weight for a specific feature from structured weights
        
        Args:
            feature_name (str): The feature to get weight for
            default_weight (float): Default weight if not found
            
        Returns:
            float: Weight value
        """
        if self.structured_weights and feature_name in self.structured_weights:
            return self.structured_weights[feature_name]
        return default_weight

    def process(self, candidate_skills: List[str], job_description: str, resume: str) -> Tuple[float, str]:
        """
        Score how well candidate's technical skills match job requirements using
        embedding-based semantic similarity with Word2Vec and weighted average
        
        Args:
            candidate_skills (list): The candidate's technical skills
            job_description (str): The job description text
            resume (str): The resume text for embedding dictionary creation
            
        Returns:
            Tuple[float, str]: (weighted_average_similarity_score, technical_match_details)
        """
        # Initialize Word2Vec model if not already provided
        if self.word2vec_model is None:
            self.initialize_word2vec_model(job_description, resume)
        
        # Extract required skills from job description
        required_skills = self.extract_required_skills(job_description)
        
        if not required_skills:
            return 0.0, "No required skills found in job description."
            
        # Preprocess candidate skills
        candidate_skills = [self.preprocess_skill(skill) for skill in candidate_skills]
        
        # Get embeddings for both sets of skills using Word2Vec
        required_embeddings = self.get_embeddings(required_skills)
        candidate_embeddings = self.get_embeddings(candidate_skills)
        
        if not required_embeddings or not candidate_embeddings:
            return 0.0, "Could not generate embeddings for skills."
            
        # Compute similarity scores
        match_scores = self.compute_skill_match_scores(required_embeddings, candidate_embeddings)
        
        # Calculate weighted average similarity score
        if not match_scores:
            return 0.0, "No match scores could be computed."
        
        # Generate weights following Gaussian distribution
        weights = self.generate_gaussian_weights(len(match_scores))
        
        # Create detailed technical match information with feature-specific weights
        technical_match_details = ""
        weighted_sum = 0.0
        total_weight = 0.0
        
        for i, (req_skill, cand_skill, similarity) in enumerate(match_scores):
            # Get weight for this skill if using structured weights
            if self.structured_weights is not None:
                weight = self.get_feature_weight(req_skill, weights[i])
            else:
                weight = weights[i]
                
            weighted_sum += weight * similarity
            total_weight += weight
            
            technical_match_details += f"Required: '{req_skill}' → Best match: '{cand_skill}' (similarity: {similarity:.3f}, weight: {weight:.3f})\n"
        
        # Calculate final score
        weighted_average_similarity = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        return weighted_average_similarity, technical_match_details