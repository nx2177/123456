import json
import re
import numpy as np
from typing import List, Dict, Tuple
from gensim.models import Word2Vec
from scipy.stats import norm

class SoftSkillsScorer:
    """
    Agent D: Score the candidate's soft skills using Word2Vec embeddings
    """
    def __init__(self, llm_model=None, weights=None, word2vec_model=None, jd_embedding=None, structured_weights=None):
        """
        Initialize the soft skills scorer agent
        
        Args:
            llm_model (str): Unused parameter kept for backward compatibility
            weights (List[float]): Optional pre-defined weights for scoring
            word2vec_model (Word2Vec): Pre-computed Word2Vec model
            jd_embedding (List[float]): Pre-computed job description embedding
            structured_weights (Dict[str, float]): Dictionary mapping features to weights
        """
        self.word2vec_model = word2vec_model
        self.weights = weights
        self.jd_embedding = jd_embedding
        self.structured_weights = structured_weights
        
        # Common soft skills we'll look for
        self.common_soft_skills = [
            "communication", "teamwork", "leadership", "problem-solving", "adaptability",
            "time management", "conflict resolution", "creativity", "emotional intelligence",
            "critical thinking", "decision making", "negotiation", "presentation", "interpersonal",
            "organization", "flexibility", "collaboration", "customer service", "mentoring",
            "initiative", "analytical"
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
    
    def extract_soft_skills(self, text):
        """
        Extract soft skills from text using keyword matching
        
        Args:
            text (str): Text to extract skills from
            
        Returns:
            list: List of identified soft skills
        """
        text = text.lower()
        found_skills = []
        
        for skill in self.common_soft_skills:
            if skill in text or skill.replace('-', ' ') in text:
                found_skills.append(skill)
                
        return found_skills
    
    def get_word2vec_embedding(self, text: str) -> List[float]:
        """
        Get Word2Vec embedding for text
        
        Args:
            text (str): The text to get embedding for
            
        Returns:
            List[float]: Embedding vector
        """
        # Split the text into individual words
        words = text.lower().split()
        
        # Get embeddings for words that are in the model's vocabulary
        word_vectors = []
        for word in words:
            if word in self.word2vec_model.wv:
                word_vectors.append(self.word2vec_model.wv[word])
        
        # If no word vectors were found, return a zero vector
        if not word_vectors:
            return np.zeros(self.word2vec_model.vector_size)
        
        # Average the word vectors to get a text vector
        text_vector = np.mean(word_vectors, axis=0)
        return text_vector.tolist()
    
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
        
        # Ensure all weights are positive
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
    
    def get_embeddings_for_sections(self, sections):
        """
        Get embeddings for a list of text sections
        
        Args:
            sections (List[str]): List of text sections
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        embeddings = []
        for section in sections:
            embedding = self.get_word2vec_embedding(section)
            # Check if embedding exists and is not all zeros
            if isinstance(embedding, list) or (isinstance(embedding, np.ndarray) and np.any(embedding)):
                embeddings.append(embedding)
        return embeddings
    
    def process(self, resume_text, job_description):
        """
        Evaluate soft skills evidence in the resume compared to job requirements using Word2Vec
        
        Args:
            resume_text (str): The full resume text
            job_description (str): The job description text
            
        Returns:
            tuple: (score, justification) where score is between 0 and 1
        """
        # Initialize Word2Vec model if not already provided
        if self.word2vec_model is None:
            self.initialize_word2vec_model(job_description, resume_text)
        
        # Extract soft skills from job description and resume
        job_soft_skills = self.extract_soft_skills(job_description)
        resume_soft_skills = self.extract_soft_skills(resume_text)
        
        if not job_soft_skills:
            return 0.5, "No specific soft skills identified in job description."
            
        # Get embeddings for job description sections
        job_sections = job_description.split('\n\n')
        job_section_embeddings = self.get_embeddings_for_sections(job_sections)
        
        # Get embeddings for resume sections
        resume_sections = resume_text.split('\n\n')
        resume_section_embeddings = self.get_embeddings_for_sections(resume_sections)
                  
        # Calculate similarities between each job section and resume sections
        similarities = []
        matched_sections = []
        
        for i, job_emb in enumerate(job_section_embeddings):
            best_sim = 0.0
            best_idx = -1
            
            for j, resume_emb in enumerate(resume_section_embeddings):
                sim = self.compute_cosine_similarity(job_emb, resume_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_idx = j
            
            if best_idx >= 0:
                similarities.append(best_sim)
                matched_sections.append((job_sections[i][:50] + "...", resume_sections[best_idx][:50] + "..."))
        
        if not similarities:
            return 0.5, "Unable to compute similarity between job and resume sections."
        
        # Calculate weighted average similarity
        weighted_sum = 0.0
        total_weight = 0.0
        
        justification = "Soft skills assessment: "
        matched_skills = []
        
        # Handle potential size mismatch between job_soft_skills and similarities
        # Since similarities are based on job sections, not directly on soft skills
        if len(similarities) > 0:
            # Generate weights for each similarity score
            weights = self.generate_gaussian_weights(len(similarities))
            
            for i, similarity in enumerate(similarities):
                # Use a default weight if not enough weights
                weight_idx = min(i, len(weights)-1) if weights else 0
                weight = weights[weight_idx]
                
                # If structured weights are available, try to apply them
                if self.structured_weights:
                    # Apply weights to similarity scores based on sections
                    section_content = matched_sections[i][0] if i < len(matched_sections) else ""
                    
                    # Check if any soft skill appears in this section
                    for skill in job_soft_skills:
                        if skill.lower() in section_content.lower():
                            weight = self.get_feature_weight(skill, weight)
                            break
                
                weighted_sum += weight * similarity
                total_weight += weight
                
                # Track matched skills for justification
                if similarity >= 0.6:
                    # Find soft skills in this section
                    for skill in job_soft_skills:
                        if skill not in matched_skills and skill.lower() in matched_sections[i][0].lower():
                            matched_skills.append(skill)
        
        # Directly identified soft skills (exact matches)
        direct_matches = [skill for skill in resume_soft_skills if skill in job_soft_skills]
        for skill in direct_matches:
            if skill not in matched_skills:
                matched_skills.append(skill)
        
        # Calculate final score
        score = weighted_sum / total_weight if total_weight > 0 else 0.5
        
        # Complete justification
        if matched_skills:
            justification += f"Found evidence of {len(matched_skills)} soft skills: {', '.join(matched_skills)}. "
        else:
            justification += "Limited evidence of required soft skills. "
            
        justification += f"Overall soft skills score: {score:.2f}/1.00"
        
        return score, justification 