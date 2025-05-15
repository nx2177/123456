import json
import re
import numpy as np
from typing import List, Dict, Tuple
from gensim.models import Word2Vec
from scipy.stats import norm

class ExperienceRelevanceScorer:
    """
    Agent C: Score the relevance of candidate's work experience using Word2Vec embeddings
    """
    def __init__(self, llm_model=None, weights=None, word2vec_model=None, jd_embedding=None, structured_weights=None):
        """
        Initialize the experience relevance scorer agent
        
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
        
        # Keywords related to experience relevance
        self.relevance_keywords = [
            "experience", "years", "role", "responsibility", "managed", "developed", 
            "led", "implemented", "created", "designed", "achieved", "improved",
            "increased", "decreased", "optimized", "maintained", "coordinated",
            "industry", "sector", "field", "technology", "method", "process"
        ]
        
        # Feature to keyword mapping
        self.feature_keywords = {
            "years_experience": ["years", "experience", "duration"],
            "relevant_role": ["role", "position", "title", "job"],
            "industry_match": ["industry", "sector", "field", "domain"],
            "responsibilities_match": ["responsibility", "duties", "tasks"],
            "senior_leadership": ["senior", "leadership", "executive", "chief", "director"],
            "team_management": ["team", "manage", "supervise", "lead", "direct"],
            "project_management": ["project", "program", "initiative", "delivery"],
            "technical_expertise": ["technical", "expert", "specialist", "proficient"],
            "domain_knowledge": ["domain", "knowledge", "expertise", "specialize"],
            "achievements": ["achieve", "accomplish", "success", "result", "impact"],
            "career_progression": ["career", "progression", "promotion", "advance"],
            "company_tier": ["company", "organization", "enterprise", "startup"]
        }
    
    def initialize_word2vec_model(self, job_description, experience_text):
        """
        Initialize Word2Vec model based on job description and experience text if not already provided
        
        Args:
            job_description (str): The job description text
            experience_text (str): The experience text
            
        Returns:
            None
        """
        # Skip initialization if model was provided in constructor
        if self.word2vec_model is not None:
            return
            
        # Combine both texts
        combined_text = job_description + " " + experience_text
        
        # Tokenize combined text
        sentences = [re.findall(r'\w+', combined_text.lower())]
        
        # Train Word2Vec model
        self.word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    
    def format_experience_text(self, job_experience):
        """
        Format job experience into a single text string
        
        Args:
            job_experience (list): List of job experience dictionaries
            
        Returns:
            str: Formatted experience text
        """
        experience_str = ""
        for i, experience in enumerate(job_experience):
            experience_str += f"Experience {i+1}:\n"
            for key, value in experience.items():
                experience_str += f"{key}: {value}\n"
            experience_str += "\n"
        
        return experience_str
    
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
            # Extract values from structured weights based on extracted section content
            weights = []
            for i in range(n):
                # Default weight
                weights.append(1.0)
            return weights
            
        mean = 1.0
        variance = 0.05
        std_dev = np.sqrt(variance)
        
        # Generate weights from Gaussian distribution
        weights = np.random.normal(mean, std_dev, n)
        
        # Ensure all weights are positive
        weights = np.maximum(weights, 0.1)
        
        return weights.tolist()
    
    def get_feature_weight(self, section_text, default_weight=1.0):
        """
        Get appropriate weight for a section based on its content
        
        Args:
            section_text (str): The text content of the section
            default_weight (float): Default weight if no matching feature is found
            
        Returns:
            float: Weight value for the section
        """
        if not self.structured_weights:
            return default_weight
            
        # Find the most relevant feature for this section
        section_text = section_text.lower()
        best_match = None
        best_score = 0
        
        for feature, keywords in self.feature_keywords.items():
            score = sum(1 for keyword in keywords if keyword in section_text)
            if score > best_score:
                best_score = score
                best_match = feature
        
        # Return weight for the best matching feature, or default if none found
        if best_match and best_match in self.structured_weights:
            return self.structured_weights[best_match]
            
        return default_weight
    
    def extract_key_sections(self, text, section_length=100):
        """
        Extract key sections from text based on relevance keywords
        
        Args:
            text (str): Text to extract from
            section_length (int): Length of section to extract
            
        Returns:
            list: List of key sections
        """
        sections = []
        text_lower = text.lower()
        
        for keyword in self.relevance_keywords:
            pos = text_lower.find(keyword)
            while pos != -1:
                start = max(0, pos - section_length//2)
                end = min(len(text), pos + section_length//2)
                section = text[start:end]
                sections.append(section)
                
                # Find next occurrence
                pos = text_lower.find(keyword, pos + 1)
                
        return sections if sections else [text[:section_length]]  # Return at least one section
    
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
    
    def process(self, job_experience, job_description):
        """
        Evaluate how relevant the candidate's work experience is to the job description using Word2Vec
        
        Args:
            job_experience (list): The candidate's job experience
            job_description (str): The job description text
            
        Returns:
            tuple: (score, justification) where score is between 0 and 1
        """
        # Format job experience into a readable string
        experience_str = self.format_experience_text(job_experience)
        
        # Initialize Word2Vec model if not already provided
        if self.word2vec_model is None:
            self.initialize_word2vec_model(job_description, experience_str)
        
        # Extract key sections from job description and experience
        job_sections = self.extract_key_sections(job_description)
        exp_sections = []
        
        for exp in job_experience:
            # Extract responsibilities as key sections
            if 'responsibilities' in exp:
                if isinstance(exp['responsibilities'], list):
                    for resp in exp['responsibilities']:
                        exp_sections.append(resp)
                else:
                    exp_sections.append(str(exp['responsibilities']))
            
            # Also include title and company
            if 'title' in exp:
                exp_sections.append(str(exp['title']))
            if 'company' in exp:
                exp_sections.append(str(exp['company']))
        
        # Get embeddings for job sections
        job_embeddings = self.get_embeddings_for_sections(job_sections)
        
        # Get embeddings for experience sections
        exp_embeddings = self.get_embeddings_for_sections(exp_sections)
        
        if not job_embeddings or not exp_embeddings:
            return 0.5, "Insufficient data to compute relevance score."
        
        # Calculate similarities between job sections and experience sections
        section_matches = []
        for job_emb, job_section in zip(job_embeddings, job_sections):
            best_sim = 0.0
            best_match = ""
            best_idx = -1
            
            for i, exp_emb in enumerate(exp_embeddings):
                sim = self.compute_cosine_similarity(job_emb, exp_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_match = exp_sections[i][:30] + "..." if len(exp_sections[i]) > 30 else exp_sections[i]
                    best_idx = i
            
            section_matches.append((best_sim, best_match, job_section))
        
        # Apply weights based on structured weights if available
        weighted_sum = 0.0
        total_weight = 0.0
        
        for sim, match, section in section_matches:
            # Get appropriate weight for this section
            weight = self.get_feature_weight(section, 1.0)
            weighted_sum += weight * sim
            total_weight += weight
        
        score = weighted_sum / total_weight if total_weight > 0 else 0.5
        
        # Generate justification
        strong_matches = [(match, section) for sim, match, section in section_matches if sim > 0.7]
        moderate_matches = [(match, section) for sim, match, section in section_matches if 0.5 <= sim <= 0.7]
        weak_matches = [(match, section) for sim, match, section in section_matches if sim < 0.5]
        
        justification = f"Experience relevance score: {score:.2f}. "
        
        if strong_matches:
            justification += f"Strong matches found in: {len(strong_matches)} areas. "
        
        if moderate_matches:
            justification += f"Moderate relevance in: {len(moderate_matches)} areas. "
            
        if weak_matches:
            justification += f"Weak matches in: {len(weak_matches)} areas. "
        
        # Add years of experience if available
        years_exp = 0
        for exp in job_experience:
            if 'duration' in exp:
                duration = exp['duration']
                # Try to extract years from duration string
                year_match = re.search(r'(\d+).*year', duration.lower())
                if year_match:
                    years_exp += int(year_match.group(1))
                    
        if years_exp > 0:
            justification += f"Candidate has approximately {years_exp} years of relevant experience."
        
        return score, justification 