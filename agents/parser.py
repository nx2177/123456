import json
import re
import numpy as np
from typing import List, Dict, Tuple
from gensim.models import Word2Vec
from scipy.stats import norm
import spacy
from collections import defaultdict

class ResumeParser:
    """
    Agent A: Parse resumes into structured data using Word2Vec embeddings
    """
    def __init__(self, word2vec_model=None):
        """
        Initialize the resume parser agent
        
        Args:
            word2vec_model (Word2Vec): Optional pre-computed Word2Vec model
        """
        self.word2vec_model = word2vec_model
        
        # Load spaCy model for NER if available, otherwise continue without it
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            self.nlp = None
            print("SpaCy model not available. Some parsing features will be limited.")
        
        # Common technical skills to identify
        self.tech_skills = [
            "python", "java", "javascript", "typescript", "c++", "c#", "ruby", "php", "swift",
            "kotlin", "go", "rust", "scala", "perl", "html", "css", "sql", "r", "matlab",
            "react", "angular", "vue", "node.js", "django", "flask", "spring", "express",
            "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy", "kubernetes", "docker",
            "aws", "azure", "gcp", "git", "jenkins", "ci/cd", "agile", "scrum", "jira",
            "mongodb", "postgresql", "mysql", "oracle", "redis", "elasticsearch", "nosql"
        ]
        
        # Common soft skills to identify
        self.soft_skills = [
            "communication", "teamwork", "leadership", "problem-solving", "creativity",
            "adaptability", "time management", "organization", "critical thinking",
            "decision-making", "project management", "collaboration", "attention to detail",
            "analytical", "interpersonal", "persuasion", "negotiation", "conflict resolution",
            "mentoring", "coaching", "customer service", "presentation", "multitasking"
        ]
    
    def initialize_word2vec_model(self, text):
        """
        Initialize Word2Vec model based on resume text
        
        Args:
            text (str): The resume text
            
        Returns:
            None
        """
        # Skip if model was already provided
        if self.word2vec_model is not None:
            return
            
        # Tokenize text
        sentences = [re.findall(r'\w+', text.lower())]
        
        # Train Word2Vec model
        self.word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    
    def extract_sections(self, resume_text):
        """
        Extract different sections from the resume by searching for exact headers
        
        Args:
            resume_text (str): The resume text
            
        Returns:
            dict: Dictionary of sections
        """
        sections = {
            "technical_skills": [],
            "experience": [],
            "soft_skills": []
        }
        
        # Split the resume text into lines
        lines = resume_text.split('\n')
        
        # Define exact headers to look for
        headers = {
            "TECHNICAL SKILLS": "technical_skills",
            "SKILLS": "technical_skills",
            "WORK EXPERIENCE": "experience",
            "EXPERIENCE": "experience", 
            "SOFT SKILLS": "soft_skills"
        }
        
        current_section = None
        
        # Process the resume line by line
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Check if this line is a header
            upper_line = line.upper()
            is_header = False
            
            for header, section_name in headers.items():
                if header in upper_line:
                    current_section = section_name
                    is_header = True
                    break
            
            # If this line is not a header and we're in a known section, add it to that section
            if not is_header and current_section:
                sections[current_section].append(line)
        
        return sections
    
    def extract_technical_skills(self, text):
        """
        Extract technical skills from text using keyword matching and Word2Vec similarity
        
        Args:
            text (str): The text to extract from
            
        Returns:
            list: List of technical skills
        """
        skills = set()
        text_lower = text.lower()
        
        # Direct keyword matching
        for skill in self.tech_skills:
            if skill in text_lower or skill.replace('.', '') in text_lower:
                skills.add(skill)
        
        # Now check for skill phrases in each line
        lines = text.split('\n')
        for line in lines:
            # If the line contains "skills" or similar keywords, it's likely a skills list
            if any(kw in line.lower() for kw in ["skills", "technologies", "tools", "languages"]):
                # Split by commas or other common separators
                items = re.split(r'[,;/&|]|\s{2,}', line)
                for item in items:
                    item = item.strip().lower()
                    if item and len(item) > 1 and item not in ["and", "or", "skills", "including"]:
                        # Check if it's a known tech skill or close to one using Word2Vec
                        if item in self.tech_skills:
                            skills.add(item)
        
        return list(skills)
    
    def extract_job_experience(self, sections):
        """
        Extract job experience from resume sections
        
        Args:
            sections (dict): Resume sections
            
        Returns:
            list: List of job experiences
        """
        experience_list = []
        
        if "experience" not in sections or not sections["experience"]:
            return experience_list
        
        experience_text = '\n'.join(sections["experience"])
        
        # Split experience text into individual job blocks
        # Look for date patterns or job titles as separators
        job_blocks = []
        current_block = []
        
        lines = experience_text.split('\n')
        for line in lines:
            # Check for potential job title or date pattern indicating a new job entry
            if re.search(r'\b(19|20)\d{2}\b.*\b(19|20)\d{2}\b|\b(19|20)\d{2}\b.*present\b', line.lower()) or \
               re.search(r'^[A-Z][a-z]+ [A-Z][a-z]+\s*[|]\s*', line) or \
               re.search(r'^[A-Z][a-z]+ [A-Z][a-z]+\s*[-]\s*', line):
                if current_block:  # If we have content in the current block
                    job_blocks.append('\n'.join(current_block))
                    current_block = []
            
            current_block.append(line)
        
        # Don't forget to add the last block
        if current_block:
            job_blocks.append('\n'.join(current_block))
        
        # Process each job block
        for block in job_blocks:
            job = {}
            
            # Extract job title
            title_match = re.search(r'^(.*?)[|;:-]', block, re.MULTILINE)
            if title_match:
                job["title"] = title_match.group(1).strip()
            else:
                # Try to find the first line that could be a title
                first_line = block.split('\n')[0].strip()
                job["title"] = first_line
            
            # Extract company
            company_match = re.search(r'[|;:-]\s*(.*?)[|;:-]', block, re.MULTILINE)
            if company_match:
                job["company"] = company_match.group(1).strip()
            else:
                # Set a placeholder if we can't find it
                job["company"] = "Unknown Company"
            
            # Extract duration
            duration_match = re.search(r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\s*[-–—]\s*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}|\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\s*[-–—]\s*Present\b', block, re.IGNORECASE)
            if duration_match:
                job["duration"] = duration_match.group(0).strip()
            else:
                # Try a simpler date pattern
                duration_match = re.search(r'\b(19|20)\d{2}\s*[-–—]\s*(19|20)\d{2}|\b(19|20)\d{2}\s*[-–—]\s*Present\b', block, re.IGNORECASE)
                if duration_match:
                    job["duration"] = duration_match.group(0).strip()
                else:
                    job["duration"] = "Not specified"
            
            # Extract responsibilities
            # Remove the title, company, and duration parts
            resp_text = re.sub(r'^.*?\n', '', block, 1)  # Remove first line
            resp_text = re.sub(r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\s*[-–—]\s*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}|\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\s*[-–—]\s*Present\b', '', resp_text, re.IGNORECASE)
            resp_text = re.sub(r'\b(19|20)\d{2}\s*[-–—]\s*(19|20)\d{2}|\b(19|20)\d{2}\s*[-–—]\s*Present\b', '', resp_text, re.IGNORECASE)
            
            # Split into bullet points if present
            bullet_points = []
            for line in resp_text.split('\n'):
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•')):
                    bullet_points.append(line.lstrip('-•').strip())
                elif line and len(bullet_points) == 0:  # If no bullet points but we have text
                    bullet_points.append(line)
            
            if bullet_points:
                job["responsibilities"] = bullet_points
            else:
                job["responsibilities"] = [resp_text.strip()]
            
            experience_list.append(job)
        
        return experience_list
    
    def extract_soft_skills(self, resume_text):
        """
        Extract soft skills from resume text
        
        Args:
            resume_text (str): The resume text
            
        Returns:
            list: List of soft skills
        """
        skills = set()
        text_lower = resume_text.lower()
        
        # Direct keyword matching
        for skill in self.soft_skills:
            if skill in text_lower:
                skills.add(skill)
        
        return list(skills)

    
    def process(self, resume_text):
        """
        Extract structured information from the resume using Word2Vec and pattern matching
        
        Args:
            resume_text (str): The resume text to parse
            
        Returns:
            dict: Structured resume information with technical_skills, job_experience, 
                and soft_skills
        """
        # Initialize Word2Vec model with resume text
        self.initialize_word2vec_model(resume_text)
        
        # Extract sections from resume
        sections = self.extract_sections(resume_text)
        
        # Extract individual components
        technical_skills = []
        if sections["technical_skills"]:
            skills_text = "\n".join(sections["technical_skills"])
            technical_skills = self.extract_technical_skills(skills_text)
        
        # If no skills found in the dedicated section, try extracting from the whole text
        if not technical_skills:
            technical_skills = self.extract_technical_skills(resume_text)
            
        job_experience = self.extract_job_experience(sections)
        
        soft_skills = []
        if sections["soft_skills"]:
            skills_text = "\n".join(sections["soft_skills"]) 
            soft_skills = self.extract_soft_skills(skills_text)
        else:
            # If no dedicated soft skills section, extract from the whole resume
            soft_skills = self.extract_soft_skills(resume_text)
        
        # Construct result
        parsed_data = {
            "technical_skills": technical_skills,
            "job_experience": job_experience,
            "soft_skills": soft_skills
        }
        
        return parsed_data