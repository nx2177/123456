import json
import re
import requests

class OllamaLLM:
    """
    LLM handler for Ollama API
    """
    def __init__(self, model="llama3.2:latest"):
        self.model = model
        self.api_url = "http://localhost:11434/api/generate"
        
    def query(self, prompt, system_prompt="", format_json=False):
        """
        Query the Ollama LLM API
        
        Args:
            prompt (str): The prompt to send to the LLM
            system_prompt (str): Optional system prompt
            format_json (bool): Whether to request the output in JSON format
            
        Returns:
            str: The response from the LLM
        """
        headers = {"Content-Type": "application/json"}
        
        if format_json:
            if system_prompt:
                system_prompt += " Your response MUST be valid JSON format only, with no additional text, no code block markers, and no explanation."
            else:
                system_prompt = "Your response MUST be valid JSON format only, with no additional text, no code block markers, and no explanation."
            
            # Add JSON format instructions to the prompt as well
            prompt = f"{prompt}\n\nIMPORTANT: Respond with ONLY the JSON object. No introductions, explanations, or code block markers."
        
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
            response_text = response.json()["response"]
            
            # Clean the response if JSON is expected
            if format_json:
                # Remove code block markers if present
                response_text = response_text.replace("```json", "").replace("```", "").strip()
                
                # Remove any text before the first { and after the last }
                first_brace = response_text.find('{')
                last_brace = response_text.rfind('}')
                if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                    response_text = response_text[first_brace:last_brace+1]
            
            return response_text
            
        except requests.exceptions.RequestException as e:
            print(f"Error querying LLM: {e}")
            return ""


class ResumeParser:
    """
    Agent A2: Parse resumes into structured data with a more detailed prompt
    """
    def __init__(self, llm_model="llama3.2:latest"):
        """
        Initialize the resume parser agent
        
        Args:
            llm_model (str): The LLM model to use
        """
        self.llm = OllamaLLM(model=llm_model)
    
    def _extract_json_from_text(self, text):
        """
        Extract JSON from text that might contain extra content
        
        Args:
            text (str): Text that might contain JSON
            
        Returns:
            dict: Extracted JSON as dict or empty dict if not found
        """
        # Try to find JSON-like content enclosed in braces
        json_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
        json_matches = re.findall(json_pattern, text)
        
        if json_matches:
            # Try each match until we find valid JSON
            for potential_json in json_matches:
                try:
                    return json.loads(potential_json)
                except json.JSONDecodeError:
                    continue
        
        return {}
    
    def process(self, resume_text):
        """
        Extract structured information from the resume with a more detailed prompt
        
        Args:
            resume_text (str): The resume text to parse
            
        Returns:
            dict: Structured resume information with technical_skills, job_experience, 
                soft_skills, and education_level
        """
        # More detailed system prompt for Agent A2
        system_prompt = """
        You are an expert resume parsing AI assistant specializing in deep information extraction.
        Your task is to meticulously analyze resumes and extract comprehensive structured information.
        
        Follow these detailed guidelines:
        
        1. Technical Skills Extraction:
           - Identify all technical skills including programming languages, frameworks, libraries, tools, platforms
           - Look for both explicitly stated skills and those implied through project or work descriptions
           - Categorize by type (languages, frameworks, databases, infrastructure, etc.) where possible
           - Exclude generic skills that aren't specifically technical in nature
           - Include version numbers or specific variations if mentioned
        
        2. Job Experience Extraction:
           - Extract all professional experiences with complete details
           - For each position, identify: 
              * Exact job title as written
              * Full company name
              * Precise duration with start and end dates
              * Comprehensive responsibilities and achievements
              * Technologies used in each role
              * Team size and management responsibilities if mentioned
              * Quantifiable achievements with metrics when available
           - Note career progression and promotions
        
        3. Soft Skills Identification:
           - Analyze the entire resume for evidence of soft skills
           - Look for explicitly mentioned soft skills
           - Infer soft skills from descriptions of teamwork, leadership, communication
           - Include soft skills demonstrated through achievements
           - Note evidence of conflict resolution, mentorship, or stakeholder management
           - Identify project management and organizational skills
        
        4. Education Level Assessment:
           - Extract the highest level of formal education achieved
           - Include the specific degree name, field of study, institution, and graduation year
           - Note any honors, distinctions, or relevant coursework
           - Include relevant certifications, bootcamps, or continuing education
        
        Your output must be a comprehensive, well-structured JSON object containing all extracted information.
        Be thorough but precise, ensuring all data is accurately extracted from the provided document.
        """
        
        # More detailed prompt for Agent A2
        prompt = f"""
        Please conduct a comprehensive analysis of the following resume, extracting all relevant structured information.
        
        I need you to thoroughly extract and organize the following categories:
        
        1. Technical Skills: Create a comprehensive list of all technical skills mentioned including programming languages, 
           frameworks, tools, platforms, methodologies, standards, and any other technical competencies. Look for skills 
           mentioned explicitly as well as those implied through project descriptions or work responsibilities.
        
        2. Job Experience: Extract all employment history with complete details including:
           - Exact job title
           - Complete company name
           - Full duration (both start and end dates)
           - All responsibilities and achievements listed
           - Any quantifiable metrics mentioned
           - Technologies and methodologies used in each position
           - Team leadership or management aspects if mentioned
        
        3. Soft Skills: Identify all soft skills explicitly mentioned as well as those demonstrated through described 
           experiences, such as leadership, communication, team collaboration, problem-solving, adaptability, time 
           management, conflict resolution, creativity, and emotional intelligence.
        
        4. Education Level: Extract the highest level of formal education including degree name, field, institution, 
           and any honors or distinctions.
        
        Return your analysis as a structured JSON object with the following keys: "technical_skills" (array of strings), 
        "job_experience" (array of objects with keys for title, company, duration, and responsibilities), 
        "soft_skills" (array of strings), and "education_level" (string).
        
        RESUME:
        {resume_text}
        """
        
        try:
            # First attempt with format_json flag
            response = self.llm.query(prompt, system_prompt, format_json=True)
            
            # Try to parse the response as JSON
            try:
                parsed_data = json.loads(response)
            except json.JSONDecodeError:
                # If direct parsing fails, try to extract JSON from the response
                parsed_data = self._extract_json_from_text(response)
                
                # If still no valid JSON, make a second attempt with stronger formatting instructions
                if not parsed_data:
                    stronger_prompt = f"""
                    Parse the following resume and return ONLY a valid JSON object. 
                    Do not include any additional text, explanations, or code blocks.
                    The response must be ONLY the JSON object itself that can be directly parsed.
                    
                    Expected format:
                    {{
                        "technical_skills": ["skill1", "skill2", ...],
                        "job_experience": [
                            {{
                                "title": "Job Title",
                                "company": "Company Name",
                                "duration": "Start - End Date",
                                "responsibilities": ["responsibility1", "responsibility2", ...]
                            }},
                            ...
                        ],
                        "soft_skills": ["skill1", "skill2", ...],
                        "education_level": "Highest degree"
                    }}
                    
                    Conduct a thorough and detailed extraction of all information from this resume:
                    
                    {resume_text}
                    """
                    response = self.llm.query(stronger_prompt, system_prompt, format_json=True)
                    try:
                        parsed_data = json.loads(response)
                    except json.JSONDecodeError:
                        parsed_data = self._extract_json_from_text(response)
            
            # Ensure we have all required keys
            required_keys = ["technical_skills", "job_experience", "soft_skills", "education_level"]
            for key in required_keys:
                if key not in parsed_data:
                    parsed_data[key] = [] if key != "education_level" else ""
            
            return parsed_data
        
        except Exception as e:
            print(f"Error parsing resume: {str(e)}")
            # Fallback in case of any issues
            return {
                "technical_skills": [],
                "job_experience": [],
                "soft_skills": [],
                "education_level": ""
            } 