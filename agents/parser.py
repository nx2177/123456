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
    Agent A: Parse resumes into structured data
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
        Extract structured information from the resume
        
        Args:
            resume_text (str): The resume text to parse
            
        Returns:
            dict: Structured resume information with technical_skills, job_experience, 
                soft_skills, and education_level
        """
        system_prompt = """
        You are a resume parsing expert. Extract structured information from the provided resume.
        Focus on these key categories:
        1. Technical Skills: List all technical skills mentioned (programming languages, tools, frameworks, etc.)
        2. Job Experience: Extract each job with title, company, duration, and key responsibilities
        3. Soft Skills: Identify soft skills like leadership, communication, teamwork, etc.
        4. Education Level: Highest education level achieved
        
        You MUST return a valid JSON object with no additional text or explanation.
        The JSON must be properly formatted with no syntax errors.
        """
        
        prompt = f"""
        Analyze the following resume and extract the structured information.
        Return ONLY a valid JSON object with these exact keys: "technical_skills" (array of strings), 
        "job_experience" (array of objects with keys "title", "company", "duration", and "responsibilities"), 
        "soft_skills" (array of strings), and "education_level" (string).
        
        Format your response as a pure JSON object with no additional text.

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
                    
                    RESUME:
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