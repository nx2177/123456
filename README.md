# Multi-Agent Resume Scoring System

A system that uses multiple specialized agents to evaluate resumes against job descriptions. This implementation uses LLaMA 3.2 via the Ollama API.

## System Overview

The system consists of four independent agents:

1. **Agent A (Resume Parser)**: Extracts structured information from a plain text resume.
2. **Agent B (Technical Skill Scorer)**: Evaluates how well the resume's technical skills match the job requirements.
3. **Agent C (Experience Relevance Scorer)**: Assesses the relevance of the candidate's work experience.
4. **Agent D (Soft Skills Scorer)**: Evaluates the candidate's soft skills.

Each agent operates independently with its own LLM interface. The system aggregates scores from all agents to produce a final score.

This system is designed with a fully decoupled, modular architecture where:

- Each agent operates independently with its own LLM handler
- No shared base classes or inheritance is used
- LLM interfaces can be easily replaced per agent
- Agents have clear, well-defined inputs and outputs
- Each agent encapsulates its own LLM logic

This mimics real-world asynchronous collaboration between different intelligent components, as if each agent is a separate entity with its own brain.

## Architecture

![64a8eee158e3d15ab80b02f189ece38](https://github.com/user-attachments/assets/f4ab136b-2faf-45e2-a8f8-50ebb1acc716)

### Multi-agent Scoring System:

The system consists of four independent agents:

1. **Agent A (Resume Parser)**: Extracts structured information from a plain text resume.
2. **Agent B (Technical Skill Scorer)**: Evaluates how well the resume's technical skills match the job requirements.
3. **Agent C (Experience Relevance Scorer)**: Assesses the relevance of the candidate's work experience.
4. **Agent D (Soft Skills Scorer)**: Evaluates the candidate's soft skills.

Each agent operates independently with its own LLM interface. The system aggregates scores from all agents to produce a final score.

This system is designed with a fully decoupled, modular architecture where:

- Each agent operates independently with its own LLM handler
- No shared base classes or inheritance is used
- LLM interfaces can be easily replaced per agent
- Agents have clear, well-defined inputs and outputs
- Each agent encapsulates its own LLM logic

This mimics real-world asynchronous collaboration between different intelligent components, as if each agent is a separate entity with its own brain.

### Performance Evaluation:

1. Simulate 5 Diverse Candidate Resumes
I've generated 5 candidate resumes with the specified distribution:
Resume 1 (ALEX ZHANG): Clear mismatch - lacks US work authorization (requires sponsorship as noted in the job description) and is missing required technical skills (Python, JavaScript, React).
Resume 2 (SARAH JOHNSON): Strong match - exceptional candidate with all required skills, appropriate experience, strong soft skills, and US citizenship.
Resume 3 (DAVID KIM): Moderate match with strength in technical skills - has excellent technical skills but less leadership experience.
Resume 4 (MICHAEL WILSON): Moderate match with strength in experience - has extensive management experience but fewer modern technical skills.
Resume 5 (RACHEL GARCIA): Moderate match with strength in soft skills - excellent at communication and user-focused development but more limited technical depth.


3. Implemented Performance Evaluation Pipeline

- Single Resume Scorer Function: Refactored the original scoring logic into a reusable function that processes a single resume.

- Ranking Accuracy Evaluator: Added a function to compare system rankings with human rankings using:
  Exact position match accuracy
  Spearman rank correlation coefficient (which considers relative positions)

- Performance Evaluation Runner: Created a function that:
  Processes all 5 candidate resumes
  Scores each one using the multi-agent system

 - Ranks candidates based on scores
  Compares to predefined human expert rankings
  Calculates and displays accuracy metrics



4. Enhanced Output
The evaluation pipeline produces detailed output including:
 - Individual scores for each candidate
 - System-generated ranking
 - Human expert ranking
 - Accuracy metrics


## Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running locally
- LLaMA 3.2 model pulled in Ollama

## Setup

1. Install Ollama by following the instructions at [https://ollama.ai/](https://ollama.ai/)
2. Pull the LLaMA 3.2 model:
   ```
   ollama pull llama3.2
   ```
3. Install required Python packages:
   ```
   pip install requests
   ```

## Usage

1. Ensure the Ollama server is running locally:
   ```
   ollama serve
   ```

2. Run the main application:
   ```
   python main.py
   ```

The system will:
1. Generate a sample resume and job description
2. Parse the resume using Agent A
3. Evaluate technical skills match using Agent B
4. Assess experience relevance using Agent C
5. Evaluate soft skills using Agent D
6. Calculate and display the final score

## Customization

To use your own resume and job description, modify the `main.py` file to read from files instead of using the sample generator.

To replace an LLM implementation for a specific agent, simply modify the LLM handler class within that agent's file.

## Implementation Details

- `agents/parser.py`: Agent A implementation with its own LLM handler
- `agents/technical_scorer.py`: Agent B implementation with its own LLM handler
- `agents/experience_scorer.py`: Agent C implementation with its own LLM handler
- `agents/soft_skills_scorer.py`: Agent D implementation with its own LLM handler
- `utils/data_generator.py`: Generates sample data for testing
- `main.py`: Orchestrates the system by calling each agent sequentially

## Note

This system demonstrates a fully decoupled multi-agent architecture. In a production environment, you might want to:
1. Implement error handling and logging
2. Add validation for input/output
3. Optimize prompts for better extraction and evaluation
4. Implement caching to reduce API calls
5. Add additional agents for more specialized evaluations 
