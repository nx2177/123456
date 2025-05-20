# Resume Scoring System

A multi-agent system for evaluating candidate resumes against job descriptions using Word2Vec embeddings and structured weights.

## Architecture

The Resume Scoring System implements a decoupled multi-agent architecture with four specialized agents:

![Resume Scoring System Architecture](https://github.com/user-attachments/assets/f4ab136b-2faf-45e2-a8f8-50ebb1acc716)

### Multi-agent Scoring System:

The system consists of four independent agents that analyze different aspects of resume-job matching:

1. **Agent A (Resume Parser)**: Extracts structured information from a plain text resume.
   - Implemented in `agents/parser.py`
   - Parses technical skills, work experience, and soft skills
   - No scoring/weights applied at this level

2. **Agent B (Technical Skill Scorer)**: Evaluates how well the resume's technical skills match the job requirements.
   - Implemented in `agents/technical_scorer.py`
   - Uses Word2Vec embeddings for semantic matching
   - Structured weights for different technical domains (e.g., programming languages, frameworks)

3. **Agent C (Experience Relevance Scorer)**: Assesses the relevance of the candidate's work experience.
   - Implemented in `agents/experience_scorer.py`
   - Analyzes experience duration, responsibilities, and role relevance
   - Structured weights for experience features (years, leadership, etc.)

4. **Agent D (Soft Skills Scorer)**: Evaluates the candidate's soft skills.
   - Implemented in `agents/soft_skills_scorer.py`
   - Identifies communication, teamwork, and other soft skills
   - Structured weights for different soft skill categories

### Implementation Details:

- **Persistent Weights Management**: Weights are stored in `weights.json` and loaded/generated as needed
- **Word2Vec Embeddings**: Used for semantic similarity calculation between job descriptions and resumes
- **Structured Weight System**: Feature-specific weights rather than flat arrays
- **UMAP & HDBSCAN Visualization**: For agent weight space analysis
- **Performance Evaluation**: Compares system rankings against human expert rankings

### Data Flow:

1. Job description and candidate resumes are processed
2. Agent A parses resume into structured data
3. Agents B, C, and D score different aspects using structured weights
4. Scores are aggregated for final candidate ranking
5. Results are analyzed with visualizations in `analysis_output/`

## Performance Evaluation:

The system evaluates performance across multiple agent combinations:

- 5 diverse candidate resumes with varying strengths
- Comparison against predefined human expert rankings
- Evaluation metrics: exact position accuracy and Spearman correlation
- Visualization of agent weight spaces and cluster analysis

## Installation

### Prerequisites

- Python 3.8+
- The following Python packages:
  - numpy
  - matplotlib
  - pandas
  - gensim
  - umap-learn
  - hdbscan
  - scikit-learn
  - spacy (optional, for enhanced parsing)

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/resume_scoring_system.git
   cd resume_scoring_system
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. (Optional) Install SpaCy model for enhanced parsing:
   ```
   python -m spacy download en_core_web_sm
   ```

## Step-by-Step Tutorial

### Installation and Environment Setup

1. **Verify Python Installation**:
   ```
   python --version
   ```
   Ensure you have Python 3.8 or higher installed.

2. **Create a Virtual Environment** (recommended):
   ```
   python -m venv venv
   ```

3. **Activate the Virtual Environment**:
   - Windows:
     ```
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```
     source venv/bin/activate
     ```

4. **Install Dependencies**:
   ```
   pip install -r requirements.txt
   ```

### Running the System

1. **Basic Usage**:
   ```
   python main.py
   ```
   This will:
   - Generate sample resumes and job descriptions
   - Create or load agent weights
   - Evaluate resume rankings
   - Generate visualizations and CSV files

2. **View Results**:
   - Check console output for ranking summary
   - Visualizations are saved to `analysis_output/visualizations/`
   - CSV analysis files are saved to `analysis_output/csv/`
   - Logs are saved to `logs/processing_details.log`

### Configuration

The main configuration parameters can be found at the top of `main.py`:

```python
# Configuration for number of agents in each category
n_A = 5  # Number of Agent A instances
n_B = 5  # Number of Agent B instances
n_C = 5  # Number of Agent C instances
n_D = 5  # Number of Agent D instances

# Weights and embedding configuration
WEIGHTS_FILE = "weights.json"
EMBEDDING_DIM = 100  # Dimension for Word2Vec embeddings
WEIGHT_MEAN = 1.0    # Mean of Gaussian distribution
WEIGHT_VARIANCE = 1.0  # Variance of Gaussian distribution
```

### Customization Examples

1. **Using Your Own Resumes and Job Descriptions**:
   
   Modify the `main.py` file to read your own files:

   ```python
   def main():
       # Read your job description
       with open("your_job_description.txt", "r") as f:
           job_description = f.read()
           
       # Read your resumes
       resumes = []
       for file in ["resume1.txt", "resume2.txt", "resume3.txt"]:
           with open(file, "r") as f:
               resumes.append(f.read())
               
       # Continue with the evaluation
       # ...
   ```

2. **Changing Agent Counts or Weights**:

   When modifying agent counts or weight parameters (n_A, n_B, etc.), you must delete the existing weights.json file to regenerate weights:

   ```
   # Delete the existing weights file
   rm weights.json  # Linux/macOS
   del weights.json  # Windows
   
   # Run the system with new parameters
   python main.py
   ```

3. **Advanced Analysis**:

   To run specific analysis on the output CSV files:

   ```python
   import pandas as pd
   
   # Load CSV data
   df = pd.read_csv("analysis_output/csv/agent_B_analysis.csv")
   
   # Perform additional analysis
   print(df.groupby("cluster")["avg_accuracy"].mean())
   ```

### Troubleshooting Guide

1. **"KeyError" in weights.json**:
   - This usually happens when you've changed agent counts without deleting weights.json
   - Solution: Delete weights.json and run again

2. **Missing Visualizations**:
   - Ensure matplotlib is properly installed
   - Check if the analysis_output directory has proper write permissions
   - Make sure you have at least 2 agents of each type for visualization

3. **SpaCy Model Missing**:
   - If you see "SpaCy model not available" warning, install the model:
   ```
   python -m spacy download en_core_web_sm
   ```

4. **Memory Issues with Large Datasets**:
   - Reduce embedding dimensions in main.py
   - Process resumes in smaller batches
   - Use a machine with more RAM

5. **UMAP/HDBSCAN Errors**:
   - These may occur with too few data points
   - Ensure you have enough agent variants (at least 5 of each type)
   - Check UMAP and HDBSCAN installation

## Advanced Usage

### Extending the System

1. **Adding New Agent Types**:
   - Create a new agent class in the agents directory
   - Update main.py to integrate the new agent
   - Modify score_single_resume_with_agents to include the new agent

2. **Custom Evaluation Metrics**:
   - Add new metrics in the evaluate_ranking_accuracy function
   - Implement calculation logic for your specific needs

3. **Integrating with External Systems**:
   - Use the score_single_resume_with_agents function as an API
   - Return structured results for integration with other systems

## Citation

If you use this system in your research, please cite:

```
@software{resume_scoring_system,
  author = {Your Name},
  title = {Resume Scoring System},
  year = {2023},
  url = {https://github.com/yourusername/resume_scoring_system}
}
```

## License

[Your chosen license]
