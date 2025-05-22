# Overview
Stochastic Multi-Agent Combination Evaluation System which is LLM-Augmented and Embedding-Based.

## Architecture

Here is a break down of the whole architecture:
<img width="989" alt="ff85b028595bff6237d3da12a269369" src="https://github.com/user-attachments/assets/abd26adc-4d61-45d9-ab4b-da539ae32203" />
<img width="990" alt="d920ec37bd4b19fd449e6ca15271b5e" src="https://github.com/user-attachments/assets/ad7dbe2a-682c-4229-9801-74baa623cf7b" />
<img width="989" alt="031a6770ff3192af90fd80ea4dde9e8" src="https://github.com/user-attachments/assets/0946c0ae-9597-4748-9c66-4f4bb32767d3" />
<img width="984" alt="7288a3fa9b8e2ec14d9fc6bef830a21" src="https://github.com/user-attachments/assets/47ceb39e-b4be-4dc4-8b64-65db7e651b02" />
<img width="989" alt="cee7426a239bd052cfffa02db04e92e" src="https://github.com/user-attachments/assets/de51e3c2-a928-46d0-9bf0-b685486cea29" />
<img width="974" alt="d506f2508a65df51cad9c453c826cdd" src="https://github.com/user-attachments/assets/34a6ef50-aead-4cc6-9f39-20ef7a95398d" />

### 🗂️ **1. Input: Resume & Job Description Simulation**

Simulate 5 candidate resumes and 1 job description using Claude.

* **5 candidate resumes** are simulated for each evaluation run:

  * 1 strong match
  * 3 moderate matches with varied strengths
  * 1 clear mismatch (fails a hard requirement)
* A **job description (JD)** is also simulated, containing structured expectations like required skills, experience, and soft qualities.

---

### 🧑‍🤝‍🧑 **2. Multi-agent Scoring System**

A multi-agent system for evaluating candidate resumes against job descriptions using Word2Vec embeddings and structured weights.
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

| Agent | Role                           | Input       | Output                                               |
| ----- | ------------------------------ | ----------- | ---------------------------------------------------- |
| **A** | Parsing agent                  | Resume + JD | Parsed data (e.g., extracted skills, experience)     |
| **B** | Technical skill evaluator      | Parsed data | Technical skill score                                |
| **C** | Job experience evaluator       | Parsed data | Experience score                                     |
| **D** | Soft skills evaluator / Ranker | Parsed data | Soft skill score                                     |

> 🔹 Agent **A** is a pure parser — it **does not** use or store any weights.

Note: Each agent operates independently with its own interface. The system aggregates scores from all agents to produce a final score.

This system is designed with a fully decoupled, modular architecture where:

- Each agent operates independently with its own handler
- No shared base classes or inheritance is used
- Interfaces can be easily replaced per agent
- Agents have clear, well-defined inputs and outputs
- Each agent encapsulates its own LLM logic

This mimics real-world asynchronous collaboration between different intelligent components, as if each agent is a separate entity with its own brain.

---

### ⚖️ **3. Stochastic Agent Weights**

Agents B, C, and D each use a **stochastic weight vector** to evaluate their respective domain:

* Agent **B**’s weights: importance of technical skills (e.g., Python: 0.9, Java: 0.7)
* Agent **C**’s weights: importance of job experience features (e.g., YearsExperience: 0.8)
* Agent **D**’s weights: importance of soft skills (e.g., Communication: 0.85)

Weights are:

* Sampled at the beginning, from i.i.d. Gaussian distributions.
* Saved in a structured `weights.json` file
* Can be reused across all evaluations to ensure **deterministic and reproducible results**

Each weight dictionary is stored like:

```json
{
  "id": "B2",
  "weights": {
    "Python": 0.92,
    "JavaScript": 0.76,...
  }
}
```

---

### 🧮 **4. Scoring and Ranking**

* Each agent processes the resume and outputs a score.
* The scores are aggregated into a final ranking of all candidates.
* This ranking is compared to a **predefined human-evaluated ground truth ranking** (e.g., `[2,4,3,5,1]`).
Note: the human-evaluated groud truth is also simulated by Claude.

---

### 🎯 **5. Evaluation Metric**

* The system uses **Accuracy**:

  * It compares the system ranking with the human ranking.
  * Accuracy: 
    - Exact position match accuracy
    - Spearman rank correlation coefficient (which considers relative positions)

---

### 🧪 **6. Combination Evaluation**

* Each agent category (B, C, D) has multiple variants (e.g., B1–B5, C1–C5…).
* The system evaluates **all possible combinations** (e.g.[B1, C2, D3]) by running them through the pipeline and recording the resulting accuracy.
* Results are sorted and ranked by performance.

---

### 📉 **7. Visualization & Interpretability**

For each agent category (B, C, D):

* The structured weight vectors are **flattened** into numerical feature vectors.
* **UMAP** is applied to reduce the vectors to 2D.
* **HDBSCAN** is applied to cluster similar agents in this weight space.
* Each point is:

  * Colored by its average accuracy (from all combinations it appeared in)
  * Labeled with its ID
  * Marked by cluster (or labeled “Noise” if unclustered)

This helps identify:

* Which **types of weight configurations** tend to perform better
* Whether there are **semantic clusters** of strong or weak agents
* How agents differ or group based on their evaluation style (as encoded in their weights)

---

### 📁 **8. Output Artifacts**

* `weights.json`: structured weight data for all agent variants
* `.csv` files per agent category: containing ID, flattened weights, accuracy, and cluster ID
* UMAP+HDBSCAN plots per agent category: for visual insight into evaluation space

---

### 🧭 **Why this design matters**

This architecture allows you to:

* Analyze **how different evaluation preferences (via weights) impact performance**
* Identify **robust evaluation strategies** via clustering
* Achieve **transparent, reproducible, and interpretable evaluation behavior**


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
  author = {Nan Xiao (nx2177)},
  title = {Resume Scoring System},
  year = {2025},
  url = {https://github.com/nx2177/123456}
}
