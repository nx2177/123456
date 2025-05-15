## Architecture

![64a8eee158e3d15ab80b02f189ece38](https://github.com/user-attachments/assets/f4ab136b-2faf-45e2-a8f8-50ebb1acc716)

### Multi-agent Scoring System:

Updates:
---

### 🗂️ **1. Input: Resume & Job Description Simulation**

* **5 candidate resumes** are simulated for each evaluation run:

  * 1 strong match
  * 3 moderate matches with varied strengths
  * 1 clear mismatch (fails a hard requirement)
* A **job description (JD)** is also simulated, containing structured expectations like required skills, experience, and soft qualities.

---

### 🧑‍🤝‍🧑 **2. Agent Pipeline (A → B → C → D)**

Each resume passes through a **multi-agent scoring pipeline**, with each agent evaluating a specific dimension:

| Agent | Role                           | Input       | Output                                               |
| ----- | ------------------------------ | ----------- | ---------------------------------------------------- |
| **A** | Parsing agent                  | Resume + JD | Structured info (e.g., extracted skills, experience) |
| **B** | Technical skill evaluator      | Parsed data | Technical skill score                                |
| **C** | Job experience evaluator       | Parsed data | Experience score                                     |
| **D** | Soft skills evaluator / Ranker | Parsed data | Final soft skill score or ranked decision            |

> 🔹 Agent **A** is a pure parser — it **does not** use or store any weights.

---

### ⚖️ **3. Agent Weights**

Agents B, C, and D each use a **custom weight vector** to evaluate their respective domain:

* Agent **B**’s weights: importance of technical skills (e.g., Python: 0.9, Java: 0.7)
* Agent **C**’s weights: importance of job experience features (e.g., YearsExperience: 0.8)
* Agent **D**’s weights: importance of soft skills (e.g., Communication: 0.85)

Weights are:

* Sampled once (e.g., from Gaussian distributions)
* Saved in a structured `weights.json` file
* Reused across all evaluations to ensure **deterministic and reproducible results**

Each weight dictionary is stored like:

```json
{
  "id": "B2",
  "weights": {
    "Python": 0.92,
    "JavaScript": 0.76,
    ...
  }
}
```

---

### 🧮 **4. Scoring and Ranking**

* Each agent processes the resume and outputs a score.
* The scores are aggregated into a final ranking of all candidates.
* This ranking is compared to a **predefined human-evaluated ground truth ranking** (e.g., `[1, 3, 2, 4, 5]`).

---

### 🎯 **5. Evaluation Metric**

* The system uses **Top-3 Set Accuracy**:

  * It compares the top-3 system-ranked resumes with the top-3 from the human ranking.
  * Accuracy = proportion of overlap between the two sets (ignoring order).

---

### 🧪 **6. Combination Evaluation**

* Each agent category (B, C, D) has multiple variants (e.g., B1–B5, C1–C5…).
* The system evaluates **all possible combinations** (e.g., \[B1, C2, D3]) by running them through the pipeline and recording the resulting accuracy.
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

------------
Original:


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

 - Resume 1 (ALEX ZHANG): Clear mismatch - lacks US work authorization (requires sponsorship as noted in the job description) and is missing required technical skills (Python, JavaScript, React).
 - Resume 2 (SARAH JOHNSON): Strong match - exceptional candidate with all required skills, appropriate experience, strong soft skills, and US citizenship.
 - Resume 3 (DAVID KIM): Moderate match with strength in technical skills - has excellent technical skills but less leadership experience.
 - Resume 4 (MICHAEL WILSON): Moderate match with strength in experience - has extensive management experience but fewer modern technical skills.
 - Resume 5 (RACHEL GARCIA): Moderate match with strength in soft skills - excellent at communication and user-focused development but more limited technical depth.


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

### Dataset

Define Agent Combinations
Evaluate the following 4 combinations:

combinations = [
   - ['A1', 'B1', 'C', 'D'],  # baseline
   - ['A2', 'B1', 'C', 'D'],  # change in prompt style (A1's prompt style is concise, A2's prompt style is detailed)
   - ['A1', 'B2', 'C', 'D'],  # change in LLM model (B1 use llama3.2, B2 use deepseek-r1:8b)
   - ['A2', 'B2', 'C', 'D'],  # change in both
]

Rank the agent combinations based on the accuracy and output the best combination. 




## Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running locally
- LLaMA 3.2 and deepseek-r1:8b pulled in Ollama

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
