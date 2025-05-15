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

7765   |   A1+B15+C15+D16   |        0.40        |        0.70
   7766   |   A1+B15+C15+D17   |        0.40        |        0.70
   7767   |   A1+B15+C15+D18   |        0.40        |        0.70
   7768   |   A1+B15+C15+D19   |        0.40        |        0.70
   7769   |   A1+B15+C15+D20   |        0.40        |        0.70
   7770   |   A1+B15+C16+D1    |        0.40        |        0.70
   7771   |   A1+B15+C16+D2    |        0.40        |        0.70
   7772   |   A1+B15+C16+D3    |        0.40        |        0.70
   7773   |   A1+B15+C16+D5    |        0.40        |        0.70
   7774   |   A1+B15+C16+D6    |        0.40        |        0.70
   7775   |   A1+B15+C16+D7    |        0.40        |        0.70
   7776   |   A1+B15+C16+D10   |        0.40        |        0.70
   7777   |   A1+B15+C16+D11   |        0.40        |        0.70
   7778   |   A1+B15+C16+D12   |        0.40        |        0.70
   7779   |   A1+B15+C16+D14   |        0.40        |        0.70
   7780   |   A1+B15+C16+D15   |        0.40        |        0.70
   7781   |   A1+B15+C16+D16   |        0.40        |        0.70
   7782   |   A1+B15+C16+D17   |        0.40        |        0.70
   7783   |   A1+B15+C16+D18   |        0.40        |        0.70
   7784   |   A1+B15+C16+D19   |        0.40        |        0.70
   7785   |   A1+B15+C16+D20   |        0.40        |        0.70
   7786   |   A1+B15+C17+D1    |        0.40        |        0.70
   7787   |   A1+B15+C17+D2    |        0.40        |        0.70
   7788   |   A1+B15+C17+D3    |        0.40        |        0.70
   7789   |   A1+B15+C17+D5    |        0.40        |        0.70
   7790   |   A1+B15+C17+D6    |        0.40        |        0.70
   7791   |   A1+B15+C17+D7    |        0.40        |        0.70
   7792   |   A1+B15+C17+D8    |        0.40        |        0.70
   7793   |   A1+B15+C17+D10   |        0.40        |        0.70
   7794   |   A1+B15+C17+D11   |        0.40        |        0.70
   7795   |   A1+B15+C17+D12   |        0.40        |        0.70
   7796   |   A1+B15+C17+D14   |        0.40        |        0.70
   7797   |   A1+B15+C17+D15   |        0.40        |        0.70
   7798   |   A1+B15+C17+D16   |        0.40        |        0.70
   7799   |   A1+B15+C17+D17   |        0.40        |        0.70
   7800   |   A1+B15+C17+D18   |        0.40        |        0.70
   7801   |   A1+B15+C17+D19   |        0.40        |        0.70
   7802   |   A1+B15+C17+D20   |        0.40        |        0.70
   7803   |   A1+B15+C18+D1    |        0.40        |        0.70
   7804   |   A1+B15+C18+D2    |        0.40        |        0.70
   7805   |   A1+B15+C18+D3    |        0.40        |        0.70
   7806   |   A1+B15+C18+D6    |        0.40        |        0.70
   7807   |   A1+B15+C18+D7    |        0.40        |        0.70
   7808   |   A1+B15+C18+D10   |        0.40        |        0.70
   7809   |   A1+B15+C18+D11   |        0.40        |        0.70
   7810   |   A1+B15+C18+D12   |        0.40        |        0.70
   7811   |   A1+B15+C18+D14   |        0.40        |        0.70
   7812   |   A1+B15+C18+D15   |        0.40        |        0.70
   7813   |   A1+B15+C18+D16   |        0.40        |        0.70
   7814   |   A1+B15+C18+D17   |        0.40        |        0.70
   7815   |   A1+B15+C18+D18   |        0.40        |        0.70
   7816   |   A1+B15+C18+D19   |        0.40        |        0.70
   7817   |   A1+B15+C18+D20   |        0.40        |        0.70
   7818   |   A1+B15+C19+D1    |        0.40        |        0.70
   7819   |   A1+B15+C19+D2    |        0.40        |        0.70
   7820   |   A1+B15+C19+D3    |        0.40        |        0.70
   7821   |   A1+B15+C19+D6    |        0.40        |        0.70
   7822   |   A1+B15+C19+D7    |        0.40        |        0.70
   7823   |   A1+B15+C19+D8    |        0.40        |        0.70
   7824   |   A1+B15+C19+D10   |        0.40        |        0.70
   7825   |   A1+B15+C19+D11   |        0.40        |        0.70
   7826   |   A1+B15+C19+D12   |        0.40        |        0.70
   7827   |   A1+B15+C19+D14   |        0.40        |        0.70
   7828   |   A1+B15+C19+D15   |        0.40        |        0.70
   7829   |   A1+B15+C19+D16   |        0.40        |        0.70
   7830   |   A1+B15+C19+D17   |        0.40        |        0.70
   7831   |   A1+B15+C19+D18   |        0.40        |        0.70
   7832   |   A1+B15+C19+D19   |        0.40        |        0.70
   7833   |   A1+B15+C19+D20   |        0.40        |        0.70
   7834   |   A1+B15+C20+D1    |        0.40        |        0.70
   7835   |   A1+B15+C20+D2    |        0.40        |        0.70
   7836   |   A1+B15+C20+D3    |        0.40        |        0.70        
   7837   |   A1+B15+C20+D4    |        0.40        |        0.70
   7838   |   A1+B15+C20+D5    |        0.40        |        0.70
   7839   |   A1+B15+C20+D6    |        0.40        |        0.70
   7840   |   A1+B15+C20+D7    |        0.40        |        0.70
   7841   |   A1+B15+C20+D8    |        0.40        |        0.70
   7842   |   A1+B15+C20+D10   |        0.40        |        0.70
   7843   |   A1+B15+C20+D11   |        0.40        |        0.70
   7844   |   A1+B15+C20+D12   |        0.40        |        0.70
   7845   |   A1+B15+C20+D14   |        0.40        |        0.70
   7846   |   A1+B15+C20+D15   |        0.40        |        0.70
   7847   |   A1+B15+C20+D16   |        0.40        |        0.70
   7848   |   A1+B15+C20+D17   |        0.40        |        0.70
   7849   |   A1+B15+C20+D18   |        0.40        |        0.70
   7850   |   A1+B15+C20+D19   |        0.40        |        0.70
   7851   |   A1+B15+C20+D20   |        0.40        |        0.70
   7852   |    A1+B19+C1+D6    |        0.40        |        0.70
   7853   |    A1+B19+C1+D7    |        0.40        |        0.70
   7854   |   A1+B19+C1+D10    |        0.40        |        0.70
   7855   |   A1+B19+C1+D12    |        0.40        |        0.70
   7856   |   A1+B19+C1+D14    |        0.40        |        0.70
   7857   |   A1+B19+C1+D15    |        0.40        |        0.70
   7858   |   A1+B19+C1+D17    |        0.40        |        0.70
   7859   |   A1+B19+C3+D12    |        0.40        |        0.70
   7860   |   A1+B19+C9+D12    |        0.40        |        0.70
   7861   |   A1+B19+C14+D6    |        0.40        |        0.70
   7862   |   A1+B19+C14+D7    |        0.40        |        0.70
   7863   |   A1+B19+C14+D10   |        0.40        |        0.70
   7864   |   A1+B19+C14+D12   |        0.40        |        0.70
   7865   |   A1+B19+C14+D14   |        0.40        |        0.70
   7866   |   A1+B19+C14+D15   |        0.40        |        0.70
   7867   |   A1+B19+C17+D6    |        0.40        |        0.70
   7868   |   A1+B19+C17+D7    |        0.40        |        0.70
   7869   |   A1+B19+C17+D10   |        0.40        |        0.70
   7870   |   A1+B19+C17+D12   |        0.40        |        0.70
   7871   |   A1+B19+C17+D14   |        0.40        |        0.70
   7872   |   A1+B19+C17+D15   |        0.40        |        0.70
   7873   |   A1+B19+C18+D7    |        0.40        |        0.70
   7874   |   A1+B19+C18+D12   |        0.40        |        0.70
   7875   |   A1+B19+C18+D15   |        0.40        |        0.70
   7876   |   A1+B19+C19+D6    |        0.40        |        0.70
   7877   |   A1+B19+C19+D7    |        0.40        |        0.70
   7878   |   A1+B19+C19+D10   |        0.40        |        0.70
   7879   |   A1+B19+C19+D12   |        0.40        |        0.70
   7880   |   A1+B19+C19+D14   |        0.40        |        0.70
   7881   |   A1+B19+C19+D15   |        0.40        |        0.70
   7882   |   A1+B19+C20+D6    |        0.40        |        0.70
   7883   |   A1+B19+C20+D7    |        0.40        |        0.70
   7884   |   A1+B19+C20+D12   |        0.40        |        0.70
   7885   |   A1+B19+C20+D14   |        0.40        |        0.70
   7886   |   A1+B19+C20+D15   |        0.40        |        0.70
   7887   |    A1+B20+C1+D5    |        0.40        |        0.70
   7888   |    A1+B20+C1+D6    |        0.40        |        0.70
   7889   |    A1+B20+C1+D7    |        0.40        |        0.70
   7890   |   A1+B20+C1+D10    |        0.40        |        0.70
   7891   |   A1+B20+C1+D12    |        0.40        |        0.70
   7892   |   A1+B20+C1+D14    |        0.40        |        0.70
   7893   |   A1+B20+C1+D15    |        0.40        |        0.70
   7894   |   A1+B20+C1+D16    |        0.40        |        0.70
   7895   |   A1+B20+C1+D17    |        0.40        |        0.70
   7896   |   A1+B20+C1+D18    |        0.40        |        0.70
   7897   |   A1+B20+C1+D19    |        0.40        |        0.70
   7898   |   A1+B20+C1+D20    |        0.40        |        0.70
   7899   |   A1+B20+C2+D12    |        0.40        |        0.70
   7900   |   A1+B20+C2+D15    |        0.40        |        0.70
   7901   |    A1+B20+C3+D6    |        0.40        |        0.70
   7902   |    A1+B20+C3+D7    |        0.40        |        0.70
   7903   |   A1+B20+C3+D10    |        0.40        |        0.70
   7904   |   A1+B20+C3+D12    |        0.40        |        0.70
   7905   |   A1+B20+C3+D14    |        0.40        |        0.70
   7906   |   A1+B20+C3+D15    |        0.40        |        0.70
   7907   |    A1+B20+C5+D6    |        0.40        |        0.70
   7908   |    A1+B20+C5+D7    |        0.40        |        0.70
   7909   |   A1+B20+C5+D12    |        0.40        |        0.70
   7910   |   A1+B20+C5+D14    |        0.40        |        0.70
   7911   |   A1+B20+C5+D15    |        0.40        |        0.70
   7912   |   A1+B20+C6+D12    |        0.40        |        0.70
   7913   |    A1+B20+C7+D6    |        0.40        |        0.70
   7914   |    A1+B20+C7+D7    |        0.40        |        0.70
   7915   |   A1+B20+C7+D12    |        0.40        |        0.70
   7916   |   A1+B20+C7+D14    |        0.40        |        0.70
   7917   |   A1+B20+C7+D15    |        0.40        |        0.70
   7918   |   A1+B20+C8+D12    |        0.40        |        0.70
   7919   |    A1+B20+C9+D6    |        0.40        |        0.70
   7920   |    A1+B20+C9+D7    |        0.40        |        0.70
   7921   |   A1+B20+C9+D10    |        0.40        |        0.70
   7922   |   A1+B20+C9+D12    |        0.40        |        0.70
   7923   |   A1+B20+C9+D14    |        0.40        |        0.70
   7924   |   A1+B20+C9+D15    |        0.40        |        0.70
   7925   |   A1+B20+C9+D17    |        0.40        |        0.70
   7926   |   A1+B20+C10+D6    |        0.40        |        0.70
   7927   |   A1+B20+C10+D7    |        0.40        |        0.70
   7928   |   A1+B20+C10+D12   |        0.40        |        0.70
   7929   |   A1+B20+C10+D14   |        0.40        |        0.70
   7930   |   A1+B20+C10+D15   |        0.40        |        0.70
   7931   |   A1+B20+C11+D12   |        0.40        |        0.70
   7932   |   A1+B20+C12+D6    |        0.40        |        0.70
   7933   |   A1+B20+C12+D7    |        0.40        |        0.70
   7934   |   A1+B20+C12+D12   |        0.40        |        0.70
   7935   |   A1+B20+C12+D14   |        0.40        |        0.70
   7936   |   A1+B20+C12+D15   |        0.40        |        0.70
   7937   |   A1+B20+C14+D5    |        0.40        |        0.70
   7938   |   A1+B20+C14+D6    |        0.40        |        0.70
   7939   |   A1+B20+C14+D7    |        0.40        |        0.70
   7940   |   A1+B20+C14+D10   |        0.40        |        0.70
   7941   |   A1+B20+C14+D12   |        0.40        |        0.70
   7942   |   A1+B20+C14+D14   |        0.40        |        0.70
   7943   |   A1+B20+C14+D15   |        0.40        |        0.70
   7944   |   A1+B20+C14+D16   |        0.40        |        0.70
   7945   |   A1+B20+C14+D17   |        0.40        |        0.70
   7946   |   A1+B20+C14+D18   |        0.40        |        0.70
   7947   |   A1+B20+C14+D19   |        0.40        |        0.70
   7948   |   A1+B20+C15+D6    |        0.40        |        0.70
   7949   |   A1+B20+C15+D7    |        0.40        |        0.70
   7950   |   A1+B20+C15+D10   |        0.40        |        0.70
   7951   |   A1+B20+C15+D12   |        0.40        |        0.70
   7952   |   A1+B20+C15+D14   |        0.40        |        0.70
   7953   |   A1+B20+C15+D15   |        0.40        |        0.70
   7954   |   A1+B20+C16+D6    |        0.40        |        0.70
   7955   |   A1+B20+C16+D7    |        0.40        |        0.70
   7956   |   A1+B20+C16+D12   |        0.40        |        0.70
   7957   |   A1+B20+C16+D14   |        0.40        |        0.70
   7958   |   A1+B20+C16+D15   |        0.40        |        0.70
   7959   |   A1+B20+C17+D6    |        0.40        |        0.70
   7960   |   A1+B20+C17+D7    |        0.40        |        0.70
   7961   |   A1+B20+C17+D10   |        0.40        |        0.70
   7962   |   A1+B20+C17+D12   |        0.40        |        0.70
   7963   |   A1+B20+C17+D14   |        0.40        |        0.70
   7964   |   A1+B20+C17+D15   |        0.40        |        0.70
   7965   |   A1+B20+C17+D17   |        0.40        |        0.70
   7966   |   A1+B20+C17+D18   |        0.40        |        0.70
   7967   |   A1+B20+C17+D19   |        0.40        |        0.70
   7968   |   A1+B20+C18+D6    |        0.40        |        0.70
   7969   |   A1+B20+C18+D7    |        0.40        |        0.70
   7970   |   A1+B20+C18+D10   |        0.40        |        0.70
   7971   |   A1+B20+C18+D12   |        0.40        |        0.70
   7972   |   A1+B20+C18+D14   |        0.40        |        0.70
   7973   |   A1+B20+C18+D15   |        0.40        |        0.70
   7974   |   A1+B20+C18+D17   |        0.40        |        0.70
   7975   |   A1+B20+C19+D6    |        0.40        |        0.70
   7976   |   A1+B20+C19+D7    |        0.40        |        0.70
   7977   |   A1+B20+C19+D10   |        0.40        |        0.70        
   7978   |   A1+B20+C19+D12   |        0.40        |        0.70
   7979   |   A1+B20+C19+D14   |        0.40        |        0.70
   7980   |   A1+B20+C19+D15   |        0.40        |        0.70
   7981   |   A1+B20+C19+D17   |        0.40        |        0.70
   7982   |   A1+B20+C19+D18   |        0.40        |        0.70
   7983   |   A1+B20+C19+D19   |        0.40        |        0.70
   7984   |   A1+B20+C20+D6    |        0.40        |        0.70
   7985   |   A1+B20+C20+D7    |        0.40        |        0.70
   7986   |   A1+B20+C20+D10   |        0.40        |        0.70
   7987   |   A1+B20+C20+D12   |        0.40        |        0.70
   7988   |   A1+B20+C20+D14   |        0.40        |        0.70
   7989   |   A1+B20+C20+D15   |        0.40        |        0.70
   7990   |   A1+B20+C20+D17   |        0.40        |        0.70
   7991   |   A1+B20+C20+D18   |        0.40        |        0.70
   7992   |   A1+B15+C18+D4    |        0.20        |        0.50
   7993   |    A1+B15+C1+D4    |        0.20        |        0.40
   7994   |    A1+B15+C1+D5    |        0.20        |        0.40
   7995   |   A1+B15+C14+D5    |        0.20        |        0.40
   7996   |   A1+B15+C17+D4    |        0.20        |        0.40
   7997   |   A1+B15+C18+D5    |        0.20        |        0.40
   7998   |   A1+B15+C19+D4    |        0.20        |        0.40
   7999   |   A1+B15+C19+D5    |        0.20        |        0.40
   8000   |   A1+B15+C14+D4    |        0.40        |        0.30
