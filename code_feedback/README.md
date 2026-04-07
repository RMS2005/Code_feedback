Official code repository for [Autograder+: A Multi-Faceted AI Framework for Rich Pedagogical Feedback in Programming Education.](https://arxiv.org/abs/2510.26402)

# Autograder+: A Multi-Faceted AI Framework for Rich Pedagogical Feedback in Programming Education

The rapid growth of programming education has outpaced traditional assessment tools, leaving faculty with limited means toprovide meaningful, scalable feedback. Conventional autograders,while efficient, act as “black-box” systems that merely indicate pass/fail status, offering little instructional value or insight into the student’s approach. Autograder+ addresses this gap through two unique features: (1) feedback generation via a fine-tuned Large Language Model (LLM), and (2) visualization of all student code submissions. The LLM is adapted via domain-specific finetuning on curated student code and expert annotations, ensuring feedback is pedagogically aligned and context-aware. In empirical
evaluation across 600 student submissions across various programming problems, Autograder+ produced feedback with an average BERTScore F1 of 0.75, demonstrating strong semantic alignment with expert-written feedback. To make visualizations meaningful,Autograder+ employs contrastively learned embeddings trained on 1,000 annotated submissions, which organize solutions into a performance-aware semantic space resulting in clusters of function-
ally similar appraoches. The framework also integrates a Prompt Pooling mechanism, allowing instructors to dynamically influence the LLM’s feedback style using a curated set of specialized prompts.
By combining advanced AI feedback generation, semantic organization, and visualization, Autograder+ reduces evaluation workload while empowering educators to deliver targeted instruction andfoster resilient learning outcomes.

### Core Features

*   **Secure & Flexible Execution:** Utilizes Docker to run student code in isolated, sandboxed environments. Natively supports both full-program scripts and function-only submissions.
*   **Tiered Analysis Pipeline:** Allows users to select the depth of analysis (`dynamic`, `embedding`, `full`) to balance speed with the richness of feedback.
*   **Semantic Code Embeddings:** Employs models like `nomic-ai/nomic-embed-code` to convert code into meaningful vector representations, enabling quantitative analysis of solution similarity.
*   **AI-Generated Pedagogical Feedback:** Leverages powerful LLMs (e.g., Qwen2) served locally via Ollama to provide structured, human-like feedback on debugging, code functionality, and conceptual understanding.
*   **Instructor Analytics Dashboard:** Generates interactive UMAP visualizations that map the entire class's solution space, allowing instructors to visually identify common strategies, shared misconceptions, and successful approaches.
*   **Fine-Tuning Capabilities:** Includes scripts for fine-tuning embedding models using advanced techniques like Multi-Label Supervised Contrastive Learning to make them "correctness-aware."

---

## Installation and Setup Guide

Follow these steps to set up and run the Autograder+ framework on your local machine.

### Step 1: Prerequisites

Before you begin, ensure you have the following software installed and configured:

1.  **Python:** Python 3.9 or higher is recommended.
2.  **Git:** For cloning the repository.
3.  **Docker:** Docker must be installed and the Docker daemon must be running.
    *   [Install Docker Engine](https://docs.docker.com/engine/install/)
    *   **Linux Users:** After installation, you must add your user to the `docker` group to run Docker commands without `sudo`.
        ```bash
        sudo usermod -aG docker $USER
        ```
        **Important:** You need to **log out and log back in** for this change to take effect. You can verify by running `docker ps`, which should execute without a permission error.
4.  **Ollama (for AI Feedback):**
    *   Ollama is required to run the generative LLMs locally. [Install Ollama](https://ollama.com/).
    *   After installation, pull the model(s) you intend to use. For example, to pull a 7-billion parameter Qwen2 instruct model, run:
        ```bash
        ollama pull qwen2:7b-instruct
        ```
    *   Ensure the model tag in `src/modules/feedback_engine.py` (e.g., `OLLAMA_MODEL_ID`) matches a model you have pulled. You can see your local models with `ollama list`.

### Step 2: Clone the Repository

Open your terminal and clone the project repository:

```bash
git clone https://github.com/your-username/autograder-plus.git
cd autograder-plus

(Replace your-username/autograder-plus.git with your actual repository URL.)

### Step 3: Set Up the Python Environment

It is strongly recommended to use a Python virtual environment to manage dependencies.

1.Create a virtual environment:   
python3 -m venv venv

2.Activate the virtual environment:

  On macOS / Linux
  
source venv/bin/activate

On Windows:
        
    .\venv\Scripts\activate

Install required packages:

The requirements.txt file contains all necessary Python libraries.

    pip install -r requirements.txt     

You are now ready to use Autograder+!


## Project Structure and File Descriptions

The project is organized into several key directories:

autograder-plus/
├── assignments/            # Assignment configurations
│   └── hwX/
│       └── config.json
├── reports/                # Generated output reports
│   └── hwX/
│       ├── Report_... .md
│       ├── Summary_... .csv
│       └── interactive_embeddings_... .html
├── submissions/            # Student code submissions
│   └── hwX/
│       ├── student_id/
│       │   └── main.py
│       └── student_id.py
├── src/                    # Source code for the autograder
│   └── modules/            # All analysis and generation modules
│       ├── ingestion.py
│       ├── static_analyzer.py
│       ├── dynamic_analyzer.py
│       ├── embedding_engine.py
|       ├── prompt_pool.py
│       ├── feedback_engine.py
│       ├── feedback_generator.py
│       └── analytics_engine.py
│   └── pipeline.py         # Main pipeline orchestrator
├──  other_module/
|     Contrastive_finetune/  
│      ├── fine_tune.py
|      ├── mnrloss.py
│      └── mul_supcon_loss.py
├── main.py                 # Main Command-Line Interface (CLI) entry point
└── requirements.txt        # Python package dependencies



Key File Descriptions

    main.py: The entry point for the application. It uses click to define the command-line interface and its arguments (--level, --config, etc.).

    src/pipeline.py: The central orchestrator. It initializes all the engine modules and runs the analysis pipeline in the correct sequence based on the selected --level.

    src/modules/:

        ingestion.py: Reads config.json and finds/loads student code from the submissions directory. Handles both student_id/code.py and student_id.py formats.

        static_analyzer.py: Performs pre-execution checks on the code using Abstract Syntax Trees (ASTs) to find syntax errors and basic code structures.

        dynamic_analyzer.py: The core execution engine. It uses Docker to securely run student code against test cases and captures the results. It generates a runner.py script on-the-fly to handle different execution_modes.

        embedding_engine.py: Loads a pre-trained or fine-tuned embedding model (e.g., nomic-embed-code) to convert code into semantic vectors.

        feedback_engine.py: Connects to the local Ollama server to send prompts (containing student code, errors, and instructor questions) to a generative LLM (e.g., Qwen2) and parses the structured feedback.

        prompt_pool.py: Collection of Prompt .

        feedback_generator.py: Consumes the results from all analysis stages and generates the final human-readable reports (aggregated .md and summary .csv).

        analytics_engine.py: Takes the embeddings from all students, performs UMAP dimensionality reduction, and generates the interactive Plotly (.html) visualization.

    finetune/: Contains standalone scripts for advanced users to fine-tune embedding models on custom datasets (e.g., using Multi-Label Supervised Contrastive Learning).

    assignments/: Instructors place their config.json files here to define new assignments.

    submissions/: Student code should be placed here, following the structure defined for the assignment.

    reports/: All output files are saved here by default.
    
    
Usage

Run the autograder from the root directory of the project.
Command Structure
code Bash

    
python main.py grade --level <LEVEL> --assignment-config <PATH> --submissions-dir <PATH> --output-dir <PATH>

  

Examples
Run the Full Pipeline (Default)

This runs all stages, including dynamic tests, embedding generation, and AI feedback generation.
code Bash

    
python main.py grade \
    --assignment-config ./assignments/hw2/config.json \
    --submissions-dir ./submissions/hw2/ \
    --output-dir ./reports/hw2_full_run

  

Run for Analytics Only

This is faster as it skips the slow generative feedback stage. It's perfect for generating the UMAP plot.
code Bash

    
python main.py grade --level embedding \
    --assignment-config ./assignments/hw2/config.json \
    --submissions-dir ./submissions/hw2/ \
    --output-dir ./reports/hw2_embedding_run

  

Run for Quick Correctness Check

This is the fastest mode, running only the classic autograder functionality.
code Bash

    
python main.py grade --level dynamic \
    --assignment-config ./assignments/hw2/config.json \
    --submissions-dir ./submissions/hw2/ \
    --output-dir ./reports/hw2_dynamic_only

  


