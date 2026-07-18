🌐 Agentic O2C Process Miner: Causal Decision Intelligence for Supply Chains

📖 Abstract

In modern Order-to-Cash (O2C) and supply chain logistics, predictive Machine Learning (ML) can successfully flag high-risk shipments, but it often fails to provide actionable, policy-compliant resolutions. Conversely, Large Language Models (LLMs) can generate prescriptive advice, but frequently suffer from generic hallucinations disconnected from the underlying mathematical data.

This project introduces an Edge-native Decision Intelligence Architecture that bridges this gap. By utilizing a Random Forest classifier validated via a Stochastic Data-Generating Process (DGP), the system extracts mathematical causal drivers using SHAP (SHapley Additive exPlanations). These causal vectors are deterministically bound to an LLM's prompt, forcing the autonomous agent to generate prescriptive actions based strictly on mathematical realities. Furthermore, the system utilizes a localized FAISS Vector Database (Edge-RAG) to securely retrieve enterprise compliance policies, ensuring all AI-generated actions are legally and financially compliant without exposing proprietary data to the cloud.

🔬 Key Methodological Contributions

1. Mathematically Sound Target Generation (No Label Leakage)

To simulate a real-world enterprise environment where ground-truth labels are noisy and unpredictable, this architecture employs a Stochastic Data-Generating Process (DGP).

Rather than hardcoding absolute rules (which leads to label leakage and artificial 100% accuracy), the system calculates risk probabilities using an adjusted sigmoid function based on causal logistics factors (shipping mode, value, weight, etc.).

Labels are assigned via a Binomial Distribution (np.random.binomial). We model a 90% probability of review for high-risk items (accounting for human error/bypasses) and a 5% probability for low-risk items (modeling random compliance audits).

The model is strictly evaluated on a 20% holdout test set, achieving a highly realistic ~85-89% generalization accuracy.

2. SHAP-Constrained Autonomous Agents

LLMs are highly prone to generating plausible but incorrect templates. To counter this, our Agentic workflow is explicitly constrained by Explainable AI (XAI).

When the ML engine flags an anomaly, shap.TreeExplainer flattens the multidimensional feature impact into a causal array.

The LLM prompt is dynamically injected with these exact causal drivers (e.g., Weight (+11.7%)). The Agent is strictly instructed to resolve only these mathematical anomalies, proving a direct pipeline from Explainability to Prescriptive Action.

3. Edge-RAG (Retrieval-Augmented Generation)

To address enterprise data privacy concerns—where uploading proprietary compliance Standard Operating Procedures (SOPs) to cloud APIs is forbidden—this system utilizes Edge-RAG.

Enterprise logistics policies (WEEE, C-TPAT, IATA) are embedded locally using HuggingFace (all-MiniLM-L6-v2).

The semantic search is executed entirely in local RAM using a FAISS (Facebook AI Similarity Search) CPU index.

This ensures that the "Actor-Critic" LLM ensemble grounds its financial and logistical decisions in verifiable, localized enterprise data.

⚙️ Technical Architecture

Predictive Layer: scikit-learn Random Forest (Class-balanced, 100 Estimators)

Explainability Layer: shap TreeExplainer for feature contribution mapping

Vector Store (Edge-RAG): faiss-cpu and sentence-transformers

Agentic Layer: langchain, langchain-groq, utilizing the Llama-3.1-8b-instant model for high-speed MoE (Mixture of Experts) inference.

Interactive UI: streamlit with custom CSS for metric visualization.

🚀 Installation & Reproducibility

To replicate this environment locally for peer review or testing:

1. Clone the Repository

git clone https://github.com/VarunJagad1811/O2CIntelligenceSystem.git
cd O2CIntelligenceSystem


2. Create a Virtual Environment (Recommended)

python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate


3. Install Dependencies

pip install -r requirements.txt


4. Configure Secure Secrets (API Keys)

This project requires a Groq API key to power the Llama 3.1 LLM inference. Create a .streamlit folder and a secrets.toml file to securely store your key (this file is ignored by git for security).

mkdir .streamlit
# Inside .streamlit, create a file named secrets.toml and add:
# GROQ_API_KEY = "your_actual_groq_api_key_here"


5. Run the Application

streamlit run app.py


📊 Dataset Notice

The dataset used (O2C_Dataset_10000_Cases_Enriched_50Features.csv) is a synthetically enriched 10,000-row logistical dataset designed specifically to mimic highly variant cross-border enterprise resource planning (ERP) systems.

🛡️ License

This project is open-source and available under the MIT License.