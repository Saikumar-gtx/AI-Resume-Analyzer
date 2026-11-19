"""
Related-evidence phrases for skills.

A resume often shows a skill without naming it: "Helm charts on EKS" is
Kubernetes work, "experiment tracking and model versioning" is MLOps. These
phrases let the scorer give *related* (partial) credit and tell the candidate to
name the skill explicitly, because ATS keyword filters only match exact terms.

Keys are lower-case skill names or aliases; values are lower-case phrases.
Embedding similarity was evaluated for this and rejected: with both
all-MiniLM-L6-v2 and bge-m3, true and false skill/bullet pairs overlapped
(e.g. a PyTorch bullet scored higher for "TensorFlow" than a BERT bullet did
for "Transformers"), so a curated, explainable list is used instead.
"""

from __future__ import annotations

CONCEPTS: dict[str, tuple[str, ...]] = {
    # Infrastructure & DevOps
    "kubernetes": ("container orchestration", "helm", "eks", "gke", "aks", "openshift", "kubectl"),
    "docker": ("containerized", "containerised", "dockerfile", "container images", "podman"),
    "ci/cd": (
        "continuous integration", "continuous delivery", "continuous deployment", "github actions",
        "gitlab ci", "jenkins", "circleci", "deployment pipeline", "build pipeline",
    ),
    "monitoring": ("observability", "alerting", "grafana", "prometheus", "datadog", "new relic", "dashboards and alerts"),
    "terraform": ("infrastructure as code", "cloudformation", "pulumi"),
    "infrastructure as code": ("terraform", "cloudformation", "pulumi", "ansible"),
    "aws": ("ec2", "s3", "aws lambda", "sagemaker", "dynamodb", "cloudformation", "amazon web services"),
    "gcp": ("google cloud", "bigquery", "cloud run", "vertex ai", "gke", "firebase"),
    "azure": ("azure functions", "cosmos db", "aks", "azure devops"),
    "cloud (aws/gcp/azure)": (
        "ec2", "s3", "aws lambda", "sagemaker", "google cloud", "cloud run", "bigquery", "vertex ai",
        "azure functions", "cloud deployment", "deployed to the cloud", "heroku", "render.com", "vercel",
    ),
    "linux": ("ubuntu", "debian", "centos", "red hat", "unix", "shell scripting"),
    "incident response": ("on-call", "postmortem", "post-mortem", "incident management", "root cause analysis"),
    "distributed systems": ("distributed computing", "horizontal scaling", "consensus", "sharding", "replication"),
    "system design": ("scalable architecture", "architected", "high availability", "load balancing", "designed the architecture"),
    # Software
    "rest": ("rest api", "restful", "http api", "api endpoints", "openapi", "swagger"),
    "rest|restful": ("rest api", "http api", "api endpoints", "openapi", "swagger"),
    "microservices": ("service-oriented", "microservice", "independently deployable services"),
    "unit testing": ("automated tests", "test coverage", "test-driven", "tdd", "integration tests", "pytest", "junit", "jest"),
    "unit test": ("automated tests", "test coverage", "test-driven", "tdd", "integration tests"),
    "authentication": ("oauth", "jwt", "single sign-on", "sso", "login system", "access tokens"),
    "git": ("github", "gitlab", "bitbucket", "version control"),
    "sql": ("postgresql", "mysql", "sqlite", "sql server", "t-sql", "pl/sql", "sql queries"),
    "accessibility": ("wcag", "screen reader", "aria attributes", "a11y"),
    "responsive design": ("mobile-first", "media queries", "responsive layout"),
    "data structures": ("algorithms", "leetcode", "dynamic programming", "graph algorithms"),
    # Data & ML
    "mlops": (
        "experiment tracking", "model versioning", "model registry", "model monitoring", "ml pipeline",
        "training pipeline", "kubeflow", "sagemaker pipelines", "vertex ai pipelines", "mlflow", "weights & biases",
    ),
    "machine learning": (
        "classification model", "regression model", "predictive model", "random forest", "gradient boosting",
        "xgboost", "scikit-learn", "sklearn", "neural network", "supervised learning",
    ),
    "deep learning": ("neural network", "cnn", "rnn", "lstm", "convolutional", "backpropagation"),
    "pytorch": ("torch",),
    "transformers": ("bert", "gpt-2", "gpt-3", "t5", "roberta", "llama", "attention mechanism", "hugging face", "encoder-decoder"),
    "llm": ("gpt-4", "gpt-3.5", "chatgpt", "llama", "mistral", "gemini", "claude", "generative ai", "genai", "language model"),
    "prompt engineering": ("prompt design", "few-shot", "chain-of-thought", "system prompt", "prompt templates", "prompting"),
    "rag": ("retrieval augmented", "retrieval-augmented", "vector search", "semantic search", "knowledge retrieval"),
    "vector database": ("vector search", "semantic search", "embedding index", "qdrant", "pgvector"),
    "nlp": ("natural language processing", "text classification", "named entity", "sentiment analysis", "tokenization"),
    "computer vision": ("image classification", "object detection", "image segmentation", "opencv", "yolo"),
    "feature engineering": ("feature extraction", "feature selection", "feature store"),
    "model deployment": ("model serving", "served the model", "inference api", "deployed the model", "deployed models"),
    "statistics": ("hypothesis testing", "regression analysis", "bayesian", "anova", "p-value", "confidence interval"),
    "a/b testing": ("controlled experiment", "split test", "multivariate test", "conversion lift", "experimentation"),
    "data visualization": ("charts", "dashboards", "plotting", "matplotlib", "seaborn", "plotly", "tableau", "power bi", "d3"),
    "etl": ("data pipeline", "data pipelines", "ingestion pipeline", "elt", "data ingestion"),
    "etl|elt": ("data pipeline", "data pipelines", "ingestion pipeline", "data ingestion"),
    "data modeling": ("schema design", "star schema", "dimensional model", "data model"),
    "spark": ("pyspark", "databricks", "spark sql"),
    "gpu|cuda": ("cuda", "gpu-accelerated", "gpu acceleration"),
    # Product & process
    "agile": ("scrum", "sprint", "kanban", "standups", "retrospectives"),
    "scrum": ("sprint planning", "scrum master", "standups", "retrospectives"),
    "stakeholder management": ("stakeholders", "cross-functional", "executive updates"),
    "user research": ("user interviews", "usability study", "personas", "customer interviews"),
    "requirements gathering": ("gathered requirements", "requirement analysis", "business requirements", "user stories"),
    "technical writing": ("documentation", "wrote docs", "api documentation", "tutorials"),
}


def concepts_for(name: str, aliases: tuple[str, ...]) -> tuple[str, ...]:
    """Related phrases for a skill, looked up by its name and each alias."""
    phrases: list[str] = []
    for key in (name, "|".join(aliases), *aliases):
        phrases.extend(CONCEPTS.get(key.lower(), ()))
    own = {a.lower() for a in aliases}
    return tuple(p for p in dict.fromkeys(phrases) if p not in own)
