import os
from dotenv import load_dotenv

load_dotenv()

# Rileva se si trova su AWS EMR o in un ambiente distribuito Hadoop/YARN
IS_EMR = os.path.exists("/etc/hadoop/conf") or "EMR_CLUSTER_ID" in os.environ or os.path.exists("/usr/lib/spark")

if not IS_EMR:
    try:
        import findspark
        findspark.init(os.getenv("SPARK_HOME", "/home/sofia/spark-3.5.8"))
    except Exception:
        pass

from pyspark.sql import SparkSession
from langchain_openai import ChatOpenAI

def get_spark_session():
    builder = SparkSession.builder.appName("TAG_Adaptive_Clinical_Analytics")

    if IS_EMR:
        # Configurazione per AWS EMR (Cluster distribuito YARN + S3)
        builder = builder \
            .master("yarn") \
            .config("spark.executor.memory", "4g") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.cores", "2") \
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
            .config("spark.hadoop.fs.s3a.aws.credentials.provider", 
                    "com.amazonaws.auth.ContainerCredentialsProvider,com.amazonaws.auth.InstanceProfileCredentialsProvider")
    else:
        # Configurazione Locale standard (VirtualBox / PC)
        builder = builder \
            .master("local[*]") \
            .config("spark.memory.fraction", "0.8") \
            .config("spark.executor.memory", "4g") \
            .config("spark.driver.memory", "2g")

    spark = builder \
        .config("spark.ui.showConsoleProgress", "false") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()
        
    return spark


class ResilientFallbackLLM:
    """
    Wrapper custom che tenta l'invocazione con il provider primario (Groq).
    In caso di errore (429, TPD, ecc.), passa in modo trasparente ai modelli di riserva su OpenRouter.
    """
    def __init__(self, primary_llm, fallback_llms=None):
        self.primary_llm = primary_llm
        self.fallback_llms = fallback_llms if fallback_llms else []

    def invoke(self, input_data, *args, **kwargs):
        # 1. Tentativo con il modello primario (Groq)
        try:
            return self.primary_llm.invoke(input_data, *args, **kwargs)
        except Exception:
            pass  # Fallback silenzioso per un terminale più pulito

        # 2. Tentativi in cascata sui modelli di fallback (OpenRouter)
        for fallback_llm in self.fallback_llms:
            try:
                return fallback_llm.invoke(input_data, *args, **kwargs)
            except Exception:
                continue

        raise RuntimeError("Tutti i modelli primari e di fallback hanno fallito.")

    def __call__(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)


def get_llm():
    # 1. Provider Principale (Groq)
    primary_api_key = os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY")
    primary_base_url = os.getenv("LLM_BASE_URL")
    primary_model = os.getenv("LLM_MODEL_NAME", "openai/gpt-oss-20b")
    
    if not primary_api_key:
        raise ValueError("ERRORE: 'LLM_API_KEY' o 'GROQ_API_KEY' non impostata.")
        
    primary_llm = ChatOpenAI(
        api_key=primary_api_key,
        base_url=primary_base_url if primary_base_url else None,
        model=primary_model,
        temperature=0.0,
        max_retries=0
    )
    
    # 2. Provider di Fallback (OpenRouter) - Lista di Modelli Gratuiti Attivi
    fallback_api_key = os.getenv("FALLBACK_API_KEY")
    fallback_base_url = os.getenv("FALLBACK_BASE_URL", "https://openrouter.ai/api/v1")
    
    # Modelli gratuiti con ruoli e provider diversi
    candidate_models = [
        "openrouter/auto",
        os.getenv("FALLBACK_MODEL_NAME", "google/gemma-4-31b-it:free"),
        "meta-llama/llama-3.3-70b-instruct:free",
        "mistralai/mistral-small-24b-instruct-2501:free",
        "deepseek/deepseek-r1:free"
    ]
    
    fallback_llms = []
    if fallback_api_key:
        for model_name in candidate_models:
            fallback_llms.append(
                ChatOpenAI(
                    api_key=fallback_api_key,
                    base_url=fallback_base_url,
                    model=model_name,
                    temperature=0.0,
                    max_retries=0
                )
            )
        return ResilientFallbackLLM(primary_llm, fallback_llms)
    
    return primary_llm