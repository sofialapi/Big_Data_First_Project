import os
import findspark  # <-- AGGIUNGI QUESTO IMPORT

# Diciamo a findspark dove si trova la tua installazione (legge la variabile d'ambiente che hai messo nel .bashrc)
findspark.init(os.getenv("SPARK_HOME", "/home/sofia/spark-3.5.8"))

from pyspark.sql import SparkSession
from langchain_openai import ChatOpenAI

def get_spark_session():
    """
    Inizializza la SparkSession con allocazione ottimale della memoria.
    Il partizionamento verrà poi calibrato dinamicamente in base al volume.
    """
    findspark.init(os.getenv("SPARK_HOME", "/home/sofia/spark-3.5.8"))
    
    spark = SparkSession.builder \
        .appName("TAG_Adaptive_Clinical_Analytics") \
        .master("local[*]") \
        .config("spark.memory.fraction", "0.8") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "2g") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()
        
    return spark

def get_llm():
    """
    Configura e restituisce il client per l'LLM esterno.
    Utilizza variabili d'ambiente per proteggere le chiavi API.
    """
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")
    model_name = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
    
    if not api_key:
        raise ValueError(
            "ERRORE: La variabile d'ambiente 'LLM_API_KEY' non è configurata. "
            "Assicurati di impostarla prima di avviare l'applicazione."
        )
        
    llm = ChatOpenAI(
        api_key=api_key,
        base_url=base_url if base_url else None,
        model=model_name,
        temperature=0.0
    )
    return llm