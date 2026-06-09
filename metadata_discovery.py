import os
from config import get_spark_session

def discover_meta_and_register_views(folder_path):
    """
    Scansiona una cartella locale, carica i file CSV in Spark,
    registra le Temporary View e genera lo schema DDL arricchito con i valori distinti
    per le colonne testuali (prevenzione allucinazioni linguistiche).
    """
    spark = get_spark_session()
    
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Errore: Il percorso specificato '{folder_path}' non esiste sulla VM.")
    
    files = os.listdir(folder_path)
    csv_files = [f for f in files if f.endswith('.csv')]
    
    if not csv_files:
        return "Nessun file CSV trovato nella cartella specificata.", {}
    
    schema_description = "Ecco lo schema del database clinico attualmente disponibile:\n\n"
    registered_tables = []
    
    for file_name in csv_files:
        table_name = os.path.splitext(file_name)[0]
        full_file_path = os.path.join(folder_path, file_name)
        
        df = spark.read.format("csv") \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .load(full_file_path)
        
        df.createOrReplaceTempView(table_name)
        registered_tables.append(table_name)
        
        schema_description += f"Tabella: {table_name}\nColumns:\n"
        for field in df.schema.fields:
            field_type = field.dataType.simpleString()
            schema_description += f"  - {field.name} ({field_type})"
            
            # Se la colonna è una stringa, estraiamo i valori unici per aiutare l'LLM nella traduzione
            if field_type == "string":
                try:
                    # Prendiamo i valori distinti (limitati a 10 per non sovraccaricare il prompt)
                    distinct_values = [row[0] for row in df.select(field.name).distinct().limit(10).collect() if row[0] is not None]
                    if distinct_values:
                        schema_description += f" [Valori reali nel DB: {distinct_values}]"
                except Exception:
                    pass # Se la tabella è vuota o Spark fallisce, procediamo senza bloccarci
            
            schema_description += "\n"
        schema_description += "\n"
        
    schema_description += "Fine dello schema.\n"
    return schema_description, registered_tables