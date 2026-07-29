import os
from config import get_spark_session

def discover_meta_and_register_views(path_input):
    """
    Scansiona un percorso (file singolo .parquet/.csv o cartella),
    registra le Temporary View in Spark mantenendo il nome dinamico della risorsa
    e genera lo schema DDL arricchito con i valori distinti per le stringhe.
    """
    spark = get_spark_session()
    
    if not os.path.exists(path_input):
        raise FileNotFoundError(f"Errore: Il percorso specificato '{path_input}' non esiste.")
    
    schema_description = "Ecco lo schema del database attualmente disponibile:\n\n"
    registered_tables = []
    
    # Clean-up helper per trasformare i nomi di file/cartelle in identificatori SQL validi
    def clean_table_name(name):
        base = os.path.splitext(os.path.basename(os.path.normpath(name)))[0]
        return base.replace("-", "_").replace(".", "_").replace(" ", "_")

    # CASO 1: Il percorso è un SINGOLO FILE (.parquet o .csv)
    if os.path.isfile(path_input):
        table_name = clean_table_name(path_input)
        
        if path_input.endswith('.parquet'):
            df = spark.read.parquet(path_input)
        elif path_input.endswith('.csv'):
            df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(path_input)
        else:
            raise ValueError("Formato file non supportato. Utilizzare file .parquet o .csv.")
            
        df.createOrReplaceTempView(table_name)
        registered_tables.append(table_name)
        schema_description += _extract_table_schema(df, table_name)

    # CASO 2: Il percorso è una CARTELLA
    elif os.path.isdir(path_input):
        files = os.listdir(path_input)
        parquet_files = [f for f in files if f.endswith('.parquet')]
        csv_files = [f for f in files if f.endswith('.csv')]
        
        # Cartella di file Parquet
        if parquet_files:
            # Se la cartella contiene più partizioni della stessa serie, creiamo una vista unica con il nome della cartella
            folder_table_name = clean_table_name(path_input)
            parquet_pattern = os.path.join(path_input, "*.parquet")
            df = spark.read.parquet(parquet_pattern)
            df.createOrReplaceTempView(folder_table_name)
            registered_tables.append(folder_table_name)
            schema_description += _extract_table_schema(df, folder_table_name)

        # Cartella di file CSV (es. multi-tabella come MIMIC-III)
        elif csv_files:
            for cf in csv_files:
                t_name = clean_table_name(cf)
                full_p = os.path.join(path_input, cf)
                df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(full_p)
                df.createOrReplaceTempView(t_name)
                registered_tables.append(t_name)
                schema_description += _extract_table_schema(df, t_name)
        else:
            return "Nessun file .parquet o .csv trovato nella cartella specificata.", []

    schema_description += "Fine dello schema.\n"
    return schema_description, registered_tables


def _extract_table_schema(df, table_name):
    """
    Helper function per estrarre lo schema DDL e campionare i valori stringa unici.
    """
    desc = f"Tabella: {table_name}\nColumns:\n"
    for field in df.schema.fields:
        field_type = field.dataType.simpleString()
        desc += f"  - {field.name} ({field_type})"
        
        # Estrazione valori unici con sampling veloce per evitare rallentamenti su grandi volumi
        if field_type == "string":
            try:
                distinct_values = [
                    row[0] for row in df.select(field.name)
                                        .limit(10000)
                                        .distinct()
                                        .limit(10)
                                        .collect() 
                    if row[0] is not None
                ]
                if distinct_values:
                    desc += f" [Valori reali nel DB: {distinct_values}]"
            except Exception:
                pass
        desc += "\n"
    desc += "\n"
    return desc