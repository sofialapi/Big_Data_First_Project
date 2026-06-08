import os
from config import get_spark_session

def discover_meta_and_register_views(folder_path):
    """
    Scansiona una cartella locale, carica i file CSV presenti in Apache Spark,
    registra le Temporary View SQL e genera lo schema DDL complessivo.
    """
    spark = get_spark_session()
    
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Errore: Il percorso specificato '{folder_path}' non esiste sulla VM.")
    
    # Lista tutti i file all'interno della cartella fornita
    files = os.listdir(folder_path)
    csv_files = [f for f in files if f.endswith('.csv')]
    
    if not csv_files:
        return "Nessun file CSV trovato nella cartella specificata.", {}
    
    schema_description = "Ecco lo schema del database clinico attualmente disponibile:\n\n"
    registered_tables = []
    
    for file_name in csv_files:
        # Il nome della tabella coinciderà con il nome del file senza estensione
        table_name = os.path.splitext(file_name)[0]
        full_file_path = os.path.join(folder_path, file_name)
        
        # Sfruttiamo Apache Spark per leggere il file e inferire automaticamente lo schema (Volume/Varietà)
        df = spark.read.format("csv") \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .load(full_file_path)
        
        # Registriamo il DataFrame come vista SQL temporanea
        df.createOrReplaceTempView(table_name)
        registered_tables.append(table_name)
        
        # Estraiamo il DDL dello schema (nomi colonne e tipi dato generati da Spark)
        schema_description += f"Tabella: {table_name}\nColumns:\n"
        for field in df.schema.fields:
            schema_description += f"  - {field.name} ({field.dataType.simpleString()})\n"
        schema_description += "\n"
        
    schema_description += "Fine dello schema.\n"
    return schema_description, registered_tables

# Piccolo blocco di test locale per verificare che Spark legga correttamente
if __name__ == "__main__":
    print("Inizializzazione del modulo di Metadata Discovery...")
    # Crea una cartella di test al volo se vuoi testarlo autonomamente
    test_folder = "./test_data"
    if os.path.exists(test_folder):
        try:
            ddl_output, tables = discover_meta_and_register_views(test_folder)
            print("--- SCHEMA ESTRATTO CON SUCCESSO ---")
            print(ddl_output)
            print("Tabelle registrate in Spark:", tables)
        except Exception as e:
            print("Si è verificato un errore durante il test:", str(e))
    else:
        print(f"Per testare questo modulo, crea una cartella '{test_folder}' e inserisci dei file .csv")