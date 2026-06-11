import sys
import re
import time
from config import get_spark_session, get_llm
from metadata_discovery import discover_meta_and_register_views
from langchain_core.prompts import ChatPromptTemplate

def generate_sql_query(llm, schema_ddl, user_question):
    """
    Invia lo schema del database e la domanda dell'utente a Groq per generare la query Spark SQL.
    """
    # Prompt per la generazione dell'SQL (Veridicità)
    system_instruction = (
        "Sei un assistente hacker/esperto in Big Data clinici e un traduttore text-to-SQL per Apache Spark.\n"
        "Il tuo unico compito è generare una query Spark SQL sintatticamente corretta basandoti ESCLUSIVAMENTE sullo schema fornito.\n\n"
        "[SCHEMA DEL DATABASE CLINICO]\n"
        "{schema_ddl}\n\n"
        "[REGOLE RIGIDE DI VERIDICITÀ]\n"
        "1. Usa solo le tabelle e le colonne elencate nello schema sopra.\n"
        "2. Se la domanda richiede colonne o tabelle non presenti, NON inventarle. Rispondi dicendo che mancano i dati.\n"
        "3. ATTENZIONE ALLA LINGUA: L'utente interroga in italiano, ma i valori testuali nel database sono in inglese. "
        "4. Controlla i [Valori reali nel DB] forniti nello schema per mappare correttamente i termini italiani dell'utente con i reali valori stringa (es. se l'utente chiede 'recupero completo' e tra i valori vedi 'Recovered', usa 'Recovered').\n"
        "5. Restituisci come risposta SOLO ed ESCLUSIVAMENTE la query SQL racchiusa dentro i tag ```sql ... ```. Non aggiungere spiegazioni.\n"
        "6. STRATEGIA DI AGGREGAZIONE: Quando l'utente chiede distribuzioni, conteggi o correlazioni tra categorie (es. trattamenti ed esiti), assicurati di calcolare aggregati significativi (usando COUNT, GROUP BY) senza ordinare ciecamente per colonne testuali che potrebbero saturare il LIMIT con un solo valore. Se una colonna contiene punteggi continui o sparsi, valuta la media (AVG) o raggruppamenti sensati per mostrare la panoramica di TUTTI i trattamenti disponibili."
    )

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        ("human", "Domanda dell'utente: {user_question}")
    ])

    # Compilazione del prompt e invocazione dell'LLM
    chain = prompt_template | llm
    response = chain.invoke({"schema_ddl": schema_ddl, "user_question": user_question})
    return response.content

def generate_natural_language_answer(llm, user_question, sql_query, query_result_str):
    """
    Prende il risultato del calcolo di Spark e lo fa tradurre in linguaggio naturale dall'LLM.
    """
    system_instruction = (
        "Sei un assistente medico-amministrativo esperto.\n"
        "Dato il risultato di un'analisi dati eseguita su un cluster Big Data, formula una risposta chiara, "
        "professionale e discorsiva in linguaggio naturale che risponda direttamente alla domanda iniziale dell'utente.\n"
        "Non menzionare i dettagli tecnici della query SQL o la struttura delle tabelle, concentrati sul significato clinico/gestionale del dato."
    )
    
    human_message = (
        "Domanda originaria dell'utente: {user_question}\n"
        "Query eseguita: {sql_query}\n"
        "Risultato grezzo restituito da Apache Spark:\n{query_result_str}\n\n"
        "Risposta in linguaggio naturale:"
    )

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        ("human", human_message)
    ])

    chain = prompt_template | llm
    response = chain.invoke({
        "user_question": user_question,
        "sql_query": sql_query,
        "query_result_str": query_result_str
    })
    return response.content

def extract_sql(llm_output):
    """
    Funzione di parsing helper per estrarre la stringa SQL pulita dai tag ```sql ```
    """
    match = re.search(r"```sql\s+(.*?)\s+```", llm_output, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return llm_output.replace("```sql", "").replace("```", "").strip()


def main():
    print("====================================================")
    print(" INTERFACCIA GENERATIVA TAG PER BIG DATA CLINICI    ")
    print("====================================================\n")
    
    # 1. Input della cartella e Metadata Discovery
    folder_path = input("Inserisci il percorso della cartella contenente i file .csv clinici: ").strip()
    
    try:
        start_discovery = time.time()
        schema_ddl, registered_tables = discover_meta_and_register_views(folder_path)
        end_discovery = time.time()
        
        print(f"\n[BENCHMARK] Tempo di Metadata Discovery: {end_discovery - start_discovery:.4f} secondi")
        print(f"[INFO] Discovery completato con successo. Registrate {len(registered_tables)} tabelle in Spark SQL.")
    except Exception as e:
        print(f"\n[ERRORE DI INIZIALIZZAZIONE] Impossibile caricare i dati: {str(e)}")
        sys.exit(1)

    # Inizializzazione motori software 
    spark = get_spark_session()
    llm = get_llm()
    
    # --- LAYER DI CONFIGURAZIONE DISTRIBUITA ADATTIVA (BIG DATA TUNING) ---
    try:
        if len(registered_tables) > 0:
            tabella_principale = registered_tables[0]
            # Calcoliamo la cardinalità fisica reale del dataset caricato
            total_rows = spark.table(tabella_principale).count()
            
            print(f"[TUNING] Dimensione totale del dataset rilevata: {total_rows:,} righe.")
            
            if total_rows > 1000000:
                # Caso Big Data: Forziamo la parallelizzazione massiva (Simulazione Cluster HDFS)
                spark.conf.set("spark.sql.shuffle.partitions", "200")
                spark.conf.set("spark.default.parallelism", "200")
                print("[TUNING] 🚀 ATTIVATA DISTRIBUZIONE BIG DATA: Configurate 200 partizioni di Shuffle.")
            else:
                # Caso Small Data: Riduciamo le partizioni a 4 per prevenire l'overhead di coordinamento
                spark.conf.set("spark.sql.shuffle.partitions", "4")
                spark.conf.set("spark.default.parallelism", "4")
                print("[TUNING] 🛴 OTTIMIZZAZIONE LOCALE: Configurate 4 partizioni per evitare l'overhead di shuffle.")
    except Exception as e:
        print(f"[WARN] Impossibile calibrare dinamicamente le partizioni di shuffle: {str(e)}")
    # ----------------------------------------------------------------------
    
    print("\nSistema pronto! Fai una domanda sui tuoi dati clinici (digita 'exit' per uscire).")
    
    # 2. Ciclo di Chat (Interfaccia Utente)
    while True:
        print("\n----------------------------------------------------")
        user_question = input("User > ").strip()
        
        if user_question.lower() in ['exit', 'quit']:
            print("Chiusura del sistema in corso... Alla prossima sessione di analisi!")
            spark.stop()
            break
            
        if not user_question:
            continue
            
        try:
            # Misura tempo GenAI (Groq) per SQL
            start_llm_sql = time.time()
            print("[LLM] Generazione della query Spark SQL in corso tramite Groq...")
            llm_output = generate_sql_query(llm, schema_ddl, user_question)
            # Estrazione e pulizia del codice SQL generato
            sql_query = extract_sql(llm_output)
            end_llm_sql = time.time()
            time_llm_sql = end_llm_sql - start_llm_sql
            print(f"\n[SPARK SQL GENERATA]:\n{sql_query}\n")
            print(f"[BENCHMARK] Tempo generazione SQL (Groq LLM): {time_llm_sql:.4f} secondi")
            
            # Misura tempo esecuzione Big Data (Apache Spark)
            print("[SPARK] Esecuzione della query sulle tabelle in memoria...")
            start_spark = time.time()
            spark_df = spark.sql(sql_query)
            # Convertiamo il DataFrame in una stringa testuale leggibile per passarla all'LLM
            # Mostriamo le prime 50 righe del risultato dell'analisi clinica
            query_result_str = spark_df._jdf.showString(50, 100, False)
            end_spark = time.time()
            
            time_spark = end_spark - start_spark
            print(f"[BENCHMARK] Tempo esecuzione Query (Apache Spark): {time_spark:.4f} secondi")
            
            # Misura tempo GenAI per la risposta discorsiva
            print("[LLM] Elaborazione del risultato e formattazione in linguaggio naturale...")
            start_llm_nl = time.time()
            final_answer = generate_natural_language_answer(llm, user_question, sql_query, query_result_str)
            end_llm_nl = time.time()
            
            time_llm_nl = end_llm_nl - start_llm_nl
            print(f"[BENCHMARK] Tempo formattazione risposta (Groq LLM): {time_llm_nl:.4f} secondi")
            print(f"[BENCHMARK] TEMPO TOTALE ELABORAZIONE: {time_llm_sql + time_spark + time_llm_nl:.4f} secondi")
            
            print(f"\nSystem > {final_answer}")
            
        except Exception as e:
            print(f"\n[ERRORE] Qualcosa è andato storto nell'elaborazione: {str(e)}")
            print("Controlla la sintassi o lo schema del database.")

if __name__ == "__main__":
    main()