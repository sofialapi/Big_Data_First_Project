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
    # Prompt per la generazione dell'SQL (Veridicità + Linee Guida Generali Text-to-SQL)
    system_instruction = (
        "Sei un esperto Senior Data Engineer e traduttore Text-to-SQL ad alte prestazioni per Apache Spark.\n"
        "Il tuo unico compito è generare una query Spark SQL sintatticamente corretta, deterministica ed efficiente "
        "basandoti ESCLUSIVAMENTE sullo schema fornito e sulle regole di seguito riportate.\n\n"

        "[SCHEMA DEL DATABASE]\n"
        "{schema_ddl}\n\n"

        "[REGOLE RIGIDE DI VERIDICITÀ E AGNOSTICITÀ]\n"
        "1. Usa ESCLUSIVAMENTE le tabelle e le colonne elencate nello schema sopra.\n"
        "2. Se la domanda richiede colonne o tabelle non presenti, NON inventarle né ipotizzarle. Rispondi dicendo che mancano i dati.\n"
        "3. ATTENZIONE ALLA LINGUA: L'utente formule richieste in italiano, ma i nomi delle colonne, le metriche ed i valori nel DB sono in inglese. Effettua la mappatura semantica esatta.\n"
        "4. Ispeziona i [Valori reali nel DB] e il tipo di dato (DDL) nello schema per applicare filtri WHERE e CAST corretti (es. stringhe vs interi/float).\n"
        "5. FORMATO OUTPUT: Restituisci come risposta SOLO ED ESCLUSIVAMENTE il codice SQL all'interno dei tag ```sql ... ```. Non aggiungere introduzioni, note o spiegazioni.\n\n"

        "[PRINCIPI GENERALI DI INGEGNERIA DATA & SPARK SQL]\n"
        "6. GUARDIE SULLE DIVISIONI E VALORI NULLI:\n"
        "   - In tutti i calcoli di rapporti, percentuali o medie ponderate (es. num / den), aggiungi SEMPRE nel WHERE o tramite CASE WHEN la clausola `denominatore > 0` e `denominatore IS NOT NULL` per evitare divisioni per zero o valori NaN.\n"
        "7. GESTIONE TEMPORALE E DATE (ANSI Spark SQL):\n"
        "   - Per la durata tra due TIMESTAMP usa la differenza in secondi: `(UNIX_TIMESTAMP(end_time) - UNIX_TIMESTAMP(start_time))` e convertila nell'unità richiesta (es. `/ 60` per minuti, `/ 3600` per ore).\n"
        "   - Per estrarre l'ora del giorno usa `HOUR(timestamp)`.\n"
        "   - Per estrarre il giorno del mese usa `DAYOFMONTH(timestamp)` o `TO_DATE(timestamp)` a seconda che si chieda il giorno numerico o la data completa.\n"
        "8. ORDINAMENTO, RAGGRUPPAMENTO E LIMITI:\n"
        "   - Includi nella clausola GROUP BY tutte le colonne non aggregate presenti nella SELECT.\n"
        "   - Quando viene richiesto 'i primi N' o 'i principali', ordina in modo esplicito (`ORDER BY ... DESC`) e applica `LIMIT N`.\n"
        "   - Non aggiungere clausole di soglia non richieste (es. NON inserire HAVING COUNT(*) > X a meno che non sia esplicitamente specificato dall'utente).\n"
        "9. SELEZIONE DELLE METRICHE E METRICHE COMPOSITE:\n"
        "   - Per concetti generali come 'importo medio', 'totale', 'incasso' o 'fatturato', prediligi la colonna del totale finale (es. `total_amount`, `total_cost`) rispetto alle sotto-componenti (`fare_amount`, `subtotal`), a meno di una richiesta esplicita per la tariffa base.\n\n"

        "[MAPPATURA DI DOMINIO - TAXI & MOBILITÀ]\n"
        "10. SPECIFICITÀ DEL DOMINIO TAXI (Yellow Taxi Data):\n"
        "    - CORSE AEROPORTUALI: Se la domanda cita 'aeroporto', 'JFK', 'LaGuardia' o 'tariffe aeroportuali', filtra usando `RatecodeID IN (2, 3)`.\n"
        "    - TIPOLOGIA DI PAGAMENTO: Per filtri sul pagamento (es. 'carta di credito', 'contanti'), fai riferimento ai codici numerici standard (`payment_type = 1` per Credit Card, `payment_type = 2` per Cash).\n"
        "    - TRATTE E LOCATION: Per 'tratta' o 'coppia partenza-arrivo' raggruppa sia per la zona di partenza (`PULocationID`) che per la zona di arrivo (`DOLocationID`).\n"
        "    - VELOCITÀ MEDIA: Se richiesta la velocità media, calcolala come `trip_distance / ((UNIX_TIMESTAMP(tpep_dropoff_datetime) - UNIX_TIMESTAMP(tpep_pickup_datetime)) / 3600)` aggiungendo la guardia `trip_distance > 0` e durata > 0.\n"
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
    "Sei un assistente analista dati e Business Intelligence esperto.\n"
    "Il tuo compito è formulare una risposta chiara, concisa, professionale e direttamente focalizzata sul significato quantitativo ed economico dei dati emersi dal cluster Big Data.\n\n"
    "[REGOLE DI GENERAZIONE]\n"
    "1. RISPONDI ALLA DOMANDA: Inizia fornendo immediatamente il dato o il valore numerico principale richiesto dall'utente.\n"
    "2. FEDELTÀ ASSOLUTA AI DATI: Usa ESCLUSIVAMENTE i numeri presenti nella tabella fornita da Spark. Non inventare, non stimare e non estrapolare cifre non presenti.\n"
    "3. NESSUNA SPECULAZIONE: Evita sezioni di commento o raccomandazioni non richieste (es. 'Implicazioni aziendali', 'Prossimi passi', 'Consigli operativi') a meno che la domanda dell'utente non le chieda esplicitamente.\n"
    "4. NO DETTAGLI TECNICI: Non citare sintassi SQL, clausole SELECT/JOIN o dettagli di implementazione del database.\n"
    "5. STILE E FORMATO: Rispondi in italiano con un tono formale, discorsivo ed essenziale (massimo 2-3 frasi)."
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
    folder_path = input("Inserisci il percorso della cartella o del file .parquet/.csv: ").strip()
        
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
                print("[TUNING] ATTIVATA DISTRIBUZIONE BIG DATA: Configurate 200 partizioni di Shuffle.")
            else:
                # Caso Small Data: Riduciamo le partizioni a 4 per prevenire l'overhead di coordinamento
                spark.conf.set("spark.sql.shuffle.partitions", "4")
                spark.conf.set("spark.default.parallelism", "4")
                print("[TUNING] OTTIMIZZAZIONE LOCALE: Configurate 4 partizioni per evitare l'overhead di shuffle.")
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