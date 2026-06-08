import sys
import re
from config import get_spark_session, get_llm
from metadata_discovery import discover_meta_and_register_views
from langchain_core.prompts import ChatPromptTemplate

def generate_sql_query(llm, schema_ddl, user_question):
    """
    Invia lo schema del database e la domanda dell'utente a Groq per generare la query Spark SQL.
    """
    # Prompt per la generazione dell'SQL (Veridicità)
    system_instruction = (
        "Sei un assistente esperto in Big Data clinici e un traduttore text-to-SQL per Apache Spark.\n"
        "Il tuo unico compito è generare una query Spark SQL sintatticamente corretta basandoti ESCLUSIVAMENTE sullo schema fornito.\n\n"
        "[SCHEMA DEL DATABASE CLINICO]\n"
        "{schema_ddl}\n\n"
        "[REGOLE RIGIDE DI VERIDICITÀ]\n"
        "1. Usa solo le tabelle e le colonne elencate nello schema sopra.\n"
        "2. Se la domanda richiede colonne o tabelle non presenti, NON inventarle. Rispondi dicendo che mancano i dati.\n"
        "3. Se nello schema vedi colonne testuali (string), usa la clausola LIKE in modo case-insensitive se appropriato.\n"
        "4. Restituisci come risposta SOLO ed ESCLUSIVAMENTE la query SQL racchiusa dentro i tag ```sql ... ```. Non aggiungere spiegazioni."
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
        schema_ddl, registered_tables = discover_meta_and_register_views(folder_path)
        print(f"\n[INFO] Discovery completato con successo. Registrate {len(registered_tables)} tabelle in Spark SQL.")
    except Exception as e:
        print(f"\n[ERRORE DI INIZIALIZZAZIONE] impossibile caricare i dati: {str(e)}")
        sys.exit(1)
        
    # Inizializziamo i motori software
    spark = get_spark_session()
    llm = get_llm()
    
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
            print("[LLM] Generazione della query Spark SQL in corso tramite Groq...")
            llm_output = generate_sql_query(llm, schema_ddl, user_question)
            
            # Estrazione e pulizia del codice SQL generato
            sql_query = extract_sql(llm_output)
            print(f"\n[SPARK SQL GENERATA]:\n{sql_query}\n")
            
            # Esecuzione della query su Apache Spark (Fase Big Data)
            print("[SPARK] Esecuzione della query sulle tabelle in memoria...")
            spark_df = spark.sql(sql_query)
            
            # Convertiamo il DataFrame in una stringa testuale leggibile per passarla all'LLM
            # Mostriamo le prime 50 righe del risultato dell'analisi clinica
            query_result_str = spark_df._jdf.showString(50, 100, False)
            
            print("[LLM] Elaborazione del risultato e formattazione in linguaggio naturale...")
            final_answer = generate_natural_language_answer(llm, user_question, sql_query, query_result_str)
            
            print(f"\nSystem > {final_answer}")
            
        except Exception as e:
            print(f"\n[ERRORE] Qualcosa è andato storto nell'elaborazione: {str(e)}")
            print("Controlla la sintassi o lo schema del database.")

if __name__ == "__main__":
    main()