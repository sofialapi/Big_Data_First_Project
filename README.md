# Big_Data_First_Project
Progetto sviluppato per il corso di Big Data nell'A.A. 2025-2026 del corso di laurea magistrale in Ingegneria Informatica dell'università di Roma Tre

# Interfaccia Generativa TAG ‘text-to-SQL’ per l'Analisi di Big Data

## Obiettivo del progetto
L'obiettivo di questo progetto è la realizzazione di un'interfaccia Text-to-SQL basata sul paradigma TAG (Table Augmented Generation). Il sistema permette di interrogare dataset tabellari di vari ambiti attraverso il linguaggio naturale. L'interfaccia sfrutta Apache Spark (Spark SQL) per l'elaborazione distribuita dei dati e l'LLM OpenAI GPT OSS 20B + sistema di fallback (tramite le API esterne di Groq e OpenRouter) per la traduzione della domanda in codice SQL e la generazione della risposta finale.

## Configurazione dell'ambiente
Il software è configurato per l'esecuzione all'interno di un ambiente virtuale Python su una macchina virtuale Linux Ubuntu. 

Entrare nella cartella principale della repository ed eseguire i seguenti comandi per creare e attivare l'ambiente virtuale:

### 1. Setup dell'Ambiente Virtuale
```bash
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install pyspark findspark langchain langchain-community langchain-core langchain-openai tkinterdnd2
```
### 2. Dipendenze di Sistema per la GUI
```bash
sudo apt update && sudo apt install python3-tk -y
```
## 3. Allineamento dei path e chiavi API
```bash
export SPARK_HOME=/home/{il_tuo_nome_utente}/spark-3.5.8
export PATH=$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH
export GROQ_API_KEY="inserisci_la_tua_api_key_di_groq"
export LLM_BASE_URL="https://api.groq.com/openai/v1"
export LLM_MODEL_NAME="openai/gpt-oss-20b"
export FALLBACK_API_KEY="inserisci_la_tua_api_key_di_openrouter"

```

Dopo aver salvato le modifiche al file, renderle attive nel terminale digitando:

```bash
source ~/.bashrc

```
## Posizionamento dei dataset
I dataset vanno collocati nella directory ./dataset_storage/. Ogni file .csv o .parquet all'interno della cartella selezionata verrà registrato automaticamente da Spark come una vista SQL. 

## Invocazione del programma da terminale

Garantito che l'ambiente virtuale venv sia attivo, l'applicazione si avvia eseguendo il file principale dal terminale:

```bash
python3 main.py

```

Il programma richiederà l'inserimento del percorso della cartella contenente i dati (ad esempio: ./dataset_storage/hospital_management_dataset). Una volta completata la fase di Metadata Discovery, il prompt sarà pronto per ricevere le domande in linguaggio naturale. Per chiudere l'applicazione e arrestare la sessione Spark, digitare exit.

## Invocazione del programma tramite Interfaccia Grafica (GUI)

Per un'esperienza d'uso totalmente disaccoppiata dal terminale è stata sviluppata un'interfaccia grafica asincrona. Questa mappa l'intero pattern architetturale (Metadata Discovery, Adaptive Tuning e Pipeline TAG) gestendo la computazione in thread separati per preservare la reattività della finestra.

Per visualizzare la GUI e interrogarla assicurarsi di lanciare il comando dal terminale nativo della VM o esportare la variabile di visualizzazione locale:

```bash
export DISPLAY=:0
python3 gui.py

```

## Modalità d'uso della GUI
è possibile selezionare la cartella desiderata con un classico drag and drop; Non appena il sistema visualizza [INFO] Sistema pronto!, la barra inferiore si sbloccherà. Sarà possibile scrivere le domande in linguaggio naturale e premere Invio o fare clic su Invia per visualizzare la risposta discorsiva del modello. Il pulsante inferiore "Concludi interrogazione" interrompe i canali della JVM, arresta la SparkSession in sicurezza per prevenire memory leak e chiude l'applicazione.
