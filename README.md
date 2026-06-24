# Big_Data_First_Project
Progetto sviluppato per il corso di Big Data nell'A.A. 2025-2026 del corso di laurea magistrale in Ingegneria Informatica dell'università di Roma Tre

# Interfaccia Generativa TAG ‘text-to-SQL’ per l'Analisi di Big Data Clinici

## Obiettivo del progetto
L'obiettivo di questo progetto è la realizzazione di un'interfaccia Text-to-SQL basata sul paradigma TAG (Table Augmented Generation). Il sistema permette di interrogare dataset clinici e gestionali in formato tabellare attraverso il linguaggio naturale. Sfrutta Apache Spark (Spark SQL) per l'elaborazione distribuita dei dati e l'LLM Llama 3.3 (tramite le API esterne di Groq) per la traduzione della domanda in codice SQL e la generazione della risposta finale.

## Configurazione dell'ambiente
Il software è configurato per l'esecuzione all'interno di un ambiente virtuale Python su una macchina virtuale Linux Ubuntu. 

Entrare nella cartella principale della repository ed eseguire i seguenti comandi per creare e attivare l'ambiente virtuale:

```bash
cd ~/Big_Data_First_Project
python3 -m venv venv
source venv/bin/activate

```

Installare le dipendenze necessarie tramite il gestore di pacchetti pip:

```bash
pip install --upgrade pip
pip install pyspark findspark langchain langchain-community langchain-core langchain-openai

```

## Allineamento dei path e chiavi API

Per consentire a Python di localizzare correttamente l'installazione di Apache Spark e di autenticare le chiamate verso Groq, è necessario aggiungere le seguenti configurazioni in fondo al file ~/.bashrc della propria macchina:

```bash
export SPARK_HOME=/home/{il_tuo_nome_utente}/spark-3.5.8
export PATH=$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH
export LLM_API_KEY="inserisci_la_tua_api_key_di_groq"
export LLM_BASE_URL="https://api.groq.com/openai/v1"
export LLM_MODEL_NAME="llama-3.3-70b-versatile"

```

Dopo aver salvato le modifiche al file, renderle attive nel terminale digitando:

```bash
source ~/.bashrc

```

## Posizionamento dei dataset

I file compressi o i file estratti in formato .csv che si desidera analizzare devono essere posizionati all'interno della struttura della repository nel seguente percorso:

```text
./dataset_storage/nome_del_dataset/

```

Ogni file .csv all'interno della cartella selezionata verrà registrato automaticamente da Spark come una tabella SQL, il cui nome coinciderà con il nome del file stesso (privato dell'estensione). Si raccomanda di utilizzare nomi di file interamente in minuscolo e privi di spazi o caratteri speciali.

## Invocazione del programma da terminale

Garantito che l'ambiente virtuale venv sia attivo, l'applicazione si avvia eseguendo il file principale dal terminale:

```bash
python3 main.py

```

Il programma richiederà l'inserimento del percorso della cartella contenente i dati (ad esempio: ./dataset_storage/hospital_management_dataset). Una volta completata la fase di Metadata Discovery, il prompt sarà pronto per ricevere le domande in linguaggio naturale. Per chiudere l'applicazione e arrestare la sessione Spark, digitare exit.

## Invocazione del programma tramite Interfaccia Grafica (GUI)

Per un'esperienza d'uso totalmente disaccoppiata dal terminale è stata sviluppata un'interfaccia grafica asincrona. Questa mappa l'intero pattern architetturale (Metadata Discovery, Adaptive Tuning e Pipeline TAG) gestendo la computazione in thread separati per preservare la reattività della finestra.

Poiché le librerie grafiche native di sistema sono separate dal core di Python su distribuzioni Linux, è necessario installare il pacchetto grafico tramite il gestore di pacchetti apt della macchina virtuale:

```bash
sudo apt update
sudo apt install python3-tk -y

```
Con l'ambiente virtuale (venv) attivato, installare l'estensione per abilitare il rilascio nativo dei file nelle finestre:

```bash
pip install tkinterdnd2

```
Per consentire a Tkinter di accedere correttamente al server grafico (X11/Wayland) del desktop di Ubuntu (evitando l'errore TclError: no display name), assicurarsi di lanciare il comando dal terminale nativo della VM o esportare la variabile di visualizzazione locale:

```bash
export DISPLAY=:0
python3 gui.py

```

## Modalità d'uso della GUI
è possibile selezionare la cartella desiderata con un classico drag and drop; Non appena il sistema visualizza [INFO] Sistema pronto!, la barra di digitazione inferiore si sbloccherà. Sarà possibile scrivere le domande in linguaggio naturale e premere Invio o fare clic su Invia per visualizzare la risposta discorsiva del modello. Il pulsante inferiore "Concludi interrogazione" interrompe i canali della JVM, arresta la SparkSession in sicurezza per prevenire memory leak e chiude l'applicazione.
