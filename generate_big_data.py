import os
import sys
import shutil
from config import get_spark_session
from pyspark.sql.functions import rand, expr

def main():
    print("====================================================")
    print("   GENERATORE DI BIG DATA CLINICI CORRETTO          ")
    print("====================================================")
    
    # Recuperiamo la sessione configurata stabilmente
    spark = get_spark_session()
        
    path_sorgente = "./dataset_storage/hospital_patient_records_dataset/hospital_data_analysis.csv"
    cartella_destinazione = "./dataset_storage/hospital_patient_records_massive"
    file_output_finale = os.path.join(cartella_destinazione, "hospital_data_massive.csv")
    
    if not os.path.exists(path_sorgente):
        print(f"[ERRORE] File sorgente non trovato in: {path_sorgente}")
        return

    print("[INFO] Lettura del dataset sorgente...")
    df_src = spark.read.csv(path_sorgente, header=True, inferSchema=True)
    
    # Estraiamo le etichette reali dal dataset originale
    genders = [row[0] for row in df_src.select('Gender').distinct().collect() if row[0]]
    conditions = [row[0] for row in df_src.select('Condition').distinct().collect() if row[0]]
    
    total_target_records = 5000000
    print(f"[INFO] Generazione di {total_target_records} record in memoria...")
    
    # Generiamo la struttura lineare vuota
    df_base = spark.range(0, total_target_records)
    
    # 1. Assegnazione ID, Età e Costo (vettorizzati)
    df_augmented = df_base.withColumn("Patient_ID", expr("id + 100000")) \
                           .withColumn("Age", (rand(seed=42) * 70 + 15).cast("int")) \
                           .withColumn("Cost", (rand(seed=24) * 49500 + 500).cast("int"))
    
    # 2. Generazione del Genere tramite indice di riga per evitare collassi logici
    # Distribuzione equa 50/50 o basata sul resto della divisione
    df_augmented = df_augmented.withColumn(
        "Gender", 
        expr(f"CASE WHEN (id % 2 = 0) THEN '{genders[0]}' ELSE '{genders[1]}' END")
    )
    
    # 3. Generazione della Patologia (Condition) distribuita matematicamente in modo uniforme
    # Dividiamo ciclicamente i record tra tutte le patologie uniche estratte
    num_conds = len(conditions)
    case_conditions = "CASE "
    for idx, cond in enumerate(conditions):
        case_conditions += f"WHEN (id % {num_conds} = {idx}) THEN '{cond}' "
    case_conditions += "END"
    
    df_augmented = df_augmented.withColumn("Condition", expr(case_conditions))
    
    # Rimuoviamo la colonna contatore temporanea 'id'
    df_final = df_augmented.drop("id")
    
    print("[INFO] Consolidamento in un singolo partizionamento (repartition)...")
    # Costringiamo Spark a unificare i blocchi in un unico file finale sul nodo centrale
    df_single_file = df_final.repartition(1)
    
    print("[INFO] Scrittura fisica del file CSV massivo in corso...")
    tmp_dir = f"{cartella_destinazione}_tmp"
    df_single_file.write.mode("overwrite").options(header="true").csv(tmp_dir)
    
    # Individuiamo l'unico vero file CSV massivo generato all'interno della cartella temporanea
    os.makedirs(cartella_destinazione, exist_ok=True)
    csv_file = [f for f in os.listdir(tmp_dir) if f.endswith('.csv') and f.startswith('part-')][0]
    
    # Spostiamo e rinominiamo direttamente sul percorso pulito desiderato
    if os.path.exists(file_output_finale):
        os.remove(file_output_finale)
        
    shutil.move(os.path.join(tmp_dir, csv_file), file_output_finale)
    
    # Pulizia radicale delle scorie e dei log CRC di Spark
    shutil.rmtree(tmp_dir)
    
    print("\n====================================================")
    print(f"[SUCCESSO] File unico da 5.000.000 di righe creato!")
    print(f"Percorso: {file_output_finale}")
    print("====================================================")
    
    spark.stop()

if __name__ == "__main__":
    main()