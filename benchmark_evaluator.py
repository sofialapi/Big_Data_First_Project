import warnings
warnings.filterwarnings("ignore")

import os
os.environ["PYSPARK_SUBMIT_ARGS"] = "--driver-java-options '-Dlog4j.logLevel=ERROR' pyspark-shell"

import json
import time
import sys
import argparse
import pandas as pd
from config import get_spark_session, get_llm
from metadata_discovery import discover_meta_and_register_views
from main import generate_sql_query, generate_natural_language_answer, extract_sql

def compare_dataframes(df_generated, df_ground_truth):
    """
    Confronta i dati dei DataFrame Spark per Execution Accuracy (EX),
    rendendo il confronto agnostico rispetto ad alias e ordinamento.
    """
    try:
        pd_gen = df_generated.toPandas()
        pd_gt = df_ground_truth.toPandas()
        
        # 1. Se la dimensione differisce, i dati sono diversi
        if pd_gen.shape != pd_gt.shape:
            return False
            
        # 2. Resettiamo le intestazioni ignorando gli alias
        pd_gen.columns = range(pd_gen.shape[1])
        pd_gt.columns = range(pd_gt.shape[1])
        
        # 3. Arrotondiamo i valori decimali per evitare disallineamenti di precisione float
        pd_gen = pd_gen.round(4)
        pd_gt = pd_gt.round(4)

        # 4. Ordiniamo le righe per confrontare unicamente il contenuto dei dati
        pd_gen_sorted = pd_gen.sort_values(by=list(pd_gen.columns)).reset_index(drop=True)
        pd_gt_sorted = pd_gt.sort_values(by=list(pd_gt.columns)).reset_index(drop=True)
        
        return pd_gen_sorted.equals(pd_gt_sorted)
    except Exception:
        return False

def evaluate_nl_answer(llm, question, spark_result_str, nl_answer):
    prompt = f"""
    Sei un valutatore severo per un sistema di Text-to-SQL.
    
    Domanda: {question}
    Dati Reali Estratti da Spark:
    {spark_result_str}
    
    Risposta in Linguaggio Naturale generata:
    {nl_answer}
    
    VALUTAZIONE:
    La risposta in linguaggio naturale riporta in modo ACCURATO i dati numerici e le entità presenti nel risultato Spark?
    - Rispondi 'YES' se i numeri e i fatti chiave menzionati nella risposta corrispondono a quelli della tabella Spark.
    - Rispondi 'NO' solo se ci sono discrepanze numeriche palesi o se la risposta dichiara il falso rispetto alla tabella Spark.
    
    Rispondi unicamente con la parola: 'YES' oppure 'NO'.
    """
    try:
        response = llm.invoke(prompt).content.strip().upper()
        return "YES" in response
    except Exception:
        return True

def run_benchmark(dataset_path, test_suite_path="test_suite.json", output_report_path="benchmark_results.json", start_id=1):
    # Inizializzazione Spark e riduzione dei Log
    spark = get_spark_session()
    spark.sparkContext.setLogLevel("ERROR")
    llm = get_llm()
    
    print("\n====================================================")
    print("     STANDARDIZED BENCHMARK RUNNER (TAG EVALUATOR)    ")
    print("====================================================")
    print(f"[DATASET PATH] {dataset_path}")
    
    # 1. Metadata Discovery
    start_disc = time.time()
    schema_ddl, registered_tables = discover_meta_and_register_views(dataset_path)
    disc_time = time.time() - start_disc
    
    # Identificazione dinamica del nome della tabella creata
    main_table_name = registered_tables[0] if registered_tables else "dataset_table"
    print(f"[DISCOVERY] Completata in {disc_time:.2f}s | Tabella Registrata: '{main_table_name}'\n")
    
    if not os.path.exists(test_suite_path):
        raise FileNotFoundError(f"File di suite '{test_suite_path}' non trovato!")
        
    with open(test_suite_path, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)
        
    # FILTRO DINAMICO START_ID
    test_cases = [t for t in test_cases if t["id"] >= start_id]
    
    total_tests = len(test_cases)
    print(f"[SUITE] Esecuzione di {total_tests} test (a partire da ID {start_id})\n")
    
    results = []
    
    valid_sql_count = 0
    exact_execution_count = 0
    nl_correct_count = 0

    # 2. Esecuzione dei Test
    for idx, test in enumerate(test_cases, 1):
        q_id = test["id"]
        diff = test.get("difficulty", "Easy")
        question = test["question"]
        
        # Iniezione dinamica del nome della tabella nella Ground Truth SQL
        gt_sql = test["ground_truth_sql"].format(table_name=main_table_name)
        
        test_record = {
            "id": q_id,
            "difficulty": diff,
            "question": question,
            "ground_truth_sql": gt_sql,
            "is_sql_valid": False,
            "is_exec_accurate": False,
            "is_nl_accurate": False,
            "latency": {}
        }
        
        # A. Generazione Query SQL (LLM)
        t0 = time.time()
        try:
            raw_llm_out = generate_sql_query(llm, schema_ddl, question)
            gen_sql = extract_sql(raw_llm_out)
            t_sql = time.time() - t0
            test_record["generated_sql"] = gen_sql
        except Exception as e:
            t_sql = time.time() - t0
            print(f"[{idx}/{total_tests}] (ID:{q_id}) [{diff}] FAIL (Gen SQL error: {e})")
            results.append(test_record)
            continue
            
        # B. Esecuzione in Spark (Query Generata + Ground Truth)
        t1 = time.time()
        try:
            spark_df_gen = spark.sql(gen_sql)
            res_str_gen = spark_df_gen._jdf.showString(10, 100, False)
            spark_df_gt = spark.sql(gt_sql)
            t_spark = time.time() - t1
            
            test_record["is_sql_valid"] = True
            valid_sql_count += 1
            
            # Valutazione EX con compare_dataframes
            is_ex = compare_dataframes(spark_df_gen, spark_df_gt)
            test_record["is_exec_accurate"] = is_ex
            if is_ex:
                exact_execution_count += 1
        except Exception:
            t_spark = time.time() - t1
            res_str_gen = "Error"
            
        # C. Generazione Risposta NL
        t2 = time.time()
        if test_record["is_sql_valid"]:
            try:
                nl_ans = generate_natural_language_answer(llm, question, gen_sql, res_str_gen)
                t_nl = time.time() - t2
                test_record["nl_answer"] = nl_ans
                
                is_nl_valid = evaluate_nl_answer(llm, question, res_str_gen, nl_ans)
                test_record["is_nl_accurate"] = is_nl_valid
                if is_nl_valid:
                    nl_correct_count += 1
            except Exception:
                t_nl = time.time() - t2
        else:
            t_nl = 0.0

        t_tot = t_sql + t_spark + t_nl
        test_record["latency"] = {
            "t_sql_sec": round(t_sql, 4),
            "t_spark_sec": round(t_spark, 4),
            "t_nl_sec": round(t_nl, 4),
            "t_total_sec": round(t_tot, 4)
        }
        
        status_sql = "OK" if test_record["is_sql_valid"] else "FAIL"
        status_ex = "OK" if test_record["is_exec_accurate"] else "FAIL"
        print(f"[{idx}/{total_tests}] (ID:{q_id}) [{diff}] SQL: {status_sql} | EX: {status_ex} | Time: {t_tot:.2f}s -> {question[:40]}...")
        
        results.append(test_record)
        time.sleep(1) # pausa per tenere più bassi i TPM 

    # 3. Metrike Finali
    if total_tests > 0:
        svr = (valid_sql_count / total_tests) * 100
        ex = (exact_execution_count / total_tests) * 100
        nl_ac = (nl_correct_count / total_tests) * 100
        
        avg_t_sql = sum(r["latency"].get("t_sql_sec", 0) for r in results) / total_tests
        avg_t_spark = sum(r["latency"].get("t_spark_sec", 0) for r in results) / total_tests
        avg_t_nl = sum(r["latency"].get("t_nl_sec", 0) for r in results) / total_tests
    else:
        svr = ex = nl_ac = avg_t_sql = avg_t_spark = avg_t_nl = 0.0

    summary = {
        "dataset_path": dataset_path,
        "main_table_name": main_table_name,
        "start_id": start_id,
        "total_queries": total_tests,
        "metrics": {
            "sql_validity_rate_SVR": f"{svr:.2f}%",
            "execution_accuracy_EX": f"{ex:.2f}%",
            "nl_answer_correctness_NL_AC": f"{nl_ac:.2f}%"
        },
        "average_latencies": {
            "avg_sql_gen_sec": round(avg_t_sql, 4),
            "avg_spark_exec_sec": round(avg_t_spark, 4),
            "avg_nl_gen_sec": round(avg_t_nl, 4),
            "avg_total_sec": round(avg_t_sql + avg_t_spark + avg_t_nl, 4)
        },
        "detailed_results": results
    }

    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n====================================================")
    print("                BENCHMARK REPORT                    ")
    print("====================================================")
    print(f"Totale Query Testate : {total_tests} (a partire da ID {start_id})")
    print(f"SQL Validity Rate    : {svr:.2f}%")
    print(f"Execution Accuracy   : {ex:.2f}%")
    print(f"NL Answer Accuracy   : {nl_ac:.2f}%")
    print("----------------------------------------------------")
    print(f"Latenze Medie: LLM SQL={avg_t_sql:.2f}s | Spark={avg_t_spark:.2f}s | LLM NL={avg_t_nl:.2f}s")
    print("====================================================")
    print(f"Report salvato in: {output_report_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TAG Benchmark Evaluator on Local or S3 Datasets.")
    parser.add_argument("--path", type=str, default=None, help="Percorso del dataset Parquet (locale o s3a://)")
    parser.add_argument("--start_id", type=int, default=1, help="ID della prima query da cui iniziare il test")
    
    args, unknown = parser.parse_known_args()
    
    # Se il parametro non viene passato via argomenti CLI, viene chiesto via prompt (compatibilità locale)
    if args.path:
        dataset_path = args.path
    else:
        # Gestione retro-compatibilità per argomenti posizionali vecchi (es: python benchmark_evaluator.py 1)
        if len(sys.argv) > 1 and sys.argv[1].replace("-", "").isdigit():
            args.start_id = int(sys.argv[1].replace("-", ""))
            
        default_path = "./dataset_storage/yellow_tripdata_2022-10.parquet"
        dataset_path = input(f"Inserisci percorso [Default: {default_path}]: ").strip() or default_path

    run_benchmark(dataset_path, start_id=args.start_id)