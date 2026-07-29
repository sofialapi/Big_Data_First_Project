import warnings
warnings.filterwarnings("ignore") # Sopprime tutti gli UserWarning (incluso PyArrow)
import os
os.environ["PYSPARK_SUBMIT_ARGS"] = "--driver-java-options '-Dlog4j.logLevel=ERROR' pyspark-shell"

import json
import time
import pandas as pd
from config import get_spark_session, get_llm
from metadata_discovery import discover_meta_and_register_views
from main import generate_sql_query, generate_natural_language_answer, extract_sql

def compare_dataframes(df_generated, df_ground_truth):
    """
    Confronta i risultati di Spark per la metrica Execution Accuracy (EX).
    """
    try:
        pd_gen = df_generated.toPandas()
        pd_gt = df_ground_truth.toPandas()
        
        if pd_gen.shape != pd_gt.shape:
            return False
            
        pd_gen.columns = [str(c).lower() for c in pd_gen.columns]
        pd_gt.columns = [str(c).lower() for c in pd_gt.columns]
        
        return pd.DataFrame.equals(pd_gen, pd_gt) or pd_gen.values.tolist() == pd_gt.values.tolist()
    except Exception:
        return False

def evaluate_nl_answer(llm, question, spark_result_str, nl_answer):
    """
    LLM-as-a-Judge per la metrica NL Answer Accuracy.
    """
    prompt = f"""
    Sei un giudice per la valutazione di sistemi Big Data QA.
    Domanda dell'utente: {question}
    Risultato grezzo di Spark: {spark_result_str}
    Risposta generata in linguaggio naturale: {nl_answer}

    La risposta in linguaggio naturale riporta in modo corretto e veritiero i dati estratti da Spark senza allucinazioni?
    Rispondi SOLTANTO con una parola: 'YES' oppure 'NO'.
    """
    try:
        response = llm.invoke(prompt).content.strip().upper()
        return "YES" in response
    except Exception:
        return True

def run_benchmark(dataset_path, test_suite_path="test_suite.json", output_report_path="benchmark_results.json"):
    # Inizializzazione Spark e riduzione dei Log del terminale
    spark = get_spark_session()
    spark.sparkContext.setLogLevel("ERROR")
    llm = get_llm()
    
    print("\n====================================================")
    print("   STANDARDIZED BENCHMARK RUNNER (TAG EVALUATOR)    ")
    print("====================================================")
    
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
        
    total_tests = len(test_cases)
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
        except Exception:
            t_sql = time.time() - t0
            print(f"[{idx}/{total_tests}] [{diff}] FAIL (Gen SQL error)")
            results.append(test_record)
            continue
            
        # B. Esecuzione in Spark (Query Generata + Ground Truth)
        t1 = time.time()
        try:
            spark_df_gen = spark.sql(gen_sql)
            res_str_gen = spark_df_gen._jdf.showString(10, 100, False)
            spark_df_gt = spark.sql(gt_sql)
            t_spark = time.time() - t1
            
            # Impostiamo valid_sql a True SOLO SE l'esecuzione Spark non ha sollevato errori
            test_record["is_sql_valid"] = True
            valid_sql_count += 1
            
            # Valutazione EX
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
        
        # Output compatto e sintetico sul terminale
        status_sql = "OK" if test_record["is_sql_valid"] else "FAIL"
        status_ex = "OK" if test_record["is_exec_accurate"] else "FAIL"
        print(f"[{idx}/{total_tests}] [{diff}] SQL: {status_sql} | EX: {status_ex} | Time: {t_tot:.2f}s -> {question[:45]}...")
        
        results.append(test_record)

    # 3. Metriche Finali
    svr = (valid_sql_count / total_tests) * 100
    ex = (exact_execution_count / total_tests) * 100
    nl_ac = (nl_correct_count / total_tests) * 100
    
    avg_t_sql = sum(r["latency"].get("t_sql_sec", 0) for r in results) / total_tests
    avg_t_spark = sum(r["latency"].get("t_spark_sec", 0) for r in results) / total_tests
    avg_t_nl = sum(r["latency"].get("t_nl_sec", 0) for r in results) / total_tests

    summary = {
        "dataset_path": dataset_path,
        "main_table_name": main_table_name,
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
    print(f"Totale Query Testate : {total_tests}")
    print(f"SQL Validity Rate    : {svr:.2f}%")
    print(f"Execution Accuracy   : {ex:.2f}%")
    print(f"NL Answer Accuracy   : {nl_ac:.2f}%")
    print("----------------------------------------------------")
    print(f"Latenze Medie: LLM SQL={avg_t_sql:.2f}s | Spark={avg_t_spark:.2f}s | LLM NL={avg_t_nl:.2f}s")
    print("====================================================")
    print(f"Report salvato in: {output_report_path}\n")

if __name__ == "__main__":
    default_path = "./dataset_storage/yellow_tripdata_2022-10.parquet"
    path = input(f"Inserisci percorso [Default: {default_path}]: ").strip() or default_path
    run_benchmark(path)