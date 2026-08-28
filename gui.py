import os
import sys
import threading
import tkinter as tk
from tkinter import messagebox, ttk
import urllib.parse
from tkinterdnd2 import DND_FILES, TkinterDnD

# Importiamo le funzioni core dal tuo main.py e config.py
from config import get_llm, get_spark_session
from main import extract_sql, generate_natural_language_answer, generate_sql_query
from metadata_discovery import discover_meta_and_register_views


class UniversalTAGGui:
    def __init__(self, root):
        self.root = root
        self.root.title("TAG - Big Data Analytics")
        self.root.geometry("700x750")
        self.root.configure(bg="#f4f6f9")

        # Variabili di stato del sistema
        self.spark = None
        self.llm = None
        self.schema_ddl = None
        self.registered_tables = None
        
        # Coda sicura per i messaggi tra thread
        self.message_queue = []

        self._build_ui()
        
        # Bind dell'evento virtuale per aggiornare la chat in sicurezza
        self.root.bind("<<UpdateChat>>", self._consume_chat_queue)

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        
        # 1. HEADER TITLE
        header = tk.Label(
            self.root, 
            text="INTERFACCIA GENERATIVA TAG PER BIG DATA ANALYTICS", 
            font=("Helvetica", 14, "bold"), 
            bg="#1e293b", 
            fg="white", 
            pady=15
        )
        header.pack(fill=tk.X)

        # 2. DROP BOX (CENTRALE)
        self.drop_frame = tk.LabelFrame(
            self.root, 
            text=" Configurazione Dataset ", 
            font=("Helvetica", 10, "bold"), 
            bg="#f4f6f9", 
            padx=10, 
            pady=10
        )
        self.drop_frame.pack(fill=tk.X, padx=15, pady=10)

        self.drop_label = tk.Label(
            self.drop_frame,
            text="Trascina qui la cartella da interrogare",
            font=("Helvetica", 11, "italic"),
            bg="#ffffff",
            fg="#64748b",
            height=5,
            bd=2,
            relief="groove",
            cursor="hand2"
        )
        self.drop_label.pack(fill=tk.X, pady=5)
        
        # Abilitazione Drag and Drop sulla label
        self.drop_label.drop_target_register(DND_FILES)
        self.drop_label.dnd_bind('<<Drop>>', self._handle_drop)

        # 3. CHAT SCREEN (AREA DI DIALOGO)
        chat_frame = tk.LabelFrame(
            self.root, 
            text=" Sessione di Analisi Dati ", 
            font=("Helvetica", 10, "bold"), 
            bg="#f4f6f9", 
            padx=10, 
            pady=10
        )
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)

        self.chat_box = tk.Text(
            chat_frame, 
            font=("Helvetica", 11), 
            bg="#ffffff", 
            state=tk.DISABLED, 
            wrap=tk.WORD,
            bd=1,
            relief="solid"
        )
        self.chat_box.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        scrollbar = ttk.Scrollbar(chat_frame, command=self.chat_box.yview)
        scrollbar.pack(fill=tk.Y, side=tk.RIGHT)
        self.chat_box.config(yscrollcommand=scrollbar.set)

        # 4. INPUT BAR (BARRA DI SCRITTURA DOMANDA)
        self.input_frame = tk.Frame(self.root, bg="#f4f6f9")
        self.input_frame.pack(fill=tk.X, padx=15, pady=10)

        self.query_entry = ttk.Entry(self.input_frame, font=("Helvetica", 11), state=tk.DISABLED)
        self.query_entry.pack(fill=tk.X, side=tk.LEFT, expand=True, ipady=5)
        self.query_entry.bind("<Return>", lambda event: self._send_question())

        self.send_btn = ttk.Button(self.input_frame, text="Invia", state=tk.DISABLED, command=self._send_question)
        self.send_btn.pack(side=tk.RIGHT, padx=5)

        # 5. BOTTOM COMMANDS (CONCLUDI INTERROGAZIONE)
        self.exit_btn = ttk.Button(self.root, text="Concludi interrogazione", command=self._exit_system)
        self.exit_btn.pack(pady=10)

    def append_to_chat(self, sender, text):
        """Inietta il messaggio nella coda e sveglia il thread grafico principale"""
        self.message_queue.append((sender, text))
        self.root.event_generate("<<UpdateChat>>", when="tail")

    def _consume_chat_queue(self, event):
        """Eseguito ESCLUSIVAMENTE dal thread principale graficamente sicuro"""
        while self.message_queue:
            sender, text = self.message_queue.pop(0)
            self.chat_box.config(state=tk.NORMAL)
            if sender == "User":
                self.chat_box.insert(tk.END, f"\nUser > {text}\n", "user_style")
            elif sender == "System":
                self.chat_box.insert(tk.END, f"\nSystem > {text}\n\n", "system_style")
                self.chat_box.insert(tk.END, "-"*60 + "\n")
            elif sender == "INFO":
                self.chat_box.insert(tk.END, f"[INFO] {text}\n", "info_style")
            
            self.chat_box.tag_config("user_style", foreground="#1d4ed8", font=("Helvetica", 11, "bold"))
            self.chat_box.tag_config("system_style", foreground="#0f766e", font=("Helvetica", 11))
            self.chat_box.tag_config("info_style", foreground="#475569", font=("Helvetica", 10, "italic"))
            self.chat_box.see(tk.END)
            self.chat_box.config(state=tk.DISABLED)

    def _handle_drop(self, event):
        raw_data = event.data.strip()
        if raw_data.startswith('{') and raw_data.endswith('}'):
            folder_path = raw_data[1:-1]
        else:
            folder_path = raw_data

        if folder_path.startswith("file://"):
            folder_path = folder_path[7:]

        folder_path = urllib.parse.unquote(folder_path)

        if not os.path.isdir(folder_path):
            messagebox.showerror("Errore", f"Il percorso rilasciato non è una cartella valida!")
            return

        self.drop_label.config(text=f"Cartella caricata:\n{os.path.basename(folder_path)}", bg="#e2e8f0")
        self.append_to_chat("INFO", f"Caricamento cartella rilevato: {folder_path}")
        
        threading.Thread(target=self._run_metadata_discovery, args=(folder_path,), daemon=True).start()

    def _run_metadata_discovery(self, folder_path):
        try:
            self.schema_ddl, self.registered_tables = discover_meta_and_register_views(folder_path)
            self.spark = get_spark_session()
            self.llm = get_llm()

            if len(self.registered_tables) > 0:
                tabella_principale = self.registered_tables[0]
                total_rows = self.spark.table(tabella_principale).count()
                
                if total_rows > 1000000:
                    self.spark.conf.set("spark.sql.shuffle.partitions", "200")
                    self.spark.conf.set("spark.default.parallelism", "200")
                else:
                    self.spark.conf.set("spark.sql.shuffle.partitions", "4")
                    self.spark.conf.set("spark.default.parallelism", "4")

            self.append_to_chat("INFO", "Sistema pronto! La barra di input è ora sbloccata.")
            self.root.after(0, self._unlock_input_bar)

        except Exception as e:
            messagebox.showerror("Errore Critico", f"Impossibile avviare il sistema: {str(e)}")

    def _unlock_input_bar(self):
        self.query_entry.config(state=tk.NORMAL)
        self.send_btn.config(state=tk.NORMAL)
        self.query_entry.focus()

    def _send_question(self):
        question = self.query_entry.get().strip()
        if not question:
            return
        
        self.query_entry.delete(0, tk.END)
        self.append_to_chat("User", question)
        
        self.query_entry.config(state=tk.DISABLED)
        self.send_btn.config(state=tk.DISABLED)

        threading.Thread(target=self._process_query_pipeline, args=(question,), daemon=True).start()

    def _process_query_pipeline(self, question):
        try:
            llm_output = generate_sql_query(self.llm, self.schema_ddl, question)
            sql_query = extract_sql(llm_output)
            
            spark_df = self.spark.sql(sql_query)
            query_result_str = spark_df._jdf.showString(50, 100, False)
            
            final_answer = generate_natural_language_answer(self.llm, question, sql_query, query_result_str)
            self.append_to_chat("System", final_answer)

        except Exception as e:
            self.append_to_chat("INFO", f"Errore durante l'elaborazione: {str(e)}")
        finally:
            self.root.after(0, self._unlock_input_bar)

    def _exit_system(self):
        if messagebox.askyesno("Uscita", "Vuoi davvero chiudere la sessione di analisi e arrestare Spark?"):
            if self.spark:
                self.spark.stop()
            self.root.destroy()
            sys.exit(0)


if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = UniversalTAGGui(root)
    root.mainloop()