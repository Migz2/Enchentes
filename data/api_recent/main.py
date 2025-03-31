import subprocess

arquivos = ["api_recente.py", "web_recent.py", "preparar_dados_recent.py"]

for arquivo in arquivos:
    print(f"Executando {arquivo}...")
    subprocess.run(["python", arquivo])  # No Windows, pode ser "python" ou "python3"
    print(f"{arquivo} finalizado!\n")