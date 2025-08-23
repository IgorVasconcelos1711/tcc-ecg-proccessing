import os
import subprocess
import sys
import venv

def create_venv(venv_dir):
    builder = venv.EnvBuilder(with_pip=True)
    builder.create(venv_dir)
    requirements_file='requirements.txt'

    python_exec = os.path.join(
        venv_dir, "Scripts", "python.exe" if os.name == "nt" else "bin/python"
    )

    print(f"[*] Instalando dependências de {requirements_file}")
    subprocess.check_call([python_exec, "-m", "pip", "install", "-r", requirements_file])

    print("Ambiente virtual pronto!")

if __name__ == "__main__":
    venv_name = 'venv'
    create_venv(venv_name)