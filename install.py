import subprocess
import sys



def install(*args, deps=True):
    arguments = [sys.executable, "-m", "pip", "install", ",".join(args)]
    if not deps:
        arguments.append("--no-deps")
    subprocess.run(arguments)

if __name__ == '__main__':
    install("wheel") # install wheel first
    install("git+https://github.com/moehle/cvxpy_codegen.git", deps=False) # dependencies are broken
    with open("requirements.txt", "r") as file:
        modules = file.read().splitlines()
        for mod in modules: # deps must be installed in the correct order and cannot be installed at the same time
            install(mod)
