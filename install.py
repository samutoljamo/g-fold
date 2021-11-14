import subprocess
import sys



def install(*args):
    pip = sys.executable + " -m pip"
    subprocess.run([pip, "install", ",".join(args)])

if __name__ == '__main__':
    install("wheel") # wheel first
    install("git+https://github.com/moehle/cvxpy_codegen.git --no-deps") # dependencies are broken
    print("installed wheel")
    with open("requirements.txt", "r") as file:
        modules = file.read().splitlines()
        for mod in modules:
            install(mod)
            print("installed", mod)
