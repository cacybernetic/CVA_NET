VENV_DIR = .venv
VENV_BIN = $(VENV_DIR)/bin
PYTHON3 = $(VENV_BIN)/python3
PYTEST = $(VENV_BIN)/pytest

install:
	test -d .venv || (python3 -m venv .venv && echo "\033[92mVirtual environment is created successfully.\033[0m")
	echo "\033[92m" && $(PYTHON3) --version && echo "\033[0m"
	$(PYTHON3) -m pip install --upgrade pip
	$(PYTHON3) -m pip install torch==2.8.0 torchvision --index-url "https://download.pytorch.org/whl/cpu" && \
	$(PYTHON3) -m pip install -r requirements.txt

gpu_install:
	test -d .venv || (python3 -m venv .venv && echo "\033[92mVirtual environment is created successfully.\033[0m")
	echo "\033[92m" && $(PYTHON3) --version && echo "\033[0m"
	$(PYTHON3) -m pip install --upgrade pip
	$(PYTHON3) -m pip install torch==2.8.0 torchvision
	$(PYTHON3) -m pip install -r requirements.txt

dev_install:
	$(PYTHON3) -m pip install -e .

test:
	$(PYTEST) tests

pep8:
	# Don't remove their commented follwing command lines:
	# autopep8 --in-place --aggressive --aggressive --recursive .
	# autopep8 --in-place --aggressive --aggressive example.py
