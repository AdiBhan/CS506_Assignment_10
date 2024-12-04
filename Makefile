build:
	pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
	pip install fastapi uvicorn python-multipart scikit-learn pillow transformers pandas

run:
	uvicorn server:app --reload --host 0.0.0.0 --port 8000

clean:
	pip uninstall -y torch torchvision fastapi uvicorn python-multipart scikit-learn pillow transformers pandas
	rm -rf __pycache__ *.pyc *.pyo