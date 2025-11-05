install:
	pip install -r requirements.txt
	pip install streamlit

run_frontend:
	cd frontend && streamlit run app/main.py
