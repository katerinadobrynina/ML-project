FROM python:3.9

# Install necessary Python libraries
RUN pip install numpy==1.24.3 scipy==1.10.1 scikit-learn==0.24.2 \
    tpot==0.11.7 auto-sklearn==0.15.0 joblib==1.4.2 distro==1.9.0 \
    typing-extensions==4.12.2 jupyter

# Expose the default Jupyter Notebook port
EXPOSE 8890

# Set the default command to launch Jupyter Notebook
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

