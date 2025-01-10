To run this code, use the following terminal command:

```sh
git clone http://github.com/xnileshtiwari/refractored_llm.git
```
```sh
python -m venv myenv
source myenv/bin/activate
```

```sh
pip install -r requirements.txt
```

```sh
dir
```

```sh
New-Item -ItemType File -Path "__init__.py"
New-Item -ItemType File -Path "llm/__init__.py"
New-Item -ItemType File -Path "Database/__init__.py"
New-Item -ItemType File -Path "pinecone_vector_database/__init__.py"
New-Item -ItemType File -Path "test/__init__.py"
```

```sh
$env:PYTHONPATH = "C:\xampp\htdocs\ai-case"
```




# Document processing
``` sh
$env:PYTHONPATH = "C:\xampp\htdocs\ai-case" <- Root 
```


# API usage 

## Document processing => upload.py

``` sh
python .\upload.py
```

Method: POST
http://127.0.0.1:5000/process-document

```json

{
    "unique_id": "sample2.pdf",
    "link": "https://pdfobject.com/pdf/sample.pdf"
}

```

x-api-key = 123



## Chat => test/cat.py

``` sh
python .\test\cat.py
```   

Method: POST 
http://127.0.0.1:5001/chat

```json
{
    "index_name": "sample2.pdf",
    "user_input": "What is the summary of this document?"
}
```

x-api-key = 1234



# Create a directory for local packages
mkdir -p ~/packages
cd ~/packages

# Create requirements.txt with core packages first
cat > requirements.txt << EOL
langchain-community
langchain-core
langchain-google-genai
langchain-pinecone
langchain-text-splitters
langserve
langsmith
pymysql
flask
langchain-text-splitters
pypdf
gunicorn
waitress
# Install packages locally
pip3 install --user -r requirements.txt

# Set PYTHONPATH to include local packages
export PYTHONPATH=$HOME/packages:$PYTHONPATH

# Test installation
python3 -c "import flask; print(flask.__version__)"