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
$env:PYTHONPATH = "C:\xampp\htdocs\ai-case" <- Root file path
python -m test.main_chat
```
# File Structure:
```sh
├── __pycache__/
├── .venv/
├── .vscode/
├── DASHBOARD/
│   ├── add_one_column.py "(Adds a new column to the cat_is_trending table)"
│   ├── one_adder.py "(Increments +1 per Query)"
│   ├── token_cost_calculator.py "(Calculates token cost)"
│   └── trend_on_date.py "(Get trending analysis for a specific date)"
├── Database/
│   ├── connection.py "(MYSQL database connection)"
│   ├── pdf_metadata.json "(PDF metadata)"
│   ├── token_usage_database_update.py "(Update or insert token usage for the current date)"
│   └── unique_id_generator.py "(Generates unique ID for PDF)"
├── Document_processing/
│   └── document_processing.py "(Document chunking, parsing and uploading to vectorstore)"
├── extras/
│   └── token_reset.py "(Reset token usage for a specific date)"
├── Ilm/
│   ├── generative_model.py "(Generates response)"
│   └── prompt.py "(Prepare prompt)"
├── pinecone_vector_database/
│   ├── index_creator.py "(Creates Pinecone index)"
│   └── query.py "(Query Pinecone index)"
├── test/
│   ├── main_chat.py "(Generates response)"
│   └── main.py
├── .env "(Environment variables)"
├── pdf_metadata.json
└── requirements.txt "(Dependencies)"

```




## Process Flow
```mermaid
graph TD
    subgraph Input
        A[PDF Document] --> B[Document Processing]
        E[User Query] --> F[Chat Interface]
    end

    subgraph Document_Processing
        B --> C[Document Chunking]
        C --> D[Parse & Extract]
    end

    subgraph Database
        D --> I[Generate Unique ID]
    end

    subgraph Vectorstore
        D --> J[Create Pinecone Index]
        J --> K[(Pinecone DB)]
    end

    subgraph Response_Generation
        F --> L[Query Processing]
        L --> M[Vector Search]
        K --> M
        M --> N[Generate Response]
        N --> O[Token Calculation]

    end

    subgraph Output
        
            
        O --> P[Format Response]
        P --> Q[User Interface]
        Q --> R[Display to User]
        O --> S[Update Usage Stats]
        S -->  H[(MySQL DB)]
    end

```
# Document processing
``` sh
$env:PYTHONPATH = "C:\xampp\htdocs\ai-case" <- Root file path
cd api
fastapi dev .\main.py
```
## Open the localhost in browser
![Screenshot](upload_image.png)

# Chat

``` sh
$env:PYTHONPATH = "C:\xampp\htdocs\ai-case" <- Root file path
cd api 
fastapi dev .\main.py
```
## Use the index name provided at the time of pdf processing
![Screenshot](image.png)

