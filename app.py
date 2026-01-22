import os
import asyncio
import re
import json
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv

from config import get_settings
from utils import get_logger
from vectorstore import QdrantVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Union

from kpi_reference import KPI_REFERENCE_TEXT

# Initialize
logger = get_logger("TelicomApp")
settings = get_settings()

# Markets list
MARKET_LIST = [
    "ARKANSAS", "ATLANTA", "AUSTIN", "BIRMINGHAM", "CHICAGO", "CINCINNATI",
    "CLEVELAND", "COLUMBUS", "DALLAS", "DENVER", "DES MOINES", "DETROIT",
    "HOUSTON", "INDIANAPOLIS", "JACKSONVILLE", "KANSAS CITY", "KNOXVILLE",
    "LOUISVILLE", "MEMPHIS", "MIAMI", "MILWAUKEE", "MINNEAPOLIS", "MOBILE",
    "MONTANA", "NASHVILLE", "OKLAHOMA CITY", "OMAHA", "ORLANDO", "PHOENIX",
    "PITTSBURGH", "PORTLAND", "PUERTO RICO", "SEATTLE", "SPOKANE", "ST LOUIS",
    "TAMPA", "WEST VIRGINIA"
]

# Nokia Cell Naming Convention:
# cellName (11 chars): B9BH0003A21 = Prefix(1) + SectorName(10)
# Sector Name (10 chars): 9BH0003A21 = SITEID(8) + SectorDigits(2)
# SITEID (8 chars): 9BH0003A
NOKIA_CELL_PATTERN = re.compile(r'^[A-Z][A-Z0-9]{8}[0-9]{2}$')

# Fake cell patterns (city/airport codes)
FAKE_CELL_PATTERNS = [
    re.compile(r'^[A-Z]{3}\d{5}[A-Z]\d{2}$', re.IGNORECASE),
    re.compile(r'^(ATL|CHI|DAL|DET|HOU|IND|MIA|PHX|SEA|TAM|NYC|LAX|SFO|DEN|BOS|MSP|DTW|ORD|DFW|IAH|EWR|JFK|PHL|CLT|MCO|TPA|FLL|JAX|BNA|MEM|STL|MCI|OMA|OKC|AUS|SAT|SLC|PDX|SAN|LAS)', re.IGNORECASE),
]


# Pydantic Models
class TableData(BaseModel):
    title: Optional[str] = None
    headers: List[str]
    rows: List[List[str]]


class TabularResponse(BaseModel):
    response: str
    tables: Optional[List[TableData]] = None
    has_tables: bool = False


class TableRowItem(BaseModel):
    number: Union[int, str]
    name: str
    description: str
    formula_calculation: str
    relationship_impact: str

    @field_validator('number', mode='before')
    @classmethod
    def convert_number(cls, v):
        return int(v) if isinstance(v, str) and v.isdigit() else v


class HierarchyRow(BaseModel):
    site_id: str = Field(description="8-char SITEID (e.g., 9BH0003A)")
    sector_name: str = Field(description="10-char Sector Name (e.g., 9BH0003A21)")
    cell_name: str = Field(description="11-char cellName (e.g., B9BH0003A21)")


class SeparateTablesResponse(BaseModel):
    kpis: List[TableRowItem] = []
    parameters: List[TableRowItem] = []
    identifiers: Optional[List[TableRowItem]] = None
    hierarchy: Optional[List[HierarchyRow]] = None
    reasoning_kpis: str = "Not applicable"
    reasoning_parameters: str = "Not applicable"
    technical_details: str = "Not applicable"
    additional_context: str = "Not applicable"
    constraints_notes: str = "Not applicable"

    class Config:
        extra = 'allow'


def extract_site_from_cell(cell_name: str) -> tuple:
    """
    Extract SITEID and Sector Name from Nokia cellName.
    
    Example: B9BH0003A21
    - cellName: B9BH0003A21 (11 chars)
    - Sector Name: 9BH0003A21 (chars 2-11, 10 chars)
    - SITEID: 9BH0003A (first 8 chars of Sector Name)
    """
    if not cell_name or len(cell_name) < 10:
        return ("N/A", "N/A")
    
    cell_name = cell_name.strip().upper()
    
    if len(cell_name) == 11:
        sector_name = cell_name[1:]      # 9BH0003A21 (10 chars)
        site_id = sector_name[:8]        # 9BH0003A (8 chars)
        return (site_id, sector_name)
    elif len(cell_name) == 10:
        return (cell_name[:8], cell_name)
    elif len(cell_name) == 8:
        return (cell_name, "N/A")
    return (cell_name[:8] if len(cell_name) >= 8 else cell_name, "N/A")


def validate_nokia_cell(cell_name: str) -> bool:
    """Validate Nokia cell naming convention."""
    if not cell_name or len(cell_name) < 10:
        return False
    return bool(NOKIA_CELL_PATTERN.match(cell_name.upper()))


def is_fake_cell(cell_name: str) -> bool:
    """Check if cell ID matches fake/hallucinated patterns."""
    cell_upper = cell_name.upper().strip()
    for pattern in FAKE_CELL_PATTERNS:
        if pattern.match(cell_upper):
            return True
    # Check for market abbreviation patterns
    for market in MARKET_LIST:
        if cell_upper.startswith(market[:3]):
            rest = cell_upper[3:]
            if sum(1 for c in rest if c.isdigit()) >= len(rest) * 0.6:
                return True
    return False


def detect_hallucinated_cells(cells_data: list) -> tuple:
    """Filter out fake cell IDs, return (valid_cells, warning)."""
    if not cells_data:
        return ([], None)
    
    valid_cells, suspicious = [], []
    
    for cell in cells_data:
        cell_name = cell.get('cell_name', '') or cell.get('cell_id', '') or ''
        cell_upper = cell_name.upper().strip()
        
        if is_fake_cell(cell_name):
            suspicious.append(cell_name)
        elif len(cell_upper) == 11 and validate_nokia_cell(cell_upper):
            siteid = cell_upper[1:9]
            if not siteid.isdigit():  # Real SITEIDs have alphanumeric mix
                valid_cells.append(cell)
            else:
                suspicious.append(cell_name)
        else:
            suspicious.append(cell_name)
    
    warning = None
    if suspicious:
        if len(suspicious) == len(cells_data):
            warning = f"⚠️ WARNING: All {len(suspicious)} cell IDs appear fabricated (e.g., '{suspicious[0]}'). Real Nokia format: 'B9BH0003A21'."
        else:
            warning = f"⚠️ WARNING: {len(suspicious)} of {len(cells_data)} cell IDs may be fabricated."
    
    return (valid_cells, warning)


def clean_markdown(text: str) -> str:
    """Remove markdown formatting."""
    if not text:
        return text
    return re.sub(r'\*+|`|_+', '', text).strip()


def extract_json(text: str) -> dict:
    """Extract JSON from text/markdown."""
    # Try code block first
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if match:
        text = match.group(1)
    
    try:
        start = text.find('{')
        if start != -1:
            end = text.rfind('}') + 1
            return json.loads(text[start:end])
    except json.JSONDecodeError:
        pass
    return {}


# Flask App
app = Flask(__name__)
CORS(app)

vector_store, embeddings, llm, reranker = None, None, None, None


async def init_components():
    global vector_store, embeddings, llm, reranker
    
    logger.info("Initializing RAG components...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder="./model_cache"
    )
    
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    vector_store = QdrantVectorStore(
        url=settings.vectorstore.url,
        collection_name=settings.vectorstore.collection_name,
        test_connection=True
    )
    await vector_store.initialize()
    
    llm = ChatOpenAI(
        model="gpt-4o",
        openai_api_key=settings.api_key,
        openai_api_base=settings.nokia_llm_gateway_url,
        default_headers={"api-key": settings.api_key, "workspaceName": settings.workspace},
        temperature=0.0,
        max_tokens=4000
    )
    
    logger.info("Components initialized.")


loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
try:
    loop.run_until_complete(init_components())
except Exception as e:
    logger.error(f"Init failed: {e}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    query = data.get("message")
    
    if not query:
        return jsonify({"error": "No message provided"}), 400
    
    try:
        # Expand "all markets" queries
        if any(alias in query.lower() for alias in ["all market", "every market", "all markets"]):
            query = f"{query}. Include Cell IDs for markets: {', '.join(MARKET_LIST)}"
        
        # RAG retrieval
        q_vector = embeddings.embed_query(query)
        results = loop.run_until_complete(
            vector_store.search(
                collection_name=settings.vectorstore.collection_name,
                query_vector=q_vector,
                limit=50,
                score_threshold=0.55
            )
        )
        
        # Rerank
        search_results = []
        if results:
            pairs = [[query, res.payload.get("text", str(res.payload))] for res in results]
            scores = reranker.predict(pairs)
            scored = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)
            search_results = [r for _, r in scored[:20]]
        
        # Build context
        contexts = []
        for idx, res in enumerate(search_results, 1):
            content = res.payload.get("text", "")
            contexts.append(f"[Doc {idx}]\n{content}")
        formatted_context = "\n\n---\n\n".join(contexts)
        
        prompt = ChatPromptTemplate.from_template("""
You are a Telecom Expert. Extract data ONLY from context. DO NOT fabricate cell IDs.

**NOKIA NAMING CONVENTION:**
| SITEID | Sector Name | cellName |
|--------|-------------|----------|
| 9BH0003A | 9BH0003A21 | B9BH0003A21 |

- cellName (11 chars): Prefix + Sector Name (e.g., B9BH0003A21)
- Sector Name (10 chars): SITEID + Sector digits (e.g., 9BH0003A21)
- SITEID (8 chars): First 8 of Sector Name (e.g., 9BH0003A)

**CONTEXT:**
{context}

**QUESTION:** {input}
""")
        
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"context": formatted_context, "input": query})
        
        return jsonify({"response": response})
    
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/chat_tabular", methods=["POST"])
def chat_tabular():
    data = request.json
    query = data.get("message")
    
    if not query:
        return jsonify({"error": "No message provided"}), 400
    
    try:
        # Expand "all markets"
        if any(alias in query.lower() for alias in ["all market", "every market", "all markets"]):
            query = f"{query}. For markets: {', '.join(MARKET_LIST)}"
        
        query_lower = query.lower()
        enhanced_query = query
        
        # Detect multi-metric query
        metrics = ['throughput', 'ho', 'handover', 'congestion', 'spectral', 'rach']
        is_multi_metric = sum(1 for m in metrics if m in query_lower) >= 3 and \
                          any(k in query_lower for k in ['cell', 'top', 'identify'])
        
        if is_multi_metric:
            enhanced_query += ' cellName SITEID "Sector Name" cell_performance'
        
        # Hybrid search
        q_vector = embeddings.embed_query(enhanced_query)
        results = loop.run_until_complete(
            vector_store.hybrid_search_rrf(
                collection_name=settings.vectorstore.collection_name,
                query_vector=q_vector,
                query_text=enhanced_query,
                limit=100
            )
        )
        
        # Rerank
        search_results = []
        if results:
            pairs = [[query, res.payload.get("text", str(res.payload))] for res in results]
            scores = reranker.predict(pairs)
            scored = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)
            search_results = [r for _, r in scored[:25]]
        
        # Build context
        contexts = [f"[Doc {i}]\n{r.payload.get('text', '')}" for i, r in enumerate(search_results, 1)]
        formatted_context = "\n\n---\n\n".join(contexts)
        
        # Select prompt based on query type
        if is_multi_metric:
            prompt_template = """
You are a Telecom Data Expert. Extract REAL cell data from context only.

**CRITICAL: DO NOT FABRICATE DATA. DO NOT USE CITY CODES (ATL, CHI, etc.)**

**NOKIA NAMING - CORRECT FORMAT:**
| SITEID | Sector Name | cellName |
|--------|-------------|----------|
| 9BH0003A | 9BH0003A21 | B9BH0003A21 |

Derivation from cellName "B9BH0003A21":
- Remove prefix "B" → Sector Name = "9BH0003A21" (10 chars)
- First 8 chars of Sector Name → SITEID = "9BH0003A" (8 chars)

**RETURN JSON:**
```json
{{
  "cell_performance": [
    {{
      "rank": 1,
      "cell_name": "B9BH0003A21",
      "site_id": "9BH0003A",
      "sector_name": "9BH0003A21",
      "market": "MARKET_NAME",
      "dl_throughput": "value or N/A",
      "ho_success_rate": "value or N/A",
      "congestion": "value or N/A",
      "spectral_efficiency": "value or N/A",
      "rach_setup_sr": "value or N/A",
      "status": "Bad/Good",
      "notes": ""
    }}
  ],
  "data_found": true,
  "technical_details": "explanation",
  "constraints_notes": "limitations"
}}
```

If no data found, return empty cell_performance array with data_found: false.

**CONTEXT:**
{context}

**QUESTION:** {input}
"""
        elif all(x in query_lower for x in ["siteid", "sector", "cell"]) or \
             ("lte" in query_lower and "site" in query_lower):
            prompt_template = """
You are a Telecom Expert. Extract Site-Sector-Cell hierarchy from context.

**NOKIA NAMING CONVENTION - TABLE FORMAT:**
| SITEID | Sector Name | cellName |
|--------|-------------|----------|
| 9BH0003A | 9BH0003A21 | B9BH0003A21 |

**Derivation Rules:**
- cellName (11 chars): e.g., B9BH0003A21
- Sector Name (10 chars): cellName without prefix = 9BH0003A21
- SITEID (8 chars): First 8 of Sector Name = 9BH0003A

**RETURN JSON:**
```json
{{
  "hierarchy": [
    {{"site_id": "9BH0003A", "sector_name": "9BH0003A21", "cell_name": "B9BH0003A21"}}
  ],
  "technical_details": "explanation",
  "constraints_notes": "notes"
}}
```

**CONTEXT:**
{context}

**QUESTION:** {input}
"""
        else:
            prompt_template = """
You are a Telecom Expert. Extract KPIs and Parameters from context.

**NOKIA CELL HIERARCHY (for reference):**
| SITEID | Sector Name | cellName |
|--------|-------------|----------|
| 9BH0003A | 9BH0003A21 | B9BH0003A21 |

**RETURN JSON:**
```json
{{
  "kpis": [{{"number": 1, "name": "", "description": "", "formula_calculation": "", "relationship_impact": ""}}],
  "parameters": [{{"number": 1, "name": "", "description": "", "formula_calculation": "", "relationship_impact": ""}}],
  "hierarchy": [{{"site_id": "", "sector_name": "", "cell_name": ""}}],
  "reasoning_kpis": "",
  "reasoning_parameters": "",
  "technical_details": "",
  "constraints_notes": ""
}}
```

**CONTEXT:**
{context}

**QUESTION:** {input}
"""
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | llm | StrOutputParser()
        llm_response = chain.invoke({"context": formatted_context, "input": query})
        
        logger.info(f"LLM Response: {llm_response[:500]}")
        
        json_data = extract_json(llm_response)
        tables = []
        response_parts = []
        
        if is_multi_metric:
            cell_performance = json_data.get('cell_performance', [])
            
            if cell_performance:
                valid_cells, warning = detect_hallucinated_cells(cell_performance)
                
                if warning:
                    response_parts.append(f"\n{warning}\n")
                
                if valid_cells:
                    # Process valid cells with correct column order
                    processed = []
                    for cell in valid_cells:
                        cell_name = cell.get('cell_name', '')
                        site_id, sector_name = extract_site_from_cell(cell_name)
                        processed.append({
                            'site_id': site_id,
                            'sector_name': sector_name,
                            'cell_name': cell_name,
                            'market': cell.get('market', 'N/A'),
                            'dl_throughput': str(cell.get('dl_throughput', 'N/A')),
                            'ho_success_rate': str(cell.get('ho_success_rate', 'N/A')),
                            'congestion': str(cell.get('congestion', 'N/A')),
                            'spectral_efficiency': str(cell.get('spectral_efficiency', 'N/A')),
                            'rach_setup_sr': str(cell.get('rach_setup_sr', 'N/A')),
                            'status': cell.get('status', 'N/A'),
                        })
                    
                    # Table with correct column order: SITEID | Sector Name | cellName | metrics...
                    tables.append(TableData(
                        title="Cell Performance Analysis",
                        headers=["SITEID", "Sector Name", "cellName", "DL Tput", "HO Succ%", "Congestion%", "SE", "RACH SR%", "Status"],
                        rows=[[
                            c['site_id'], c['sector_name'], c['cell_name'],
                            c['dl_throughput'], c['ho_success_rate'], c['congestion'],
                            c['spectral_efficiency'], c['rach_setup_sr'], c['status']
                        ] for c in processed]
                    ))
                    response_parts.append(f"### Cell Performance\nFound {len(processed)} cells.\n")
                else:
                    response_parts.append("### No Valid Cell Data Found\n")
            else:
                response_parts.append("### No Cell Data in Context\n")
            
            if json_data.get('technical_details'):
                response_parts.append(f"\n### Technical Details\n{json_data['technical_details']}\n")
        
        else:
            # Standard query processing
            if 'kpis' not in json_data:
                json_data['kpis'] = []
            if 'parameters' not in json_data:
                json_data['parameters'] = []
            
            try:
                result = SeparateTablesResponse(**json_data)
            except:
                result = SeparateTablesResponse()
            
            # Hierarchy table with correct order: SITEID | Sector Name | cellName
            if result.hierarchy:
                hier_rows = []
                for row in result.hierarchy:
                    cell_name = clean_markdown(row.cell_name)
                    site_id, sector_name = extract_site_from_cell(cell_name)
                    # Use extracted values to ensure correctness
                    hier_rows.append([site_id, sector_name, cell_name])
                
                tables.append(TableData(
                    title="Site-Sector-Cell Hierarchy",
                    headers=["SITEID", "Sector Name", "cellName"],
                    rows=hier_rows
                ))
            
            if result.kpis:
                tables.append(TableData(
                    title="KPIs",
                    headers=["#", "Name", "Description", "Formula", "Impact"],
                    rows=[[str(k.number), clean_markdown(k.name), clean_markdown(k.description),
                           clean_markdown(k.formula_calculation), clean_markdown(k.relationship_impact)]
                          for k in result.kpis]
                ))
            
            if result.parameters:
                tables.append(TableData(
                    title="Parameters",
                    headers=["#", "Name", "Description", "Formula", "Impact"],
                    rows=[[str(p.number), clean_markdown(p.name), clean_markdown(p.description),
                           clean_markdown(p.formula_calculation), clean_markdown(p.relationship_impact)]
                          for p in result.parameters]
                ))
            
            if tables:
                response_parts.append("### Analysis Results\n")
            
            for field in ['reasoning_kpis', 'reasoning_parameters', 'technical_details']:
                val = getattr(result, field, '')
                if val and len(val) > 15 and 'N/A' not in val and 'Not applicable' not in val:
                    response_parts.append(f"\n**{field.replace('_', ' ').title()}:**\n{val}\n")
        
        response_parts.append(f"\n*Source: {settings.vectorstore.collection_name}*")
        
        return jsonify(TabularResponse(
            response=''.join(response_parts),
            tables=tables,
            has_tables=len(tables) > 0
        ).model_dump())
    
    except Exception as e:
        logger.error(f"chat_tabular error: {e}", exc_info=True)
        return jsonify(TabularResponse(response=f"Error: {e}", tables=None, has_tables=False).model_dump()), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
