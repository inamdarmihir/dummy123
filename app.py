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

# Initialize Logger
logger = get_logger("TelicomApp")

from kpi_reference import KPI_REFERENCE_TEXT

# Load settings
settings = get_settings()

# List of all markets for "all markets" queries
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

# Fake cell patterns (city/airport codes) - for hallucination detection
FAKE_CELL_PATTERNS = [
    re.compile(r'^[A-Z]{3}\d{5}[A-Z]\d{2}$', re.IGNORECASE),
    re.compile(r'^(ATL|CHI|DAL|DET|HOU|IND|MIA|PHX|SEA|TAM|NYC|LAX|SFO|DEN|BOS|MSP|DTW|ORD|DFW|IAH|EWR|JFK|PHL|CLT|MCO|TPA|FLL|JAX|BNA|MEM|STL|MCI|OMA|OKC|AUS|SAT|SLC|PDX|SAN|LAS)', re.IGNORECASE),
]

# Pydantic models
class TableRow(BaseModel):
    number: int = Field(description="Row number starting from 1")
    name: str = Field(description="Exact name from context")
    description: str = Field(description="Brief description (1 sentence, max 15 words)")
    relationship: str = Field(description="Brief impact/relationship (1 sentence, max 15 words)")

class ReasoningItem(BaseModel):
    label: str = Field(description="Label like 'Parameter1', 'KPI1', etc.")
    name: str = Field(description="Actual parameter or KPI name")
    explanation: str = Field(description="Detailed explanation (1-2 sentences)")

class StructuredResponse(BaseModel):
    topic: str = Field(description="The topic being queried")
    parameters: Optional[List[TableRow]] = None
    kpis: Optional[List[TableRow]] = None
    parameter_reasoning: Optional[List[ReasoningItem]] = None
    kpi_reasoning: Optional[List[ReasoningItem]] = None
    constraints_notes: Optional[str] = None

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
    counters: Optional[List[TableRowItem]] = None
    features: Optional[List[TableRowItem]] = None
    alarms: Optional[List[TableRowItem]] = None
    reasoning_kpis: str = "Not applicable"
    reasoning_parameters: str = "Not applicable"
    technical_details: str = "Not applicable"
    additional_context: str = "Not applicable"
    constraints_notes: str = "Not applicable"

    class Config:
        extra = 'allow'


# Utility Functions
def extract_site_from_cell(cell_name: str) -> tuple:
    """
    Extract SITEID and Sector Name from Nokia cellName.
    
    Nokia Naming Convention:
    - cellName (11 chars): B9BH0003A21
    - Sector Name (10 chars): 9BH0003A21 (chars 2-11, cellName without prefix)
    - SITEID (8 chars): 9BH0003A (first 8 chars of Sector Name)
    
    Example: B9BH0003A21 -> SITEID=9BH0003A, Sector Name=9BH0003A21
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
    """Check if cell ID matches fake/hallucinated patterns (city/airport codes)."""
    cell_upper = cell_name.upper().strip()
    for pattern in FAKE_CELL_PATTERNS:
        if pattern.match(cell_upper):
            return True
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
            if not siteid.isdigit():
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
    """Remove markdown formatting from text."""
    if not text:
        return text
    text = text.replace('**', '').replace('__', '')
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = text.replace('`', '')
    return text.strip()


def extract_json(text: str) -> dict:
    """Extract JSON from text/markdown code blocks."""
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

vector_store = None
embeddings = None
llm = None
llm_structured = None
reranker = None


async def init_components():
    global vector_store, embeddings, llm, llm_structured, reranker
    
    logger.info("Initializing RAG components...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder="./model_cache"
    )
    
    logger.info("Loading Reranker Model...")
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    vector_store = QdrantVectorStore(
        url=settings.vectorstore.url,
        collection_name=settings.vectorstore.collection_name,
        test_connection=True
    )
    await vector_store.initialize()
    
    headers = {"api-key": settings.api_key, "workspaceName": settings.workspace}
    
    llm = ChatOpenAI(
        model="gpt-4o",
        openai_api_key=settings.api_key,
        openai_api_base=settings.nokia_llm_gateway_url,
        default_headers=headers,
        temperature=0.1,
        max_tokens=4000
    )
    
    llm_structured = llm.with_structured_output(StructuredResponse)
    logger.info("Components initialized successfully.")


loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
try:
    loop.run_until_complete(init_components())
except Exception as e:
    logger.error(f"Failed to initialize components: {e}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    query = data.get("message")
    use_structured = data.get("structured", False)
    
    if not query:
        return jsonify({"error": "No message provided"}), 400
    
    try:
        # Expand "all markets" queries
        if any(alias in query.lower() for alias in ["all market", "all the market", "every market", "all markets"]):
            query = f"{query}. Ensure you include Cell IDs and Sites for the following markets: {', '.join(MARKET_LIST)}"
        
        query_lower = query.lower()
        
        # Detect question type
        question_type = "general"
        if "compare" in query_lower or "vs" in query_lower or "versus" in query_lower:
            question_type = "comparison"
        elif "top" in query_lower and ("cell" in query_lower or "site" in query_lower):
            question_type = "ranking"
        elif "identify" in query_lower or "list" in query_lower or "pull" in query_lower:
            question_type = "extraction"
        
        # Extract requested types
        requested_types = []
        for kw, typ in [("parameter", "parameters"), ("kpi", "KPIs"), ("counter", "counters"), 
                        ("feature", "features"), ("alarm", "alarms")]:
            if kw in query_lower:
                requested_types.append(typ)
        if "cell" in query_lower and question_type == "ranking":
            requested_types.append("cells")
        
        # Extract metric focus
        metric_focus = ""
        metric_map = {
            "dl throughput": "DL Throughput", "downlink throughput": "DL Throughput",
            "spectral efficiency": "Spectral Efficiency", "voice dcr": "Voice DCR",
            "drop call rate": "Voice DCR", "latency": "Latency", "prb": "PRB Utilization"
        }
        for key, val in metric_map.items():
            if key in query_lower:
                metric_focus = val
                break
        
        logger.info(f"Query analysis - Type: {question_type}, Requested: {requested_types}, Metric: {metric_focus}")
        
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
            pairs = [[query, res.payload.get("text") or res.payload.get("content") or str(res.payload)] for res in results]
            scores = reranker.predict(pairs)
            scored_results = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)
            search_results = [res for _, res in scored_results[:20]]
            logger.info(f"RAG Retrieval: Selected {len(search_results)} results after reranking")
        
        # Format context
        contexts = []
        for idx, res in enumerate(search_results, 1):
            content = res.payload.get("text") or res.payload.get("content") or ""
            metadata = {k.capitalize(): res.payload[k] for k in ["type", "category", "source"] if k in res.payload}
            context_block = f"[Document {idx}]"
            if metadata:
                context_block += "\n" + " | ".join([f"{k}: {v}" for k, v in metadata.items()])
            context_block += f"\n{content}"
            contexts.append(context_block)
        
        formatted_context = "\n\n---\n\n".join(contexts)
        
        # System instruction with comprehensive formatting rules
        system_instruction = f"""
You are a Telecom Performance Expert specializing in LTE/5G NR (NSA & SA).

**NOKIA CELL NAMING CONVENTION (CRITICAL - USE THIS EXACT FORMAT):**

| SITEID | Sector Name | cellName |
|--------|-------------|----------|
| 9BH0003A | 9BH0003A21 | B9BH0003A21 |

- **cellName** (11 chars): Prefix + Sector Name (e.g., B9BH0003A21)
- **Sector Name** (10 chars): SITEID + Sector digits (e.g., 9BH0003A21)  
- **SITEID** (8 chars): First 8 chars of Sector Name (e.g., 9BH0003A)

**Prefix Decoders:**
- A: LTE Primary Carrier / Mid Band
- B: LTE 700MHz / Low Band
- J: LTE Secondary / High Band
- K: LAA / Additional Capacity
- D/L/E/Z: Technology/Band specific

**REFERENCE FORMULAS:**
1. **Combined RACH setup SR**: `100 * (RACH completions contention + RACH completions dedicated) / (RA att contention + RA att dedicated)`
2. **Handover Success Ratio**: `(intra_du_ho_att - intra_du_ho_failures) / (intra_du_ho_att + prep_failures)`
3. **Voice DCR**: `DCR = (Dropped Calls / Total Call Attempts) * 100`

**FOR CELL RANKING QUERIES:**
| Rank | cellName | SITEID | Sector Name | Metric Value | Status | Notes |

**FOR MULTI-METRIC PERFORMANCE (DL Tput, HO, Congestion, SE, RACH):**
| Rank | cellName | SITEID | Sector Name | DL Tput | HO Succ% | Congestion% | SE | RACH SR% | Status |

- **Good Cell**: High DL Tput, High HO Succ, Low Congestion, High SE, High RACH
- **Bad Cell**: Low DL Tput, Low HO Succ, High Congestion, Low SE, Low RACH

**FOR DOCUMENTATION QUERIES (Parameters/KPIs):**

| # | Name | Description | Formula/Calculation | Relationship/Impact |
|---|------|-------------|---------------------|---------------------|

**TYPE CLASSIFICATIONS:**
- PARAMETER: Configuration values, input variables
- KPI: Calculated performance metrics
- COUNTER: Raw measurements
- FEATURE: Network features (CA, MIMO)
- ALARM: Alerts/Alarms

**CRITICAL RULES:**
1. Extract ONLY from context - DO NOT fabricate cell IDs
2. Use EXACT names from context
3. NO markdown formatting (**, __, *) in table cells
4. Create SEPARATE tables for each type
5. Column order for hierarchy: SITEID | Sector Name | cellName

**Source:** Collection: {settings.vectorstore.collection_name}

**REFERENCE KNOWLEDGE:**
{KPI_REFERENCE_TEXT}
"""

        prompt = ChatPromptTemplate.from_template("""
{system_instruction}

**QUERY ANALYSIS:**
- Question Type: {question_type}
- Requested Items: {requested_types}
- Metric Focus: {metric_focus}

**CONTEXT:**
{context}

**USER QUESTION:** {input}

**YOUR RESPONSE:**
""")
        
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({
            "system_instruction": system_instruction,
            "question_type": question_type,
            "requested_types": ", ".join(requested_types) if requested_types else "All relevant items",
            "metric_focus": metric_focus if metric_focus else "General telecom metrics",
            "context": formatted_context,
            "input": query
        })
        
        # Clean markdown from table cells
        lines = response.split('\n')
        cleaned_lines = []
        for line in lines:
            if '|' in line and not line.strip().startswith('**'):
                line = re.sub(r'\*\*([^*]+)\*\*', r'\1', line)
                line = re.sub(r'__([^_]+)__', r'\1', line)
            cleaned_lines.append(line)
        response = '\n'.join(cleaned_lines)
        
        return jsonify({"response": response})
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        return jsonify({
            "error": "An error occurred",
            "message": str(e),
            "response": "Error processing query. Please try again."
        }), 500


@app.route("/chat_tabular", methods=["POST"])
def chat_tabular():
    """Enhanced endpoint returning structured JSON with tables."""
    data = request.json
    query = data.get("message")
    
    if not query:
        return jsonify({"error": "No message provided"}), 400
    
    try:
        # Expand "all markets"
        if any(alias in query.lower() for alias in ["all market", "all the market", "every market", "all markets"]):
            query = f"{query}. Specifically for markets: {', '.join(MARKET_LIST)}"
        
        query_lower = query.lower()
        enhanced_query = query
        
        # Enhanced query for Voice DCR
        if 'dcr' in query_lower or 'dropped call' in query_lower or 'voice' in query_lower:
            enhanced_query += ' "Dropped Calls" "Total Call Attempts" "Handover Success Rate" "Congestion Rate" cellName SITEID'
        
        # Enhanced query for bad/poor performance
        if any(term in query_lower for term in ['bad', 'poor', 'low', 'high', 'worst', 'degraded']):
            if 'throughput' in query_lower:
                enhanced_query += ' "PDCP_SDU_VOL_DL" "PRB_USED_PDSCH" "User Throughput"'
            if 'ho' in query_lower or 'handover' in query_lower:
                enhanced_query += ' "Handover Success Rate" "HO Failure"'
            if 'congestion' in query_lower:
                enhanced_query += ' "Congestion Rate" "RRC Congestion"'
            if 'spectral' in query_lower or 'se' in query_lower:
                enhanced_query += ' "Spectral Efficiency" "NR_5108e"'
            if 'rach' in query_lower:
                enhanced_query += ' "RACH Setup Success Rate" "Preamble"'
        
        # Extract cell IDs for exact matching
        id_pattern = re.compile(r'[A-Z0-9]{8,20}')
        id_matches = id_pattern.findall(query.upper())
        for match in id_matches:
            if len(match) >= 8:
                enhanced_query += f' "{match}"'
                if len(match) >= 11:
                    enhanced_query += f' "{match[1:9]}"'  # SITEID
        
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
        
        logger.info(f"Retrieved {len(results)} documents")
        
        # Rerank
        search_results = []
        if results:
            pairs = [[query, res.payload.get("text") or str(res.payload)] for res in results]
            scores = reranker.predict(pairs)
            scored_results = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)
            search_results = [res for _, res in scored_results[:20]]
            logger.info(f"Top reranking scores: {[float(s) for s, _ in scored_results[:5]]}")
        
        # Format context
        contexts = []
        for idx, res in enumerate(search_results, 1):
            content = res.payload.get("text") or res.payload.get("content") or ""
            metadata = {k.capitalize(): res.payload[k] for k in ["type", "category", "source"] if k in res.payload}
            context_block = f"[Document {idx}]"
            if metadata:
                context_block += "\n" + " | ".join([f"{k}: {v}" for k, v in metadata.items()])
            context_block += f"\n{content}"
            contexts.append(context_block)
        
        formatted_context = "\n\n---\n\n".join(contexts)
        logger.info(f"Context length: {len(formatted_context)} chars")
        
        # Detect query type
        is_cell_query = any(w in query_lower for w in ['cell', 'site', 'enodeb', 'gnodeb', 'top', 'worst', 'best'])
        is_multi_metric = is_cell_query and all(k in query_lower for k in ['throughput', 'ho', 'congestion'])
        is_hierarchy_query = (all(x in query_lower for x in ["siteid", "sector", "cell"])) or \
                            ("lte" in query_lower and "siteid" in query_lower)
        is_definition_query = "what is" in query_lower and ("kpi" in query_lower or "parameter" in query_lower) and len(query.split()) < 10
        
        # Select appropriate prompt
        if is_multi_metric:
            prompt_template = """
You are a Telecom Performance Expert. Analyze cells based on: DL Throughput, HO Success, Congestion, Spectral Efficiency, RACH Setup SR.

**CRITICAL: DO NOT FABRICATE CELL IDs. Extract ONLY from context.**

**NOKIA NAMING CONVENTION - CORRECT TABLE FORMAT:**
| SITEID | Sector Name | cellName |
|--------|-------------|----------|
| 9BH0003A | 9BH0003A21 | B9BH0003A21 |

Derivation from cellName "B9BH0003A21":
- Remove prefix "B" → Sector Name = "9BH0003A21" (10 chars)
- First 8 of Sector Name → SITEID = "9BH0003A" (8 chars)

Return JSON:
```json
{{
  "cell_performance": [
    {{"rank": 1, "cell_name": "B9BH0003A21", "site_id": "9BH0003A", "sector_name": "9BH0003A21", 
      "market": "MARKET", "dl_throughput": "val", "ho_success_rate": "val", "congestion": "val",
      "spectral_efficiency": "val", "rach_setup_sr": "val", "status": "Good/Bad", "notes": ""}}
  ],
  "technical_details": "Analysis explanation",
  "constraints_notes": "Limitations"
}}
```

**Performance Criteria:**
- Good: High DL Tput (>100Mbps), High HO (>98%), Low Congestion (<20%), High SE (>5), High RACH (>99%)
- Bad: Low DL Tput (<50Mbps), Low HO (<95%), High Congestion (>80%), Low SE (<2), Low RACH (<95%)

CONTEXT: {context}
QUESTION: {input}
"""
        elif is_hierarchy_query:
            prompt_template = """
You are a Telecom Expert. Extract Site-Sector-Cell hierarchy from context.

**NOKIA NAMING CONVENTION - CORRECT TABLE FORMAT:**
| SITEID | Sector Name | cellName |
|--------|-------------|----------|
| 9BH0003A | 9BH0003A21 | B9BH0003A21 |

**STRICT DERIVATION RULES:**
- cellName (11 chars): e.g., BDA01442D41
- Sector Name: EXACT last 10 chars of cellName (e.g., DA01442D41)
- SITEID: EXACT first 8 chars of Sector Name (e.g., DA01442D)
- NEVER include prefix (A/B/J/K) in SITEID

**Prefix Decoders:**
- A: LTE Primary / Mid Band
- B: LTE 700MHz / Low Band
- J: LTE Secondary / High Band
- K: LAA / Additional Capacity

Return JSON:
```json
{{
  "hierarchy": [
    {{"site_id": "9BH0003A", "sector_name": "9BH0003A21", "cell_name": "B9BH0003A21"}}
  ],
  "reasoning_parameters": "Layer analysis",
  "technical_details": "Technology summary"
}}
```

CONTEXT: {context}
QUESTION: {input}
"""
        elif is_definition_query:
            prompt_template = """
You are a Telecom Expert. Define the concept clearly.

Return JSON:
```json
{{
  "kpis": [],
  "parameters": [],
  "reasoning_kpis": "DEFINITION: [What a KPI is in telecom - monitoring, optimization, benchmarking]",
  "reasoning_parameters": "DEFINITION: [What a Parameter is - configuration vs measurement]",
  "technical_details": "KPIs are calculated using counters and influenced by parameters.",
  "additional_context": "For specific KPIs, please specify the topic (e.g., '5G KPIs').",
  "constraints_notes": "N/A"
}}
```

QUESTION: {input}
"""
        elif is_cell_query and ('top' in query_lower or 'best' in query_lower or 'worst' in query_lower):
            prompt_template = """
You are a Telecom Expert. Extract cell performance data from context.

**NOKIA NAMING - TABLE FORMAT:**
| SITEID | Sector Name | cellName | Metric | Status |
|--------|-------------|----------|--------|--------|
| 9BH0003A | 9BH0003A21 | B9BH0003A21 | Value | Good/Bad |

**VOICE DCR ANALYSIS:**
- Formula: DCR = (Dropped Calls / Total Call Attempts) * 100
- Good: LOW DCR (ascending order)
- Include: HO Success, Congestion impact

Return JSON with kpis array (for cells) or parameters array:
```json
{{
  "kpis": [{{"number": 1, "name": "B9BH0003A21", "description": "DCR: X%", 
             "formula_calculation": "HO: Y%, Congestion: Z%", "relationship_impact": "Analysis"}}],
  "reasoning_kpis": "DCR formula and cell analysis",
  "technical_details": "DCR = (Dropped Calls / Total Attempts) * 100",
  "constraints_notes": "Data limitations"
}}
```

CONTEXT: {context}
QUESTION: {input}
"""
        else:
            prompt_template = """
You are a Telecom Expert. Extract KPIs and Parameters from context.

**NOKIA NAMING CONVENTION:**
| SITEID | Sector Name | cellName |
|--------|-------------|----------|
| 9BH0003A | 9BH0003A21 | B9BH0003A21 |

**CLASSIFICATION:**
- IDENTIFIERS: Cell Name, Cell ID, Site ID, ECGI, gNB, eNB, TAC, PCI → put in `identifiers`
- KPIs: Calculated metrics (Success Rates, Throughput) → put in `kpis`
- PARAMETERS: Configuration values → put in `parameters`

**FORMULAS:**
1. RACH SR: `100 * (RACH completions) / (RA attempts)`
2. HO Success: `(HO attempts - HO failures) / (HO attempts + prep failures)`
3. Voice DCR: `(Dropped Calls / Total Attempts) * 100`

Return JSON:
```json
{{
  "kpis": [{{"number": 1, "name": "KPI_NAME", "description": "Brief", "formula_calculation": "Formula", "relationship_impact": "Impact"}}],
  "parameters": [{{"number": 1, "name": "PARAM_NAME", "description": "Brief", "formula_calculation": "N/A", "relationship_impact": "Impact"}}],
  "identifiers": [{{"number": 1, "name": "Site ID", "description": "Physical Site", "formula_calculation": "9BH0003A", "relationship_impact": "Site Level"}}],
  "hierarchy": [{{"site_id": "9BH0003A", "sector_name": "9BH0003A21", "cell_name": "B9BH0003A21"}}],
  "reasoning_kpis": "KPI explanations",
  "reasoning_parameters": "Parameter explanations",
  "technical_details": "Technical context",
  "additional_context": "Additional notes",
  "constraints_notes": "Limitations"
}}
```

CONTEXT: {context}
QUESTION: {input}

RULES:
1. Extract ONLY from context - DO NOT fabricate
2. Use exact names from context
3. Put identifiers (Cell Name, Site ID) in `identifiers`, NOT `parameters`
"""
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | llm | StrOutputParser()
        
        llm_response = chain.invoke({"context": formatted_context, "input": query})
        logger.info(f"LLM Response (first 300 chars): {llm_response[:300]}")
        
        # Parse JSON
        json_data = extract_json(llm_response)
        
        # Also try extracting from multiple code blocks
        if not json_data:
            code_blocks = re.findall(r'```(?:json)?\s*\n?(.*?)\n?```', llm_response, re.DOTALL)
            for block in code_blocks:
                try:
                    json_data = json.loads(block.strip())
                    break
                except:
                    continue
        
        # Ensure lists exist
        if 'kpis' not in json_data:
            json_data['kpis'] = []
        if 'parameters' not in json_data:
            json_data['parameters'] = []
        
        tables = []
        response_parts = []
        
        # Handle multi-metric query
        if is_multi_metric and 'cell_performance' in json_data:
            cell_performance = json_data.get('cell_performance', [])
            
            if cell_performance:
                valid_cells, warning = detect_hallucinated_cells(cell_performance)
                
                if warning:
                    response_parts.append(f"\n{warning}\n")
                
                if valid_cells:
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
                    
                    # Correct column order: SITEID | Sector Name | cellName
                    tables.append(TableData(
                        title="Cell Performance Analysis",
                        headers=["SITEID", "Sector Name", "cellName", "DL Tput", "HO%", "Cong%", "SE", "RACH%", "Status"],
                        rows=[[c['site_id'], c['sector_name'], c['cell_name'], c['dl_throughput'],
                               c['ho_success_rate'], c['congestion'], c['spectral_efficiency'],
                               c['rach_setup_sr'], c['status']] for c in processed]
                    ))
                    response_parts.append(f"### Cell Performance\nFound {len(processed)} cells.\n")
        
        # Standard processing
        try:
            structured_result = SeparateTablesResponse(**json_data)
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            structured_result = SeparateTablesResponse(kpis=[], parameters=[], constraints_notes=str(e))
        
        # Move identifiers from parameters if misclassified
        identifier_keywords = ["Cell Name", "Cell ID", "Site ID", "ECGI", "gNB", "eNB", "TAC", "PCI", "Sector ID", "Sector Name", "Market"]
        
        if structured_result.parameters:
            if structured_result.identifiers is None:
                structured_result.identifiers = []
            
            items_to_move = []
            for i in range(len(structured_result.parameters) - 1, -1, -1):
                param = structured_result.parameters[i]
                clean_name = param.name.lower().replace('*', '').replace('_', '').strip()
                
                for kw in identifier_keywords:
                    kw_clean = kw.lower().replace(' ', '')
                    name_clean = clean_name.replace(' ', '')
                    if kw_clean == name_clean or kw_clean in name_clean:
                        if "count" not in clean_name and "rate" not in clean_name:
                            items_to_move.append(structured_result.parameters.pop(i))
                            break
            
            for item in reversed(items_to_move):
                if not item.relationship_impact or item.relationship_impact == "Parameter":
                    item.relationship_impact = "Identifier"
                structured_result.identifiers.append(item)
        
        # Create Hierarchy table - CORRECT ORDER: SITEID | Sector Name | cellName
        if structured_result.hierarchy:
            hier_rows = []
            for row in structured_result.hierarchy:
                cell_name = clean_markdown(row.cell_name)
                site_id, sector_name = extract_site_from_cell(cell_name)
                hier_rows.append([site_id, sector_name, cell_name])
            
            tables.insert(0, TableData(
                title="Site-Sector-Cell Hierarchy",
                headers=["SITEID", "Sector Name", "cellName"],
                rows=hier_rows
            ))
        
        # Create KPIs table
        if structured_result.kpis:
            tables.append(TableData(
                title="KPIs Related to Query",
                headers=["#", "Name", "Description", "Formula/Calculation", "Relationship/Impact"],
                rows=[[str(k.number), clean_markdown(k.name), clean_markdown(k.description),
                       clean_markdown(k.formula_calculation), clean_markdown(k.relationship_impact)]
                      for k in structured_result.kpis]
            ))
        
        # Create Parameters table
        if structured_result.parameters:
            tables.append(TableData(
                title="Parameters Related to Query",
                headers=["#", "Name", "Description", "Formula/Calculation", "Relationship/Impact"],
                rows=[[str(p.number), clean_markdown(p.name), clean_markdown(p.description),
                       clean_markdown(p.formula_calculation), clean_markdown(p.relationship_impact)]
                      for p in structured_result.parameters]
            ))
        
        # Create Identifiers table
        if structured_result.identifiers:
            tables.append(TableData(
                title="Identifiers / Network Hierarchy",
                headers=["#", "Identifier Name", "Value", "Description", "Level/Notes"],
                rows=[[str(i.number), clean_markdown(i.name), clean_markdown(i.formula_calculation),
                       clean_markdown(i.description), clean_markdown(i.relationship_impact)]
                      for i in structured_result.identifiers]
            ))
        
        # Create Counters table
        if structured_result.counters:
            tables.append(TableData(
                title="Counters Related to Query",
                headers=["#", "Name", "Description", "Formula/Calculation", "Relationship/Impact"],
                rows=[[str(c.number), clean_markdown(c.name), clean_markdown(c.description),
                       clean_markdown(c.formula_calculation), clean_markdown(c.relationship_impact)]
                      for c in structured_result.counters]
            ))
        
        # Build response text
        if tables:
            response_parts.append("Based on the retrieved context, here is the analysis:\n")
        
        def is_meaningful(text):
            return text and len(str(text)) > 10 and "N/A" not in str(text) and "Not applicable" not in str(text)
        
        if is_meaningful(structured_result.reasoning_kpis) or is_meaningful(structured_result.reasoning_parameters):
            response_parts.append("\n---\n### Reasoning and Relationships\n")
            if is_meaningful(structured_result.reasoning_kpis):
                response_parts.append(f"\n**KPIs**:\n{structured_result.reasoning_kpis}\n")
            if is_meaningful(structured_result.reasoning_parameters):
                response_parts.append(f"\n**Parameters**:\n{structured_result.reasoning_parameters}\n")
        
        if is_meaningful(structured_result.technical_details):
            response_parts.append(f"\n---\n### Technical Details\n{structured_result.technical_details}\n")
        
        if is_meaningful(structured_result.additional_context):
            response_parts.append(f"\n---\n### Additional Context\n{structured_result.additional_context}\n")
        
        if is_meaningful(structured_result.constraints_notes):
            response_parts.append(f"\n---\n### Constraints\n{structured_result.constraints_notes}\n")
        
        response_parts.append(f"\n---\n*Source: {settings.vectorstore.collection_name} | Search: Hybrid/Cosine*")
        
        response_text = ''.join(response_parts)
        
        logger.info(f"Final response: {len(tables)} tables, {len(response_text)} chars")
        
        return jsonify(TabularResponse(
            response=response_text,
            tables=tables,
            has_tables=len(tables) > 0
        ).model_dump())
        
    except Exception as e:
        import traceback
        logger.error(f"Error in chat_tabular: {e}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return jsonify(TabularResponse(response=f"Error: {e}", tables=None, has_tables=False).model_dump()), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
