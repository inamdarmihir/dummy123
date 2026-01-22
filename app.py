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

# Nokia Cell Name Pattern Regex - for validation
# Format: 1 letter prefix + 8 char SITEID + 2 digit sector = 11 chars total
# Valid example: B9BH0003A21 where B=prefix, 9BH0003A=SITEID, 21=sector
NOKIA_CELL_PATTERN = re.compile(r'^[A-Z][A-Z0-9]{8}[0-9]{2}$')
NOKIA_SITEID_PATTERN = re.compile(r'^[A-Z0-9]{8}$')

# Patterns that indicate FAKE/hallucinated cell IDs (city/airport codes)
FAKE_CELL_PATTERNS = [
    # 3-letter city/airport codes followed by numbers (ATL00123A11, CHI04567B21)
    re.compile(r'^[A-Z]{3}\d{5}[A-Z]\d{2}$', re.IGNORECASE),
    # Common US city/airport codes at start
    re.compile(r'^(ATL|CHI|DAL|DET|HOU|IND|MIA|PHX|SEA|TAM|NYC|LAX|SFO|DEN|BOS|MSP|DTW|ORD|DFW|IAH|EWR|JFK|LGA|PHL|CLT|MCO|TPA|FLL|RSW|JAX|BNA|MEM|STL|MCI|OMA|OKC|AUS|SAT|ABQ|SLC|PDX|SAN|SMF|OAK|SJC|LAS|RNO|BOI|GEG|ANC|HNL)', re.IGNORECASE),
    # Market name abbreviations (ATLANTA -> ATL pattern)
    re.compile(r'^(ATLANTA|CHICAGO|DALLAS|DETROIT|HOUSTON|INDIANAPOLIS|MIAMI|PHOENIX|SEATTLE|TAMPA|DENVER|BOSTON|AUSTIN|ORLANDO|MEMPHIS|NASHVILLE|PORTLAND|CLEVELAND|COLUMBUS|PITTSBURGH|MILWAUKEE|MINNEAPOLIS|JACKSONVILLE|BIRMINGHAM|LOUISVILLE|KNOXVILLE|MOBILE|MONTANA|OMAHA|SPOKANE)', re.IGNORECASE),
]


# Pydantic models for structured output
class TableRow(BaseModel):
    """Represents a single row in the parameters/KPIs table"""
    number: int = Field(description="Row number starting from 1")
    name: str = Field(description="Exact name from context (preserve capitalization, underscores)")
    description: str = Field(description="Brief description (1 sentence, max 15 words)")
    relationship: str = Field(description="Brief impact/relationship (1 sentence, max 15 words)")


class ReasoningItem(BaseModel):
    """Represents reasoning for a parameter or KPI"""
    label: str = Field(description="Label like 'Parameter1', 'KPI1', etc.")
    name: str = Field(description="Actual parameter or KPI name")
    explanation: str = Field(description="Detailed explanation (1-2 sentences)")


class StructuredResponse(BaseModel):
    """Complete structured response with separate tables and reasoning"""
    topic: str = Field(description="The topic being queried (e.g., 'DL Spectral Efficiency')")
    parameters: Optional[List[TableRow]] = Field(default=None, description="List of parameters in table format")
    kpis: Optional[List[TableRow]] = Field(default=None, description="List of KPIs in table format")
    parameter_reasoning: Optional[List[ReasoningItem]] = Field(default=None, description="Reasoning for parameters")
    kpi_reasoning: Optional[List[ReasoningItem]] = Field(default=None, description="Reasoning for KPIs")
    constraints_notes: Optional[str] = Field(default=None, description="Any constraints or additional notes")


# Pydantic models for tabular endpoint
class TableData(BaseModel):
    """Represents a parsed markdown table"""
    title: Optional[str] = Field(default=None, description="Title/label for the table (e.g., 'KPIs', 'Parameters')")
    headers: List[str] = Field(description="Table column headers")
    rows: List[List[str]] = Field(description="Table rows, each row is a list of cell values")


class TabularResponse(BaseModel):
    """Response model for tabular endpoint with separated tables and text"""
    response: str = Field(description="Text content without tables")
    tables: Optional[List[TableData]] = Field(default=None, description="Parsed tables from the response")
    has_tables: bool = Field(default=False, description="Whether tables are present in the response")


# NEW: Structured models for separate tables
class TableRowItem(BaseModel):
    """Single row in a table"""
    number: Union[int, str] = Field(description="Row number (1, 2, 3...) or item identifier")
    name: str = Field(description="Name of the item (without markdown formatting)")
    description: str = Field(description="Brief description (plain text, no markdown)")
    formula_calculation: str = Field(description="Formula or N/A for parameters (plain text)")
    relationship_impact: str = Field(description="Relationship/Impact (plain text)")

    @field_validator('number', mode='before')
    @classmethod
    def convert_number(cls, v):
        """Convert number to int if it's a string digit, otherwise keep as string"""
        if isinstance(v, str):
            if v.isdigit():
                return int(v)
            return v
        return v


class HierarchyRow(BaseModel):
    """Specific row for Site-Sector-Cell hierarchy"""
    site_id: str = Field(description="8-char SITEID (e.g. 9BH0003A)")
    sector_name: str = Field(description="10-char Sector Name (e.g. 9BH0003A11)")
    cell_name: str = Field(description="Prefix + ID Cell Name (e.g. A9BH0003A11)")


# NEW: Multi-metric cell performance row
class CellPerformanceRow(BaseModel):
    """Row for multi-metric cell performance analysis"""
    rank: int = Field(description="Ranking position (1 = worst/best depending on query)")
    cell_name: str = Field(description="Full 11-char cell name (e.g., B9BH0003A21)")
    site_id: str = Field(description="8-char SITEID derived from cell name")
    sector_name: str = Field(description="10-char Sector Name")
    market: Optional[str] = Field(default=None, description="Market name if available")
    dl_throughput: Optional[str] = Field(default="N/A", description="DL Throughput value")
    ho_success_rate: Optional[str] = Field(default="N/A", description="Handover Success Rate")
    congestion: Optional[str] = Field(default="N/A", description="Congestion percentage")
    spectral_efficiency: Optional[str] = Field(default="N/A", description="Spectral Efficiency")
    rach_setup_sr: Optional[str] = Field(default="N/A", description="RACH Setup Success Rate")
    status: str = Field(default="N/A", description="Performance status (Good/Bad/Poor)")
    notes: Optional[str] = Field(default="", description="Additional notes")


class SeparateTablesResponse(BaseModel):
    """Response with explicitly separate tables for each type"""
    kpis: List[TableRowItem] = Field(default=[], description="List of KPIs with details")
    parameters: List[TableRowItem] = Field(default=[], description="List of Parameters with details")
    identifiers: Optional[List[TableRowItem]] = Field(default=None, description="List of Identifiers (Cell Name, Cell ID, etc.)")
    hierarchy: Optional[List[HierarchyRow]] = Field(default=None, description="LTE/5G Site-Sector-Cell Hierarchy mapping")
    cell_performance: Optional[List[CellPerformanceRow]] = Field(default=None, description="Cell performance data for multi-metric queries")
    counters: Optional[List[TableRowItem]] = Field(default=None, description="List of Counters if applicable")
    features: Optional[List[TableRowItem]] = Field(default=None, description="List of Features if applicable")
    alarms: Optional[List[TableRowItem]] = Field(default=None, description="List of Alarms if applicable")

    reasoning_kpis: str = Field(default="Not applicable", description="Explain each KPI and its significance")
    reasoning_parameters: str = Field(default="Not applicable", description="Explain each parameter and how it impacts the topic")
    technical_details: str = Field(default="Not applicable", description="Technical context, formulas, calculations")
    additional_context: str = Field(default="Not applicable", description="Constraints, vendor notes, technology specifics")
    constraints_notes: str = Field(default="Not applicable", description="Limitations, missing information, considerations")
    data_quality_warning: Optional[str] = Field(default=None, description="Warning if data might be incomplete or fabricated")

    class Config:
        extra = 'allow'


def validate_nokia_cell_name(cell_name: str) -> bool:
    """Validate if a cell name follows Nokia naming convention."""
    if not cell_name or len(cell_name) < 10:
        return False
    # Check pattern: 1 letter prefix + site/sector characters
    return bool(NOKIA_CELL_PATTERN.match(cell_name.upper()))


def extract_site_from_cell(cell_name: str) -> tuple:
    """
    Extract SITEID and Sector Name from Nokia cell name.

    Nokia naming convention:
    - cellName: 11 chars (e.g., B9BH0003A21)
    - Prefix: 1st char (B, A, J, K, etc.) - indicates layer/band
    - Sector Name: last 10 chars (e.g., 9BH0003A21)
    - SITEID: first 8 chars of Sector Name (e.g., 9BH0003A)

    Returns: (site_id, sector_name)
    """
    if not cell_name or len(cell_name) < 10:
        return ("N/A", "N/A")

    cell_name = cell_name.strip().upper()

    if len(cell_name) == 11:
        # Standard format: Prefix + Sector Name
        sector_name = cell_name[1:]  # Last 10 chars (without prefix)
        site_id = sector_name[:8]     # First 8 chars of sector name
        return (site_id, sector_name)
    elif len(cell_name) == 10:
        # Already a sector name (no prefix)
        site_id = cell_name[:8]
        return (site_id, cell_name)
    elif len(cell_name) == 8:
        # Just a SITEID
        return (cell_name, "N/A")
    else:
        return (cell_name[:8] if len(cell_name) >= 8 else cell_name, "N/A")


def detect_hallucinated_cells(cells_data: list) -> tuple:
    """
    Detect potentially hallucinated/fabricated cell IDs.
    
    Fake cell IDs typically follow patterns like:
    - ATL00123A11 (city code + sequential numbers)
    - CHI04567B21 (airport code pattern)
    
    Real Nokia cell IDs follow pattern like:
    - B9BH0003A21 (prefix + alphanumeric SITEID + sector digits)

    Returns: (valid_cells, warning_message)
    """
    if not cells_data:
        return ([], None)

    valid_cells = []
    suspicious_cells = []

    for cell in cells_data:
        cell_name = cell.get('cell_name', '') or cell.get('cell_id', '') or cell.get('name', '')
        cell_name_upper = cell_name.upper().strip()
        
        is_suspicious = False
        reason = ""

        # Check against known fake patterns FIRST
        for fake_pattern in FAKE_CELL_PATTERNS:
            if fake_pattern.match(cell_name_upper):
                is_suspicious = True
                reason = "matches city/airport code pattern"
                break
        
        # Additional heuristic checks for fake cells
        if not is_suspicious:
            # Check for sequential digit patterns (00123, 04567, etc.) - common in hallucinated IDs
            sequential_digits = re.search(r'\d{5,}', cell_name_upper)
            if sequential_digits:
                digits = sequential_digits.group()
                # Check if digits are mostly sequential or repetitive
                if len(set(digits)) <= 2:  # Very repetitive like 00000 or 00011
                    is_suspicious = True
                    reason = "contains repetitive digit sequence"
            
            # Check for market name embedded in cell ID
            for market in MARKET_LIST:
                market_abbrev = market[:3].upper()
                if cell_name_upper.startswith(market_abbrev) and len(cell_name_upper) >= 10:
                    # Check if it's followed by mostly digits (hallucination pattern)
                    rest = cell_name_upper[3:]
                    digit_count = sum(1 for c in rest if c.isdigit())
                    if digit_count >= len(rest) * 0.6:  # More than 60% digits after city code
                        is_suspicious = True
                        reason = f"starts with market abbreviation '{market_abbrev}'"
                        break

        # Validate against Nokia pattern if not already suspicious
        if not is_suspicious:
            if len(cell_name_upper) >= 10 and len(cell_name_upper) <= 12:
                if validate_nokia_cell_name(cell_name_upper):
                    # Additional check: first char after prefix should not be all same digits
                    if len(cell_name_upper) == 11:
                        siteid_part = cell_name_upper[1:9]  # SITEID portion
                        # Real SITEIDs have mixed alphanumeric, not just digits
                        if not siteid_part.isdigit():
                            valid_cells.append(cell)
                        else:
                            is_suspicious = True
                            reason = "SITEID portion is all digits (unusual)"
                    else:
                        valid_cells.append(cell)
                else:
                    is_suspicious = True
                    reason = "does not match Nokia cell naming convention"
            else:
                is_suspicious = True
                reason = f"invalid length ({len(cell_name_upper)} chars, expected 10-12)"

        if is_suspicious:
            suspicious_cells.append({'cell_name': cell_name, 'reason': reason})
            logger.warning(f"Potentially fabricated cell ID detected: {cell_name} - {reason}")

    warning = None
    suspicious_count = len(suspicious_cells)
    
    if suspicious_count > 0:
        if suspicious_count == len(cells_data):
            warning = f"⚠️ WARNING: All {suspicious_count} cell IDs appear to be fabricated/hallucinated (e.g., '{suspicious_cells[0]['cell_name']}' - {suspicious_cells[0]['reason']}). No matching cell data found in the database. Real Nokia cell IDs follow format like 'B9BH0003A21'. Please verify your data source contains actual cell performance records."
        else:
            warning = f"⚠️ WARNING: {suspicious_count} of {len(cells_data)} cell IDs may be fabricated. Only {len(valid_cells)} valid cells will be shown."

    return (valid_cells, warning)


app = Flask(__name__)
CORS(app)

# Global components
vector_store = None
embeddings = None
llm = None
llm_structured = None
reranker = None


async def init_components():
    global vector_store, embeddings, llm, llm_structured, reranker

    logger.info("Initializing RAG components...")

    # 1. Initialize Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder="./model_cache"
    )

    # 2. Initialize Reranker (Cross-Encoder)
    logger.info("Loading Reranker Model...")
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    # 3. Initialize Vector Store
    vector_store = QdrantVectorStore(
        url=settings.vectorstore.url,
        collection_name=settings.vectorstore.collection_name,
        test_connection=True
    )
    await vector_store.initialize()

    # 4. Initialize LLM
    headers = {
        "api-key": settings.api_key,
        "workspaceName": settings.workspace
    }

    llm = ChatOpenAI(
        model="gpt-4o",
        openai_api_key=settings.api_key,
        openai_api_base=settings.nokia_llm_gateway_url,
        default_headers=headers,
        temperature=0.0,  # Set to 0 for more deterministic extraction
        max_tokens=4000
    )

    # Create structured output LLM
    llm_structured = llm.with_structured_output(StructuredResponse)

    logger.info("Components initialized successfully.")


# Run initialization
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
        # Check for "all markets" and expand query if needed
        market_aliases = ["all market", "all the market", "every market", "all markets"]
        if any(alias in query.lower() for alias in market_aliases):
            markets_str = ", ".join(MARKET_LIST)
            logger.info(f"Expanding 'all markets' query with {len(MARKET_LIST)} markets")
            query = f"{query}. Ensure you include Cell IDs and Sites for the following markets: {markets_str}"

        query_lower = query.lower()

        # Detect question type
        question_type = "general"
        if "compare" in query_lower or "vs" in query_lower or "versus" in query_lower:
            question_type = "comparison"
        elif "top" in query_lower and ("cell" in query_lower or "site" in query_lower):
            question_type = "ranking"
        elif "identify" in query_lower or "list" in query_lower or "pull" in query_lower:
            question_type = "extraction"

        # Extract requested item types
        requested_types = []
        if "parameter" in query_lower:
            requested_types.append("parameters")
        if "kpi" in query_lower:
            requested_types.append("KPIs")
        if "counter" in query_lower:
            requested_types.append("counters")
        if "feature" in query_lower:
            requested_types.append("features")
        if "alarm" in query_lower:
            requested_types.append("alarms")
        if "cell" in query_lower and question_type == "ranking":
            requested_types.append("cells")

        # Extract metric focus
        metric_focus = ""
        if "dl throughput" in query_lower or "downlink throughput" in query_lower:
            metric_focus = "DL Throughput"
        elif "spectral efficiency" in query_lower:
            metric_focus = "Spectral Efficiency"
        elif "voice dcr" in query_lower or "drop call rate" in query_lower:
            metric_focus = "Voice DCR"
        elif "latency" in query_lower:
            metric_focus = "Latency"
        elif "prb" in query_lower or "resource block" in query_lower:
            metric_focus = "PRB Utilization"

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

        # Rerank results
        search_results = []
        if results:
            pairs = []
            for res in results:
                content = res.payload.get("text") or res.payload.get("content") or str(res.payload)
                pairs.append([query, content])

            scores = reranker.predict(pairs)
            scored_results = [(scores[i], res) for i, res in enumerate(results)]
            scored_results.sort(key=lambda x: x[0], reverse=True)
            search_results = [res for score, res in scored_results[:20]]

            logger.info(f"RAG Retrieval: Selected {len(search_results)} results after reranking")

        # Format context
        contexts = []
        for idx, res in enumerate(search_results, 1):
            content = res.payload.get("text") or res.payload.get("content") or ""
            metadata = {}
            if "type" in res.payload:
                metadata["Type"] = res.payload["type"]
            if "category" in res.payload:
                metadata["Category"] = res.payload["category"]
            if "source" in res.payload:
                metadata["Source"] = res.payload["source"]

            context_block = f"[Document {idx}]"
            if metadata:
                meta_str = " | ".join([f"{k}: {v}" for k, v in metadata.items()])
                context_block += f"\n{meta_str}"
            context_block += f"\n{content}"
            contexts.append(context_block)

        formatted_context = "\n\n---\n\n".join(contexts)

        # Standard text-based response
        system_instruction = f"""
You are a Telecom Performance Expert specializing in LTE/5G NR (NSA & SA).

**CRITICAL INSTRUCTION - DATA EXTRACTION ONLY:**
You MUST extract data ONLY from the provided context.
DO NOT fabricate, invent, or hallucinate any cell IDs, site IDs, or performance values.

**NOKIA CELL NAMING CONVENTION (MUST FOLLOW):**
- **cellName** (11 chars): e.g., B9BH0003A21
  - 1st char = Layer Prefix (A=Primary LTE, B=700MHz, J=Secondary, K=LAA)
  - Chars 2-9 = SITEID (8 chars): e.g., 9BH0003A
  - Chars 10-11 = Sector (e.g., 21 = Sector 2, Sub-sector 1)
- **Sector Name** (10 chars): Last 10 chars of cellName (e.g., 9BH0003A21)
- **SITEID** (8 chars): First 8 chars of Sector Name (e.g., 9BH0003A)

**IF NO MATCHING CELL DATA IS FOUND IN CONTEXT:**
Respond with: "No cell performance data matching the requested criteria was found in the database. The context does not contain cell-level metrics for [requested parameters]."

**FOR CELL RANKING QUERIES:**
| Rank | cellName | SITEID | Sector Name | DL Tput | HO Succ | Congestion | SE | RACH SR | Status | Notes |
|------|----------|--------|-------------|---------|---------|------------|----|---------| -------|-------|

**VERIFICATION REQUIRED:**
Before outputting any cell data, verify:
1. The cellName follows Nokia format (letter + 8 alphanumeric + 2 digits)
2. The SITEID is derived correctly (chars 2-9 of cellName)
3. All values are extracted from context, not generated

**Source:** Collection: {settings.vectorstore.collection_name}
"""

        prompt = ChatPromptTemplate.from_template("""
{system_instruction}

**CONTEXT:**
{context}

**USER QUESTION:** {input}

**YOUR RESPONSE:**
""")

        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({
            "system_instruction": system_instruction,
            "context": formatted_context,
            "input": query
        })

        return jsonify({"response": response})

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        return jsonify({
            "error": "An error occurred while processing your request",
            "message": str(e),
            "response": "I encountered an error processing your query. Please try again or rephrase your question."
        }), 500


@app.route("/chat_tabular", methods=["POST"])
def chat_tabular():
    """
    Enhanced endpoint that returns structured JSON for tabular data queries.
    """
    data = request.json
    query = data.get("message")

    if not query:
        return jsonify({"error": "No message provided"}), 400

    try:
        # Check for "all markets" and expand query if needed
        market_aliases = ["all market", "all the market", "every market", "all markets"]
        if any(alias in query.lower() for alias in market_aliases):
            markets_str = ", ".join(MARKET_LIST)
            logger.info(f"Expanding 'all markets' query with {len(MARKET_LIST)} markets")
            query = f"{query}. Specifically for the following markets: {markets_str}"

        query_lower = query.lower()
        enhanced_query = query

        # Detect multi-metric cell performance query
        is_multi_metric_query = False
        multi_metric_keywords = ['throughput', 'ho', 'handover', 'congestion', 'spectral', 'rach']
        matching_keywords = sum(1 for kw in multi_metric_keywords if kw in query_lower)

        if matching_keywords >= 3 and ('cell' in query_lower or 'top' in query_lower or 'identify' in query_lower):
            is_multi_metric_query = True
            logger.info("Detected multi-metric cell performance query")
            # Enhance query for better retrieval
            enhanced_query += ' cellName SITEID "DL Throughput" "HO Success" "Congestion" "Spectral Efficiency" "RACH" cell_performance sector'

        # Enhance for Voice DCR queries
        if 'dcr' in query_lower or 'dropped call' in query_lower or 'voice' in query_lower:
            enhanced_query += " \"Dropped Calls\" \"Total Call Attempts\" cellName SITEID sector cell_performance"

        # Use Hybrid Search for better exact ID matching
        q_vector = embeddings.embed_query(enhanced_query)
        results = loop.run_until_complete(
            vector_store.hybrid_search_rrf(
                collection_name=settings.vectorstore.collection_name,
                query_vector=q_vector,
                query_text=enhanced_query,
                limit=100
            )
        )

        logger.info(f"Retrieved {len(results)} documents from vector store")

        # Log sample of retrieved content for debugging
        if results and len(results) > 0:
            sample_content = results[0].payload.get("text", "")[:300]
            logger.info(f"Sample retrieved content: {sample_content}")

        search_results = []
        if results:
            pairs = [[query, res.payload.get("text") or res.payload.get("content") or str(res.payload)] for res in results]
            scores = reranker.predict(pairs)
            scored_results = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)
            search_results = [res for score, res in scored_results[:25]]
            logger.info(f"Using top {len(search_results)} documents after reranking")

        # Format context
        contexts = []
        for idx, res in enumerate(search_results, 1):
            content = res.payload.get("text") or res.payload.get("content") or ""
            metadata = {}
            for key in ["type", "category", "source", "market"]:
                if key in res.payload:
                    metadata[key.capitalize()] = res.payload[key]

            context_block = f"[Document {idx}]"
            if metadata:
                context_block += "\n" + " | ".join([f"{k}: {v}" for k, v in metadata.items()])
            context_block += f"\n{content}"
            contexts.append(context_block)

        formatted_context = "\n\n---\n\n".join(contexts)

        logger.info(f"Context length: {len(formatted_context)} characters")

        # MULTI-METRIC CELL PERFORMANCE QUERY PROMPT
        if is_multi_metric_query:
            simplified_prompt = ChatPromptTemplate.from_template("""
You are a Telecom Data Extraction Expert. Your task is to extract REAL cell performance data from the provided context.

**CRITICAL RULES - READ CAREFULLY:**

1. **EXTRACTION ONLY**: You MUST extract cell IDs and values EXACTLY as they appear in the context.
2. **NO FABRICATION**: DO NOT invent, generate, or hallucinate ANY cell IDs or values.
3. **NO GENERIC PATTERNS**: Cell IDs like "ATL00123A11", "CHI04567B21" are INVALID - these follow city airport codes, not Nokia format.

**NOKIA CELL NAMING CONVENTION:**
- **cellName** (11 chars): Format is [Prefix][SITEID][Sector#][Sub#]
  - Example: B9BH0003A21
  - Prefix (1 char): A, B, J, K, D, E, L, Z (indicates layer/band)
  - SITEID (8 chars): e.g., 9BH0003A (extracted from chars 2-9)
  - Sector (2 chars): e.g., 21 (Sector 2, Sub-sector 1)
- **Sector Name** (10 chars): cellName without prefix (e.g., 9BH0003A21)
- **SITEID** (8 chars): First 8 chars of Sector Name (e.g., 9BH0003A)

**HOW TO DERIVE SITEID FROM cellName:**
- If cellName = "B9BH0003A21", then:
  - Remove prefix "B" → "9BH0003A21" (Sector Name)
  - Take first 8 chars → "9BH0003A" (SITEID)

**EXPECTED OUTPUT FORMAT:**
Return a JSON object with:
```json
{{
  "cell_performance": [
    {{
      "rank": 1,
      "cell_name": "EXACT cellName from context",
      "site_id": "DERIVED 8-char SITEID",
      "sector_name": "DERIVED 10-char Sector Name",
      "market": "Market name if available",
      "dl_throughput": "Value from context or N/A",
      "ho_success_rate": "Value from context or N/A",
      "congestion": "Value from context or N/A",
      "spectral_efficiency": "Value from context or N/A",
      "rach_setup_sr": "Value from context or N/A",
      "status": "Bad/Poor/Good based on values",
      "notes": "Brief observation"
    }}
  ],
  "data_found": true/false,
  "technical_details": "Explanation of the data and metrics",
  "constraints_notes": "Any limitations or missing data"
}}
```

**IF NO CELL DATA IS FOUND IN CONTEXT:**
Return:
```json
{{
  "cell_performance": [],
  "data_found": false,
  "technical_details": "No cell performance data matching the criteria was found in the retrieved context.",
  "constraints_notes": "The database may not contain cell-level performance metrics for the requested query. Please verify the data source."
}}
```

**RANKING CRITERIA (for "bad" performance):**
- LOW DL Throughput (< 50 Mbps is poor)
- LOW HO Success Rate (< 95% is poor)
- HIGH Congestion (> 80% is poor)
- LOW Spectral Efficiency (< 2 bps/Hz is poor)
- LOW RACH Setup Success Rate (< 95% is poor)

**USER QUESTION:** {input}

**CONTEXT TO EXTRACT FROM:**
{context}

**IMPORTANT VERIFICATION:**
Before including ANY cell in your response:
1. Verify the cellName exists VERBATIM in the context above
2. Verify it follows Nokia format (not airport/city codes)
3. Extract actual metric values from context, don't estimate

**YOUR JSON RESPONSE:**
""")

        elif "what is" in query_lower and ("kpi" in query_lower or "parameter" in query_lower) and len(query.split()) < 10:
            # General definition query
            simplified_prompt = ChatPromptTemplate.from_template("""
You are a Telecom Performance Expert. Define the concept clearly.

User Question: {input}

Return a JSON with:
- kpis: []
- parameters: []
- reasoning_kpis: Clear definition of KPI in telecom context
- reasoning_parameters: Clear definition of Parameter in telecom context
- technical_details: Relationship between KPIs and parameters
- additional_context: "This is a general definition."
- constraints_notes: "N/A"
""")

        elif (all(x in query_lower for x in ["siteid", "sector", "cellname"])) or \
             ("lte" in query_lower and "siteid" in query_lower):
            # Hierarchy query
            simplified_prompt = ChatPromptTemplate.from_template("""
You are a Telecom Performance Expert. Extract the Site-Sector-Cell hierarchy from the context.

Return a JSON with:
- hierarchy: Array of {{site_id, sector_name, cell_name}}
- reasoning_parameters: Analysis of the site configuration
- technical_details: Summary of the technology

NOKIA NAMING RULES:
- cellName (11 chars): e.g., B9BH0003A21
- Sector Name: Last 10 chars of cellName (e.g., 9BH0003A21)
- SITEID: First 8 chars of Sector Name (e.g., 9BH0003A)

CONTEXT: {context}
USER QUESTION: {input}
""")

        else:
            # Standard KPI/Parameter query
            simplified_prompt = ChatPromptTemplate.from_template("""
You are a Telecom Performance Expert. Extract KPIs and Parameters from context.

Return a structured JSON with:
- kpis: Array of {{number, name, description, formula_calculation, relationship_impact}}
- parameters: Array of {{number, name, description, formula_calculation, relationship_impact}}
- identifiers: Array for Cell Name, Cell ID, Site ID, etc.
- reasoning_kpis: String explaining the KPIs
- reasoning_parameters: String explaining the parameters
- technical_details: Technical context and formulas
- additional_context: Additional notes
- constraints_notes: Limitations

CONTEXT: {context}
USER QUESTION: {input}

RULES:
1. Extract ONLY items from context - DO NOT fabricate
2. Use exact names as they appear in context
3. For cell-related identifiers, follow Nokia naming convention
""")

        def clean_markdown_formatting(text: str) -> str:
            """Remove markdown formatting symbols from text"""
            if not text:
                return text
            text = text.replace('**', '').replace('__', '')
            text = re.sub(r'\*([^*]+)\*', r'\1', text)
            text = text.replace('`', '')
            return text.strip()

        def extract_json_from_markdown(text: str) -> str:
            """Extract JSON from markdown code blocks if present."""
            json_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
            match = re.search(json_block_pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()
            return text.strip()

        # Execute LLM chain
        chain = simplified_prompt | llm | StrOutputParser()

        llm_response = chain.invoke({
            "context": formatted_context,
            "input": query
        })

        logger.info(f"LLM Response (first 500 chars): {llm_response[:500]}")

        # Parse JSON response
        json_str = extract_json_from_markdown(llm_response)
        json_data = {}

        try:
            # Try to find and parse JSON
            start_brace = json_str.find('{')
            if start_brace != -1:
                end_brace = json_str.rfind('}') + 1
                json_data = json.loads(json_str[start_brace:end_brace])
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            # Try to extract from code blocks
            code_blocks = re.findall(r'```(?:json)?\s*\n?(.*?)\n?```', llm_response, re.DOTALL)
            for block in code_blocks:
                try:
                    json_data = json.loads(block.strip())
                    break
                except:
                    continue

        # Process multi-metric query results
        tables = []
        data_quality_warning = None

        if is_multi_metric_query:
            cell_performance = json_data.get('cell_performance', [])
            data_found = json_data.get('data_found', len(cell_performance) > 0)

            if cell_performance:
                # Validate extracted cells - CRITICAL: filters out hallucinated/fake cell IDs
                valid_cells, warning = detect_hallucinated_cells(cell_performance)
                data_quality_warning = warning
                
                logger.info(f"Cell validation: {len(cell_performance)} input, {len(valid_cells)} valid, {len(cell_performance) - len(valid_cells)} rejected")

                if not valid_cells:
                    # All cells are fabricated - don't show the table
                    logger.warning("All extracted cells appear to be fabricated - no valid Nokia cell IDs found")
                    cell_performance = []
                else:
                    # Process ONLY valid cells - derive SITEID correctly using Nokia naming convention
                    processed_cells = []
                    for cell in valid_cells:  # FIXED: Use valid_cells, not cell_performance
                        cell_name = cell.get('cell_name', '')

                        # Derive SITEID and Sector Name correctly per Nokia convention
                        # cellName (11 chars): B9BH0003A21
                        # Sector Name (10 chars): 9BH0003A21 (chars 2-11)
                        # SITEID (8 chars): 9BH0003A (first 8 of Sector Name)
                        site_id, sector_name = extract_site_from_cell(cell_name)

                        processed_cells.append({
                            'rank': cell.get('rank', len(processed_cells) + 1),
                            'cell_name': cell_name,
                            'site_id': site_id,
                            'sector_name': sector_name,
                            'market': cell.get('market', 'N/A'),
                            'dl_throughput': str(cell.get('dl_throughput', 'N/A')),
                            'ho_success_rate': str(cell.get('ho_success_rate', 'N/A')),
                            'congestion': str(cell.get('congestion', 'N/A')),
                            'spectral_efficiency': str(cell.get('spectral_efficiency', 'N/A')),
                            'rach_setup_sr': str(cell.get('rach_setup_sr', 'N/A')),
                            'status': cell.get('status', 'N/A'),
                            'notes': cell.get('notes', '')
                        })

                    cell_performance = processed_cells
                    logger.info(f"Processed {len(cell_performance)} valid cells for display")

            # Create cell performance table
            if cell_performance:
                cell_table = TableData(
                    title="Cell Performance Analysis (Multi-Metric)",
                    headers=["Rank", "cellName", "SITEID", "Sector Name", "DL Tput (Mbps)", "HO Succ (%)", "Congestion (%)", "SE (bps/Hz)", "RACH SR (%)", "Status", "Notes"],
                    rows=[[
                        str(cell['rank']),
                        cell['cell_name'],
                        cell['site_id'],
                        cell['sector_name'],
                        cell['dl_throughput'],
                        cell['ho_success_rate'],
                        cell['congestion'],
                        cell['spectral_efficiency'],
                        cell['rach_setup_sr'],
                        cell['status'],
                        cell['notes']
                    ] for cell in cell_performance]
                )
                tables.append(cell_table)

            # Build response text
            response_parts = []

            if data_quality_warning:
                response_parts.append(f"\n{data_quality_warning}\n")

            if cell_performance:
                response_parts.append(f"### Cell Performance Analysis\n\nFound {len(cell_performance)} cells matching the criteria.\n")
            else:
                response_parts.append("### No Matching Cell Data Found\n\nThe database does not contain cell performance records matching your query criteria.\n")
                response_parts.append("\n**Possible reasons:**\n")
                response_parts.append("- The collection may not contain cell-level performance metrics\n")
                response_parts.append("- The specific metrics requested may not be available\n")
                response_parts.append("- Try querying for specific cell IDs if you have them\n")

            if json_data.get('technical_details'):
                response_parts.append(f"\n---\n### Technical Details\n{json_data['technical_details']}\n")

            if json_data.get('constraints_notes'):
                response_parts.append(f"\n---\n### Notes\n{json_data['constraints_notes']}\n")

            response_parts.append(f"\n---\n*Source: {settings.vectorstore.collection_name}*")

            response_text = ''.join(response_parts)

        else:
            # Standard processing for non-multi-metric queries
            # Ensure lists exist
            if 'kpis' not in json_data:
                json_data['kpis'] = []
            if 'parameters' not in json_data:
                json_data['parameters'] = []

            try:
                structured_result = SeparateTablesResponse(**json_data)
            except Exception as e:
                logger.error(f"Validation failed: {e}")
                structured_result = SeparateTablesResponse(
                    kpis=[],
                    parameters=[],
                    constraints_notes=str(e)
                )

            # Move identifiers from parameters if misclassified
            identifier_keywords = ["Cell Name", "Cell ID", "Site ID", "ECGI", "gNB", "eNB", "TAC", "PCI", "Sector ID", "Sector Name", "Market"]

            if structured_result.parameters:
                if structured_result.identifiers is None:
                    structured_result.identifiers = []

                items_to_move = []
                for i in range(len(structured_result.parameters) - 1, -1, -1):
                    param = structured_result.parameters[i]
                    clean_name = param.name.lower().replace('*', '').replace('_', '').strip()

                    is_match = False
                    for kw in identifier_keywords:
                        kw_clean = kw.lower().replace(' ', '')
                        name_clean = clean_name.replace(' ', '')

                        if kw_clean == name_clean or kw_clean in name_clean:
                            if "count" not in clean_name and "rate" not in clean_name:
                                is_match = True
                                break

                    if is_match:
                        items_to_move.append(structured_result.parameters.pop(i))

                for item in reversed(items_to_move):
                    if not item.relationship_impact or item.relationship_impact == "Parameter":
                        item.relationship_impact = "Identifier"
                    structured_result.identifiers.append(item)

            # Create tables
            if structured_result.kpis:
                kpi_table = TableData(
                    title="KPIs Related to the Query",
                    headers=["#", "Name", "Description", "Formula/Calculation", "Relationship/Impact"],
                    rows=[[
                        str(kpi.number),
                        clean_markdown_formatting(kpi.name),
                        clean_markdown_formatting(kpi.description),
                        clean_markdown_formatting(kpi.formula_calculation),
                        clean_markdown_formatting(kpi.relationship_impact)
                    ] for kpi in structured_result.kpis]
                )
                tables.append(kpi_table)

            if structured_result.parameters:
                param_table = TableData(
                    title="Parameters Related to the Query",
                    headers=["#", "Name", "Description", "Formula/Calculation", "Relationship/Impact"],
                    rows=[[
                        str(param.number),
                        clean_markdown_formatting(param.name),
                        clean_markdown_formatting(param.description),
                        clean_markdown_formatting(param.formula_calculation),
                        clean_markdown_formatting(param.relationship_impact)
                    ] for param in structured_result.parameters]
                )
                tables.append(param_table)

            if structured_result.identifiers:
                ident_table = TableData(
                    title="Identifiers / Network Hierarchy",
                    headers=["#", "Identifier Name", "Value", "Description", "Level/Notes"],
                    rows=[[
                        str(ident.number),
                        clean_markdown_formatting(ident.name),
                        clean_markdown_formatting(ident.formula_calculation),
                        clean_markdown_formatting(ident.description),
                        clean_markdown_formatting(ident.relationship_impact)
                    ] for ident in structured_result.identifiers]
                )
                tables.append(ident_table)

            if structured_result.hierarchy:
                hier_table = TableData(
                    title="Site-Sector-Cell Hierarchy Mapping",
                    headers=["SITEID", "Sector Name", "cellName"],
                    rows=[[
                        clean_markdown_formatting(row.site_id),
                        clean_markdown_formatting(row.sector_name),
                        clean_markdown_formatting(row.cell_name)
                    ] for row in structured_result.hierarchy]
                )
                tables.insert(0, hier_table)

            # Build response text
            response_parts = []

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

            response_parts.append(f"\n---\n*Source: {settings.vectorstore.collection_name}*")

            response_text = ''.join(response_parts)

        # Create final response
        tabular_response = TabularResponse(
            response=response_text,
            tables=tables,
            has_tables=len(tables) > 0
        )

        logger.info(f"Final response: {len(tables)} tables, {len(response_text)} chars of text")

        return jsonify(tabular_response.model_dump())

    except Exception as e:
        import traceback
        logger.error(f"Error in chat_tabular endpoint: {e}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        error_response = TabularResponse(
            response=f"Error: {str(e)}",
            tables=None,
            has_tables=False
        )
        return jsonify(error_response.model_dump()), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
