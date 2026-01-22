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
            # If it's not a digit, it's probably a KPI/parameter name, keep as string
            return v
        return v

class HierarchyRow(BaseModel):
    """Specific row for Site-Sector-Cell hierarchy"""
    site_id: str = Field(description="8-char SITEID (e.g. 9BH0003A)")
    sector_name: str = Field(description="10-char Sector Name (e.g. 9BH0003A11)")
    cell_name: str = Field(description="Prefix + ID Cell Name (e.g. A9BH0003A11)")

class SeparateTablesResponse(BaseModel):
    """Response with explicitly separate tables for each type"""
    kpis: List[TableRowItem] = Field(default=[], description="List of KPIs with details")
    parameters: List[TableRowItem] = Field(default=[], description="List of Parameters with details")
    identifiers: Optional[List[TableRowItem]] = Field(default=None, description="List of Identifiers (Cell Name, Cell ID, etc.)")
    hierarchy: Optional[List[HierarchyRow]] = Field(default=None, description="LTE/5G Site-Sector-Cell Hierarchy mapping")
    counters: Optional[List[TableRowItem]] = Field(default=None, description="List of Counters if applicable")
    features: Optional[List[TableRowItem]] = Field(default=None, description="List of Features if applicable")
    alarms: Optional[List[TableRowItem]] = Field(default=None, description="List of Alarms if applicable")
    
    reasoning_kpis: str = Field(default="Not applicable", description="Explain each KPI and its significance")
    reasoning_parameters: str = Field(default="Not applicable", description="Explain each parameter and how it impacts the topic")
    technical_details: str = Field(default="Not applicable", description="Technical context, formulas, calculations")
    additional_context: str = Field(default="Not applicable", description="Constraints, vendor notes, technology specifics")
    constraints_notes: str = Field(default="Not applicable", description="Limitations, missing information, considerations")

    class Config:
        extra = 'allow'




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
        temperature=0.1,  # Slightly increased for better natural language
        max_tokens=4000   # Increased for comprehensive tabular responses
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
    use_structured = data.get("structured", False)  # Optional: use structured output
    
    if not query:
        return jsonify({"error": "No message provided"}), 400
    
    try:
        # Check for "all markets" and expand query if needed
        market_aliases = ["all market", "all the market", "every market", "all markets"]
        if any(alias in query.lower() for alias in market_aliases):
            markets_str = ", ".join(MARKET_LIST)
            logger.info(f"Expanding 'all markets' query with {len(MARKET_LIST)} markets")
            # We append it to the query to ensure the LLM sees it
            query = f"{query}. Ensure you include Cell IDs and Sites for the following markets: {markets_str}"

        # Enhanced query processing - extract key terms and intent
        query_lower = query.lower()
        
        # Detect question type and key terms
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
        
        # IMPROVED RAG LOGIC: Multi-stage retrieval with better context formatting
        
        # Stage 1: Fetch more candidates for better coverage
        # Enhance query with key terms for better retrieval
        q_vector = embeddings.embed_query(query)
        results = loop.run_until_complete(
            vector_store.search(
                collection_name=settings.vectorstore.collection_name,
                query_vector=q_vector,
                limit=50,
                score_threshold=0.55  # Cosine Similarity Threshold
            )
        )
        
        # Stage 2: Rerank with Cross-Encoder for relevance
        search_results = []
        if results:
            pairs = []
            for res in results:
                content = res.payload.get("text") or res.payload.get("content") or str(res.payload)
                pairs.append([query, content])
            
            # Predict relevance scores
            scores = reranker.predict(pairs)
            
            # Combine results with scores
            scored_results = []
            for i, res in enumerate(results):
                scored_results.append((scores[i], res))
            
            # Sort by relevance score (descending)
            scored_results.sort(key=lambda x: x[0], reverse=True)
            
            # Take Top 20
            search_results = [res for score, res in scored_results[:20]]
            
            logger.info(f"RAG Retrieval: Selected {len(search_results)} results after reranking")
        
        # Stage 3: Enhanced context formatting with metadata
        contexts = []
        for idx, res in enumerate(search_results, 1):
            content = res.payload.get("text") or res.payload.get("content") or ""
            
            # Extract metadata if available
            metadata = {}
            if "type" in res.payload:
                metadata["Type"] = res.payload["type"]
            if "category" in res.payload:
                metadata["Category"] = res.payload["category"]
            if "source" in res.payload:
                metadata["Source"] = res.payload["source"]
            
            # Format context with metadata
            context_block = f"[Document {idx}]"
            if metadata:
                meta_str = " | ".join([f"{k}: {v}" for k, v in metadata.items()])
                context_block += f"\n{meta_str}"
            context_block += f"\n{content}"
            
            contexts.append(context_block)
        
        formatted_context = "\n\n---\n\n".join(contexts)
        
        # Choose between structured or text output
        if use_structured:
            try:
                # Check if llm_structured is available
                if llm_structured is None:
                    logger.warning("Structured LLM not initialized, falling back to text-based output")
                    use_structured = False
                else:
                    # Use structured output with Pydantic
                    structured_prompt = ChatPromptTemplate.from_template("""
You are a Telecom Performance Expert. Extract parameters and KPIs from the context and organize them into SEPARATE lists.

**CONTEXT:**
{context}

**USER QUESTION:** {input}

**INSTRUCTIONS:**
1. Identify the topic from the user question
2. Extract ALL relevant parameters from context - create a list with:
   - number (starting from 1)
   - name (exact from context)
   - description (1 sentence, max 15 words)
   - relationship (1 sentence, max 15 words)
3. Extract ALL relevant KPIs from context - create a separate list with same structure
4. For reasoning, use labels like "Parameter1", "Parameter2", "KPI1", "KPI2"
5. Provide detailed explanations (1-2 sentences) for each item in reasoning

Return structured data with:
- topic
- parameters (list of TableRow)
- kpis (list of TableRow)
- parameter_reasoning (list with label, name, explanation)
- kpi_reasoning (list with label, name, explanation)
""")
                
                    chain = structured_prompt | llm_structured
                    structured_result = chain.invoke({
                        "context": formatted_context,
                        "input": query
                    })
                    
                    # Format structured output as markdown with separate tables
                    response = f"Here is a tabular representation of the **parameters and KPIs** related to {structured_result.topic}, based on the retrieved context:\n\n"
                    
                    # Create Parameters table
                    if structured_result.parameters and len(structured_result.parameters) > 0:
                        response += "**Parameters:**\n\n"
                        response += "| # | Parameter Name | Description | Relationship to " + structured_result.topic + " |\n"
                        response += "|---|----------------|-------------|" + "-" * (len(structured_result.topic) + 18) + "|\n"
                        
                        for row in structured_result.parameters:
                            # Clean any markdown from table cells
                            clean_name = str(row.name).replace('**', '').replace('__', '')
                            clean_desc = str(row.description).replace('**', '').replace('__', '')
                            clean_rel = str(row.relationship).replace('**', '').replace('__', '')
                            response += f"| {row.number} | {clean_name} | {clean_desc} | {clean_rel} |\n"
                        response += "\n"
                    
                    # Create KPIs table
                    if structured_result.kpis and len(structured_result.kpis) > 0:
                        response += "**KPIs:**\n\n"
                        response += "| # | KPI Name | Description | Relationship to " + structured_result.topic + " |\n"
                        response += "|---|----------|-------------|" + "-" * (len(structured_result.topic) + 18) + "|\n"
                        
                        for row in structured_result.kpis:
                            # Clean any markdown from table cells
                            clean_name = str(row.name).replace('**', '').replace('__', '')
                            clean_desc = str(row.description).replace('**', '').replace('__', '')
                            clean_rel = str(row.relationship).replace('**', '').replace('__', '')
                            response += f"| {row.number} | {clean_name} | {clean_desc} | {clean_rel} |\n"
                        response += "\n"
                    
                    # Add reasoning section
                    response += "**Reasoning and Relationships**\n\n"
                    
                    if structured_result.parameter_reasoning:
                        response += "1. **Parameters:**\n"
                        for item in structured_result.parameter_reasoning:
                            response += f"   - **{item.label} ({item.name})**: {item.explanation}\n"
                        response += "\n"
                    
                    if structured_result.kpi_reasoning:
                        response += "2. **KPIs:**\n"
                        for item in structured_result.kpi_reasoning:
                            response += f"   - **{item.label} ({item.name})**: {item.explanation}\n"
                        response += "\n"
                    
                    if structured_result.constraints_notes:
                        response += f"**Constraints and Notes:**\n{structured_result.constraints_notes}\n\n"
                    
                    response += f"**Source:** Collection: {settings.vectorstore.collection_name}"
                
            except Exception as e:
                logger.error(f"Structured output failed: {e}, falling back to text-based output")
                use_structured = False
        
        if not use_structured:
            # Use text-based output (original approach)
            system_instruction = f"""
You are a Telecom Performance Expert specializing in LTE/5G NR (NSA & SA).

**CRITICAL: Analyze the user's question and provide DIFFERENT responses for DIFFERENT questions.**

**STEP 1: IDENTIFY WHAT IS REQUESTED**
Read the question carefully and identify:
- Is this about CELLS/SITES? (cell performance, top cells, cell ranking, good/bad cells)
- OR is this about DOCUMENTATION? (parameters, KPIs, counters, features, alarms)
- What metric/topic? (DL Throughput, Spectral Efficiency, Voice DCR, etc.)

**STEP 2: CREATE APPROPRIATE TABLES**

**FOR CELL/SITE QUESTIONS (top cells, cell performance, cell ranking):**

**Top [N] Cells - [Metric Name]**

| Rank | Cell ID/Name | [Metric] Value | Performance Status | Notes |
|------|--------------|----------------|-------------------|-------|
| 1 | JDA03866A11 | 95.5% | Good | Brief note about performance |
| 2 | JDA03866A21 | 94.2% | Good | Brief note about performance |

**IMPORTANT: CELL NAME FORMATTING**
- Extract cell names EXACTLY as they appear (e.g., JDA03866A11).
- Do not truncate or modify the cell ID.

**FOR MULTI-METRIC PERFORMANCE RANKING (DL Tput, HO, Congestion, SE, RACH):**

If the user asks for "Top 10" comparison based on DL Throughput, HO Success, Congestion, Spectral Efficiency, AND RACH Setup:

**Format Requirement:**
| Rank | Cell ID/Name | DL Throughput (Mbps) | HO Success Rate (%) | Congestion (%) | Spectral Efficiency (bps/Hz) | RACH Setup Success Rate (%) | Performance Status | Notes |
|------|--------------|----------------------|---------------------|----------------|------------------------------|-----------------------------|--------------------|-------|
| 1 | [Cell ID] | [Val] | [Val] | [Val] | [Val] | [Val] | [Status] | [Note] |

- **Good Cell Criteria**: High DL Tput, High HO Succ, Low Congestion, High SE, High RACH.
- **Bad Cell Criteria**: Low DL Tput, Low HO Succ, High Congestion, Low SE, Low RACH.

**SPECIFIC ANALYSIS FORMULAS & CALCULATIONS:**

1. **Combined RACH setup SR**: 
   - Formula: `100 * (Num of RACH setup completions for contention-based preambles + Num of RACH setup completions for dedicated preambles) / (RA setup att for contention-based preambles + RA setup att for dedicated preambles)`

2. **Handover Success Ratio**:
   - Formula: `(intra_du_ho_att_nsa_mnb + intra_du_ho_att_nsa_srb3 - intra_du_ho_fai_t304_exp_nsa - intra_du_ho_fai_tdc_exp - intra_du_hopre_fai_menb_ref) / (intra_du_ho_att_nsa_mnb + intra_du_ho_att_nsa_srb3 + intra_du_hopre_e1_fai_nsa + intra_du_hopre_f1_fai_nsa + intra_du_hopre_fai_res_nsa)`

3. **Voice DCR (Dropped Call Rate)**:
   - Formula: `DCR = (Dropped Calls / Total Call Attempts) * 100`
   - **Goal**: Rank cells by **LOWEST** DCR (Ascending) for "good/top" performance.
   - **Mandatory Output**: Always include this formula and explain the impact of dropped calls and attempts.
   - **Threshold**: Use `Percentile_Flag` (>1% is poor). Filter out high congestion/low HO success if requested.

**FOR DOCUMENTATION QUESTIONS (parameters, KPIs, counters, features, alarms):**

**If Parameters are requested:**

**Parameters:**

| Type | Parameter Name | Description | Relationship to [Topic] |
|------|----------------|-------------|-------------------------|
| Parameter | EXACT_NAME | Brief description (1 sentence) | Brief impact (1 sentence) |
| Parameter | EXACT_NAME | Brief description (1 sentence) | Brief impact (1 sentence) |
| Parameter | EXACT_NAME | Brief description (1 sentence) | Brief impact (1 sentence) |

**If KPIs are requested:**

**KPIs:**

| Type | KPI Name | Description | Relationship to [Topic] |
|------|----------|-------------|-------------------------|
| KPI | EXACT_NAME | Brief description (1 sentence) | Brief impact (1 sentence) |
| KPI | EXACT_NAME | Brief description (1 sentence) | Brief impact (1 sentence) |
| KPI | EXACT_NAME | Brief description (1 sentence) | Brief impact (1 sentence) |

**If Counters are requested:**

**Counters:**

| Type | Counter Name | Description | Relationship to [Topic] |
|------|--------------|-------------|-------------------------|
| Counter | EXACT_NAME | Brief description (1 sentence) | Brief impact (1 sentence) |
| Counter | EXACT_NAME | Brief description (1 sentence) | Brief impact (1 sentence) |

**If Features are requested:**

**Features:**

| Type | Feature Name | Description | Relationship to [Topic] |
|------|--------------|-------------|-------------------------|
| Feature | EXACT_NAME | Brief description (1 sentence) | Brief impact (1 sentence) |
| Feature | EXACT_NAME | Brief description (1 sentence) | Brief impact (1 sentence) |

**If Alarms are requested:**

**Alarms:**

| Type | Alarm Name | Description | Relationship to [Topic] |
|------|------------|-------------|-------------------------|
| Alarm | EXACT_NAME | Brief description (1 sentence) | Brief impact (1 sentence) |
| Alarm | EXACT_NAME | Brief description (1 sentence) | Brief impact (1 sentence) |

**For Comparison Questions (Good vs Bad):**

**Comparison: Good vs Bad Cell/Site**

| Type | Item Name | Good Cell/Site | Bad Cell/Site | Impact |
|------|-----------|----------------|---------------|--------|
| Parameter | NAME | Value | Value | Explanation |
| KPI | NAME | Value | Value | Explanation |

**STEP 3: ADD REASONING SECTION (ONLY FOR DOCUMENTATION QUESTIONS)**

**For Cell Questions:** Skip reasoning section or provide brief cell analysis

**For Documentation Questions:**

**Reasoning and Relationships**

1. **Parameters:**
   - **Parameter 1 (EXACT_NAME)**: [1-2 sentences explaining impact and technical significance]
   - **Parameter 2 (EXACT_NAME)**: [1-2 sentences explaining impact and technical significance]

2. **KPIs:**
   - **KPI 1 (EXACT_NAME)**: [1-2 sentences explaining what it measures and why it matters]
   - **KPI 2 (EXACT_NAME)**: [1-2 sentences explaining what it measures and why it matters]

3. **Counters:** (if applicable)
   - **Counter 1 (EXACT_NAME)**: [Explanation]

4. **Features:** (if applicable)
   - **Feature 1 (EXACT_NAME)**: [Explanation]

**CRITICAL RULES:**

1. **DETECT QUESTION TYPE FIRST**
   - Cell questions: "top cells", "cell performance", "best cells", "worst cells", "cell ranking", "cells performing good"
     → Create CELL RANKING table, NOT parameter/KPI tables
   - Documentation questions: "parameters", "KPIs", "counters", "features", "alarms"
     → Create DOCUMENTATION tables with Parameter 1, KPI 1, etc.

2. **DIFFERENT QUESTIONS = DIFFERENT ANSWERS**
   - "parameters and KPIs for DL Spectral Efficiency" → Extract items related to Spectral Efficiency
   - "counters and KPIs for DL Throughput" → Extract items related to Throughput (DIFFERENT from above)
   - "top 10 cells for Voice DCR" → Extract CELL data, NOT parameters/KPIs

3. **TYPE COLUMN FORMAT**
   - First column is "Type" (not "#")
   - For Parameters table: Every row has "Parameter" in Type column
   - For KPIs table: Every row has "KPI" in Type column
   - For Counters table: Every row has "Counter" in Type column
   - For Features table: Every row has "Feature" in Type column
   - For Alarms table: Every row has "Alarm" in Type column
   - For Cells: Use "Rank" column with numbers 1, 2, 3...

4. **CREATE SEPARATE TABLES** - One table per type (Parameters, KPIs, Counters, etc.)

5. **EXTRACT ONLY WHAT'S REQUESTED**
   - If question asks for "parameters and KPIs" → Create 2 tables (Parameters + KPIs)
   - If question asks for "top 10 cells" → Create 1 CELL table, NO parameter/KPI tables
   - If question asks for "counters and KPIs" → Create 2 tables (Counters + KPIs)

6. **FOCUS ON THE SPECIFIC METRIC**
   - DL Spectral Efficiency → Extract items affecting spectral efficiency
   - DL Throughput → Extract items affecting throughput
   - Voice DCR → Extract items affecting voice drop call rate
   - Each metric has DIFFERENT relevant parameters/KPIs

7. **TABLE FORMATTING**
   - Use EXACT names from context (preserve capitalization/underscores)
   - SHORT descriptions (1 sentence, max 15 words)
   - **NO markdown formatting (**, __, *, _) inside table cells**
   - Plain text only in all cells

8. **CONTEXT-BASED EXTRACTION**
   - Extract 5-15 items per type if available
   - If "top 10" is specified, extract exactly 10
   - Do NOT invent information—use ONLY what's in context
   - If a type has no relevant data, state "No [type] data found in context"

**EXAMPLE RESPONSES:**

**Question 1:** "parameters and KPIs for DL Spectral Efficiency"
**Response:** 2 tables (Parameters table with Type="Parameter" in each row + KPIs table with Type="KPI" in each row)

**Question 2:** "top 10 cells performing good in Voice DCR"
**Response:** 1 CELL RANKING table with Rank 1-10, Cell IDs, Voice DCR values (NO parameter/KPI tables)

**Question 3:** "counters and KPIs impacting DL Throughput"  
**Response:** 2 tables (Counters table with Type="Counter" + KPIs table with Type="KPI")

**Source:** Collection: {settings.vectorstore.collection_name}

**REFERENCE KNOWLEDGE (Basic RAN Parameters & KPIs):**
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
            
            # Post-process: Remove markdown formatting from table cells
            # This ensures no ** ** or other markdown appears in table values
            lines = response.split('\n')
            cleaned_lines = []
            in_table = False
            
            for line in lines:
                # Detect if we're in a table (lines with |)
                if '|' in line and not line.strip().startswith('**'):
                    in_table = True
                    # Remove markdown formatting only from table cells
                    # Remove ** bold **
                    line = re.sub(r'\*\*([^*]+)\*\*', r'\1', line)
                    # Remove __ bold __
                    line = re.sub(r'__([^_]+)__', r'\1', line)
                    # Remove * italic *
                    line = re.sub(r'(?<!\*)\*(?!\*)([^*]+)\*(?!\*)', r'\1', line)
                    # Remove _ italic _
                    line = re.sub(r'(?<!_)_(?!_)([^_]+)_(?!_)', r'\1', line)
                elif not '|' in line:
                    in_table = False
                
                cleaned_lines.append(line)
            
            response = '\n'.join(cleaned_lines)
        
        return jsonify({"response": response})
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Full traceback: {error_details}")
        
        # Return user-friendly error message
        return jsonify({
            "error": "An error occurred while processing your request",
            "message": str(e),
            "response": "I encountered an error processing your query. Please try again or rephrase your question."
        }), 500

@app.route("/chat_tabular", methods=["POST"])
def chat_tabular():
    """
    Enhanced endpoint that returns structured JSON for tabular data queries.
    Parses markdown tables and returns them separately from text content.
    
    Returns:
        {
            "response": "text content without tables",
            "tables": [{"headers": [...], "rows": [[...]]}],
            "has_tables": boolean
        }
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

        # Enhance query for better retrieval accuracy
        query_lower = query.lower()
        enhanced_query = query
        
        # Specific keywords expansion for Voice DCR
        if 'dcr' in query_lower or 'dropped call' in query_lower or 'voice' in query_lower:
            enhanced_query += " \"Dropped Calls\" \"Total Call Attempts\" \"Handover Success Rate\" \"Congestion Rate\" \"NG_FLOW_SETUP_SUCC_TOT_5QI1\" \"NG_FLOW_REL_ABNORMAL_5QI1\""
            # Also add cell-specific terms
            enhanced_query += " \"Cell\" \"Site\" \"cellName\" \"Cell ID\" \"Performance\" \"Ranking\""
            logger.info(f"Voice DCR query detected. Enhanced query: {enhanced_query}")

        # Specific keywords expansion for Bad/Poor Performance Analysis
        if any(term in query_lower for term in ['bad', 'poor', 'low', 'high', 'worst', 'degraded']):
            if 'throughput' in query_lower:
                enhanced_query += " \"PDCP_SDU_VOL_DL\" \"PRB_USED_PDSCH\" \"L.Thomp.DL\" \"User Throughput\""
            if 'ho' in query_lower or 'handover' in query_lower:
                enhanced_query += " \"Handover Success Rate\" \"L.HHO.SuccRate\" \"HO Failure\""
            if 'congestion' in query_lower:
                enhanced_query += " \"Congestion Rate\" \"L.Cell.Unavail.Dur\" \"RRC Congestion\""
            if 'spectral efficiency' in query_lower or 'se' in query_lower:
                enhanced_query += " \"Spectral Efficiency\" \"NR_5108e\" \"NR_571a\" \"L.ChMeas.PRB.DL.Usage\""
            if 'rach' in query_lower:
                enhanced_query += " \"RACH Setup Success Rate\" \"L.RA.SuccRate\" \"Preamble\""
            
            logger.info(f"Bad Performance query detected. Enhanced query: {enhanced_query}")
            
        # Specific keywords expansion for Hierarchy/ID Search (Exact IDs)
        id_pattern = re.compile(r'[A-Z0-9]{8,20}')
        id_matches = id_pattern.findall(query.upper())
        added_ids = set()
        for match in id_matches:
            if len(match) >= 8 and match not in added_ids:
                enhanced_query += f' "{match}"'
                added_ids.add(match)
                # If it looks like a cell name with prefix (11 chars), also add the site id (last 10 chars, then take first 8 etc)
                if len(match) >= 11:
                    site_candidate = match[1:9] # Common Nokia pattern
                    enhanced_query += f' "{site_candidate}"'
        
        if added_ids:
            logger.info(f"Detected potential IDs: {added_ids}. Enhanced query for exact matching.")

        # Use Hybrid Search (Semantic + Keyword) for better exact ID matching (e.g. B9BH0003A21)
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
        
        search_results = []
        if results:
            pairs = [[query, res.payload.get("text") or res.payload.get("content") or str(res.payload)] for res in results]
            scores = reranker.predict(pairs)
            scored_results = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)
            
            # Take Top 20 results
            search_results = [res for score, res in scored_results[:20]]
            
            # Log top scores for debugging
            logger.info(f"Top 5 reranking scores: {[float(score) for score, _ in scored_results[:5]]}")
            logger.info(f"Using top {len(search_results)} documents after reranking")
        
        # Format context
        contexts = []
        for idx, res in enumerate(search_results, 1):
            content = res.payload.get("text") or res.payload.get("content") or ""
            metadata = {}
            for key in ["type", "category", "source"]:
                if key in res.payload:
                    metadata[key.capitalize()] = res.payload[key]
            
            context_block = f"[Document {idx}]"
            if metadata:
                context_block += "\n" + " | ".join([f"{k}: {v}" for k, v in metadata.items()])
            context_block += f"\n{content}"
            contexts.append(context_block)
        
        formatted_context = "\n\n---\n\n".join(contexts)
        
        # Log context preview for debugging
        logger.info(f"Context length: {len(formatted_context)} characters")
        logger.info(f"Context preview (first 500 chars): {formatted_context[:500]}")
        
        # Generate response
        system_instruction = f"""
You are a Telecom Performance Expert specializing in LTE/5G NR (NSA & SA).

**TASK:** Extract KPIs and Parameters from context and present them in separate tables with reasoning.

**REFERENCE FORMULAS (Use these exact formulas if the KPI is requested):**
1. **Combined RACH setup SR**: `100 * (Num of RACH setup completions for contention-based preambles + Num of RACH setup completions for dedicated preambles) / (RA setup att for contention-based preambles + RA setup att for dedicated preambles)`
2. **Handover Success Ratio**: `(intra_du_ho_att_nsa_mnb + intra_du_ho_att_nsa_srb3 - intra_du_ho_fai_t304_exp_nsa - intra_du_ho_fai_tdc_exp - intra_du_hopre_fai_menb_ref) / (intra_du_ho_att_nsa_mnb + intra_du_ho_att_nsa_srb3 + intra_du_hopre_e1_fai_nsa + intra_du_hopre_f1_fai_nsa + intra_du_hopre_fai_res_nsa)`
3. **Voice DCR**: `DCR = (Dropped Calls / Total Call Attempts) * 100`

**FOR MULTI-METRIC PERFORMANCE RANKING (DL Tput, HO, Congestion, SE, RACH):**

If the user asks for "Top 10" comparison based on DL Throughput, HO Success, Congestion, Spectral Efficiency, AND RACH Setup:

**Format Requirement:**
| Rank | Cell ID/Name | DL Throughput (Mbps) | HO Success Rate (%) | Congestion (%) | Spectral Efficiency (bps/Hz) | RACH Setup Success Rate (%) | Performance Status | Notes |
|------|--------------|----------------------|---------------------|----------------|------------------------------|-----------------------------|--------------------|-------|
| 1 | [Cell ID] | [Val] | [Val] | [Val] | [Val] | [Val] | [Status] | [Note] |

- **Good Cell Criteria**: High DL Tput, High HO Succ, Low Congestion, High SE, High RACH.
- **Bad Cell Criteria**: Low DL Tput, Low HO Succ, High Congestion, Low SE, Low RACH.

**MANDATORY PARAMETER EXTRACTION:**
When listing any KPI with a formula, you MUST:
1. Identify ALL parameters/variables in the formula
2. Add each parameter as a SEPARATE entry in the Parameters table
3. This is REQUIRED for every KPI with a formula

**TYPE CLASSIFICATIONS:**
- **PARAMETER**: Configuration values, input variables (e.g., PRB_USED_PDSCH, PDCP_SDU_VOL_DL)
- **KPI**: Calculated performance metrics (e.g., NR_5108e, NR_571a)
- **COUNTER**: Raw measurements
- **FEATURE**: Network features (e.g., Carrier Aggregation, MIMO)
- **ALARM**: Alerts/Alarms

**OUTPUT STRUCTURE:**

Start with: "Based on the retrieved context, here is a comprehensive analysis:"

**1. Identifiers Table (IF requested):**

### Identifiers Related to [Topic]:

| # | Name | Description | Example Value | Category |
|---|------|-------------|---------------|----------|
| 1 | Cell Name | Logical name of the cell | ATL00123A11 | Identifier |
| 2 | Cell ID | Unique ID for a cell | 310-410-12345 | Identifier |

**2. KPIs Table:**

**REFERENCE KNOWLEDGE (Basic RAN Parameters & KPIs):**
{KPI_REFERENCE_TEXT}

### KPIs Related to [Topic]:

| # | Name | Description | Formula/Calculation | Relationship/Impact |
|---|------|-------------|---------------------|---------------------|
| 1 | NR_5108e | 5G DL spectral efficiency | (PDCP_SDU_VOL_DL + PDCP_PDU_X2_DL_SCG) * 8 / PRB_USED_PDSCH / 180 | Primary DL spectral efficiency KPI |
| 2 | NR_571a | 5G DL spectral efficiency (RLC) | SUM(DL_RLC_VOL_RX_L_QOS_GRP_01..20) / (180 * PRB_USED_PDSCH) | RLC layer efficiency measurement |

**2. Parameters Table:**

### Parameters Related to [Topic]:

| # | Name | Description | Formula/Calculation | Relationship/Impact |
|---|------|-------------|---------------------|---------------------|
| 1 | PRB_USED_PDSCH | Physical Resource Blocks for PDSCH | N/A | Measures resource utilization for spectral efficiency |
| 2 | PDCP_SDU_VOL_DL | PDCP SDU volume in DL | N/A | Payload data volume for efficiency calculation |

**3. Additional Tables (if applicable):**
- ### Counters Related to [Topic]:
- ### Features Related to [Topic]:
- ### Alarms Related to [Topic]:

**CRITICAL FORMATTING RULES:**
🚫 NO markdown formatting (**, __, *, _) in table cells
🚫 NO combining KPIs and Parameters in one table
✅ Separate tables with separate headers
✅ Number rows starting from 1 in each table
✅ Plain text only in all cells

**POST-TABLE SECTIONS (Required):**

---

### Reasoning and Relationships:

**KPIs:**
- [KPI Name]: Explain significance and measurement purpose

**Parameters:**
- [Parameter Name]: Explain impact and role in calculations

---

### Technical Details:
[Technical context, formulas, calculations, units, thresholds]

---

### Additional Context:
[Vendor notes, technology specifics, related information]

---

### Constraints and Notes:
[Limitations, missing data, special considerations]

---

**Source:** Collection: {settings.vectorstore.collection_name} | Distance Metric: cosine | Search: Hybrid RRF + Reranking
"""

        # Detect query type to determine appropriate response format
        query_lower = query.lower()
        is_cell_query = any(word in query_lower for word in ['cell', 'site', 'enodeb', 'gnodeb', 'top', 'worst', 'best'])
        
        # Use structured output - OpenAI will handle JSON parsing automatically
        # The with_structured_output() method uses function calling to ensure valid JSON
        
        # Dynamic prompt based on query type
        if is_cell_query and ('top' in query_lower or 'best' in query_lower or 'worst' in query_lower or 'performing' in query_lower):
            # Cell performance query
            simplified_prompt = ChatPromptTemplate.from_template("""
You are a Telecom Performance Expert. Extract cell performance data or KPIs from context.

You must return a structured response with the following fields:
- kpis: List of KPI items (each has number, name, description, formula_calculation, relationship_impact)
- parameters: List of Parameter items (same structure as KPIs)
- reasoning_kpis: Text explanation of KPIs
- reasoning_parameters: Text explanation of parameters
- technical_details: Technical context and formulas
- additional_context: Additional notes or vendor specifics
- constraints_notes: Limitations or considerations

USER QUESTION: {input}
CONTEXT: {context}

IMPORTANT INSTRUCTIONS FOR CELL DATA:
1. If context contains a list of cells/sites, put them in the 'kpis' array.
2. **CELL NAME FORMATTING**: Ensure cell names are extracted exactly as they appear (e.g., JDA03866A11, JDAYD604A71). 
   - Identify the cell name by patterns like 3 letters + 5 numbers + 1 letter + 2 numbers (e.g., JDA03866A11).
   - DO NOT truncate or alter the cell identifiers.
3. Field mapping for 'kpis' array items:
   - number: Rank (1, 2, 3...)
   - name: Cell Name (e.g., JDA03866A11)
   - description: Metric Value (e.g., "98.5%") or Status
   - formula_calculation: "N/A"
   - relationship_impact: Performance Note (e.g., "Top performer", "High DCR")

**SPECIFIC ANALYSIS FOR VOICE DCR (Dropped Call Rate):**
If the query involves "Voice DCR" or "Dropped Call Rate":
1. **Goal**: Rank cells by **LOWEST** DCR (Ascending order) for "good/top" performance.
2. **Calculation**: `DCR = (Dropped Calls / Total Call Attempts) * 100`.
3. **Field Mapping for 'kpis' array**:
   - **name**: Cell Name/ID (e.g., JDA03866A11) - MUST extract from context
   - **description**: "DCR: [Value]% | Dropped: [Cnt] | Attempts: [Cnt]"
   - **formula_calculation**: "HO Success: [Val]% | Congestion: [Val]%" (If available)
   - **relationship_impact**: Actionable Insight (e.g., "Check HO failure", "Congestion high")
4. **MANDATORY OUTPUT REQUIREMENTS**:
   - **technical_details**: MUST include the DCR formula: "DCR = (Dropped Calls / Total Call Attempts) * 100"
   - **reasoning_kpis**: MUST explain what DCR means, why these cells perform well/poorly, and analysis of the data
   - **reasoning_parameters**: MUST explain the impact of parameters (HO Success, Congestion, etc.) on DCR
5. **Exclusions**: IGNORE unrelated features (ANR, CA, RAT1) unless directly affecting DCR.
6. **Threshold**: Use `Percentile_Flag` (>1% is poor). Filter out high congestion/low HO success if requested.
7. **VoNR Counters**: Look for `NG_FLOW_SETUP_SUCC_TOT_5QI1` (Success) and `NG_FLOW_REL_ABNORMAL_5QI1` (Abnormal Release) to validate DCR content.

Otherwise: extract KPIs/parameters normally.

IMPORTANT: 
- ALL fields are REQUIRED. Use "N/A" or "Not applicable" for empty fields.
- For empty lists, use empty array []
- No markdown formatting in any field values
""")


        elif (all(x in query_lower for x in ["siteid", "sector", "cellname"])) or \
             (all(x in query_lower for x in ["siteid", "sector", "cell name"])) or \
             (query_lower.strip() == "lte" or ( "lte" in query_lower and "siteid" in query_lower)):
            # HIERARCHY / LTE TOPOLOGY QUERY
            simplified_prompt = ChatPromptTemplate.from_template("""
You are a Telecom Performance Expert. Extract the Site-Sector-Cell hierarchy from the context.

You must return a JSON with:
- hierarchy: Array of objects, each with: site_id (string), sector_name (string), cell_name (string)
- reasoning_parameters: Analysis of the site configuration and technical layers (A, J, K, B, D, L, E, Z etc.)
- technical_details: Summary of the technology (LTE/5G) and carrier architecture

CONTEXT: {context}
USER QUESTION: {input}

RULES:
1. **HIERARCHY EXTRACTION**: Extract rows EXACTLY as site-sector-cell mappings.
2. **NOKIA RAN NAMING LOGIC (STRICT STRING SLICING)**:
   - **cellName** (e.g., BDA01442D41): The full 11-character unique cell identifier.
   - **Sector Name**: The **EXACT last 10 characters** of the cellName (e.g., DA01442D41). Always 10 chars.
   - **SITEID**: The **EXACT first 8 characters of the Sector Name** (e.g., DA01442D). Always 8 chars. **NEVER include the first letter prefix (A/B/J/K) in SITEID.**
   - **Prefix Decoders**:
     - **A**: LTE Primary Carrier / Mid Band
     - **B**: LTE 700MHz / Low Band layer
     - **J**: LTE Secondary / High Band layer
     - **K**: LAA / Additional Capacity layer
3. If context contains "BDA01442D41", map it to SITEID=DA01442D, Sector=DA01442D41, Cell=BDA01442D41.
4. If context contains "JA1O0236A31", map it to SITEID=A1O0236A, Sector=A1O0236A31, Cell=JA1O0236A31.
5. Every cell MUST have its associated SITEID and Sector Name columns.
6. **FORMAT**: Ensure the 3-column mapping (SITEID, Sector Name, cellName) is strictly followed in the output table.
7. Use "reasoning_parameters" to explain the layer (Prefix) and "technical_details" for the Site configuration.

EXAMPLE JSON:
{{
  "hierarchy": [
    {{ "site_id": "DA01442D", "sector_name": "DA01442D41", "cell_name": "BDA01442D41" }},
    {{ "site_id": "A1O0236A", "sector_name": "A1O0236A31", "cell_name": "JA1O0236A31" }}
  ],
  "reasoning_parameters": "Site has 700MHz (B) and High-Band (J) layers.",
  "technical_details": "Nokia LTE technology with multi-layer architecture."
}}
""")
        elif "what is" in query_lower and ("kpi" in query_lower or "parameter" in query_lower) and len(query.split()) < 10:
            # GENERAL DEFINITION QUERY
            simplified_prompt = ChatPromptTemplate.from_template("""
You are a Telecom Performance Expert. Define the concept clearly.

User Question: {input}

Your goal is to explain the CONCEPT of what a KPI or Parameter IS in the context of telecommunications.
DO NOT output a table of random examples unless specifically asked for "examples".

Return a JSON with:
- kpis: [] (Empty list, do not extract random examples)
- parameters: [] (Empty list)
- reasoning_kpis: **DEFINITION**: [Provide a clear, high-quality definition of what a Key Performance Indicator is in telecom]. Explain its purpose (monitoring, optimization, benchmarking).
- reasoning_parameters: **DEFINITION**: [Provide a clear definition of what a Parameter is]. Explain how it differs from a KPI (configuration vs measurement).
- technical_details: Explain the relationship (KPIs are often calculated using counters and influenced by parameters).
- additional_context: "This is a general definition. If you want specific KPIs for a technology (e.g., '5G KPIs'), please specify the topic."
- constraints_notes: "N/A"
""")
        else:
            # KPI/Parameter documentation query
            simplified_prompt = ChatPromptTemplate.from_template("""
You are a Telecom Performance Expert. Extract KPIs and Parameters from context.

You must return a structured response with these fields:
- kpis: Array of KPI objects, each with: number (row number as integer: 1, 2, 3...), name (string), description (string), formula_calculation (string), relationship_impact (string)
- parameters: Array of Parameter objects with same structure
- identifiers: Array of Identifier objects with same structure (Cell Name, Cell ID, etc.)
- reasoning_kpis: String explaining the KPIs
- reasoning_parameters: String explaining the parameters
- technical_details: String with technical context and formulas
- additional_context: String with additional notes
- constraints_notes: String with limitations or considerations

CONTEXT: {context}
USER QUESTION: {input}

RULES:
1. **STRICT RELEVANCE**: Extract ONLY items DIRECTLY related to the user's query.
2. **CLASSIFICATION RULES**:
   - **IDENTIFIERS**: Put "Cell Name", "Cell ID", "Site ID", "ECGI", "gNB", "eNB", "TAC", "PCI" and similar ID fields into the `identifiers` list. **DO NOT** put them in `parameters`.
   - **KPIs**: Put calculated metrics (Success Rates, Drop Rates, Throughput) in `kpis`.
   - **PARAMETERS**: Put configuration/input variables in `parameters`.
3. **NAMING CONVENTION ANALYSIS (Nokia/RAN)**:
   - **SITEID** (8 chars, e.g. 9BH0003A): Physical Site.
   - **Sector Name** (10 chars, e.g. 9BH0003A21):
     - 9th digit (1,2,3) -> Sector (1=Alpha, 2=Beta, 3=Gamma).
     - 10th digit (1,2) -> Sub-sector / Beam.
   - **Cell Name** (Prefix + Sector, e.g. A9BH0003A21):
     - **Prefixes**:
       - **A**: LTE low/mid band primary carrier.
       - **J**: LTE mid/high band secondary carrier.
       - **K**: Additional LTE carrier / LAA / CBRS / refarmed layer.
       - **B/D/L/E/Z**: Technology/Band specific (LTE).
     - Use this logic to populate "Description" (e.g. "Sector 2, Layer A (Primary Carrier)").
4. **NO GENERIC FILLER**: If context lacks specific items, return empty arrays.
5. Extract parameters from KPI formulas where applicable.
6. **JSON ONLY**: Return strictly valid JSON.

EXAMPLE CORRECT FORMAT:
{{
  "kpis": [],
  "parameters": [],
  "identifiers": [
    {{"number": 1, "name": "Site ID", "description": "Physical Site", "formula_calculation": "9BH0003A", "relationship_impact": "Site Level"}},
    {{"number": 2, "name": "Sector Name", "description": "Sector 2, Sub-sector 1", "formula_calculation": "9BH0003A21", "relationship_impact": "Sector Level"}},
    {{"number": 3, "name": "Cell Name", "description": "Sector 2, Layer A (Primary)", "formula_calculation": "A9BH0003A21", "relationship_impact": "LTE Cell Layer"}},
    {{"number": 4, "name": "Cell Name", "description": "Sector 2, Layer J (Secondary)", "formula_calculation": "J9BH0003A21", "relationship_impact": "LTE Cell Layer"}},
    {{"number": 5, "name": "Cell Name", "description": "Sector 2, Layer K (LAA/Refarmed)", "formula_calculation": "K9BH0003A21", "relationship_impact": "LTE Cell Layer"}}
  ],
  "reasoning_kpis": "...",
  "reasoning_parameters": "...",
  "technical_details": "...",
  "additional_context": "...",
  "constraints_notes": "..."
}}
""")
        
        # Check for Multi-Metric Query (specific case requested by user)
        # "DL throughput is high , HO success rate is high ,congestion low ,High Spectral Efficiency RACH setup success rate high"
        is_multi_metric_query = False
        if is_cell_query and 'throughput' in query_lower and 'ho' in query_lower and 'congestion' in query_lower:
            is_multi_metric_query = True
            simplified_prompt = ChatPromptTemplate.from_template("""
You are a Telecom Performance Expert. Analyze cells based on 5 parameters: DL Throughput, HO Success Rate, Congestion, Spectral Efficiency, and RACH Setup Success Rate.

You must return a JSON object with:
- good_cells: List of objects (rank, cell_id, dl_throughput, ho_success_rate, congestion, spectral_efficiency, rach_setup_sr, status, notes)
- bad_cells: List of objects (same fields)
- technical_details: String
- additional_context: String
- constraints_notes: String

CONTEXT: {context}
USER QUESTION: {input}

RULES:
1. Extract cells and values EXACTLY from context.
2. **Format Requirement**:
   - rank: integer (1, 2, 3...)
   - cell_id: string (e.g., JDA03866A11)
   - dl_throughput: number/string (e.g. 150.2)
   - ho_success_rate: number/string (e.g. 98.5)
   - congestion: number/string (e.g. 0.5)
   - spectral_efficiency: number/string (e.g. 7.8)
   - rach_setup_sr: number/string (e.g. 99.2)
   - status: string ("Excellent", "Good", "Bad", "Poor")
   - notes: string (e.g., "High DL throughput")

3. If the user asks for "Good" cells (High Tput, Low Congestion), put them in 'good_cells'.
4. If the user asks for "Bad" cells (Low Tput, High Congestion), put them in 'bad_cells'.
5. Return ONLY valid JSON.

EXAMPLE JSON:
{{
  "good_cells": [
    {{ "rank": 1, "cell_id": "ATL001", "dl_throughput": "150.2", "ho_success_rate": "98.5", "congestion": "0.5", "spectral_efficiency": "7.8", "rach_setup_sr": "99.2", "status": "Excellent", "notes": "Top performer" }}
  ],
  "bad_cells": [],
  "technical_details": "Analysis based on 5 metrics",
  "additional_context": "None",
  "constraints_notes": "None"
}}
""")
        
        def clean_markdown_formatting(text: str) -> str:
            """Remove markdown formatting symbols from text"""
            if not text:
                return text
            # Remove bold (**text** or __text__)
            text = text.replace('**', '').replace('__', '')
            # Remove italic (*text* or _text_) - be careful with underscores in names
            # Only remove standalone formatting, not underscores in variable names
            # Remove italic markers at word boundaries
            text = re.sub(r'\*([^*]+)\*', r'\1', text)
            # Remove other common markdown
            text = text.replace('`', '')
            return text.strip()
        
        def extract_json_from_markdown(text: str) -> str:
            """Extract JSON from markdown code blocks if present."""
            # Try to find JSON within markdown code blocks
            json_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
            match = re.search(json_block_pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()
            return text.strip()
        
        # Use standard LLM output and parse manually
        # This gives us more control over handling markdown-wrapped JSON
        chain = simplified_prompt | llm | StrOutputParser()
        
        llm_response = chain.invoke({
            "context": formatted_context,
            "input": query
        })
        
        logger.info(f"LLM Response (first 200 chars): {llm_response[:200]}")
        
        # Extract JSON from markdown if present
        json_str = extract_json_from_markdown(llm_response)
        logger.info(f"Extracted JSON (first 200 chars): {json_str[:200]}")
        
        # Robust Multi-Block JSON Parsing
        json_data = {}
        
        # Strategy 1: regex for code blocks
        code_blocks = re.findall(r'```(?:json)?\s*\n?(.*?)\n?```', llm_response, re.DOTALL)
        
        if code_blocks:
            logger.info(f"Found {len(code_blocks)} code blocks")
            
            extracted_lists = []
            extracted_dicts = []
            
            for block in code_blocks:
                try:
                    parsed = json.loads(block.strip())
                    if isinstance(parsed, list):
                        extracted_lists.append(parsed)
                    elif isinstance(parsed, dict):
                        extracted_dicts.append(parsed)
                except:
                    continue
            
            # Merge findings
            # If we found a dict, it's likely the main structure
            for d in extracted_dicts:
                json_data.update(d)
                
            # If we found lists, try to categorize them if not already in dict
            if not json_data.get('kpis') and not json_data.get('parameters'):
                # Heuristic: First list is often KPIs, second Parameters (based on prompt order)
                # Or check content
                for lst in extracted_lists:
                    if not lst: continue
                    first_item = lst[0]
                    # Check if it looks like a parameter or KPI (often hard to tell, but we can default)
                    # For now, if we have 2 lists and empty json_data, map 1st->KPIs, 2nd->Params
                    if not json_data.get('kpis'):
                        json_data['kpis'] = lst
                    elif not json_data.get('parameters'):
                        json_data['parameters'] = lst
        
        else:
            # Strategy 2: Try finding valid JSON substrings if no code blocks
            try:
                # Find outermost brackets
                start_brace = llm_response.find('{')
                start_bracket = llm_response.find('[')
                
                if start_brace != -1:
                    end_brace = llm_response.rfind('}') + 1
                    json_data = json.loads(llm_response[start_brace:end_brace])
                elif start_bracket != -1:
                    end_bracket = llm_response.rfind(']') + 1
                    # It's a list, probably just KPIs
                    json_data['kpis'] = json.loads(llm_response[start_bracket:end_bracket])
            except:
                 pass

        # Strategy 3: Extract Text Fields (Reasoning, Context) using Regex
        # The LLM often puts these outside JSON. We need to grab them.
        
        text_fields = {
            'reasoning_kpis': r'(?:\*\*|#|\b)Reasoning (?:for|of) KPIs(?:\*\*|:)?\s*(.*?)(?=(?:\*\*|#|\b)Reasoning|\Z)',
            'reasoning_parameters': r'(?:\*\*|#|\b)Reasoning (?:for|of) Parameters(?:\*\*|:)?\s*(.*?)(?=(?:\*\*|#|\b)Technical|\Z)',
            'technical_details': r'(?:\*\*|#|\b)Technical Details(?:\*\*|:)?\s*(.*?)(?=(?:\*\*|#|\b)Additional|\Z)',
            'additional_context': r'(?:\*\*|#|\b)Additional Context(?:\*\*|:)?\s*(.*?)(?=(?:\*\*|#|\b)Constraints|\Z)',
            'constraints_notes': r'(?:\*\*|#|\b)Constraints(?:\*\*|:)?\s*(.*?)(?=$)',
        }
        
        for field, pattern in text_fields.items():
            if not json_data.get(field) or json_data.get(field) == "Not applicable":
                match = re.search(pattern, llm_response, re.DOTALL | re.IGNORECASE)
                if match:
                    json_data[field] = match.group(1).strip()
        
        # Final Verification
        try:
             # Ensure lists exist
            if 'kpis' not in json_data: json_data['kpis'] = []
            if 'parameters' not in json_data: json_data['parameters'] = []
            
            structured_result = SeparateTablesResponse(**json_data)
            logger.info(f"Successfully constructed structured response from fragments")
            
        except Exception as e:
            logger.error(f"Reasoning extraction/validation failed: {e}")
            problematic_content = llm_response[:500] if llm_response else 'Empty String'
            
            # Write to file for debugging
            try:
                with open('json_error_log.txt', 'w', encoding='utf-8') as f:
                    f.write(f"Error: {e}\n")
                    f.write(f"Raw LLM Response: {llm_response}\n")
            except:
                pass
            
            # Fallback
            structured_result = SeparateTablesResponse(
                kpis=[],
                parameters=[],
                reasoning_kpis=f"Error parsing response. See logs.",
                reasoning_parameters="...",
                technical_details="...",
                additional_context="...",
                constraints_notes=str(e)
            )
        
        if is_multi_metric_query:
            # specialized parsing to TableData
            try:
                # Convert extracted JSON lists to TableData
                if 'good_cells' in json_data and json_data['good_cells']:
                    good_table = TableData(
                        title="Top Performing Cells (Good Performance)",
                        headers=["Rank", "Cell ID", "DL Throughput (Mbps)", "HO Success Rate (%)", "Congestion (%)", "Spectral Efficiency (bps/Hz)", "RACH Setup Success Rate (%)", "Status", "Notes"],
                        rows=[[
                            str(item.get('rank', i+1)),
                            str(item.get('cell_id', 'N/A')),
                            str(item.get('dl_throughput', 'N/A')),
                            str(item.get('ho_success_rate', 'N/A')),
                            str(item.get('congestion', 'N/A')),
                            str(item.get('spectral_efficiency', 'N/A')),
                            str(item.get('rach_setup_sr', 'N/A')),
                            str(item.get('status', 'Excellent')),
                            str(item.get('notes', ''))
                        ] for i, item in enumerate(json_data['good_cells'])]
                    )
                    tables.append(good_table)
                
                if 'bad_cells' in json_data and json_data['bad_cells']:
                    bad_table = TableData(
                        title="Low Performing Cells (Bad Performance)",
                        headers=["Rank", "Cell ID", "DL Throughput (Mbps)", "HO Success Rate (%)", "Congestion (%)", "Spectral Efficiency (bps/Hz)", "RACH Setup Success Rate (%)", "Status", "Notes"],
                        rows=[[
                            str(item.get('rank', i+1)),
                            str(item.get('cell_id', 'N/A')),
                            str(item.get('dl_throughput', 'N/A')),
                            str(item.get('ho_success_rate', 'N/A')),
                            str(item.get('congestion', 'N/A')),
                            str(item.get('spectral_efficiency', 'N/A')),
                            str(item.get('rach_setup_sr', 'N/A')),
                            str(item.get('status', 'Poor')),
                            str(item.get('notes', ''))
                        ] for i, item in enumerate(json_data['bad_cells'])]
                    )
                    tables.append(bad_table)
                    
                structured_result = SeparateTablesResponse(
                    kpis=[], parameters=[], 
                    technical_details=json_data.get('technical_details', ''),
                    additional_context=json_data.get('additional_context', ''),
                    constraints_notes=json_data.get('constraints_notes', '')
                )
            except Exception as e:
                logger.error(f"Failed to parse multi-metric table: {e}")

        logger.info(f"Structured output received: {len(structured_result.kpis)} KPIs, {len(structured_result.parameters)} Parameters")
        
        # Convert structured output to table format for frontend
        tables = []

        # CORRECTION LOGIC: Move Identifiers from Parameters to Identifiers list if misclassified
        # This fixes the issue where "Cell Name" or "Cell ID" are put in Parameters table
        identifier_keywords = ["Cell Name", "Cell ID", "Site ID", "ECGI", "gNB", "eNB", "TAC", "PCI", "Sector ID", "Sector Name", "Market"]
        
        # We iterate backwards to safely remove items
        if structured_result.parameters:
            if structured_result.identifiers is None:
                structured_result.identifiers = []
                
            items_to_move = []
            for i in range(len(structured_result.parameters) - 1, -1, -1):
                param = structured_result.parameters[i]
                # Checking params
                clean_name = param.name.lower().replace('*', '').replace('_', '').strip()


                # Check if name contains any identifier keyword (case insensitive)
                # We check against cleaned name
                is_match = False
                for kw in identifier_keywords:
                    kw_clean = kw.lower().replace(' ', '')
                    name_clean = clean_name.replace(' ', '')
                    
                    # Exact match or contained
                    if kw_clean == name_clean or kw_clean in name_clean:
                        # Exclude if "count" or "rate" or "throughput" is in the name (to avoid false positives like "Cell ID Counter")
                        if "count" not in clean_name and "rate" not in clean_name and "throughput" not in clean_name:
                            is_match = True
                            with open("debug_correction.log", "a") as f:
                                f.write(f"  MATCHED keyword: '{kw}'\n")
                            break
                
                if is_match:
                    logger.info(f"Moving '{param.name}' to Identifiers")
                    items_to_move.append(structured_result.parameters.pop(i))
            
            # Add to identifiers list (reverse order to maintain original relative order if possible, though exact order less critical)
            for item in reversed(items_to_move):
                # Update category if needed
                if not item.relationship_impact or item.relationship_impact == "Parameter":
                    item.relationship_impact = "Identifier"
                structured_result.identifiers.append(item)
                
        # Create KPIs table
        if structured_result.kpis:
            # Use context-aware title
            if is_cell_query and ('top' in query_lower or 'best' in query_lower or 'worst' in query_lower or 'performing' in query_lower):
                table_title = "Cell Performance Data"
                # Customize headers for cell ranking
                headers = ["Rank", "Cell Name", "DCR Metric (Dropped/Attempts)", "Secondary Metrics (HO/Congestion)", "Insights/Note"]
            else:
                table_title = "KPIs Related to the Query"
                headers = ["#", "Name", "Description", "Formula/Calculation", "Relationship/Impact"]
                
            kpi_table = TableData(
                title=table_title,
                headers=headers,
                rows=[[
                    str(kpi.number),
                    clean_markdown_formatting(kpi.name),
                    clean_markdown_formatting(kpi.description),
                    clean_markdown_formatting(kpi.formula_calculation),
                    clean_markdown_formatting(kpi.relationship_impact)
                ] for kpi in structured_result.kpis]
            )
            tables.append(kpi_table)
            logger.info(f"Created table '{table_title}' with {len(structured_result.kpis)} rows")
        
        # Create Parameters table
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
            logger.info(f"Created Parameters table with {len(structured_result.parameters)} rows")
        
        # Create Counters table if present
        if structured_result.counters:
            counter_table = TableData(
                title="Counters Related to the Query",
                headers=["#", "Name", "Description", "Formula/Calculation", "Relationship/Impact"],
                rows=[[
                    str(counter.number),
                    clean_markdown_formatting(counter.name),
                    clean_markdown_formatting(counter.description),
                    clean_markdown_formatting(counter.formula_calculation),
                    clean_markdown_formatting(counter.relationship_impact)
                ] for counter in structured_result.counters]
            )
            tables.append(counter_table)
            tables.append(counter_table)
            logger.info(f"Created Counters table with {len(structured_result.counters)} rows")

        # Create Identifiers table if present
        if structured_result.identifiers:
            ident_table = TableData(
                title="Identifiers / Network Hierarchy",
                headers=["#", "Identifier Name", "Value / Full Form", "Description", "Level / Notes"],
                rows=[[
                    str(ident.number),
                    clean_markdown_formatting(ident.name),                  # Identifier Name (e.g. Site ID)
                    clean_markdown_formatting(ident.formula_calculation),   # Value / Full Form
                    clean_markdown_formatting(ident.description),           # Description
                    clean_markdown_formatting(ident.relationship_impact)    # Level / Notes
                ] for ident in structured_result.identifiers]
            )
            tables.append(ident_table)
            logger.info(f"Created Identifiers table with {len(structured_result.identifiers)} rows")

        # Create Hierarchy table if present
        if structured_result.hierarchy:
            hier_titles = ["SITEID", "Sector Name", "cellName"]
            hier_table = TableData(
                title="Site-Sector-Cell Hierarchy Mapping",
                headers=hier_titles,
                rows=[[
                    clean_markdown_formatting(row.site_id),
                    clean_markdown_formatting(row.sector_name),
                    clean_markdown_formatting(row.cell_name)
                ] for row in structured_result.hierarchy]
            )
            tables.insert(0, hier_table) # Put it at the top
            logger.info(f"Created Hierarchy table with {len(structured_result.hierarchy)} rows")
        
        # Build response text from structured data
        response_parts = []
        
        # Only add intro if we have tables
        if tables:
            response_parts.append("Based on the retrieved context, here is the analysis:\n")
        
        if structured_result.kpis:
            response_parts.append(f"\n### {len(structured_result.kpis)} KPIs Found (see table above)\n")
        
        if structured_result.parameters:
            response_parts.append(f"\n### {len(structured_result.parameters)} Parameters Found (see table above)\n")
        
        if structured_result.counters:
            response_parts.append(f"\n### {len(structured_result.counters)} Counters Found (see table above)\n")
        
        # Dynamic Text Sections - Only show if meaningful content exists
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
        
        # Create final response
        tabular_response = TabularResponse(
            response=response_text,
            tables=tables,
            has_tables=len(tables) > 0
        )
        
        logger.info(f"Final response: {len(tables)} tables, {len(response_text)} chars of text")
        
        #  Return structured response with separated tables and text
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