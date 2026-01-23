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
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Optional, Union

# Initialize Logger
logger = get_logger("TelicomApp")

# Load settings
settings = get_settings()


# Pydantic models with Pydantic v2 compatibility
class TableRow(BaseModel):
    """Represents a single row in the parameters/KPIs table"""
    number: int = Field(description="Row number starting from 1")
    name: str = Field(description="Exact name from context")
    description: str = Field(description="Brief description (1 sentence, max 15 words)")
    relationship: str = Field(description="Brief impact/relationship (1 sentence, max 15 words)")


class ReasoningItem(BaseModel):
    """Represents reasoning for a parameter or KPI"""
    label: str = Field(description="Label like 'Parameter1', 'KPI1', etc.")
    name: str = Field(description="Actual parameter or KPI name")
    explanation: str = Field(description="Detailed explanation (1-2 sentences)")


class StructuredResponse(BaseModel):
    """Complete structured response with separate tables and reasoning"""
    topic: str = Field(description="The topic being queried")
    parameters: Optional[List[TableRow]] = Field(default=None)
    kpis: Optional[List[TableRow]] = Field(default=None)
    parameter_reasoning: Optional[List[ReasoningItem]] = Field(default=None)
    kpi_reasoning: Optional[List[ReasoningItem]] = Field(default=None)
    constraints_notes: Optional[str] = Field(default=None)


class TableData(BaseModel):
    """Represents a parsed table"""
    title: Optional[str] = Field(default=None)
    headers: List[str]
    rows: List[List[str]]


class TabularResponse(BaseModel):
    """Response model for tabular endpoint"""
    response: str
    tables: Optional[List[TableData]] = Field(default=None)
    has_tables: bool = Field(default=False)


class TableRowItem(BaseModel):
    """Single row in a table"""
    number: Union[int, str]
    name: str
    description: str
    formula_calculation: str = Field(default="N/A")
    relationship_impact: str = Field(default="N/A")

    @field_validator('number', mode='before')
    @classmethod
    def convert_number(cls, v):
        """Convert number to int if it's a string digit"""
        if isinstance(v, str) and v.isdigit():
            return int(v)
        return v


class HierarchyRow(BaseModel):
    """Row for Site-Sector-Cell hierarchy extracted from Qdrant"""
    site_id: str = Field(description="Site ID from Qdrant payload")
    sector_name: str = Field(description="Sector Name from Qdrant payload")
    cell_name: str = Field(description="Cell Name from Qdrant payload")


class SeparateTablesResponse(BaseModel):
    """Response with separate tables for each type"""
    kpis: List[TableRowItem] = Field(default_factory=list)
    parameters: List[TableRowItem] = Field(default_factory=list)
    identifiers: Optional[List[TableRowItem]] = Field(default=None)
    hierarchy: Optional[List[HierarchyRow]] = Field(default=None)
    counters: Optional[List[TableRowItem]] = Field(default=None)
    features: Optional[List[TableRowItem]] = Field(default=None)
    alarms: Optional[List[TableRowItem]] = Field(default=None)
    reasoning_kpis: str = Field(default="Not applicable")
    reasoning_parameters: str = Field(default="Not applicable")
    technical_details: str = Field(default="Not applicable")
    additional_context: str = Field(default="Not applicable")
    constraints_notes: str = Field(default="Not applicable")

    # Pydantic v2: use model_config instead of class Config
    model_config = ConfigDict(extra='allow')


# Utility Functions
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


def extract_identifiers_from_payload(payload: dict) -> dict:
    """
    Extract identifiers (Site ID, Cell ID, Cell Name, Sector Name, etc.) 
    directly from Qdrant payload without any LLM generation.
    
    This ensures all identifiers come from the actual stored data.
    """
    identifiers = {}
    
    # Common field names for identifiers in Qdrant payloads
    id_fields = {
        'site_id': ['site_id', 'siteid', 'SITEID', 'siteId', 'site'],
        'cell_id': ['cell_id', 'cellid', 'CELLID', 'cellId', 'cell_identity'],
        'cell_name': ['cell_name', 'cellname', 'cellName', 'CELLNAME', 'cell'],
        'sector_name': ['sector_name', 'sectorname', 'sectorName', 'SECTORNAME', 'sector'],
        'market': ['market', 'MARKET', 'region', 'REGION'],
        'enb_id': ['enb_id', 'enbid', 'eNB_ID', 'enb'],
        'gnb_id': ['gnb_id', 'gnbid', 'gNB_ID', 'gnb'],
        'pci': ['pci', 'PCI', 'physical_cell_id'],
        'tac': ['tac', 'TAC', 'tracking_area_code'],
        'ecgi': ['ecgi', 'ECGI', 'e_utran_cell_global_id'],
    }
    
    for id_type, field_names in id_fields.items():
        for field in field_names:
            if field in payload and payload[field]:
                identifiers[id_type] = str(payload[field])
                break
    
    return identifiers


def extract_hierarchy_from_results(results: list) -> List[dict]:
    """
    Extract Site-Sector-Cell hierarchy directly from Qdrant results.
    No LLM generation - pure data extraction from payloads.
    """
    hierarchy = []
    seen = set()
    
    for res in results:
        payload = res.payload if hasattr(res, 'payload') else res
        
        ids = extract_identifiers_from_payload(payload)
        
        site_id = ids.get('site_id', '')
        cell_name = ids.get('cell_name', '')
        sector_name = ids.get('sector_name', '')
        
        # Only add if we have at least one identifier
        if site_id or cell_name or sector_name:
            key = f"{site_id}|{sector_name}|{cell_name}"
            if key not in seen:
                seen.add(key)
                hierarchy.append({
                    'site_id': site_id or 'N/A',
                    'sector_name': sector_name or 'N/A',
                    'cell_name': cell_name or 'N/A'
                })
    
    return hierarchy


def extract_metrics_from_payload(payload: dict) -> dict:
    """
    Extract performance metrics directly from Qdrant payload.
    No LLM generation - pure data extraction.
    """
    metrics = {}
    
    # Common metric field names
    metric_fields = {
        'dl_throughput': ['dl_throughput', 'dl_tput', 'downlink_throughput', 'DL_Throughput'],
        'ul_throughput': ['ul_throughput', 'ul_tput', 'uplink_throughput', 'UL_Throughput'],
        'ho_success_rate': ['ho_success_rate', 'handover_success', 'HO_SR', 'ho_sr'],
        'congestion': ['congestion', 'prb_utilization', 'PRB_Util'],
        'spectral_efficiency': ['spectral_efficiency', 'se', 'SE', 'dl_se'],
        'rach_sr': ['rach_sr', 'rach_success_rate', 'RACH_SR'],
        'voice_dcr': ['voice_dcr', 'dcr', 'dropped_call_rate', 'DCR'],
        'latency': ['latency', 'rtt', 'round_trip_time'],
    }
    
    for metric_type, field_names in metric_fields.items():
        for field in field_names:
            if field in payload and payload[field] is not None:
                metrics[metric_type] = payload[field]
                break
    
    return metrics


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
        temperature=0.0,  # Deterministic for formatting
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
    
    if not query:
        return jsonify({"error": "No message provided"}), 400
    
    try:
        # RAG retrieval from Qdrant
        q_vector = embeddings.embed_query(query)
        results = loop.run_until_complete(
            vector_store.search(
                collection_name=settings.vectorstore.collection_name,
                query_vector=q_vector,
                limit=50,
                score_threshold=0.55
            )
        )
        
        logger.info(f"QDRANT RETRIEVAL: Found {len(results) if results else 0} documents")
        
        # If no results, return immediately - NO LLM generation
        if not results or len(results) == 0:
            logger.warning(f"No documents retrieved from Qdrant for query: {query}")
            return jsonify({
                "response": "No relevant information found in the database. The query did not match any documents in the collection."
            })
        
        # Rerank
        search_results = []
        if results:
            pairs = [[query, res.payload.get("text") or res.payload.get("content") or str(res.payload)] for res in results]
            scores = reranker.predict(pairs)
            scored_results = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)
            search_results = [res for _, res in scored_results[:20]]
            logger.info(f"Reranked to {len(search_results)} results")
        
        if not search_results:
            logger.warning("No results after reranking")
            return jsonify({
                "response": "No relevant information found in the database after reranking."
            })
        
        # Extract identifiers directly from Qdrant payloads (NO LLM)
        all_identifiers = []
        for res in search_results:
            ids = extract_identifiers_from_payload(res.payload)
            if ids:
                all_identifiers.append(ids)
        
        # Format retrieved context
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
        logger.info(f"Context prepared: {len(formatted_context)} characters from {len(contexts)} documents")
        
        # Use LLM ONLY for formatting/summarizing the retrieved text
        # NOT for generating identifiers like Site ID, Cell ID, etc.
        system_instruction = f"""
You are a text formatter. Your ONLY job is to organize and present the retrieved documents clearly.

STRICT RULES:
1. ONLY quote text that exists in the provided documents
2. DO NOT invent, generate, or fabricate any identifiers (Site ID, Cell ID, Cell Name, Sector Name, etc.)
3. If the documents don't contain the requested information, say "Not found in retrieved documents"
4. Format the existing text into readable paragraphs or bullet points

Source: {settings.vectorstore.collection_name}
"""

        prompt = ChatPromptTemplate.from_template("""
{system_instruction}

Retrieved Documents:
{context}

User Question: {input}

Format the relevant information from the documents above. DO NOT generate any new data:
""")
        
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({
            "system_instruction": system_instruction,
            "context": formatted_context,
            "input": query
        })
        
        # Clean markdown formatting
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
    """
    Endpoint returning structured JSON with tables.
    Data comes ONLY from Qdrant - LLM only formats into tables.
    """
    data = request.json
    query = data.get("message")
    
    if not query:
        return jsonify({"error": "No message provided"}), 400
    
    try:
        # Hybrid search from Qdrant
        q_vector = embeddings.embed_query(query)
        results = loop.run_until_complete(
            vector_store.hybrid_search_rrf(
                collection_name=settings.vectorstore.collection_name,
                query_vector=q_vector,
                query_text=query,
                limit=100
            )
        )
        
        logger.info(f"QDRANT RETRIEVAL (Hybrid): Found {len(results) if results else 0} documents")
        
        # If no results, return immediately - NO LLM generation
        if not results or len(results) == 0:
            logger.warning(f"No documents retrieved from Qdrant for query: {query}")
            return jsonify(TabularResponse(
                response="No relevant information found in the database. The query did not match any documents in the collection.",
                tables=None,
                has_tables=False
            ).model_dump())
        
        # Rerank
        search_results = []
        if results:
            pairs = [[query, res.payload.get("text") or str(res.payload)] for res in results]
            scores = reranker.predict(pairs)
            scored_results = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)
            search_results = [res for _, res in scored_results[:20]]
            logger.info(f"Reranked to {len(search_results)} results")
        
        if not search_results:
            logger.warning("No results after reranking")
            return jsonify(TabularResponse(
                response="No relevant information found in the database after reranking.",
                tables=None,
                has_tables=False
            ).model_dump())
        
        # ============================================
        # EXTRACT DATA DIRECTLY FROM QDRANT PAYLOADS
        # NO LLM GENERATION OF IDENTIFIERS
        # ============================================
        
        # Extract hierarchy (Site ID, Cell Name, Sector Name) directly from payloads
        hierarchy_data = extract_hierarchy_from_results(search_results)
        
        # Extract all identifiers from payloads
        all_identifiers = []
        for res in search_results:
            ids = extract_identifiers_from_payload(res.payload)
            if ids:
                all_identifiers.append(ids)
        
        # Extract metrics from payloads
        all_metrics = []
        for res in search_results:
            metrics = extract_metrics_from_payload(res.payload)
            if metrics:
                ids = extract_identifiers_from_payload(res.payload)
                all_metrics.append({**ids, **metrics})
        
        # Format context for LLM (only for text formatting, not ID generation)
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
        logger.info(f"Context prepared: {len(formatted_context)} characters")
        
        # LLM prompt - ONLY for formatting text content into structured tables
        # NOT for generating identifiers
        prompt_template = """
You are a table formatter. Extract KPIs, parameters, and technical details from the documents and format them.

STRICT RULES:
1. Extract ONLY items that exist in the provided documents
2. DO NOT generate, invent, or fabricate any data
3. For each KPI/parameter, use the EXACT name as it appears in the documents
4. If information is not in the documents, use "N/A"

Documents:
{context}

Question: {input}

Return a JSON with:
- kpis: Array of {{number, name, description, formula_calculation, relationship_impact}}
- parameters: Array of {{number, name, description, formula_calculation, relationship_impact}}
- reasoning_kpis: Brief explanation of listed KPIs (from documents only)
- reasoning_parameters: Brief explanation of listed parameters (from documents only)
- technical_details: Technical context from documents
- constraints_notes: Any limitations mentioned in documents

JSON:
"""
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | llm | StrOutputParser()
        
        llm_response = chain.invoke({"context": formatted_context, "input": query})
        logger.info(f"LLM Response (first 300 chars): {llm_response[:300]}")
        
        # Parse JSON
        json_data = extract_json(llm_response)
        
        if not json_data:
            code_blocks = re.findall(r'```(?:json)?\s*\n?(.*?)\n?```', llm_response, re.DOTALL)
            for block in code_blocks:
                try:
                    json_data = json.loads(block.strip())
                    break
                except json.JSONDecodeError:
                    continue
        
        # Ensure required fields exist
        if 'kpis' not in json_data:
            json_data['kpis'] = []
        if 'parameters' not in json_data:
            json_data['parameters'] = []
        
        tables = []
        response_parts = []
        
        # Parse structured response
        try:
            structured_result = SeparateTablesResponse(**json_data)
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            structured_result = SeparateTablesResponse(kpis=[], parameters=[], constraints_notes=str(e))
        
        # ============================================
        # BUILD TABLES FROM QDRANT DATA DIRECTLY
        # ============================================
        
        # Create Hierarchy table from QDRANT DATA (not LLM generated)
        if hierarchy_data:
            tables.insert(0, TableData(
                title="Site-Sector-Cell Hierarchy (from Database)",
                headers=["SITEID", "Sector Name", "cellName"],
                rows=[[h['site_id'], h['sector_name'], h['cell_name']] for h in hierarchy_data]
            ))
        
        # Create Cell Metrics table from QDRANT DATA (not LLM generated)
        if all_metrics:
            metric_rows = []
            for m in all_metrics:
                row = [
                    m.get('site_id', 'N/A'),
                    m.get('cell_name', 'N/A'),
                    str(m.get('dl_throughput', 'N/A')),
                    str(m.get('ho_success_rate', 'N/A')),
                    str(m.get('congestion', 'N/A')),
                    str(m.get('spectral_efficiency', 'N/A')),
                ]
                metric_rows.append(row)
            
            if metric_rows:
                tables.append(TableData(
                    title="Cell Performance Metrics (from Database)",
                    headers=["SITEID", "Cell Name", "DL Throughput", "HO Success Rate", "Congestion", "Spectral Efficiency"],
                    rows=metric_rows
                ))
        
        # Create KPIs table (LLM formatted from document text)
        if structured_result.kpis:
            tables.append(TableData(
                title="KPIs from Retrieved Documents",
                headers=["#", "Name", "Description", "Formula/Calculation", "Relationship/Impact"],
                rows=[[str(k.number), clean_markdown(k.name), clean_markdown(k.description),
                       clean_markdown(k.formula_calculation), clean_markdown(k.relationship_impact)]
                      for k in structured_result.kpis]
            ))
        
        # Create Parameters table (LLM formatted from document text)
        if structured_result.parameters:
            tables.append(TableData(
                title="Parameters from Retrieved Documents",
                headers=["#", "Name", "Description", "Formula/Calculation", "Relationship/Impact"],
                rows=[[str(p.number), clean_markdown(p.name), clean_markdown(p.description),
                       clean_markdown(p.formula_calculation), clean_markdown(p.relationship_impact)]
                      for p in structured_result.parameters]
            ))
        
        # Create Identifiers table from QDRANT DATA (not LLM generated)
        if all_identifiers:
            # Deduplicate and format identifiers
            seen_ids = set()
            unique_identifiers = []
            for ids in all_identifiers:
                key = json.dumps(ids, sort_keys=True)
                if key not in seen_ids:
                    seen_ids.add(key)
                    unique_identifiers.append(ids)
            
            if unique_identifiers:
                id_rows = []
                for idx, ids in enumerate(unique_identifiers[:20], 1):  # Limit to 20 rows
                    id_rows.append([
                        str(idx),
                        ids.get('site_id', 'N/A'),
                        ids.get('cell_name', 'N/A'),
                        ids.get('sector_name', 'N/A'),
                        ids.get('market', 'N/A')
                    ])
                
                tables.append(TableData(
                    title="Network Identifiers (from Database)",
                    headers=["#", "Site ID", "Cell Name", "Sector Name", "Market"],
                    rows=id_rows
                ))
        
        # Create Counters table (LLM formatted from document text)
        if structured_result.counters:
            tables.append(TableData(
                title="Counters from Retrieved Documents",
                headers=["#", "Name", "Description", "Formula/Calculation", "Relationship/Impact"],
                rows=[[str(c.number), clean_markdown(c.name), clean_markdown(c.description),
                       clean_markdown(c.formula_calculation), clean_markdown(c.relationship_impact)]
                      for c in structured_result.counters]
            ))
        
        # Build response text
        if tables:
            response_parts.append("Based on the retrieved data from the database:\n")
        
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
        
        response_parts.append(f"\n---\n*Source: {settings.vectorstore.collection_name} | All identifiers extracted directly from database*")
        
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
