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

# Load settings
settings = get_settings()

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
    site_id: str = Field(description="Site ID from context")
    sector_name: str = Field(description="Sector Name from context")
    cell_name: str = Field(description="Cell Name from context")

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
        
        logger.info(f"📊 QDRANT RETRIEVAL: Found {len(results) if results else 0} documents from collection '{settings.vectorstore.collection_name}'")
        print(f"📊 QDRANT RETRIEVAL: Found {len(results) if results else 0} documents")
        
        # If no results, return immediately - NO LLM generation
        if not results or len(results) == 0:
            logger.warning(f"⚠️ No documents retrieved from Qdrant for query: {query}")
            print(f"⚠️ No documents retrieved from Qdrant")
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
            logger.info(f"✅ Reranked to {len(search_results)} results")
            print(f"✅ Reranked to {len(search_results)} results")
        
        # Double-check after reranking
        if not search_results or len(search_results) == 0:
            logger.warning(f"⚠️ No results after reranking")
            print(f"⚠️ No results after reranking")
            return jsonify({
                "response": "No relevant information found in the database after reranking."
            })
        
        # Format retrieved context only
        contexts = []
        for idx, res in enumerate(search_results, 1):
            content = res.payload.get("text") or res.payload.get("content") or ""
            metadata = {k.capitalize(): res.payload[k] for k in ["type", "category", "source"] if k in res.payload}
            context_block = f"[Document {idx}]"
            if metadata:
                context_block += "\n" + " | ".join([f"{k}: {v}" for k, v in metadata.items()])
            context_block += f"\n{content}"
            contexts.append(context_block)
            
            # Print first 200 chars of each retrieved document
            print(f"  Document {idx}: {content[:200]}...")
        
        formatted_context = "\n\n---\n\n".join(contexts)
        logger.info(f"📄 Context prepared: {len(formatted_context)} characters from {len(contexts)} documents")
        print(f"📄 Context prepared: {len(formatted_context)} characters from {len(contexts)} documents")
        print(f"📄 First 500 chars of context: {formatted_context[:500]}")
        
        # Use LLM ONLY for formatting the retrieved data
        system_instruction = f"""
You must ONLY quote text that exists in the documents below. You cannot generate, invent, or create any identifiers, names, or numbers.
If the documents don't contain the requested information, say "No data found in documents."

Source: {settings.vectorstore.collection_name}
"""

        prompt = ChatPromptTemplate.from_template("""
{system_instruction}

Documents:
{context}

Question: {input}

Answer using ONLY text from the documents above:
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
    """Endpoint returning structured JSON with tables from retrieved data only."""
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
        
        logger.info(f"📊 QDRANT RETRIEVAL (Hybrid): Found {len(results) if results else 0} documents from collection '{settings.vectorstore.collection_name}'")
        print(f"📊 QDRANT RETRIEVAL (Hybrid): Found {len(results) if results else 0} documents")
        
        # If no results, return immediately - NO LLM generation
        if not results or len(results) == 0:
            logger.warning(f"⚠️ No documents retrieved from Qdrant for query: {query}")
            print(f"⚠️ No documents retrieved from Qdrant")
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
            logger.info(f"✅ Reranked to {len(search_results)} results")
            print(f"✅ Reranked to {len(search_results)} results")
            logger.info(f"Top reranking scores: {[float(s) for s, _ in scored_results[:5]]}")
        
        # Double-check after reranking
        if not search_results or len(search_results) == 0:
            logger.warning(f"⚠️ No results after reranking")
            print(f"⚠️ No results after reranking")
            return jsonify(TabularResponse(
                response="No relevant information found in the database after reranking.",
                tables=None,
                has_tables=False
            ).model_dump())
        
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
            
            # Print first 200 chars of each retrieved document
            print(f"  Document {idx}: {content[:200]}...")
        
        formatted_context = "\n\n---\n\n".join(contexts)
        logger.info(f"📄 Context prepared: {len(formatted_context)} characters from {len(contexts)} documents")
        print(f"📄 Context prepared: {len(formatted_context)} characters from {len(contexts)} documents")
        
        # Single prompt - LLM only formats retrieved data into structured JSON
        prompt_template = """
Extract data from the context below and return as JSON. Only use values from the context.

CONTEXT: {context}
QUESTION: {input}

Return JSON:
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
        
        # Parse structured response
        try:
            structured_result = SeparateTablesResponse(**json_data)
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            structured_result = SeparateTablesResponse(kpis=[], parameters=[], constraints_notes=str(e))
        
        # Create Hierarchy table (if exists in retrieved data)
        if structured_result.hierarchy:
            hier_rows = []
            for row in structured_result.hierarchy:
                hier_rows.append([
                    clean_markdown(row.site_id).strip(),
                    clean_markdown(row.sector_name).strip(),
                    clean_markdown(row.cell_name).strip()
                ])
            
            if hier_rows:
                tables.insert(0, TableData(
                    title="Site-Sector-Cell Hierarchy",
                    headers=["SITEID", "Sector Name", "cellName"],
                    rows=hier_rows
                ))
        
        # Create KPIs table (from retrieved data)
        if structured_result.kpis:
            tables.append(TableData(
                title="KPIs from Retrieved Data",
                headers=["#", "Name", "Description", "Formula/Calculation", "Relationship/Impact"],
                rows=[[str(k.number), clean_markdown(k.name), clean_markdown(k.description),
                       clean_markdown(k.formula_calculation), clean_markdown(k.relationship_impact)]
                      for k in structured_result.kpis]
            ))
        
        # Create Parameters table (from retrieved data)
        if structured_result.parameters:
            tables.append(TableData(
                title="Parameters from Retrieved Data",
                headers=["#", "Name", "Description", "Formula/Calculation", "Relationship/Impact"],
                rows=[[str(p.number), clean_markdown(p.name), clean_markdown(p.description),
                       clean_markdown(p.formula_calculation), clean_markdown(p.relationship_impact)]
                      for p in structured_result.parameters]
            ))
        
        # Create Identifiers table (from retrieved data)
        if structured_result.identifiers:
            tables.append(TableData(
                title="Identifiers from Retrieved Data",
                headers=["#", "Identifier Name", "Value", "Description", "Level/Notes"],
                rows=[[str(i.number), clean_markdown(i.name), clean_markdown(i.formula_calculation),
                       clean_markdown(i.description), clean_markdown(i.relationship_impact)]
                      for i in structured_result.identifiers]
            ))
        
        # Create Counters table (from retrieved data)
        if structured_result.counters:
            tables.append(TableData(
                title="Counters from Retrieved Data",
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
